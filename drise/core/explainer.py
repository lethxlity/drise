"""
D-RISE explainer.

Core algorithm that generates saliency maps for object detector
predictions by:
    1. Generating random masks.
    2. Running the detector on masked images.
    3. Computing similarity between target detections and proposals.
    4. Computing a weighted sum of masks to produce saliency maps.

The method is black-box: it only requires access to the detector's
inputs and outputs, not its weights, gradients, or architecture.
"""

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .detection import Detection
from .masks import MaskGenerator
from .similarity import DetectionSimilarity
from .wrappers.base import DetectorWrapper


class DRISE:
    """
    D-RISE: Detector Randomized Input Sampling for Explanation.

    Generates saliency maps that highlight image regions most
    important for a detector's prediction of a specific detection
    (bounding box + category).
    """

    def __init__(
        self,
        detector: DetectorWrapper,
        num_masks: int = 5000,
        mask_res: tuple[int, int] = (16, 16),
        mask_prob: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            detector: Detector wrapper implementing the detect() interface.
            num_masks: Number of random masks N to sample.
            mask_res: Low-resolution mask dimensions (h, w).
            mask_prob: Probability of preserving a pixel p.
            device: Device for computation ('cpu' or 'cuda').
        """
        self.detector = detector
        self.num_masks = num_masks
        self.mask_res = mask_res
        self.mask_prob = mask_prob
        self.device = device

    @staticmethod
    def _create_target_vector(
        box: torch.Tensor,
        label: int,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a target detection vector d_t with a one-hot
        class probability vector and objectness set to 1.

        Args:
            box: (4,) bounding box [x1, y1, x2, y2].
            label: Class index.
            num_classes: Total number of classes C.

        Returns
        -------
            target_box: (1, 4) bounding box tensor.
            target_probs: (1, C) one-hot probability vector.
        """
        target_box = box.unsqueeze(0).float()
        target_probs = torch.zeros(1, num_classes)
        target_probs[0, label] = 1.0
        return target_box, target_probs

    @staticmethod
    def _detections_to_tensors(
        detections: list[Detection],
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Converts a list of Detection objects into batched tensors.

        Args:
            detections: List of Detection objects.
            num_classes: Number of classes C.

        Returns
        -------
            boxes: (N, 4) bounding boxes.
            probs: (N, C) class probabilities.
            objectness: (N,) objectness scores, or None.
        """
        if len(detections) == 0:
            return (
                torch.zeros(0, 4),
                torch.zeros(0, num_classes),
                torch.zeros(0),
            )

        boxes = torch.stack([d.box for d in detections])
        probs = torch.stack([d.class_probs for d in detections])

        if detections[0].objectness is not None:
            objectness = torch.tensor([d.objectness for d in detections])
        else:
            objectness = None

        return boxes, probs, objectness

    @staticmethod
    def _compute_mask_weight(
        target_boxes: torch.Tensor,
        target_probs: torch.Tensor,
        proposal_boxes: torch.Tensor,
        proposal_probs: torch.Tensor,
        proposal_objectness: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Computes the mask weight for each target detection.

        For each target, the weight is the maximum similarity
        between the target detection vector and all proposal
        vectors produced for the masked image:

            S(d_t, f(M_i * I)) = max_{d_j in f(M_i * I)} s(d_t, d_j)

        Args:
            target_boxes: (T, 4) target bounding boxes.
            target_probs: (T, C) target class probabilities (one-hot).
            proposal_boxes: (P, 4) proposal bounding boxes.
            proposal_probs: (P, C) proposal class probabilities.
            proposal_objectness: (P,) objectness scores, or None.

        Returns
        -------
            weights: (T,) maximum similarity for each target.
        """
        if proposal_boxes.shape[0] == 0:
            return torch.zeros(target_boxes.shape[0])

        sim_matrix = DetectionSimilarity.compute_similarity(
            target_boxes,
            target_probs,
            proposal_boxes,
            proposal_probs,
            proposal_objectness,
        )

        weights, _ = sim_matrix.max(dim=1)
        return weights

    def explain(
        self,
        image: torch.Tensor,
        target_detections: list[dict[str, Any]],
        num_classes: int = 91,
        verbose: bool = True,
    ) -> list[np.ndarray]:
        """
        Generates saliency maps for the given target detections.

        The target detections do not have to come from the detector
        itself. This allows explaining missed detections or
        arbitrary hypothetical detection vectors.

        Args:
            image: (3, H, W) image tensor with values in [0, 1].
            target_detections: List of dicts, each with keys:
                - 'box': torch.Tensor of shape (4,) [x1, y1, x2, y2]
                - 'label': int (class index)
            num_classes: Number of classes in the detector.
            verbose: Whether to show a progress bar.

        Returns
        -------
            saliency_maps: List of numpy arrays of shape (H, W),
                one per target detection, with values in [0, 1].
        """
        _, H, W = image.shape
        T = len(target_detections)

        # Step 1: Build target detection vectors
        target_boxes_list = []
        target_probs_list = []
        for td in target_detections:
            tb, tp = self._create_target_vector(td["box"], td["label"], num_classes)
            target_boxes_list.append(tb)
            target_probs_list.append(tp)

        target_boxes = torch.cat(target_boxes_list, dim=0)  # (T, 4)
        target_probs = torch.cat(target_probs_list, dim=0)  # (T, C)

        # Step 2: Generate random masks
        mask_gen = MaskGenerator(
            image_height=H,
            image_width=W,
            mask_height=self.mask_res[0],
            mask_width=self.mask_res[1],
            prob=self.mask_prob,
        )
        masks = mask_gen.generate(self.num_masks, device="cpu")  # (N, 1, H, W)

        # Step 3: Run detector on each masked image and compute weights
        all_weights = torch.zeros(self.num_masks, T)

        iterator = range(self.num_masks)
        if verbose:
            iterator = tqdm(iterator, desc="D-RISE: processing masks")

        for i in iterator:
            mask_i = masks[i]  # (1, H, W)
            masked_image = image * mask_i  # (3, H, W)

            proposals = self.detector.detect(masked_image)

            prop_boxes, prop_probs, prop_obj = self._detections_to_tensors(proposals, num_classes)

            weights = self._compute_mask_weight(
                target_boxes,
                target_probs,
                prop_boxes,
                prop_probs,
                prop_obj,
            )

            all_weights[i] = weights

        # Step 4: Compute weighted sum of masks
        masks_2d = masks.squeeze(1)  # (N, H, W)

        saliency_maps = []
        for t in range(T):
            w = all_weights[:, t]  # (N,)
            saliency = torch.einsum("n,nhw->hw", w, masks_2d)  # (H, W)

            # Min-max normalization
            saliency_np = saliency.numpy()
            s_min, s_max = saliency_np.min(), saliency_np.max()
            if s_max > s_min:
                saliency_np = (saliency_np - s_min) / (s_max - s_min)

            saliency_maps.append(saliency_np)

        return saliency_maps
