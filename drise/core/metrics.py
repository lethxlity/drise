"""
Evaluation metrics for saliency maps.

Implements three metrics from the paper:
    - Pointing Game: checks if the maximum saliency point
      falls within the ground truth region.
    - Deletion: measures how fast similarity drops when
      removing pixels in decreasing order of saliency.
    - Insertion: measures how fast similarity rises when
      adding pixels in decreasing order of saliency.
"""


import numpy as np
import torch

from .similarity import DetectionSimilarity
from .wrappers.base import DetectorWrapper


class DRISEMetrics:
    """Evaluation metrics for D-RISE saliency maps."""

    @staticmethod
    def pointing_game(
        saliency_map: np.ndarray,
        gt_mask: np.ndarray,
    ) -> bool:
        """
        Pointing Game metric with a segmentation mask.

        A hit is scored if the point of maximum saliency lies
        within the ground truth object segmentation mask.

        Args:
            saliency_map: (H, W) saliency map.
            gt_mask: (H, W) binary ground truth mask.

        Returns
        -------
            hit: True if the maximum saliency point is inside the mask.
        """
        max_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        return bool(gt_mask[max_idx] > 0)

    @staticmethod
    def pointing_game_bbox(
        saliency_map: np.ndarray,
        bbox: list[float],
    ) -> bool:
        """
        Pointing Game metric with a bounding box.

        A hit is scored if the point of maximum saliency lies
        within the ground truth bounding box.

        Args:
            saliency_map: (H, W) saliency map.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns
        -------
            hit: True if the maximum saliency point is inside the box.
        """
        max_pos = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        y, x = max_pos
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def deletion_insertion(
        image: torch.Tensor,
        saliency_map: np.ndarray,
        detector: DetectorWrapper,
        target_box: torch.Tensor,
        target_label: int,
        num_classes: int = 91,
        num_steps: int = 100,
        mode: str = "deletion",
    ) -> tuple[list[float], float]:
        """
        Deletion or Insertion metric.

        Deletion: Sequentially removes pixels in order of decreasing
        saliency and measures how quickly the detector's output
        similarity drops. Lower AUC is better.

        Insertion: Sequentially adds pixels in order of decreasing
        saliency to a blank image and measures how quickly the
        similarity rises. Higher AUC is better.

        Args:
            image: (3, H, W) image tensor in [0, 1].
            saliency_map: (H, W) saliency map.
            detector: Detector wrapper.
            target_box: (4,) target bounding box.
            target_label: Target class index.
            num_classes: Number of classes.
            num_steps: Number of evaluation steps.
            mode: Either 'deletion' or 'insertion'.

        Returns
        -------
            scores: List of similarity scores at each step.
            auc: Area under the curve.
        """
        _, H, W = image.shape
        total_pixels = H * W

        # Sort pixel indices by decreasing saliency
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(-flat_saliency)

        # Build target vectors
        target_box_t = target_box.unsqueeze(0).float()
        target_probs = torch.zeros(1, num_classes)
        target_probs[0, target_label] = 1.0

        step_size = max(total_pixels // num_steps, 1)
        scores = []

        for step in range(num_steps + 1):
            n_pixels = min(step * step_size, total_pixels)

            if mode == "deletion":
                # Remove the n_pixels most salient pixels
                modified = image.clone().view(3, -1)
                if n_pixels > 0:
                    indices = sorted_indices[:n_pixels]
                    modified[:, indices] = 0
                modified = modified.view(3, H, W)

            elif mode == "insertion":
                # Add the n_pixels most salient pixels to a blank image
                modified = torch.zeros_like(image).view(3, -1)
                if n_pixels > 0:
                    indices = sorted_indices[:n_pixels]
                    modified[:, indices] = image.view(3, -1)[:, indices]
                modified = modified.view(3, H, W)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'deletion' or 'insertion'.")

            # Run detector on the modified image
            proposals = detector.detect(modified)

            if len(proposals) == 0:
                scores.append(0.0)
                continue

            # Compute maximum similarity with the target
            prop_boxes = torch.stack([d.box for d in proposals])
            prop_probs = torch.stack([d.class_probs for d in proposals])
            prop_obj = None
            if proposals[0].objectness is not None:
                prop_obj = torch.tensor([d.objectness for d in proposals])

            sim = DetectionSimilarity.compute_similarity(
                target_box_t,
                target_probs,
                prop_boxes,
                prop_probs,
                prop_obj,
            )
            max_sim = sim.max().item()
            scores.append(max_sim)

        # Compute AUC using the trapezoidal rule
        auc = float(np.trapz(scores, dx=1.0 / num_steps))

        return scores, auc
