"""
Detection similarity metric.

Computes pairwise similarity between target detection vectors
and proposal detection vectors using three components:
    s(d_t, d_j) = s_L * s_P * s_O

where:
    s_L = IoU(L_t, L_j)          -- localization similarity
    s_P = cosine_sim(P_t, P_j)   -- classification similarity
    s_O = O_j                    -- objectness score
"""


import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


class DetectionSimilarity:
    """
    Computes similarity between target and proposal detection vectors.

    The similarity metric accounts for localization (IoU),
    classification (cosine similarity of class probabilities),
    and objectness score components.
    """

    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Computes IoU between two sets of bounding boxes.

        Args:
            boxes1: (N, 4) tensor in [x1, y1, x2, y2] format.
            boxes2: (M, 4) tensor in [x1, y1, x2, y2] format.

        Returns
        -------
            iou_matrix: (N, M) IoU matrix.
        """
        return box_iou(boxes1, boxes2)

    @staticmethod
    def compute_class_similarity(probs1: torch.Tensor, probs2: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between class probability distributions.

        Args:
            probs1: (N, C) class probabilities for targets.
            probs2: (M, C) class probabilities for proposals.

        Returns
        -------
            sim_matrix: (N, M) cosine similarity matrix.
        """
        norm1 = F.normalize(probs1, p=2, dim=1)
        norm2 = F.normalize(probs2, p=2, dim=1)
        return torch.mm(norm1, norm2.t())

    @staticmethod
    def compute_similarity(
        target_boxes: torch.Tensor,
        target_probs: torch.Tensor,
        proposal_boxes: torch.Tensor,
        proposal_probs: torch.Tensor,
        proposal_objectness: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the full similarity metric s(d_t, d_j).

        The scalar product of the three components is used to model
        a logical AND: if any single component is low, the total
        similarity is also low.

        Args:
            target_boxes: (T, 4) bounding boxes of target detections.
            target_probs: (T, C) one-hot class probability vectors.
            proposal_boxes: (P, 4) bounding boxes of proposals.
            proposal_probs: (P, C) class probabilities of proposals.
            proposal_objectness: (P,) objectness scores (optional,
                used for detectors like YOLO that produce them).

        Returns
        -------
            similarity: (T, P) similarity matrix.
        """
        s_L = DetectionSimilarity.compute_iou(target_boxes, proposal_boxes)
        s_P = DetectionSimilarity.compute_class_similarity(target_probs, proposal_probs)

        similarity = s_L * s_P

        if proposal_objectness is not None:
            similarity = similarity * proposal_objectness.unsqueeze(0)

        return similarity
