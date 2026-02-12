"""
Detection data structure.

Represents a single object detection consisting of:
    - Bounding box coordinates [x1, y1, x2, y2]
    - Class probability distribution
    - Objectness score (optional, used by detectors like YOLO)
    - Class label index
    - Confidence score
"""

from dataclasses import dataclass

import torch


@dataclass
class Detection:
    """
    Represents a single detection from an object detector.

    This structure encodes the detection vector d_i = [L_i, O_i, P_i]
    as defined in the paper, where:
        - L_i: localization info (bounding box corners)
        - O_i: objectness score
        - P_i: class probability vector
    """

    box: torch.Tensor
    """Bounding box coordinates (4,) in [x1, y1, x2, y2] format."""

    class_probs: torch.Tensor
    """Class probability distribution (C,)."""

    objectness: float | None = None
    """Objectness score in [0, 1]. None if the detector does not produce one."""

    label: int | None = None
    """Predicted class index."""

    score: float | None = None
    """Detection confidence score."""
