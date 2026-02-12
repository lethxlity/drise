"""
Base class for detector wrappers.

Any object detector can be integrated with D-RISE by subclassing
DetectorWrapper and implementing the detect() method.
"""

import torch

from ..detection import Detection


class DetectorWrapper:
    """
    Abstract base class for object detector wrappers.

    Subclasses must implement the detect() method, which takes
    an image tensor and returns a list of Detection objects.
    This is the only interface that D-RISE requires, making
    it truly black-box with respect to the detector.
    """

    def detect(self, image: torch.Tensor) -> list[Detection]:
        """
        Runs the detector on an image.

        Args:
            image: (3, H, W) image tensor with values in [0, 1].

        Returns
        -------
            detections: List of Detection objects.
        """
        raise NotImplementedError("Subclasses must implement the detect() method.")
