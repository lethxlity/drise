"""
Wrapper for torchvision Faster R-CNN models.

Faster R-CNN is a two-stage detector that does not produce
an explicit objectness score. Instead, the detection confidence
is incorporated into the class probability vector to ensure
the similarity metric accounts for detection certainty.
"""


import torch

from ..detection import Detection
from .base import DetectorWrapper


class TorchvisionFasterRCNNWrapper(DetectorWrapper):
    """
    Wrapper for Faster R-CNN from torchvision.

    Converts the model's output into a list of Detection objects.
    Since Faster R-CNN does not produce an explicit objectness
    score, the confidence is embedded in the class probability
    vector to provide a smooth similarity signal.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        score_threshold: float = 0.0,
        num_classes: int = 91,
    ):
        """
        Args:
            model: A torchvision Faster R-CNN model instance.
            device: Device to run inference on ('cpu' or 'cuda').
            score_threshold: Minimum confidence threshold for detections.
                For D-RISE explanation, use a low value (e.g. 0.0)
                to capture all proposals.
            num_classes: Number of classes (91 for COCO in torchvision).
        """
        self.model = model
        self.device = device
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> list[Detection]:
        """
        Runs Faster R-CNN on an image.

        Args:
            image: (3, H, W) tensor with values in [0, 1].

        Returns
        -------
            detections: List of Detection objects.
        """
        image = image.to(self.device)
        outputs = self.model([image])[0]

        detections = []
        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        for i in range(len(boxes)):
            if scores[i] < self.score_threshold:
                continue

            # Build a class probability vector with the confidence
            # placed at the predicted class index
            class_probs = torch.zeros(self.num_classes, device=self.device)
            class_probs[labels[i]] = scores[i].item()

            det = Detection(
                box=boxes[i].cpu(),
                class_probs=class_probs.cpu(),
                # Use confidence as objectness so the similarity metric
                # can weight proposals by their detection certainty.
                # Without this, Faster R-CNN similarity is just IoU
                # (binary class match), losing the confidence signal.
                objectness=scores[i].item(),
                label=labels[i].item(),
                score=scores[i].item(),
            )
            detections.append(det)

        return detections
