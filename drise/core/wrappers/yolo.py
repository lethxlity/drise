"""
Wrapper for YOLO models (ultralytics).

YOLO is a one-stage detector that produces an explicit objectness
score, which is incorporated into the similarity metric.

IMPORTANT: Ultralytics YOLO applies its own internal confidence
threshold (default 0.25) during post-processing. For D-RISE to
work correctly, this threshold must be overridden by passing the
desired value directly to the model call. Otherwise, proposals
from masked images (which naturally have lower confidence) are
silently discarded inside the model, and the wrapper's own
score_threshold has no effect.
"""


import numpy as np
import torch

from ..detection import Detection
from .base import DetectorWrapper


class YOLOWrapper(DetectorWrapper):
    """
    Wrapper for YOLO models from the ultralytics library.

    Converts the model's output into a list of Detection objects.
    The objectness score is included since YOLO produces one.
    """

    def __init__(
        self,
        model,
        device: str = "cpu",
        score_threshold: float = 0.5,
        num_classes: int = 80,
        nms_iou_threshold: float = 0.7,
        max_detections: int = 300,
    ):
        """
        Args:
            model: A YOLO model instance from ultralytics.
            device: Device string ('cpu' or 'cuda').
            score_threshold: Confidence threshold passed directly
                to the YOLO model's internal post-processing.
                For D-RISE explanation, use a low value (e.g. 0.01).
                For displaying final detections, use a higher value
                (e.g. 0.5).
            num_classes: Number of classes (80 for COCO).
            nms_iou_threshold: IoU threshold for NMS inside the model.
            max_detections: Maximum number of detections per image.
        """
        self.model = model
        self.device = device
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> list[Detection]:
        """
        Runs YOLO on an image.

        The score_threshold is passed directly to the model's
        inference call to override the internal default (0.25).
        Without this, the model silently discards low-confidence
        proposals that are essential for D-RISE to work correctly
        on masked images.

        Args:
            image: (3, H, W) tensor with values in [0, 1].

        Returns
        -------
            detections: List of Detection objects.
        """
        # Convert tensor to numpy array for ultralytics API
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Pass conf and iou thresholds directly to the model.
        # This overrides the model's internal defaults (conf=0.25,
        # iou=0.7) and is the ONLY way to control filtering —
        # our wrapper cannot recover proposals already discarded
        # inside the model.
        results = self.model(
            img_np,
            verbose=False,
            conf=self.score_threshold,
            iou=self.nms_iou_threshold,
            max_det=self.max_detections,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):
                box = result.boxes.xyxy[i].cpu()
                conf = result.boxes.conf[i].item()
                cls = int(result.boxes.cls[i].item())

                class_probs = torch.zeros(self.num_classes)
                class_probs[cls] = conf

                det = Detection(
                    box=box,
                    class_probs=class_probs,
                    objectness=conf,
                    label=cls,
                    score=conf,
                )
                detections.append(det)

        return detections
