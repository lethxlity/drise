"""
Average saliency map computation.

Computes per-category average saliency maps by cropping,
resizing, and averaging individual saliency maps across
many detections. This provides a holistic view of what
regions the model typically relies on for each class.
"""


import numpy as np
from PIL import Image as PILImage


class AverageSaliencyComputer:
    """
    Computes average saliency maps per object category.

    Individual saliency maps are cropped around the detection
    bounding box (with surrounding context), resized to a
    common size, and averaged. This reveals common patterns
    in how the model uses different image regions for each class.
    """

    def __init__(self, output_size: tuple[int, int] = (128, 128)):
        """
        Args:
            output_size: Common (H, W) size for resizing cropped maps.
        """
        self.output_size = output_size
        self.accumulators: dict[int, list[np.ndarray]] = {}

    def add(
        self,
        saliency_map: np.ndarray,
        bbox: list[float],
        label: int,
        context_factor: float = 1.5,
    ):
        """
        Adds a saliency map to the accumulator for the given class.

        The saliency map is cropped around the bounding box with
        a context margin, resized, and stored for later averaging.

        Args:
            saliency_map: (H, W) full saliency map.
            bbox: Bounding box [x1, y1, x2, y2].
            label: Class index.
            context_factor: Factor by which to expand the crop
                beyond the bounding box (1.0 = no context).
        """
        H, W = saliency_map.shape
        x1, y1, x2, y2 = bbox

        # Expand the bounding box by the context factor
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = bw * context_factor
        new_h = bh * context_factor

        crop_x1 = max(0, int(cx - new_w / 2))
        crop_y1 = max(0, int(cy - new_h / 2))
        crop_x2 = min(W, int(cx + new_w / 2))
        crop_y2 = min(H, int(cy + new_h / 2))

        cropped = saliency_map[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped.size == 0:
            return

        # Resize to the common output size
        cropped_pil = PILImage.fromarray((cropped * 255).astype(np.uint8)).resize(
            (self.output_size[1], self.output_size[0]),
            PILImage.BILINEAR,
        )
        resized = np.array(cropped_pil).astype(np.float32) / 255.0

        if label not in self.accumulators:
            self.accumulators[label] = []
        self.accumulators[label].append(resized)

    def compute_average(self, label: int) -> np.ndarray | None:
        """
        Computes the average saliency map for a given class.

        Args:
            label: Class index.

        Returns
        -------
            average_map: (H, W) averaged saliency map, or None
                if no maps have been accumulated for this class.
        """
        if label not in self.accumulators or len(self.accumulators[label]) == 0:
            return None
        return np.mean(self.accumulators[label], axis=0)

    def compute_all_averages(self) -> dict[int, np.ndarray]:
        """
        Computes average saliency maps for all accumulated classes.

        Returns
        -------
            averages: Dict mapping class index to (H, W) average map.
        """
        averages = {}
        for label in self.accumulators:
            avg = self.compute_average(label)
            if avg is not None:
                averages[label] = avg
        return averages

    @property
    def class_counts(self) -> dict[int, int]:
        """Returns the number of accumulated maps per class."""
        return {label: len(maps) for label, maps in self.accumulators.items()}
