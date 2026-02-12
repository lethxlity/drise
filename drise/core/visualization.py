"""
Visualization utilities for D-RISE saliency maps.

Provides functions to overlay saliency heatmaps on images,
draw bounding boxes, and create publication-quality figures.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

# COCO class names for torchvision Faster R-CNN (91 categories)
COCO_CLASSES_91 = [
    "_background_",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "trafficlight",
    "firehydrant",
    "streetsign",
    "stopsign",
    "parkingmeter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eyeglasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sportsball",
    "kite",
    "baseballbat",
    "baseballglove",
    "skateboard",
    "surfboard",
    "tennisracket",
    "bottle",
    "plate",
    "wineglass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "mirror",
    "diningtable",
    "window",
    "desk",
    "toilet",
    "door",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cellphone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddybear",
    "hairdrier",
    "toothbrush",
    "hairbrush",
]


class DRISEVisualizer:
    """Visualization utilities for D-RISE explanations."""

    @staticmethod
    def overlay_saliency(
        image: np.ndarray,
        saliency_map: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlays a saliency heatmap on an image.

        Args:
            image: (H, W, 3) RGB image with values in [0, 255].
            saliency_map: (H, W) normalized saliency map in [0, 1].
            alpha: Blending factor for the overlay.
            colormap: Matplotlib colormap name.

        Returns
        -------
            overlay: (H, W, 3) blended image with values in [0, 255].
        """
        cmap = plt.get_cmap(colormap)
        heatmap = cmap(saliency_map)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        image_float = image.astype(np.float32)
        heatmap_float = heatmap.astype(np.float32)

        overlay = (1 - alpha) * image_float + alpha * heatmap_float
        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def visualize_explanations(
        image: torch.Tensor,
        target_detections: list[dict[str, Any]],
        saliency_maps: list[np.ndarray],
        class_names: list[str] | None = None,
        figsize: tuple[int, int] = (16, 5),
        save_path: str | None = None,
    ):
        """
        Creates a figure with the original image and saliency overlays.

        The leftmost panel shows the original image with bounding boxes.
        Subsequent panels show saliency maps overlaid on the image,
        one per target detection.

        Args:
            image: (3, H, W) image tensor in [0, 1].
            target_detections: List of dicts with 'box' and 'label' keys.
            saliency_maps: List of (H, W) saliency arrays.
            class_names: Optional list of class name strings.
            figsize: Figure size (width, height).
            save_path: If provided, saves the figure to this path.
        """
        if class_names is None:
            class_names = COCO_CLASSES_91

        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        T = len(target_detections)
        num_panels = T + 1

        fig, axes = plt.subplots(1, num_panels, figsize=figsize)
        if num_panels == 1:
            axes = [axes]

        # Panel 0: Original image with bounding boxes
        axes[0].imshow(img_np)
        axes[0].set_title("Original + Detections", fontsize=12)
        for td in target_detections:
            box = td["box"].numpy()
            label = td["label"]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[0].add_patch(rect)
            name = class_names[label] if label < len(class_names) else str(label)
            axes[0].text(
                x1,
                y1 - 5,
                name,
                color="lime",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
            )
        axes[0].axis("off")

        # Panels 1..T: Saliency overlays
        for t in range(T):
            overlay = DRISEVisualizer.overlay_saliency(img_np, saliency_maps[t])
            axes[t + 1].imshow(overlay)

            box = target_detections[t]["box"].numpy()
            label = target_detections[t]["label"]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[t + 1].add_patch(rect)
            name = class_names[label] if label < len(class_names) else str(label)
            axes[t + 1].set_title(f"Saliency: {name}", fontsize=12)
            axes[t + 1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        plt.show()

    @staticmethod
    def visualize_saliency_only(
        saliency_map: np.ndarray,
        title: str = "Saliency Map",
        save_path: str | None = None,
    ):
        """
        Visualizes a single saliency map with a colorbar.

        Args:
            saliency_map: (H, W) saliency array.
            title: Plot title.
            save_path: If provided, saves the figure to this path.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(saliency_map, cmap="jet")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()
