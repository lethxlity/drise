"""Utility functions."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_image(
    image_path: str,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Loads an image from disk and converts it to a tensor.

    Args:
        image_path: Path to the image file.
        target_size: Optional (H, W) to resize the image.

    Returns
    -------
        image: (3, H, W) tensor with values in [0, 1].
    """
    img = Image.open(image_path).convert("RGB")

    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]))

    transform = transforms.ToTensor()
    return transform(img)


def resize_image_for_detector(
    image: torch.Tensor,
    detector_type: str = "yolo",
    max_size: int | None = None,
) -> torch.Tensor:
    """
    Resizes an image to match the detector's native input size.

    Args:
        image: (3, H, W) tensor with values in [0, 1].
        detector_type: 'yolo' or 'fasterrcnn'.
        max_size: Override the target size.

    Returns
    -------
        resized: (3, H', W') tensor with values in [0, 1].
    """
    _, H, W = image.shape

    if max_size is not None:
        scale = max_size / max(H, W)
        new_h = int(H * scale)
        new_w = int(W * scale)
    elif detector_type == "yolo":
        new_h = 640
        new_w = 640
    elif detector_type == "fasterrcnn":
        if H < W:
            scale = 800 / H
            new_h = 800
            new_w = min(int(W * scale), 1333)
        else:
            scale = 800 / W
            new_w = 800
            new_h = min(int(H * scale), 1333)
    else:
        return image

    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return resized


def upscale_saliency_map(
    saliency_map: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    Upscales a saliency map to the target resolution using
    bilinear interpolation.

    Args:
        saliency_map: (H, W) saliency map with values in [0, 1].
        target_height: Desired output height.
        target_width: Desired output width.

    Returns
    -------
        upscaled: (target_height, target_width) saliency map.
    """
    tensor = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0)
    upscaled = F.interpolate(
        tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return upscaled.squeeze(0).squeeze(0).numpy()


def rescale_boxes(
    detections_list: list,
    src_height: int,
    src_width: int,
    dst_height: int,
    dst_width: int,
) -> list:
    """
    Rescales bounding boxes from source resolution to destination
    resolution.

    Args:
        detections_list: List of dicts with 'box' (torch.Tensor (4,))
            and 'label' (int) keys.
        src_height: Height of the source image.
        src_width: Width of the source image.
        dst_height: Height of the destination image.
        dst_width: Width of the destination image.

    Returns
    -------
        rescaled: New list of dicts with rescaled 'box' tensors.
    """
    scale_y = dst_height / src_height
    scale_x = dst_width / src_width

    rescaled = []
    for det in detections_list:
        box = det["box"].clone().float()
        box[0] *= scale_x  # x1
        box[1] *= scale_y  # y1
        box[2] *= scale_x  # x2
        box[3] *= scale_y  # y2
        rescaled.append({"box": box, "label": det["label"]})

    return rescaled


def get_repo_path() -> Path:
    """Get repo path."""
    current_path = Path(__file__).resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError("Repo path not found.")


def get_data_path() -> Path:
    """Get data path."""
    return get_repo_path() / "data"
