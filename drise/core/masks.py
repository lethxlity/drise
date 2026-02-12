"""
Random mask generation following the RISE methodology.

Masks are generated in three steps:
    1. Sample binary masks at a low resolution.
    2. Upsample using bilinear interpolation.
    3. Randomly crop to the target image size.

This produces smooth, randomly positioned masks that cover
varying regions of the input image.
"""

import numpy as np
import torch
import torch.nn.functional as F


class MaskGenerator:
    """
    Generates random binary masks following the RISE approach.

    The masks are created at a low resolution and upsampled
    with bilinear interpolation to produce smooth occlusion patterns.
    Random cropping ensures that mask boundaries are not aligned
    with a fixed grid.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        mask_height: int = 16,
        mask_width: int = 16,
        prob: float = 0.5,
    ):
        """
        Args:
            image_height: Target image height H.
            image_width: Target image width W.
            mask_height: Low-resolution mask height h.
            mask_width: Low-resolution mask width w.
            prob: Probability of keeping a pixel (preservation probability p).
        """
        self.image_height = image_height
        self.image_width = image_width
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.prob = prob

        # Cell size in the upsampled mask.
        # Use ceiling division to guarantee the upsampled mask
        # is always large enough to crop from at any valid offset.
        self.cell_h = int(np.ceil(image_height / mask_height))
        self.cell_w = int(np.ceil(image_width / mask_width))

    def generate(self, num_masks: int, device: str = "cpu") -> torch.Tensor:
        """
        Generates N random masks.

        Args:
            num_masks: Number of masks N to generate.
            device: Target device ('cpu' or 'cuda').

        Returns
        -------
            masks: (N, 1, H, W) tensor of masks with values in [0, 1].
        """
        # Step 1: Generate low-resolution binary masks
        small_masks = (torch.rand(num_masks, 1, self.mask_height, self.mask_width, device=device) < self.prob).float()

        # Step 2: Upsample to an enlarged size.
        # The upsampled size must be at least (image_size + cell_size)
        # so that after cropping with any valid offset we still have
        # enough pixels to fill the target image dimensions.
        upsampled_h = (self.mask_height + 1) * self.cell_h
        upsampled_w = (self.mask_width + 1) * self.cell_w

        upsampled_masks = F.interpolate(
            small_masks,
            size=(upsampled_h, upsampled_w),
            mode="bilinear",
            align_corners=False,
        )

        # Step 3: Randomly crop to the target image size.
        # Maximum valid offset ensures the crop window fits entirely
        # within the upsampled mask.
        max_offset_h = upsampled_h - self.image_height
        max_offset_w = upsampled_w - self.image_width

        masks = torch.zeros(num_masks, 1, self.image_height, self.image_width, device=device)

        for i in range(num_masks):
            offset_h = np.random.randint(0, max_offset_h + 1)
            offset_w = np.random.randint(0, max_offset_w + 1)

            masks[i] = upsampled_masks[
                i,
                :,
                offset_h : offset_h + self.image_height,
                offset_w : offset_w + self.image_width,
            ]

        return masks
