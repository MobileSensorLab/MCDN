"""Augmentation pipelines for synchronized RGB/mask transforms.

Provides the training and validation albumentations pipelines, including the
custom ``MaskCorruption`` transform that degrades the binary mask to prevent
the model from over-trusting a priori vector polygons. The production
ImageNet normalization is performed in-place on the GPU tensor inside
``src/data/dataset.py``; numpy-based normalization helpers used by the
documentation live under ``doc/scripts/normalize.py``.
"""

import albumentations
import cv2
import random

import numpy as np


class MaskCorruption(albumentations.DualTransform):
    """Randomly degrades the binary mask to prevent network over-reliance.

    Applies morphological erosion, dilation, and spatial translation specifically
    to the mask channel while leaving the paired RGB image untouched.

    Args:
        max_shift_px: Maximum translation shift in pixels (x and y).
        always_apply: Set 'True' to always apply the transform, or 'False' (default) to apply probabilistically.
        p: Probability of applying the transform.
    """

    def __init__(self, max_shift_px: int = 15, always_apply: bool = False, p: float = 0.5) -> None:
        """Initialize mask corruption hyperparameters."""

        super().__init__(always_apply, p)
        self.max_shift_px = max_shift_px

    def apply(self, img: np.ndarray, **_params: object) -> np.ndarray:
        """Passes the image through untouched."""

        return img

    def apply_to_mask(self, mask: np.ndarray, **_params: object) -> np.ndarray:
        """Applies morphological corruption and translation to the mask."""

        corrupted = mask.copy()

        # Random morphological dilation or erosion
        if random.random() < 0.5:
            k_size = random.choice([3, 5, 7])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            corrupted = cv2.dilate(corrupted, kernel, iterations=1) if random.random() < 0.5 else cv2.erode(corrupted, kernel, iterations=1)

        # Random translation (simulating footprint registration lag / GPS drift)
        if random.random() < 0.5:
            dx = random.randint(-self.max_shift_px, self.max_shift_px)
            dy = random.randint(-self.max_shift_px, self.max_shift_px)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            corrupted = cv2.warpAffine(
                corrupted,
                matrix,
                (corrupted.shape[1], corrupted.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        return corrupted


def get_train_transforms() -> albumentations.Compose:
    """Constructs the synchronized augmentation pipeline for training."""

    return albumentations.Compose([

        # Standard spatial degradation
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomRotate90(p=0.5),

        # Sensor + environmental degradation
        albumentations.MotionBlur(blur_limit=(3, 11), p=0.4),
        albumentations.GaussNoise(std_range=(0.02, 0.05), p=0.4),

        # Lighting variance
        albumentations.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),

        # Severe spatial occlusion
        albumentations.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(0.06, 0.2),  # Roughly 30px to 100px on a 512px chip
            hole_width_range=(0.06, 0.2),
            fill=0,
            fill_mask=0,
            p=0.5
        ),

        # Footprint Registration Lag Simulation
        MaskCorruption(max_shift_px=20, p=0.5),
    ])


def get_val_transforms() -> albumentations.Compose:
    """Constructs the deterministic augmentation pipeline for validation.

    Returns:
        An empty albumentations composition (identity transform).
    """

    return albumentations.Compose([])
