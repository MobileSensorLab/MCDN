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


class SyntheticGsdDegradation(albumentations.ImageOnlyTransform):
    """Simulates a coarser sensor ground-sample distance on the RGB chip.

    Anti-aliased blur-then-decimate: the chip is downsampled by ``factor`` with area
    averaging (``cv2.INTER_AREA``, which low-pass filters before decimation), then
    restored to the original pixel grid with bilinear interpolation. The result carries
    the information content of a sensor with ``factor``-times coarser GSD while chip
    geometry, instance inventory, and the footprint mask channel stay identical to the
    native-resolution pipeline — the controlled resolution-degradation arm. As an
    ``ImageOnlyTransform`` the vector-derived mask is untouched by construction.

    Args:
        factor: Linear GSD degradation factor (e.g. 3.0 maps 5 cm to 15 cm).
        p: Probability of applying the transform. Defaults to 1.0 — the knob models a
            sensor property, not a stochastic augmentation. Keyword-passed to the
            albumentations 2.x base, whose constructor takes ``p`` only (the legacy
            positional ``(always_apply, p)`` call silently mis-binds under 2.x).
    """

    def __init__(self, factor: float, p: float = 1.0) -> None:
        """Initialize the degradation factor."""

        super().__init__(p=p)
        if factor < 1.0:
            raise ValueError("factor must be >= 1.0.")
        self.factor = factor

    def apply(self, img: np.ndarray, **_params: object) -> np.ndarray:
        """Downsample by the configured factor with anti-aliasing, then restore the pixel grid."""

        if self.factor == 1.0:
            return img
        h, w = img.shape[:2]
        low_w = max(1, round(w / self.factor))
        low_h = max(1, round(h / self.factor))
        low = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def get_train_transforms(synthetic_gsd_factor: float = 1.0) -> albumentations.Compose:
    """Constructs the synchronized augmentation pipeline for training.

    Args:
        synthetic_gsd_factor: Optional GSD degradation factor (see ``SyntheticGsdDegradation``).
            Applied first, so downstream augmentations operate on the already-degraded imagery,
            mirroring capture-time physics. ``1.0`` disables it.
    """

    transforms: list[albumentations.BasicTransform] = []
    if synthetic_gsd_factor > 1.0:
        transforms.append(SyntheticGsdDegradation(factor=synthetic_gsd_factor))

    return albumentations.Compose([
        *transforms,

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


def get_val_transforms(synthetic_gsd_factor: float = 1.0) -> albumentations.Compose:
    """Constructs the deterministic augmentation pipeline for validation.

    Args:
        synthetic_gsd_factor: Optional GSD degradation factor (see ``SyntheticGsdDegradation``).
            The degradation models the sensor, so evaluation applies it too — deterministic,
            which keeps validation-tensor caching valid. ``1.0`` disables it.

    Returns:
        An albumentations composition: the GSD degradation when enabled, otherwise identity.
    """

    if synthetic_gsd_factor > 1.0:
        return albumentations.Compose([SyntheticGsdDegradation(factor=synthetic_gsd_factor)])
    return albumentations.Compose([])
