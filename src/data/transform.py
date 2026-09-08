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
import math
import random

import numpy as np


def gaussian_sigma_for_mtf(factor: float, mtf_at_nyquist: float) -> float:
    """Gaussian blur sigma (in source-pixel units) hitting a target MTF at the new Nyquist.

    For a Gaussian PSF the modulation transfer function is MTF(f) = exp(-2 pi^2 sigma^2 f^2).
    Decimating by ``factor`` puts the new Nyquist at f = 1/(2*factor) cycles per source pixel;
    solving for sigma gives sigma = 2 * factor * sqrt(-ln(MTF) / (2 pi^2)). At MTF 0.3 this
    reduces to the sigma ~= 0.5 * factor rule of thumb (0.494 exactly); 0.15 gives 0.62 * factor
    and 0.45 gives 0.40 * factor, bracketing realistic sensor quality.

    Args:
        factor: Linear GSD degradation factor (decimation ratio).
        mtf_at_nyquist: Target system MTF at the post-decimation Nyquist frequency, in (0, 1).

    Returns:
        Gaussian sigma in source (high-resolution) pixel units.
    """

    if not 0.0 < mtf_at_nyquist < 1.0:
        raise ValueError("mtf_at_nyquist must lie strictly between 0 and 1.")
    return 2.0 * factor * math.sqrt(-math.log(mtf_at_nyquist) / (2.0 * math.pi**2))


# Unsharp-mask support for the product-referenced sharpening stage, in low-resolution
# (post-decimation) pixels. At sigma 1.0 the Gaussian's response at the low-res Nyquist
# is ~0.007, so the stage's transfer there is ~(1 + amount) — one interpretable knob.
POST_SHARPEN_SIGMA_LR_PX = 1.0


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

    Two fidelity levels, selected by ``mtf_at_nyquist``:

    - ``None`` (sampling-only): area-averaged decimation (``cv2.INTER_AREA``) then bilinear
      restoration to the original grid. The box average models the detector pixel aperture
      *only* (MTF ~= 0.66 at the new Nyquist for factor 3) — an idealized best-case sensor
      whose sole blur source is pixel integration. Real coarse-GSD systems stack optics PSF,
      defocus, atmosphere, and platform motion on top, so this arm bounds degradation from
      the optimistic side.
    - A float in (0, 1) (MTF-matched): Gaussian pre-blur at sigma = 2 * factor *
      sqrt(-ln(MTF)/(2 pi^2)) source pixels (0.494 * factor at the 0.3 default) before the
      same decimate/restore, modeling system optics at a target MTF-at-Nyquist. The
      subsequent box decimation contributes its own aperture MTF (~0.66 at Nyquist), so the
      effective system MTF is the product (~0.20 for a 0.3 target) — mildly pessimistic,
      bounding realism from the other side of the sampling-only arm. The native imagery's
      own PSF (sigma ~0.5 px) would tighten sigma by ~6% in quadrature; ignored as
      negligible against sensor-quality uncertainty.

    A third, independent stage models the *delivered product* rather than the raw sensor:
    ``post_sharpen_amount`` applies an unsharp mask on the low-resolution grid after
    decimation, mirroring the MTF-compensation/deconvolution step documented in operational
    ortho-production chains (e.g. Pleiades ground restoration, QuickBird MTF resampling).
    The stage's transfer at the low-res Nyquist is ~(1 + amount); the amount is regressed
    against measured transfer functions of genuinely overlapping crewed deliverables.

    Chip geometry, instance inventory, and the footprint mask channel stay identical to the
    native-resolution pipeline in both modes. As an ``ImageOnlyTransform`` the vector-derived
    mask is untouched by construction.

    Args:
        factor: Linear GSD degradation factor (e.g. 3.0 maps 5 cm to 15 cm).
        mtf_at_nyquist: Optional target system MTF at the post-decimation Nyquist. ``None``
            preserves the sampling-only behavior byte-for-byte.
        post_sharpen_amount: Optional unsharp-mask amount applied on the low-resolution grid
            after decimation (product-referenced MTF compensation). ``None`` disables the stage.
        p: Probability of applying the transform. Defaults to 1.0 — the knob models a
            sensor property, not a stochastic augmentation. Keyword-passed to the
            albumentations 2.x base, whose constructor takes ``p`` only (the legacy
            positional ``(always_apply, p)`` call silently mis-binds under 2.x).
    """

    def __init__(self, factor: float, mtf_at_nyquist: float | None = None,
                 post_sharpen_amount: float | None = None, p: float = 1.0) -> None:
        """Initialize the degradation factor, optional MTF blur, and optional product sharpening."""

        super().__init__(p=p)
        if factor < 1.0:
            raise ValueError("factor must be >= 1.0.")
        if post_sharpen_amount is not None and post_sharpen_amount <= 0.0:
            raise ValueError("post_sharpen_amount must be positive when set.")
        self.factor = factor
        self.mtf_at_nyquist = mtf_at_nyquist
        self.blur_sigma = gaussian_sigma_for_mtf(factor, mtf_at_nyquist) if mtf_at_nyquist is not None else 0.0
        self.post_sharpen_amount = post_sharpen_amount

    def apply(self, img: np.ndarray, **_params: object) -> np.ndarray:
        """Optionally MTF-blur, downsample, optionally product-sharpen, then restore the pixel grid."""

        if self.factor == 1.0:
            return img
        if self.blur_sigma > 0.0:
            # Kernel size (0, 0) lets OpenCV derive the support from sigma.
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)
        h, w = img.shape[:2]
        low_w = max(1, round(w / self.factor))
        low_h = max(1, round(h / self.factor))
        low = cv2.resize(img, (low_w, low_h), interpolation=cv2.INTER_AREA)
        if self.post_sharpen_amount is not None:
            # Unsharp mask on the product grid; uint8 saturation clips like a real 8-bit deliverable.
            blurred = cv2.GaussianBlur(low, (0, 0), sigmaX=POST_SHARPEN_SIGMA_LR_PX, sigmaY=POST_SHARPEN_SIGMA_LR_PX)
            low = cv2.addWeighted(low, 1.0 + self.post_sharpen_amount, blurred, -self.post_sharpen_amount, 0.0)
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def get_train_transforms(synthetic_gsd_factor: float = 1.0,
                         synthetic_gsd_mtf_at_nyquist: float | None = None,
                         synthetic_gsd_post_sharpen: float | None = None) -> albumentations.Compose:
    """Constructs the synchronized augmentation pipeline for training.

    Args:
        synthetic_gsd_factor: Optional GSD degradation factor (see ``SyntheticGsdDegradation``).
            Applied first, so downstream augmentations operate on the already-degraded imagery,
            mirroring capture-time physics. ``1.0`` disables it.
        synthetic_gsd_mtf_at_nyquist: Optional MTF-matched blur target for the degradation
            (see ``SyntheticGsdDegradation``). ``None`` selects sampling-only decimation.
        synthetic_gsd_post_sharpen: Optional product-referenced unsharp amount for the
            degradation (see ``SyntheticGsdDegradation``). ``None`` disables the stage.
    """

    transforms: list[albumentations.BasicTransform] = []
    if synthetic_gsd_factor > 1.0:
        transforms.append(SyntheticGsdDegradation(factor=synthetic_gsd_factor, mtf_at_nyquist=synthetic_gsd_mtf_at_nyquist,
                                                  post_sharpen_amount=synthetic_gsd_post_sharpen))

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


def get_val_transforms(synthetic_gsd_factor: float = 1.0,
                       synthetic_gsd_mtf_at_nyquist: float | None = None,
                       synthetic_gsd_post_sharpen: float | None = None) -> albumentations.Compose:
    """Constructs the deterministic augmentation pipeline for validation.

    Args:
        synthetic_gsd_factor: Optional GSD degradation factor (see ``SyntheticGsdDegradation``).
            The degradation models the sensor, so evaluation applies it too — deterministic,
            which keeps validation-tensor caching valid. ``1.0`` disables it.
        synthetic_gsd_mtf_at_nyquist: Optional MTF-matched blur target for the degradation
            (see ``SyntheticGsdDegradation``). ``None`` selects sampling-only decimation.
        synthetic_gsd_post_sharpen: Optional product-referenced unsharp amount for the
            degradation (see ``SyntheticGsdDegradation``). ``None`` disables the stage.

    Returns:
        An albumentations composition: the GSD degradation when enabled, otherwise identity.
    """

    if synthetic_gsd_factor > 1.0:
        return albumentations.Compose([
            SyntheticGsdDegradation(factor=synthetic_gsd_factor, mtf_at_nyquist=synthetic_gsd_mtf_at_nyquist,
                                    post_sharpen_amount=synthetic_gsd_post_sharpen)
        ])
    return albumentations.Compose([])
