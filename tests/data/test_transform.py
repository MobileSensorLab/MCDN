"""Tests for the data augmentation pipelines and mask corruption transform."""

import pytest

import albumentations as A
import numpy as np

from src.data.transform import (
    MaskCorruption,
    SyntheticGsdDegradation,
    get_train_transforms,
    get_val_transforms
)


def test_mask_corruption_leaves_image_untouched() -> None:
    """Verifies that the custom transform strictly targets the mask."""
    transform = MaskCorruption(always_apply=True)
    rng = np.random.default_rng(seed=42)

    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    augmented = transform(image=img, mask=mask)

    np.testing.assert_array_equal(augmented["image"], img)


def test_mask_corruption_modifies_mask() -> None:
    """Verifies that the mask is shifted/dilated/eroded."""
    transform = MaskCorruption(always_apply=True, max_shift_px=10)

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1

    changed = False
    for _ in range(10):
        augmented = transform(image=img, mask=mask)
        if not np.array_equal(augmented["mask"], mask):
            changed = True
            break

    assert changed, "MaskCorruption failed to modify the mask over 10 iterations."


def test_pipeline_builders_return_compose() -> None:
    """Verifies the pipeline builder functions return valid A.Compose objects."""
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    assert isinstance(train_transform, A.Compose)
    assert isinstance(val_transform, A.Compose)
    assert len(train_transform.transforms) > 0
    assert len(val_transform.transforms) == 0


def _checkerboard_rgb(size: int = 96) -> np.ndarray:
    """Build a 1 px black/white checkerboard chip — maximal high-frequency content."""

    coords = np.indices((size, size)).sum(axis=0)
    board = ((coords % 2) * 255).astype(np.uint8)
    return np.stack([board, board, board], axis=-1)


def test_synthetic_gsd_degradation_rejects_invalid_factor() -> None:
    """Factors below 1.0 (upsampling) are not a resolution degradation."""

    with pytest.raises(ValueError, match="factor must be >= 1.0"):
        SyntheticGsdDegradation(factor=0.5)


def test_synthetic_gsd_degradation_identity_at_factor_one() -> None:
    """Factor 1.0 passes the chip through byte-for-byte."""

    transform = SyntheticGsdDegradation(factor=1.0)
    img = _checkerboard_rgb()

    np.testing.assert_array_equal(transform(image=img)["image"], img)


def test_synthetic_gsd_degradation_removes_high_frequency_detail() -> None:
    """A 3x degradation collapses pixel-scale detail while preserving shape, dtype, and brightness."""

    transform = SyntheticGsdDegradation(factor=3.0)
    img = _checkerboard_rgb()

    degraded = transform(image=img)["image"]

    assert degraded.shape == img.shape
    assert degraded.dtype == np.uint8
    # The 1 px checkerboard lies beyond the 3x-coarser Nyquist limit: area averaging melts
    # it toward uniform gray, so per-pixel dispersion collapses while mean brightness holds.
    assert degraded.std() < 0.25 * img.std()
    assert abs(float(degraded.mean()) - float(img.mean())) < 5.0


def test_synthetic_gsd_degradation_preserves_constant_image() -> None:
    """Degradation removes detail, not brightness: a flat chip is unchanged."""

    transform = SyntheticGsdDegradation(factor=3.0)
    img = np.full((96, 96, 3), 137, dtype=np.uint8)

    np.testing.assert_array_equal(transform(image=img)["image"], img)


def test_synthetic_gsd_degradation_leaves_mask_untouched() -> None:
    """The footprint mask is a vector prior, not sensor imagery — it must never degrade."""

    transform = A.Compose([SyntheticGsdDegradation(factor=3.0)])
    img = _checkerboard_rgb()
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:60, 30:60] = 1

    augmented = transform(image=img, mask=mask)

    np.testing.assert_array_equal(augmented["mask"], mask)
    assert not np.array_equal(augmented["image"], img)


def test_pipeline_builders_wire_gsd_degradation() -> None:
    """Both pipelines include the degradation when enabled and omit it at the default factor."""

    train_degraded = get_train_transforms(synthetic_gsd_factor=3.0)
    val_degraded = get_val_transforms(synthetic_gsd_factor=3.0)

    assert isinstance(train_degraded.transforms[0], SyntheticGsdDegradation)
    assert len(val_degraded.transforms) == 1
    assert isinstance(val_degraded.transforms[0], SyntheticGsdDegradation)
    assert not any(isinstance(t, SyntheticGsdDegradation) for t in get_train_transforms().transforms)
    assert len(get_val_transforms().transforms) == 0


def test_val_pipeline_gsd_degradation_is_deterministic() -> None:
    """Validation degradation must be reproducible so tensor caching stays valid."""

    pipeline = get_val_transforms(synthetic_gsd_factor=3.0)
    rng = np.random.default_rng(seed=7)
    img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)

    first = pipeline(image=img)["image"]
    second = pipeline(image=img)["image"]

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, img)
