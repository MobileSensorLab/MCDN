"""Tests for the data augmentation pipelines and mask corruption transform."""

import pytest

import albumentations as A
import numpy as np

from src.data.transform import (
    MaskCorruption,
    SyntheticGsdDegradation,
    gaussian_sigma_for_mtf,
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

    with pytest.raises(ValueError, match=r"factor must be >= 1\.0"):
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


def test_gaussian_sigma_for_mtf_matches_rule_of_thumb() -> None:
    """MTF 0.3 at Nyquist recovers sigma ~= 0.494 * factor; the bracket endpoints scale correctly."""

    assert gaussian_sigma_for_mtf(3.0, 0.3) == pytest.approx(0.494 * 3.0, abs=0.01)
    assert gaussian_sigma_for_mtf(3.0, 0.15) == pytest.approx(0.62 * 3.0, abs=0.02)
    assert gaussian_sigma_for_mtf(3.0, 0.45) == pytest.approx(0.40 * 3.0, abs=0.01)
    # Sigma scales linearly with the decimation factor at a fixed MTF target.
    assert gaussian_sigma_for_mtf(6.0, 0.3) == pytest.approx(2.0 * gaussian_sigma_for_mtf(3.0, 0.3))


def test_gaussian_sigma_for_mtf_rejects_degenerate_targets() -> None:
    """MTF targets at or beyond the [0, 1] endpoints have no finite Gaussian solution."""

    with pytest.raises(ValueError, match="mtf_at_nyquist"):
        gaussian_sigma_for_mtf(3.0, 0.0)
    with pytest.raises(ValueError, match="mtf_at_nyquist"):
        gaussian_sigma_for_mtf(3.0, 1.0)


def test_mtf_matched_degradation_blurs_more_than_sampling_only() -> None:
    """The optics pre-blur removes strictly more mid-frequency detail than pure decimation.

    A 6 px-period checkerboard survives 3x area decimation (it sits below the new Nyquist)
    but is strongly attenuated by the sigma ~= 1.48 px Gaussian, so the MTF-matched output
    must show lower contrast than the sampling-only output.
    """

    size = 96
    coords = np.indices((size, size)).sum(axis=0)
    board = (((coords // 3) % 2) * 255).astype(np.uint8)
    img = np.stack([board, board, board], axis=-1)

    sampling_only = SyntheticGsdDegradation(factor=3.0)(image=img)["image"]
    mtf_matched = SyntheticGsdDegradation(factor=3.0, mtf_at_nyquist=0.3)(image=img)["image"]

    assert mtf_matched.std() < 0.8 * sampling_only.std()
    # Both preserve mean brightness: blur redistributes energy, it does not remove it.
    assert abs(float(mtf_matched.mean()) - float(img.mean())) < 5.0


def test_mtf_matched_degradation_is_deterministic_and_mask_safe() -> None:
    """MTF-matched mode stays deterministic (cache-safe) and never touches the mask channel."""

    pipeline = get_val_transforms(synthetic_gsd_factor=3.0, synthetic_gsd_mtf_at_nyquist=0.3)
    rng = np.random.default_rng(seed=11)
    img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:60, 30:60] = 1

    first = pipeline(image=img, mask=mask)
    second = pipeline(image=img, mask=mask)

    np.testing.assert_array_equal(first["image"], second["image"])
    np.testing.assert_array_equal(first["mask"], mask)
    assert not np.array_equal(first["image"], img)


def test_pipeline_builders_thread_mtf_target() -> None:
    """Both pipeline builders forward the MTF target into the degradation transform."""

    train_pipe = get_train_transforms(synthetic_gsd_factor=3.0, synthetic_gsd_mtf_at_nyquist=0.3)
    val_pipe = get_val_transforms(synthetic_gsd_factor=3.0, synthetic_gsd_mtf_at_nyquist=0.3)

    assert train_pipe.transforms[0].mtf_at_nyquist == 0.3
    assert train_pipe.transforms[0].blur_sigma == pytest.approx(0.494 * 3.0, abs=0.01)
    assert val_pipe.transforms[0].mtf_at_nyquist == 0.3
    # Default remains sampling-only: no blur.
    assert get_val_transforms(synthetic_gsd_factor=3.0).transforms[0].blur_sigma == 0.0


def test_post_sharpen_restores_high_frequency_content() -> None:
    """The product-referenced unsharp mask boosts detail the plain decimation attenuates.

    Sharpening operates on the low-resolution grid, so the deliverable-matched output must
    carry more residual contrast than the sampling-only output on the same textured chip,
    while leaving a flat chip untouched (unsharp of a constant is the constant).
    """

    rng = np.random.default_rng(seed=23)
    img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)

    sampling_only = SyntheticGsdDegradation(factor=7.0)(image=img)["image"]
    deliverable = SyntheticGsdDegradation(factor=7.0, post_sharpen_amount=0.2)(image=img)["image"]

    assert deliverable.std() > sampling_only.std()
    assert not np.array_equal(deliverable, sampling_only)

    flat = np.full((96, 96, 3), 137, dtype=np.uint8)
    np.testing.assert_array_equal(SyntheticGsdDegradation(factor=7.0, post_sharpen_amount=0.2)(image=flat)["image"], flat)


def test_post_sharpen_rejects_non_positive_amounts() -> None:
    """A zero or negative unsharp amount is a configuration error, not a silent no-op."""

    with pytest.raises(ValueError, match="post_sharpen_amount"):
        SyntheticGsdDegradation(factor=7.0, post_sharpen_amount=0.0)
    with pytest.raises(ValueError, match="post_sharpen_amount"):
        SyntheticGsdDegradation(factor=7.0, post_sharpen_amount=-0.2)


def test_post_sharpen_is_deterministic_and_mask_safe() -> None:
    """Deliverable-matched mode stays deterministic (cache-safe) and never touches the mask channel."""

    pipeline = get_val_transforms(synthetic_gsd_factor=7.0, synthetic_gsd_post_sharpen=0.2)
    rng = np.random.default_rng(seed=29)
    img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:60, 30:60] = 1

    first = pipeline(image=img, mask=mask)
    second = pipeline(image=img, mask=mask)

    np.testing.assert_array_equal(first["image"], second["image"])
    np.testing.assert_array_equal(first["mask"], mask)
    assert not np.array_equal(first["image"], img)


def test_pipeline_builders_thread_post_sharpen() -> None:
    """Both pipeline builders forward the unsharp amount into the degradation transform."""

    train_pipe = get_train_transforms(synthetic_gsd_factor=7.0, synthetic_gsd_post_sharpen=0.2)
    val_pipe = get_val_transforms(synthetic_gsd_factor=7.0, synthetic_gsd_post_sharpen=0.2)

    assert train_pipe.transforms[0].post_sharpen_amount == 0.2
    assert val_pipe.transforms[0].post_sharpen_amount == 0.2
    # Default remains un-sharpened.
    assert get_val_transforms(synthetic_gsd_factor=7.0).transforms[0].post_sharpen_amount is None
