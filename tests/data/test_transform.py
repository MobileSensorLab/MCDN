"""Tests for the data augmentation pipelines and mask corruption transform."""

import albumentations as A
import numpy as np

from src.data.transform import (
    MaskCorruption,
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
