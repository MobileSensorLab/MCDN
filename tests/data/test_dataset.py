"""Tests for the CRASARUnitemporalDataset class and tensor generation logic."""

import albumentations as A
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import CRASARUnitemporalDataset


def test_dataset_initialization_extracts_instances(sample_manifest: pd.DataFrame) -> None:
    """Verifies that the dataset iterates over polygons, not just orthomosaics."""
    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=16)

    # We wrote two polygons with valid labels in conftest
    assert len(dataset) == 2

    # Verify the instance metadata is correctly cached
    instance1 = dataset.instances[0]
    assert instance1["damage_label"] in ["minor damage", "destroyed"]
    assert "centroid_px" in instance1
    assert instance1["event_name"] == "Hurricane Ian"


def test_dataset_typology_mapping(sample_manifest: pd.DataFrame) -> None:
    """Verifies the string-to-one-hot context vector logic."""
    dataset = CRASARUnitemporalDataset(sample_manifest)

    # Known event mappings
    assert torch.equal(dataset._get_typology_vector("Hurricane Ian"), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(dataset._get_typology_vector("Mayfield Tornado"), torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(dataset._get_typology_vector("Mussett Bayou Fire"), torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # Unknown fallback
    assert torch.equal(dataset._get_typology_vector("Alien Invasion"), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_dataset_getitem_returns_valid_tensors(sample_manifest: pd.DataFrame) -> None:
    """Verifies that __getitem__ returns a correctly shaped 4-channel tensor."""
    chip_size = 32
    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=chip_size)

    item = dataset[0]

    assert "image" in item
    assert "label" in item
    assert "context" in item

    # Check tensor shapes (4 channels: R, G, B, Mask). The dataset returns uint8 to keep
    # dataloader -> GPU bandwidth low; ImageNet normalization is applied GPU-side by the
    # trainer in UnitemporalTrainer._prepare_batch.
    image = item["image"]
    assert image.shape == (4, chip_size, chip_size)
    assert image.dtype == torch.uint8

    # Check label and context
    assert item["label"].dim() == 0  # 0D Scalar for CrossEntropy
    assert item["context"].shape == (4,)


def test_dataset_with_albumentations_transforms(sample_manifest: pd.DataFrame) -> None:
    """Verifies that spatial transforms are applied successfully to both image and mask."""

    # A.Lambda only passes the specific target (image) to this function
    def mock_image(image: np.ndarray, **_kwargs: object) -> np.ndarray:
        return np.zeros_like(image)

    transform = A.Compose([A.Lambda(image=mock_image)])

    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=16, transform=transform)
    item = dataset[0]

    image = item["image"]

    # The first 3 channels (RGB) are zeroed by the Lambda transform; the dataset emits
    # them as uint8 without normalization. ImageNet shift / scale is performed GPU-side
    # by the trainer, so the dataset-level expectation is just zero-bytes for RGB.
    assert image.dtype == torch.uint8
    assert image[:3].sum().item() == 0


def test_dataset_extraction_window_jitter(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that spatial jitter is applied during training but not validation."""

    mock_randint_calls = []

    def mock_randint(low: int, high: int) -> int:
        mock_randint_calls.append((low, high))
        return high  # always return max jitter

    import random
    monkeypatch.setattr(random, "randint", mock_randint)

    # Validation Dataset (No Jitter)
    val_dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, is_train=False)
    _ = val_dataset[0]
    assert len(mock_randint_calls) == 0, "Jitter should not be applied when is_train=False"

    # Training Dataset (Jitter Applied)
    train_dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, is_train=True)
    _ = train_dataset[0]
    assert len(mock_randint_calls) == 2, "Jitter should be applied twice (x and y) when is_train=True"

    # Max jitter for chip_size 32 is 32 // 4 = 8
    assert mock_randint_calls[0] == (-8, 8)
    assert mock_randint_calls[1] == (-8, 8)
