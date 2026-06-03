"""Tests for dataset sampling and splitting strategies."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from torch.utils.data import WeightedRandomSampler

from src.data.sampling import generate_loeo_splits, create_weighted_sampler


# --- LOEO Split Tests ---

def test_generate_loeo_splits_missing_event_column() -> None:
    """Raises ValueError if 'event' column is missing from the manifest."""
    df = pd.DataFrame({"image_path": ["a.tif", "b.tif"]})
    with pytest.raises(ValueError, match="Manifest must contain an 'event' column"):
        list(generate_loeo_splits(df))


def test_generate_loeo_splits_success() -> None:
    """Yields correct train/val splits strictly isolated by event."""
    df = pd.DataFrame({
        "image_path": ["1.tif", "2.tif", "3.tif", "4.tif", "5.tif"],
        "event": ["Hurricane Ian", "Hurricane Ian", "Mayfield Tornado", "Mayfield Tornado", "Mussett Bayou Fire"]
    })

    splits = list(generate_loeo_splits(df))
    assert len(splits) == 3  # 3 unique events

    # Isolate the tornado holdout
    _holdout, train_df, val_df = next(s for s in splits if s[0] == "Mayfield Tornado")

    assert len(val_df) == 2
    assert len(train_df) == 3

    # Verify no geographic leakage occurred
    assert "Mayfield Tornado" not in train_df["event"].values
    assert "Mayfield Tornado" in val_df["event"].values


# --- Weighted Sampler Tests ---

def test_create_weighted_sampler_empty_dataset() -> None:
    """Raises ValueError if the dataset has no parsed instances."""
    mock_ds = MagicMock()
    mock_ds.instances = []
    with pytest.raises(ValueError, match="Dataset contains no instances"):
        create_weighted_sampler(mock_ds)


def test_create_weighted_sampler_calculates_correct_weights() -> None:
    """Assigns proportionally higher sampling weights to minority classes."""
    mock_ds = MagicMock()

    # Extreme imbalance: 4 structurally sound buildings, 1 destroyed building
    mock_ds.instances = [
        {"damage_label": "no damage"},
        {"damage_label": "no damage"},
        {"damage_label": "no damage"},
        {"damage_label": "no damage"},
        {"damage_label": "destroyed"}
    ]

    sampler = create_weighted_sampler(mock_ds)

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 5

    # 'no damage' frequency = 4 -> weight = 1/4 = 0.25
    # 'destroyed' frequency = 1 -> weight = 1/1 = 1.00
    expected_weights = [0.25, 0.25, 0.25, 0.25, 1.0]

    # Convert PyTorch tensor to list for direct comparison
    actual_weights = sampler.weights.tolist()
    assert actual_weights == pytest.approx(expected_weights)
