"""Tests for dataset sampling and splitting strategies."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from torch.utils.data import WeightedRandomSampler

from src.data.sampling import build_multi_event_fold, generate_loeo_splits, create_weighted_sampler


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


# --- Multi-Event Fold Tests ---

def _multi_event_manifest() -> pd.DataFrame:
    """Manifest with four events at two rows each for composite-fold tests."""

    events = ["Hurricane Ian", "Hurricane Michael", "Mayfield Tornado", "Mussett Bayou Fire"]
    return pd.DataFrame({
        "image_path": [f"{i}.tif" for i in range(8)],
        "event": [event for event in events for _ in range(2)]
    })


def test_build_multi_event_fold_partitions_events() -> None:
    """Listed events form the validation pool; every other event trains; no leakage."""

    manifest = _multi_event_manifest()
    fold_name, train_df, val_df, selection = build_multi_event_fold(
        manifest=manifest, holdout_events=["Hurricane Michael", "Mayfield Tornado"]
    )

    assert fold_name == "Hurricane Michael+Mayfield Tornado"
    assert selection == "explicit_multi"
    assert len(val_df) == 4
    assert len(train_df) == 4
    assert set(val_df["event"]) == {"Hurricane Michael", "Mayfield Tornado"}
    assert set(train_df["event"]) == {"Hurricane Ian", "Mussett Bayou Fire"}


def test_build_multi_event_fold_canonicalizes_requested_names() -> None:
    """Case / hyphen variants resolve to manifest labels, matching select_fold semantics."""

    manifest = _multi_event_manifest()
    fold_name, _train_df, val_df, _selection = build_multi_event_fold(
        manifest=manifest, holdout_events=["hurricane-michael", "MAYFIELD tornado"]
    )

    assert fold_name == "Hurricane Michael+Mayfield Tornado"
    assert set(val_df["event"]) == {"Hurricane Michael", "Mayfield Tornado"}


def test_build_multi_event_fold_deduplicates_matched_events() -> None:
    """Requests resolving to the same manifest event collapse to one validation pool entry."""

    manifest = _multi_event_manifest()
    fold_name, _train_df, val_df, _selection = build_multi_event_fold(
        manifest=manifest, holdout_events=["Hurricane Michael", "hurricane michael"]
    )

    assert fold_name == "Hurricane Michael"
    assert len(val_df) == 2


def test_build_multi_event_fold_unknown_event_error() -> None:
    """An unmatched event name raises with the available events listed."""

    manifest = _multi_event_manifest()
    with pytest.raises(ValueError, match="not found in manifest"):
        build_multi_event_fold(manifest=manifest, holdout_events=["Hurricane Michael", "Hurricane Sandy"])


def test_build_multi_event_fold_rejects_holdout_of_all_events() -> None:
    """Holding out every event leaves no training pool and raises."""

    manifest = _multi_event_manifest()
    all_events = ["Hurricane Ian", "Hurricane Michael", "Mayfield Tornado", "Mussett Bayou Fire"]
    with pytest.raises(ValueError, match="no training events"):
        build_multi_event_fold(manifest=manifest, holdout_events=all_events)


def test_build_multi_event_fold_missing_event_column() -> None:
    """Raises ValueError if 'event' column is missing from the manifest."""

    manifest = pd.DataFrame({"image_path": ["a.tif", "b.tif"]})
    with pytest.raises(ValueError, match="must contain an 'event' column"):
        build_multi_event_fold(manifest=manifest, holdout_events=["Hurricane Michael"])


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
