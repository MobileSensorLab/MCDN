"""Tests for dataset sampling and splitting strategies."""

import numpy as np
import pandas as pd
import pytest
import rasterio

from pathlib import Path
from rasterio.transform import from_origin
from torch.utils.data import WeightedRandomSampler
from unittest.mock import MagicMock

from src.data import sampling
from src.data.sampling import (
    _assert_one_event_per_image,
    _attach_source_metadata,
    _ensure_event_column,
    _filter_manifest_for_sensor_profile,
    build_multi_event_fold,
    build_spatial_split_manifest,
    build_valid_manifest,
    create_weighted_sampler,
    generate_loeo_splits,
    select_fold,
)


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


# --- Spatial-block fallback ---

def _write_raster(path: Path, left: float) -> None:
    """Write a 10x10 EPSG:4326 raster whose west edge sits at ``left`` degrees longitude."""

    profile = {"driver": "GTiff", "height": 10, "width": 10, "count": 1, "dtype": rasterio.uint8, "crs": "EPSG:4326",
               "transform": from_origin(left, 20.0, 0.01, 0.01)}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.uint8))


def test_single_event_manifest_falls_back_to_spatial_block_split(tmp_path: Path) -> None:
    """One event triggers the west-to-east 80/20 split, named Spatial_Block_East, sorted by raster centroid."""

    rows = []
    for name, left in (("east", 12.0), ("west", 10.0), ("mid_a", 11.0), ("mid_b", 11.5), ("far_east", 13.0)):
        path = tmp_path / f"{name}.tif"
        _write_raster(path, left)
        rows.append({"image_path": str(path), "label_path": f"{name}.json", "event": "Spatial_Block"})
    manifest = pd.DataFrame(rows)

    splits = list(generate_loeo_splits(manifest))
    assert len(splits) == 1
    holdout, train_df, val_df = splits[0]
    assert holdout == "Spatial_Block_East"
    assert "cx" not in train_df.columns
    assert [Path(p).stem for p in train_df["image_path"]] == ["west", "mid_a", "mid_b", "east"]
    assert [Path(p).stem for p in val_df["image_path"]] == ["far_east"]


def test_spatial_block_split_tolerates_unreadable_rasters(tmp_path: Path) -> None:
    """Rasters that cannot be opened sort at longitude 0.0 rather than aborting the split."""

    good = tmp_path / "good.tif"
    _write_raster(good, 10.0)
    manifest = pd.DataFrame([
        {"image_path": str(good), "label_path": "good.json", "event": "Only"},
        {"image_path": str(tmp_path / "missing.tif"), "label_path": "missing.json", "event": "Only"}
    ])
    _holdout, train_df, val_df = next(iter(generate_loeo_splits(manifest)))
    assert Path(train_df["image_path"].iloc[0]).name == "missing.tif"
    assert Path(val_df["image_path"].iloc[0]).name == "good.tif"


def test_loeo_skips_folds_with_empty_partitions() -> None:
    """Rows without an event are dropped from the event list, and single-sided folds are not emitted."""

    manifest = pd.DataFrame([
        {"image_path": "a.tif", "label_path": "a.json", "event": "Hurricane Ian"},
        {"image_path": "b.tif", "label_path": "b.json", "event": "Hurricane Ida"},
        {"image_path": "c.tif", "label_path": "c.json", "event": None}
    ])
    folds = {holdout: (len(train_df), len(val_df)) for holdout, train_df, val_df in generate_loeo_splits(manifest)}
    assert folds == {"Hurricane Ian": (2, 1), "Hurricane Ida": (2, 1)}


# --- Event-column derivation and consistency ---

def test_ensure_event_column_derives_from_image_name() -> None:
    """Missing or blank events derive from the image-name prefix, with a fallback for prefix-less names."""

    derived = _ensure_event_column(pd.DataFrame({"image_name": ["Ian_01", "plain"]}))
    assert derived["event"].tolist() == ["Ian", "Test_Event"]

    partial = pd.DataFrame({"image_name": ["Ian_01", "Ida_02", "solo"], "event": ["Hurricane Ian", " ", None]})
    filled = _ensure_event_column(partial)
    assert filled["event"].tolist() == ["Hurricane Ian", "Ida", "Test_Event"]

    complete = pd.DataFrame({"image_name": ["Ian_01"], "event": ["Hurricane Ian"]})
    assert _ensure_event_column(complete)["event"].tolist() == ["Hurricane Ian"]


def test_assert_one_event_per_image_reports_offenders() -> None:
    """Images mapped to more than one event are rejected with examples in the message."""

    consistent = pd.DataFrame({"image_name": ["a", "a", "b"], "event": ["Ian", "Ian", "Ida"]})
    _assert_one_event_per_image(consistent)

    inconsistent = pd.DataFrame({"image_name": ["a", "a", "b"], "event": ["Ian", "Ida", "Ida"]})
    with pytest.raises(ValueError, match=r"Found 1 inconsistent image\(s\). Examples -> a: \['Ian', 'Ida'\]"):
        _assert_one_event_per_image(inconsistent)


# --- Source metadata and sensor-profile filtering ---

def _scan_manifest() -> pd.DataFrame:
    return pd.DataFrame({"image_name": ["uas_01", "crewed_01", "unknown_01"], "event": ["Ian", "Ian", None]})


def test_attach_source_metadata_without_statistics_is_identity(tmp_path: Path) -> None:
    """No statistics.csv (or no image_name column) leaves the manifest untouched."""

    manifest = _scan_manifest()
    assert _attach_source_metadata(manifest=manifest, data_dir=str(tmp_path)) is manifest
    nameless = pd.DataFrame({"event": ["Ian"]})
    (tmp_path / "statistics.csv").write_text("Orthomosaic,Source\n", encoding="utf-8")
    assert _attach_source_metadata(manifest=nameless, data_dir=str(tmp_path)) is nameless


def test_attach_source_metadata_maps_source_and_fills_event(tmp_path: Path) -> None:
    """Source and Event columns are joined on Orthomosaic; statistics events win, scan events fill gaps."""

    (tmp_path / "statistics.csv").write_text(
        "Orthomosaic,Source,Event\nuas_01,UAS,Hurricane Michael\ncrewed_01,Crewed,\nunknown_01,,Mayfield\n",
        encoding="utf-8")
    output = _attach_source_metadata(manifest=_scan_manifest(), data_dir=str(tmp_path))
    assert output["source"].tolist()[:2] == ["UAS", "Crewed"]
    assert output["event"].tolist() == ["Hurricane Michael", "Ian", "Mayfield"]

    eventless = _attach_source_metadata(manifest=_scan_manifest().drop(columns=["event"]), data_dir=str(tmp_path))
    assert eventless["event"].tolist()[0] == "Hurricane Michael"
    assert pd.isna(eventless["event"].iloc[1])


def test_attach_source_metadata_warns_on_bad_statistics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A statistics.csv without the Orthomosaic column, or one that fails to parse, is skipped with a warning."""

    manifest = _scan_manifest()
    (tmp_path / "statistics.csv").write_text("Mosaic,Source\nuas_01,UAS\n", encoding="utf-8")
    assert _attach_source_metadata(manifest=manifest, data_dir=str(tmp_path)) is manifest
    assert "missing 'Orthomosaic' column" in capsys.readouterr().out

    (tmp_path / "statistics.csv").write_bytes(b"")
    assert _attach_source_metadata(manifest=manifest, data_dir=str(tmp_path)) is manifest
    assert "Failed to parse source metadata" in capsys.readouterr().out


def test_filter_manifest_for_sensor_profile_branches(capsys: pytest.CaptureFixture[str]) -> None:
    """Each profile keeps its own sources; missing or empty metadata skips filtering with a warning."""

    with pytest.raises(ValueError, match=r"Unsupported data\.sensor_profile"):
        _filter_manifest_for_sensor_profile(manifest=_scan_manifest(), sensor_profile="satellite", data_dir="data")

    no_source = _scan_manifest()
    assert _filter_manifest_for_sensor_profile(manifest=no_source, sensor_profile="uas_5cm", data_dir="data") is no_source
    assert "No source metadata available" in capsys.readouterr().out

    empty_source = _scan_manifest().assign(source=[None, "", None])
    assert _filter_manifest_for_sensor_profile(manifest=empty_source, sensor_profile="uas_5cm", data_dir="data") is empty_source
    assert "Source metadata is empty" in capsys.readouterr().out

    sourced = _scan_manifest().assign(source=["UAS", "Crewed", None])
    uas = _filter_manifest_for_sensor_profile(manifest=sourced, sensor_profile="uas_5cm", data_dir="data")
    assert uas["image_name"].tolist() == ["uas_01"]
    crewed = _filter_manifest_for_sensor_profile(manifest=sourced, sensor_profile="manned_15cm", data_dir="data")
    assert crewed["image_name"].tolist() == ["crewed_01"]

    with pytest.raises(ValueError, match=r"No records matched data\.sensor_profile='manned_15cm'"):
        _filter_manifest_for_sensor_profile(manifest=sourced.iloc[[0]], sensor_profile="manned_15cm", data_dir="data")


# --- Fold selection ---

def _splits(*names: str) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    frame = pd.DataFrame({"image_path": ["x.tif"]})
    return [(name, frame, frame) for name in names]


def test_select_fold_explicit_matching_and_errors() -> None:
    """Exact labels win, unique normalized matches are accepted, and ambiguity or absence is rejected."""

    assert select_fold(_splits("Hurricane Ian", "Hurricane Ida"), holdout_event="Hurricane Ida")[3] == "explicit"
    assert select_fold(_splits("Hurricane Ian", "Hurricane Ida"), holdout_event="hurricane-ida")[0] == "Hurricane Ida"
    with pytest.raises(ValueError, match="matched multiple folds"):
        select_fold(_splits("Hurricane Ida", "hurricane_ida"), holdout_event="HURRICANE IDA")
    with pytest.raises(ValueError, match="not found in manifest"):
        select_fold(_splits("Hurricane Ian"), holdout_event="Hurricane Ida")


def test_select_fold_default_spatial_resolution() -> None:
    """Without an explicit holdout, exactly one spatial-block fold must exist."""

    holdout, _train, _val, selection = select_fold(_splits("Hurricane Ian", "Spatial_Block_East"), holdout_event=None)
    assert (holdout, selection) == ("Spatial_Block_East", "default_spatial")
    with pytest.raises(ValueError, match="ambiguous"):
        select_fold(_splits("Spatial_Block_East", "spatial-block-east-b"), holdout_event=None)
    with pytest.raises(ValueError, match="No default spatial holdout event found"):
        select_fold(_splits("Hurricane Ian"), holdout_event=None)


def test_build_spatial_split_manifest_prefers_encoded_blocks() -> None:
    """Explicit spatial_block labels in image names are used; any gap collapses to one synthetic event."""

    encoded = pd.DataFrame({"image_name": ["Ian_Spatial-Block-East_01", "Ian_spatial_block_west_02"]})
    assert build_spatial_split_manifest(encoded)["event"].tolist() == ["spatial_block_east", "spatial_block_west"]
    mixed = pd.DataFrame({"image_name": ["Ian_spatial_block_east_01", "Ian_02"]})
    assert build_spatial_split_manifest(mixed)["event"].tolist() == ["Spatial_Block", "Spatial_Block"]


def test_build_valid_manifest_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The scan is filtered to valid rows, corrupt counts are reported, and events are normalized."""

    scanned = pd.DataFrame({
        "image_path": ["a.tif", "b.tif"], "label_path": ["a.json", "b.json"],
        "image_name": ["Ian_01", "Ida_01"], "valid": [True, False], "event": ["Hurricane Ian", None]
    })
    monkeypatch.setattr(sampling, "scan_dataset", lambda _root: scanned)
    monkeypatch.setattr(sampling, "validate_integrity", lambda _manifest: {"corrupt": 1})

    valid = build_valid_manifest(data_dir=str(tmp_path), sensor_profile="uas_5cm")
    assert valid["image_name"].tolist() == ["Ian_01"]
    assert valid["event"].tolist() == ["Hurricane Ian"]
    assert "Found 1 corrupt images" in capsys.readouterr().out

    monkeypatch.setattr(sampling, "scan_dataset", lambda _root: scanned.drop(columns=["valid"]))
    with pytest.raises(ValueError, match="required manifest columns"):
        build_valid_manifest(data_dir=str(tmp_path), sensor_profile="uas_5cm")

    monkeypatch.setattr(sampling, "scan_dataset", lambda _root: scanned.assign(valid=[False, False]))
    monkeypatch.setattr(sampling, "validate_integrity", lambda _manifest: {"corrupt": 0})
    with pytest.raises(ValueError, match="No valid records"):
        build_valid_manifest(data_dir=str(tmp_path), sensor_profile="uas_5cm")
