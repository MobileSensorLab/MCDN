"""Tests for `src.data.io` parsing and inventory scanning."""

import json

from pathlib import Path

import pandas as pd

from src.data.io import parse_crasar_json, scan_dataset


def test_parse_crasar_json_missing_file_returns_empty() -> None:
    """Missing JSON path returns an empty GeoDataFrame."""

    gdf = parse_crasar_json("missing_file.json")
    assert gdf.empty
    assert str(gdf.crs) == "EPSG:4326"


def test_parse_crasar_json_directory_path_returns_empty(tmp_path: Path) -> None:
    """Directory input raises OSError internally and returns empty data."""

    gdf = parse_crasar_json(tmp_path)
    assert gdf.empty
    assert str(gdf.crs) == "EPSG:4326"


def test_parse_crasar_json_skips_bad_items_and_repairs_bowtie(tmp_path: Path) -> None:
    """Parser skips malformed items and keeps valid or repaired polygons."""

    payload = [
        "not-a-dict",
        {"EPSG:4326": [{"lon": 0.0, "lat": 0.0}, {"lon": 1.0, "lat": 1.0}]},
        {"EPSG:4326": [{"lon": 0.0, "lat": 0.0}, {"lon": 1.0, "lat": 1.0}, {"lon": 1.0, "lat": 0.0}, {"lon": 0.0, "lat": 1.0}], "label": "minor damage"},
        {"EPSG:4326": [{"lon": "bad", "lat": 0.0}, {"lon": 1.0, "lat": 0.0}, {"lon": 1.0, "lat": 1.0}], "label": "destroyed"},
        {"EPSG:4326": [{"lon": 2.0, "lat": 2.0}, {"lon": 3.0, "lat": 2.0}, {"lon": 3.0, "lat": 3.0}], "label": "no damage"}
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gdf = parse_crasar_json(path)
    assert not gdf.empty
    assert len(gdf) == 2
    assert set(gdf["label"]) == {"minor damage", "no damage"}


def test_parse_crasar_json_no_valid_polygons_returns_empty(tmp_path: Path) -> None:
    """Parser returns empty data when every candidate is filtered out."""

    payload = [
        {"wrong_key": []},
        {"EPSG:4326": [{"lon": 0.0, "lat": 0.0}, {"lon": 1.0, "lat": 1.0}]}
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    gdf = parse_crasar_json(path)
    assert gdf.empty


def test_scan_dataset_defaults_with_missing_dirs_returns_empty(tmp_path: Path) -> None:
    """Default split/sensor scan warns and returns empty schema when dirs are missing."""

    df = scan_dataset(tmp_path)
    assert df.empty
    assert list(df.columns) == ["image_path", "label_path", "alignment_path", "split", "sensor", "image_name", "valid"]


def test_scan_dataset_no_images_found_returns_empty(tmp_path: Path) -> None:
    """Existing image directory with no tif files returns empty schema."""

    image_dir = tmp_path / "train" / "imagery" / "UAS"
    image_dir.mkdir(parents=True)

    df = scan_dataset(tmp_path, splits=["train"], sensor_types=["UAS"])
    assert df.empty


def test_scan_dataset_orphans_and_valid_pairs(tmp_path: Path) -> None:
    """Scanner correctly handles orphan imagery, labels, and alignment files."""

    image_dir = tmp_path / "train" / "imagery" / "UAS"
    label_dir = tmp_path / "train" / "annotations" / "UAS" / "building_damage_assessment"
    align_dir = tmp_path / "train" / "annotations" / "UAS" / "building_alignment_adjustments"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True)

    valid_image = image_dir / "valid.geo.tif"
    orphan_image = image_dir / "orphan.geo.tif"
    valid_image.write_bytes(b"")
    orphan_image.write_bytes(b"")

    valid_label = label_dir / "valid.geo.tif.json"
    valid_align = align_dir / "valid.geo.tif.json"
    valid_label.write_text("[]", encoding="utf-8")
    valid_align.write_text("[]", encoding="utf-8")

    strict_df = scan_dataset(tmp_path, splits=["train"], sensor_types=["UAS"], include_orphans=False)
    assert len(strict_df) == 1
    assert strict_df.iloc[0]["image_name"] == "valid.geo.tif"
    assert bool(strict_df.iloc[0]["valid"]) is True
    assert strict_df.iloc[0]["label_path"] is not None
    assert strict_df.iloc[0]["alignment_path"] is not None

    lax_df = scan_dataset(tmp_path, splits=["train"], sensor_types=["UAS"], include_orphans=True)
    assert len(lax_df) == 2
    orphan_row = lax_df[lax_df["image_name"] == "orphan.geo.tif"].iloc[0]
    assert bool(orphan_row["valid"]) is False
    assert pd.isna(orphan_row["label_path"])
    assert pd.isna(orphan_row["alignment_path"])


def test_scan_dataset_orphans_without_annotation_dirs(tmp_path: Path) -> None:
    """Orphan rows are still included when annotation folders do not exist."""

    image_dir = tmp_path / "test" / "imagery" / "UAS"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.geo.tif"
    image_path.write_bytes(b"")

    df = scan_dataset(tmp_path, splits=["test"], sensor_types=["UAS"], include_orphans=True)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["image_name"] == "sample.geo.tif"
    assert bool(row["valid"]) is False
    assert pd.isna(row["label_path"])
    assert pd.isna(row["alignment_path"])


def test_scan_dataset_auto_discovers_sensor_dirs_when_unspecified(tmp_path: Path) -> None:
    """Default scan discovers all sensor subdirectories under split imagery roots."""

    uas_image_dir = tmp_path / "train" / "imagery" / "UAS"
    sat_image_dir = tmp_path / "train" / "imagery" / "SAT"
    uas_label_dir = tmp_path / "train" / "annotations" / "UAS" / "building_damage_assessment"
    sat_label_dir = tmp_path / "train" / "annotations" / "SAT" / "building_damage_assessment"
    uas_image_dir.mkdir(parents=True)
    sat_image_dir.mkdir(parents=True)
    uas_label_dir.mkdir(parents=True)
    sat_label_dir.mkdir(parents=True)

    uas_image = uas_image_dir / "u_sample.geo.tif"
    sat_image = sat_image_dir / "s_sample.geo.tif"
    uas_image.write_bytes(b"")
    sat_image.write_bytes(b"")
    (uas_label_dir / "u_sample.geo.tif.json").write_text("[]", encoding="utf-8")
    (sat_label_dir / "s_sample.geo.tif.json").write_text("[]", encoding="utf-8")

    df = scan_dataset(tmp_path, splits=["train"], sensor_types=None, include_orphans=False)
    assert len(df) == 2
    assert set(df["sensor"]) == {"UAS", "SAT"}
