import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon


def write_seed_dir(seed_dir: Path, *, checkpoint_bytes: bytes, sidecars: tuple[str, ...] = ("metrics.json", "config_resolved.yaml")) -> None:
    """Populate one ``seed_XX`` directory with a fake checkpoint and the named sidecars."""

    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "best_model.pt").write_bytes(checkpoint_bytes)
    for name in sidecars:
        (seed_dir / name).write_text(f"{seed_dir.name}:{name}\n", encoding="utf-8")


@pytest.fixture
def ablation_tree(tmp_path: Path) -> Path:
    """Fake ``outputs/ablation`` root: two arms, an excluded fold, an underscore dir, and a seed without a checkpoint."""

    root = tmp_path / "ablation"
    rng = np.random.default_rng(0)
    for arm, folds in {"all_features": ("Hurricane_Ida", "Spatial_Block_East"), "rgb_only": ("Hurricane_Ida",)}.items():
        for fold in folds:
            for seed in ("seed_00", "seed_11"):
                write_seed_dir(root / arm / fold / seed, checkpoint_bytes=rng.bytes(4096))
    (root / "rgb_only" / "Hurricane_Ida" / "seed_22").mkdir()  # no checkpoint: must be ignored
    (root / "_ensembles").mkdir()
    (root / "_ensembles" / "probs.pt").write_bytes(b"x")
    return root


@pytest.fixture
def mock_dataset_root(tmp_path: Path) -> Path:
    """Creates a temporary directory structure mimicking the CRASAR dataset."""
    root = tmp_path / "data"
    root.mkdir()

    for split in ["train", "test"]:
        (root / split / "imagery" / "UAS").mkdir(parents=True)
        (root / split / "annotations" / "UAS" / "building_damage_assessment").mkdir(parents=True)

    return root


@pytest.fixture
def sample_geotiff(mock_dataset_root: Path) -> Path:
    """Creates a dummy GeoTIFF image in standard EPSG:4326 (Lat/Lon)."""
    img_dir = mock_dataset_root / "train" / "imagery" / "UAS"
    img_path = img_dir / "test_image.geo.tif"

    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 3,
        "dtype": rasterio.uint8,
        "crs": "EPSG:4326",
        # Start at Lon 10.0, Lat 20.0, with a coarse 0.01 degree resolution
        "transform": from_origin(10.0, 20.0, 0.01, 0.01),
    }

    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(np.zeros((3, 100, 100), dtype=rasterio.uint8))

    return img_path


def write_crasar_json(path: Path, polygons: list[Polygon], labels: list[str] | None = None) -> None:
    """Helper to write polygons in the CRASAR custom format, with optional labels."""
    data = []
    for i, poly in enumerate(polygons):
        coords = [{"lon": x, "lat": y} for x, y in poly.exterior.coords]
        item = {"EPSG:4326": coords}
        if labels and i < len(labels):
            item["label"] = labels[i]
        data.append(item)

    with open(path, "w") as f:
        json.dump(data, f)


@pytest.fixture
def sample_geojson(mock_dataset_root: Path, sample_geotiff: Path) -> Path:
    """Creates a dummy Label matching the sample_geotiff."""
    lbl_dir = mock_dataset_root / "train" / "annotations" / "UAS" / "building_damage_assessment"
    lbl_path = lbl_dir / f"{sample_geotiff.name}.json"

    # Coordinates inside the 10.0-11.0 Lon and 19.0-20.0 Lat bounding box
    poly1 = Polygon([(10.4, 19.4), (10.6, 19.4), (10.6, 19.6), (10.4, 19.6), (10.4, 19.4)])
    poly2 = Polygon([(10.1, 19.8), (10.2, 19.8), (10.2, 19.9), (10.1, 19.9), (10.1, 19.8)])

    write_crasar_json(lbl_path, [poly1, poly2], labels=["minor damage", "destroyed"])
    return lbl_path


@pytest.fixture
def sample_manifest(sample_geotiff: Path, sample_geojson: Path) -> pd.DataFrame:
    """Creates a dummy manifest DataFrame for dataset instantiation."""
    return pd.DataFrame([{
        "image_path": str(sample_geotiff),
        "label_path": str(sample_geojson),
        "alignment_path": None,
        "event": "Hurricane Ian"
    }])

@pytest.fixture
def utm_geotiff(mock_dataset_root: Path) -> Path:
    """Creates a dummy GeoTIFF in UTM (EPSG:32616) to test mismatch logic."""
    img_dir = mock_dataset_root / "train" / "imagery" / "UAS"
    img_path = img_dir / "utm_image.geo.tif"
    profile = {
        "driver": "GTiff", "height": 100, "width": 100, "count": 3,
        "dtype": rasterio.uint8, "crs": "EPSG:32616",
        "transform": from_origin(500000, 4000000, 1.0, 1.0),
    }
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(np.zeros((3, 100, 100), dtype=rasterio.uint8))
    return img_path

@pytest.fixture
def mismatch_crs_geojson(mock_dataset_root: Path, utm_geotiff: Path) -> Path:
    """Creates a Label with mismatched CRS (implicitly EPSG:4326 via parser)."""
    lbl_dir = mock_dataset_root / "train" / "annotations" / "UAS" / "building_damage_assessment"
    lbl_path = lbl_dir / f"{utm_geotiff.name}_mismatch.json"
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    write_crasar_json(lbl_path, [poly])
    return lbl_path
