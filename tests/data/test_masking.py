"""Tests for `src.data.masking` rasterization and black-edge nodata behavior."""

import json

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

from src.data.masking import edge_black_nodata_mask, polygonize_black_regions, rasterize_vectors


def _write_test_raster(
    path: Path,
    *,
    width: int = 20,
    height: int = 20,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 20.0, 20.0),
    crs: str = "EPSG:4326"
) -> None:
    """Write a small RGB GeoTIFF for deterministic masking tests."""

    transform = from_bounds(*bounds, width, height)
    rgb = np.full((3, height, width), 255, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=3,
        width=width,
        height=height,
        dtype=np.uint8,
        transform=transform,
        crs=crs
    ) as dst:
        dst.write(rgb)


def _write_vector_json(path: Path, coordinates: list[dict[str, float]], label: str = "minor damage") -> None:
    """Write one CRASAR-style polygon feature JSON file."""

    payload = [{"EPSG:4326": coordinates, "label": label}]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rasterize_vectors_full_image_has_positive_pixels(tmp_path: Path) -> None:
    """rasterize_vectors creates a non-empty full-image mask for intersecting polygons."""

    raster_path = tmp_path / "image.tif"
    vector_path = tmp_path / "labels.json"
    _write_test_raster(raster_path)
    _write_vector_json(
        vector_path,
        coordinates=[
            {"lon": 5.0, "lat": 5.0},
            {"lon": 8.0, "lat": 5.0},
            {"lon": 8.0, "lat": 8.0},
            {"lon": 5.0, "lat": 8.0}
        ]
    )

    mask, meta = rasterize_vectors(raster_path, vector_path)

    assert mask.shape == (20, 20)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) > 0
    assert meta["height"] == 20
    assert meta["width"] == 20


def test_rasterize_vectors_window_branch_respects_window_shape(tmp_path: Path) -> None:
    """Windowed rasterization uses window transform and output dimensions."""

    raster_path = tmp_path / "image.tif"
    vector_path = tmp_path / "labels.json"
    _write_test_raster(raster_path)
    _write_vector_json(
        vector_path,
        coordinates=[
            {"lon": 1.0, "lat": 16.0},
            {"lon": 3.0, "lat": 16.0},
            {"lon": 3.0, "lat": 18.0},
            {"lon": 1.0, "lat": 18.0}
        ]
    )

    window = Window(col_off=0, row_off=0, width=8, height=6)
    mask, _ = rasterize_vectors(raster_path, vector_path, window=window)

    assert mask.shape == (6, 8)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) > 0


def test_rasterize_vectors_returns_empty_mask_for_non_intersecting_polygon(tmp_path: Path) -> None:
    """Non-overlapping vectors produce an all-zero mask."""

    raster_path = tmp_path / "image.tif"
    vector_path = tmp_path / "labels.json"
    _write_test_raster(raster_path)
    _write_vector_json(
        vector_path,
        coordinates=[
            {"lon": 30.0, "lat": 30.0},
            {"lon": 35.0, "lat": 30.0},
            {"lon": 35.0, "lat": 35.0},
            {"lon": 30.0, "lat": 35.0}
        ]
    )

    mask, _ = rasterize_vectors(raster_path, vector_path)

    assert mask.shape == (20, 20)
    assert np.all(mask == 0)


def test_polygonize_black_regions_detects_edge_abutting_only(tmp_path: Path) -> None:
    """polygonize_black_regions separates all pure-black and edge-abutting sets."""

    raster_path = tmp_path / "black_regions.tif"
    width, height = 20, 20
    bounds = (0.0, 0.0, 20.0, 20.0)
    transform = from_bounds(*bounds, width, height)

    rgb = np.full((3, height, width), 255, dtype=np.uint8)
    rgb[:, 0:4, 0:4] = 0
    rgb[:, 8:12, 8:12] = 0
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=width,
        height=height,
        dtype=np.uint8,
        transform=transform,
        crs="EPSG:4326"
    ) as dst:
        dst.write(rgb)

    all_black, edge_abutting, raster_box = polygonize_black_regions(raster_path)

    assert len(all_black) >= 2
    assert len(edge_abutting) == 1
    assert raster_box.bounds == bounds


def test_edge_black_nodata_mask_marks_only_edge_black_regions() -> None:
    """edge_black_nodata_mask burns edge-touching black regions but keeps interior valid."""

    height, width = 20, 20
    bounds = (0.0, 0.0, 20.0, 20.0)
    transform = from_bounds(*bounds, width, height)
    rgb = np.full((3, height, width), 255, dtype=np.uint8)
    rgb[:, 0:4, 0:4] = 0
    rgb[:, 8:12, 8:12] = 0

    mask = edge_black_nodata_mask(rgb=rgb, transform=transform, bounds=bounds)

    assert mask.shape == (height, width)
    assert mask.dtype == np.uint8
    assert mask[0, 0] == 0
    assert mask[9, 9] == 255


def test_edge_black_nodata_mask_without_edge_black_returns_all_valid() -> None:
    """If black regions are interior-only, nodata mask remains all 255."""

    height, width = 20, 20
    bounds = (0.0, 0.0, 20.0, 20.0)
    transform = from_bounds(*bounds, width, height)
    rgb = np.full((3, height, width), 255, dtype=np.uint8)
    rgb[:, 8:12, 8:12] = 0

    mask = edge_black_nodata_mask(rgb=rgb, transform=transform, bounds=bounds)
    assert np.all(mask == 255)


def test_polygonize_and_edge_mask_consistency_on_shared_scene(tmp_path: Path) -> None:
    """polygonize_black_regions and edge_black_nodata_mask agree on edge-black pixels."""

    raster_path = tmp_path / "consistency.tif"
    width, height = 24, 24
    bounds = (0.0, 0.0, 24.0, 24.0)
    transform = from_bounds(*bounds, width, height)

    rgb = np.full((3, height, width), 255, dtype=np.uint8)
    rgb[:, 0:3, 0:3] = 0
    rgb[:, 12:16, 12:16] = 0
    rgb[:, 20:24, 20:24] = 0
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=width,
        height=height,
        dtype=np.uint8,
        transform=transform,
        crs="EPSG:4326"
    ) as dst:
        dst.write(rgb)

    _, edge_abutting, _ = polygonize_black_regions(raster_path)
    nodata_mask = edge_black_nodata_mask(rgb=rgb, transform=transform, bounds=bounds)

    assert len(edge_abutting) == 2
    assert int(np.count_nonzero(nodata_mask == 0)) > 0
    assert nodata_mask[1, 1] == 0
    assert nodata_mask[13, 13] == 255
    assert nodata_mask[22, 22] == 0
