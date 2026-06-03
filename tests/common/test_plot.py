"""Tests for common.plot."""

from pathlib import Path


import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon

from src.common.plot import (
    SEVERITY_COLORS,
    fill_nodata_white,
    geometry_to_pixel,
    overlay_building_boundaries,
    plot_alignment_verification,
    plot_ortho,
    rgb_band_first_to_channel_last,
)


def test_fill_nodata_white_uses_valid_mask() -> None:
    """Pixels where valid_mask == 0 become white; others unchanged."""
    rgb = np.full((3, 3, 3), 100, dtype=np.uint8)
    valid_mask = np.ones((3, 3), dtype=np.uint8) * 255
    valid_mask[0, 0] = 0
    valid_mask[2, 2] = 0
    out = fill_nodata_white(rgb, valid_mask)
    assert out.shape == (3, 3, 3)
    np.testing.assert_array_equal(out[1, 1], [100, 100, 100])
    np.testing.assert_array_equal(out[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(out[2, 2], [255, 255, 255])


def test_rgb_band_first_to_channel_last_shape() -> None:
    """Band-first (3, H, W) becomes (H, W, 3)."""
    rgb = np.zeros((3, 10, 20), dtype=np.uint8)
    out = rgb_band_first_to_channel_last(rgb)
    assert out.shape == (10, 20, 3)


def test_rgb_band_first_to_channel_last_already_channel_last() -> None:
    """Channel-last (H, W, 3) is returned unchanged."""
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    out = rgb_band_first_to_channel_last(rgb)
    assert out.shape == (10, 20, 3)
    assert np.shares_memory(out, rgb) or np.array_equal(out, rgb)


def test_severity_colors_four_keys() -> None:
    """SEVERITY_COLORS has exactly the four ordinal labels."""
    expected = {"no damage", "minor damage", "major damage", "destroyed"}
    assert set(SEVERITY_COLORS.keys()) == expected
    assert all(isinstance(v, str) for v in SEVERITY_COLORS.values())


def test_geometry_to_pixel_none_empty() -> None:
    """geometry_to_pixel returns None for None or empty geometry."""
    transform = from_bounds(0, 0, 10, 10, 100, 100)
    assert geometry_to_pixel(None, transform) is None
    empty = Polygon()
    assert geometry_to_pixel(empty, transform) is None


def test_geometry_to_pixel_simple() -> None:
    """geometry_to_pixel maps a small polygon into pixel range."""
    transform = from_bounds(0, 0, 10, 10, 100, 100)
    poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])
    out = geometry_to_pixel(poly, transform)
    assert out is not None
    assert not out.is_empty
    xs, ys = out.exterior.xy
    assert all(0 <= x <= 100 for x in xs)
    assert all(0 <= y <= 100 for y in ys)


def test_overlay_building_boundaries_smoke() -> None:
    """overlay_building_boundaries runs without error and adds artists."""
    transform = from_bounds(0, 0, 10, 10, 50, 50)
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    gdf = gpd.GeoDataFrame(geometry=[poly], data={"label": ["no damage"]})
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.zeros((50, 50, 3), dtype=np.uint8), origin="upper", extent=[0, 50, 50, 0])
    overlay_building_boundaries(ax, gdf, transform)
    handles, _ = ax.get_legend_handles_labels()
    assert len(handles) >= 1
    plt.close(fig)


def test_overlay_building_boundaries_multipolygon() -> None:
    """overlay_building_boundaries draws MultiPolygon parts (one per poly in geoms)."""
    transform = from_bounds(0, 0, 10, 10, 50, 50)
    p1 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])
    p2 = Polygon([(4, 4), (5, 4), (5, 5), (4, 5), (4, 4)])
    multi = MultiPolygon([p1, p2])
    gdf = gpd.GeoDataFrame(geometry=[multi], data={"label": ["minor damage"]})
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.zeros((50, 50, 3), dtype=np.uint8), origin="upper", extent=[0, 50, 50, 0])
    overlay_building_boundaries(ax, gdf, transform)
    handles, _ = ax.get_legend_handles_labels()
    assert len(handles) >= 1
    plt.close(fig)


def test_overlay_building_boundaries_no_label_column() -> None:
    """overlay_building_boundaries does not raise when label column is missing; skips per-label loop."""
    transform = from_bounds(0, 0, 10, 10, 50, 50)
    poly = Polygon([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    gdf = gpd.GeoDataFrame(geometry=[poly])  # no "label" column
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.zeros((50, 50, 3), dtype=np.uint8), origin="upper", extent=[0, 50, 50, 0])
    overlay_building_boundaries(ax, gdf, transform)
    plt.close(fig)


def test_plot_alignment_verification_smoke() -> None:
    """plot_alignment_verification returns fig, ax and does not raise."""
    rgb = np.zeros((3, 40, 40), dtype=np.uint8)
    transform = from_bounds(0, 0, 40, 40, 40, 40)
    poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)])
    gdf = gpd.GeoDataFrame(geometry=[poly], data={"label": ["destroyed"]})
    fig, ax = plot_alignment_verification(rgb, transform, gdf, title="Test")
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_ortho_smoke(tmp_path: Path) -> None:
    """plot_ortho returns fig, ax; with extents adds a patch."""
    raster_path = tmp_path / "small.tif"
    w, h = 50, 50
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=w,
        height=h,
        dtype=np.uint8,
        transform=from_bounds(0, 0, 100, 100, w, h),
        crs="EPSG:32616",
    ) as dst:
        dst.write(np.zeros((3, h, w), dtype=np.uint8))

    fig, ax = plot_ortho(raster_path)
    assert fig is not None
    assert ax is not None
    plt.close(fig)

    fig2, ax2 = plot_ortho(raster_path, extents=(10, 10, 30, 30))
    assert len(ax2.patches) == 1
    plt.close(fig2)

    win = Window(10, 10, 20, 20)
    fig3, ax3 = plot_ortho(raster_path, window=win)
    assert ax3.images
    plt.close(fig3)


def test_plot_ortho_existing_ax(tmp_path: Path) -> None:
    """plot_ortho with ax= draws on given axes and returns same figure."""
    raster_path = tmp_path / "small.tif"
    w, h = 50, 50
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=w,
        height=h,
        dtype=np.uint8,
        transform=from_bounds(0, 0, 100, 100, w, h),
        crs="EPSG:32616",
    ) as dst:
        dst.write(np.zeros((3, h, w), dtype=np.uint8))

    fig, ax = plt.subplots(1, 1)
    fig_out, ax_out = plot_ortho(raster_path, ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    assert ax.images
    plt.close(fig)


def test_plot_ortho_downscale_max(tmp_path: Path) -> None:
    """plot_ortho with downscale_max uses reduced resolution read path."""
    raster_path = tmp_path / "large.tif"
    w, h = 200, 200
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=w,
        height=h,
        dtype=np.uint8,
        transform=from_bounds(0, 0, 100, 100, w, h),
        crs="EPSG:32616",
    ) as dst:
        dst.write(np.zeros((3, h, w), dtype=np.uint8))

    fig, ax = plot_ortho(raster_path, downscale_max=100)
    assert fig is not None
    assert ax is not None
    assert ax.images
    plt.close(fig)


def test_plot_ortho_no_edge_black_nodata(tmp_path: Path) -> None:
    """plot_ortho with apply_edge_black_nodata=False skips edge-abutting black mask."""
    raster_path = tmp_path / "small.tif"
    w, h = 50, 50
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=3,
        width=w,
        height=h,
        dtype=np.uint8,
        transform=from_bounds(0, 0, 100, 100, w, h),
        crs="EPSG:32616",
    ) as dst:
        dst.write(np.zeros((3, h, w), dtype=np.uint8))

    fig, ax = plot_ortho(raster_path, apply_edge_black_nodata=False)
    assert fig is not None
    assert ax is not None
    assert ax.images
    plt.close(fig)
