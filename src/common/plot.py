"""Atomic plotting helpers for RGB chips and building-footprint overlays."""

import rasterio

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pathlib import Path
from rasterio.transform import Affine, from_bounds
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.data.masking import edge_black_nodata_mask

# Ordinal damage labels to display color (FEMA-inspired severity).
SEVERITY_COLORS: dict[str, str] = {
    "no damage": "lime",
    "minor damage": "gold",
    "major damage": "darkorange",
    "destroyed": "red",
}


def rgb_band_first_to_channel_last(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB from (C, H, W) to (H, W, C) for matplotlib imshow.

    Args:
        rgb: Array of shape (3, H, W) or (H, W, 3). If already channel-last, returned unchanged.

    Returns:
        Array of shape (H, W, 3).
    """

    if rgb.ndim == 3 and rgb.shape[-1] in (3, 4):
        return np.asarray(rgb)
    return np.moveaxis(rgb, 0, -1)


def fill_nodata_white(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Replace nodata pixels with white using rasterio-style valid data mask.

    Uses the GDAL convention: valid_mask has 0 = nodata, non-zero (typically 255) = valid.
    Pixels where valid_mask == 0 are set to white for display.

    Args:
        rgb: RGB array (H, W, 3), channel-last, uint8 or float.
        valid_mask: 2D uint8 from rasterio dataset_mask(); 0 = nodata, 255 = valid.

    Returns:
        Copy of rgb with nodata pixels set to white.
    """

    out = np.array(rgb, copy=True)
    is_dtype_float = np.issubdtype(out.dtype, np.floating)
    white = 1.0 if is_dtype_float else 255
    out[valid_mask == 0, :] = white
    return out


def geometry_to_pixel(geom: BaseGeometry | None, transform: Affine) -> BaseGeometry | None:
    """Transform a Shapely geometry from map coordinates to pixel coordinates.

    Pixel convention matches imshow(..., origin="upper", extent=[0, width, height, 0]): (col, row) with row 0 at top.

    Args:
        geom: Shapely geometry in map CRS (same as transform).
        transform: Rasterio Affine transform (map -> pixel).

    Returns:
        New geometry in pixel (col, row) coordinates, or None if geom is None/empty.
    """

    if geom is None or geom.is_empty:
        return None

    inv = ~transform

    def map_to_pixel(x: float, y: float) -> tuple[float, float]:
        col, row = inv * (x, y)
        return float(col), float(row)

    return shapely_transform(map_to_pixel, geom)


def overlay_building_boundaries(ax: Axes, gdf: gpd.GeoDataFrame, transform: Affine, label_column: str = "label",
                                color_map: dict[str, str] | None = None, linewidth: int = 2) -> None:
    """Draw building polygon boundaries on an axes in pixel space.

    Transforms geometries using transform, then plots exteriors by label.
    Legend entries are deduplicated by label.

    Args:
        ax: Matplotlib axes (e.g. from imshow with extent=[0, w, h, 0], origin="upper").
        gdf: GeoDataFrame with geometry in map CRS and a label_column column.
        transform: Rasterio Affine (map -> pixel).
        label_column: Column name for color-by-label.
        color_map: Label -> color. Defaults to SEVERITY_COLORS.
        linewidth: Line width for boundaries.
    """

    if color_map is None:
        color_map = SEVERITY_COLORS

    for label in color_map:
        if label_column not in gdf.columns:
            continue
        subset = gdf[gdf[label_column] == label]
        if subset.empty:
            continue
        color = color_map[label]
        for geom in subset.geometry:
            px_geom = geometry_to_pixel(geom, transform)
            if px_geom is None or px_geom.is_empty:
                continue
            if px_geom.geom_type == "Polygon":
                xs, ys = px_geom.exterior.xy
                ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)
            elif px_geom.geom_type == "MultiPolygon":
                for poly in px_geom.geoms:
                    xs, ys = poly.exterior.xy
                    ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)

    handles, labels_seen = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_seen, handles, strict=True))
    ax.legend(by_label.values(), [key.title() for key in by_label], loc="upper left")
    # ax.legend(by_label.values(), by_label.keys(), loc="upper left")


def plot_ortho(raster_path: Path | str, window: Window | None = None, extents: tuple[float, float, float, float] | None = None,
               rgb_indexes: tuple[int, ...] = (1, 2, 3), downscale_max: int | None = None, rectangle_color: str = "red",
               ax: Axes | None = None, fig_size: tuple[float, float] = (7, 7), apply_edge_black_nodata: bool = True) -> tuple[Figure, Axes]:
    """Plot an orthomosaic (or a window) in map coordinates, optionally with a rectangle for extents.

    Args:
        raster_path: Path to the GeoTIFF.
        window: If set, read only this rasterio window (subsection view).
        extents: Optional (minx, miny, maxx, maxy) in map CRS; if set, draw a rectangle for this box.
        rgb_indexes: 1-based band indices for RGB. Default (1, 2, 3).
        downscale_max: If set and full raster is larger, read at reduced resolution so max dimension is this.
        rectangle_color: Color for the extents' rectangle. Ignored if extents is None.
        ax: Axis to draw upon; otherwise create a new figure.
        fig_size: Figure size in inches when ax is None.
        apply_edge_black_nodata: If True, for 3-band rasters with no nodata metadata, treat edge-abutting
            pure-black regions as nodata (filled white). Set False to show raw pixels.

    Returns:
        (fig, ax) for further use or plt.show().
    """

    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        count = src.count
        if window is not None:
            rgb = src.read(window=window, indexes=rgb_indexes)
            valid_mask = src.dataset_mask(window=window)
            bounds = src.window_bounds(window)
            transform = src.window_transform(window)
        else:
            h, w = src.height, src.width
            if downscale_max is not None and max(h, w) > downscale_max:
                scale = downscale_max / max(h, w)
                out_h = max(1, int(h * scale))
                out_w = max(1, int(w * scale))
                out_shape = (3, out_h, out_w)
                rgb = src.read(indexes=rgb_indexes, out_shape=out_shape)
                valid_mask = src.dataset_mask(out_shape=(out_h, out_w))
                bounds = src.bounds
                minx, miny, maxx, maxy = bounds
                transform = from_bounds(minx, miny, maxx, maxy, out_w, out_h)
            else:
                rgb = src.read(indexes=rgb_indexes)
                valid_mask = src.dataset_mask()
                bounds = src.bounds
                transform = src.transform
        minx, miny, maxx, maxy = bounds

    if apply_edge_black_nodata and count == 3 and np.all(valid_mask == 255):
        edge_mask = edge_black_nodata_mask(rgb, transform, (minx, miny, maxx, maxy))
        valid_mask = np.minimum(valid_mask, edge_mask)

    rgb_display = rgb_band_first_to_channel_last(rgb)
    rgb_display = fill_nodata_white(rgb_display, valid_mask)
    extent = (minx, maxx, miny, maxy)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        fig = ax.get_figure()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(rgb_display, origin="upper", extent=extent)
    if extents is not None:
        ex, ey, ex2, ey2 = extents
        rect = Rectangle(
            (ex, ey),
            ex2 - ex,
            ey2 - ey,
            fill=False,
            edgecolor=rectangle_color,
            linewidth=2,
        )
        ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


def plot_alignment_verification(rgb: np.ndarray, transform: Affine, gdf: gpd.GeoDataFrame,
                                title: str = "Building footprints by damage severity (2px outline)",
                                fig_size: tuple[float, float] = (7, 7)) -> tuple[Figure, Axes]:
    """Plot an RGB chip with building boundaries overlaid in pixel space.

    Args:
        rgb: RGB array (3, H, W) or (H, W, 3).
        transform: Rasterio Affine for the chip (map -> pixel).
        gdf: GeoDataFrame with geometry in map CRS and "label" column.
        title: Axes title.
        fig_size: Figure size in inches.

    Returns:
        (fig, ax) for further use or plt.show().
    """

    rgb_display = rgb_band_first_to_channel_last(rgb)
    height, width = rgb_display.shape[0], rgb_display.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.imshow(rgb_display, origin="upper", extent=[0, width, height, 0])
    overlay_building_boundaries(ax, gdf, transform, linewidth=2)
    ax.set_title(title)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig, ax
