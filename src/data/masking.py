"""Mask-building utilities for vector rasterization and edge no-data handling."""

import rasterio

import numpy as np

from affine import Affine
from pathlib import Path
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import Polygon, box, shape

from src.data.geometry import clip_polygons_to_bounds
from src.data.io import parse_crasar_json


def _extract_black_polygon_features(rgb: np.ndarray, transform: Affine, bounds: tuple[float, ...]) -> tuple[list[dict], list[dict], list[Polygon], Polygon]:
    """Extract pure-black polygons and edge-abutting polygon geometries.

    Args:
        rgb: RGB array with shape [3, H, W].
        transform: Affine transform for converting pixel and world coordinates.
        bounds: Raster bounds as (minx, miny, maxx, maxy).

    Returns:
        Tuple of (all_polygons, edge_abutting, edge_polygons, raster_box) where:
        - all_polygons are GeoJSON-like shape/value dictionaries.
        - edge_abutting are feature dictionaries that intersect the raster boundary.
        - edge_polygons are valid polygon geometries intersecting the raster boundary.
        - raster_box is the raster extent polygon.
    """

    red, green, blue = rgb[0], rgb[1], rgb[2]
    pure_black = ((red == 0) & (green == 0) & (blue == 0)).astype(np.uint8)
    all_polygons = [
        {"geom": geom, "value": int(value)}
        for geom, value in features.shapes(pure_black, mask=pure_black.astype(bool), transform=transform)
    ]

    raster_box = box(*bounds)
    raster_boundary = raster_box.boundary
    edge_abutting: list[dict] = []
    edge_polygons: list[Polygon] = []
    for item in all_polygons:
        polygon = shape(item["geom"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if raster_boundary is not None and polygon.intersects(raster_boundary):
            edge_abutting.append(item)
            edge_polygons.append(polygon)

    return all_polygons, edge_abutting, edge_polygons, raster_box


def rasterize_vectors(image_path: str | Path, vector_path: str | Path, window: Window | None = None) -> tuple[np.ndarray, dict]:
    """Rasterize vector footprints to full-image or window-aligned mask.

    Args:
        image_path: Path to reference raster.
        vector_path: Path to CRASAR vector JSON.
        window: Optional window for chip-level rasterization.

    Returns:
        Tuple of (mask, meta) where mask is uint8 [H, W].
    """

    image = Path(image_path)
    vector = Path(vector_path)
    with rasterio.open(image) as source:
        meta = source.meta.copy()
        image_crs = source.crs
        if window is not None:
            height = int(window.height)
            width = int(window.width)
            transform = source.window_transform(window)
            image_bounds = box(*source.window_bounds(window))
        else:
            height, width = source.shape
            transform = source.transform
            image_bounds = box(*source.bounds)

    gdf = parse_crasar_json(vector)
    if not gdf.empty and gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)

    gdf_intersecting = gdf[gdf.intersects(image_bounds)] if not gdf.empty else gdf
    gdf_clipped = clip_polygons_to_bounds(gdf=gdf_intersecting, bounds=image_bounds.bounds)
    if gdf_clipped.empty:
        return np.zeros((height, width), dtype=np.uint8), meta

    shapes = ((geometry, 1) for geometry in gdf_clipped.geometry)
    mask = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    return mask, meta


def polygonize_black_regions(raster_path: Path | str, out_shape: tuple[int, int] | None = None) -> tuple[list[dict], list[dict], Polygon]:
    """Extract pure-black RGB polygons and identify edge-abutting regions.

    Args:
        raster_path: Path to the source raster.
        out_shape: Optional (height, width) for resampled extraction space.

    Returns:
        Tuple of (all_polygons, edge_abutting, raster_box) where:
        - all_polygons are all pure-black polygon features.
        - edge_abutting are pure-black polygons intersecting raster boundary.
        - raster_box is the raster extent polygon.
    """

    path = Path(raster_path)
    with rasterio.open(path) as source:
        bounds = source.bounds
        minx, miny, maxx, maxy = bounds
        if out_shape is not None:
            out_height, out_width = out_shape
            rgb = source.read(indexes=(1, 2, 3), out_shape=(3, out_height, out_width))
            transform = from_bounds(minx, miny, maxx, maxy, out_width, out_height)
        else:
            rgb = source.read(indexes=(1, 2, 3))
            transform = source.transform

    all_polygons, edge_abutting, _, raster_box = _extract_black_polygon_features(rgb=rgb, transform=transform, bounds=bounds)
    return all_polygons, edge_abutting, raster_box


def edge_black_nodata_mask(rgb: np.ndarray, transform: Affine, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Build a valid-data mask using edge-abutting pure-black regions as nodata.

    Args:
        rgb: RGB array with shape [3, H, W].
        transform: Affine transform used for vectorization/rasterization.
        bounds: Raster bounds as (minx, miny, maxx, maxy).

    Returns:
        A uint8 mask of shape [H, W] with 255 as valid data and 0 as nodata.
    """

    _, _, edge_polygons, _ = _extract_black_polygon_features(rgb=rgb, transform=transform, bounds=bounds)

    height, width = rgb.shape[1], rgb.shape[2]
    if not edge_polygons:
        return np.full((height, width), 255, dtype=np.uint8)

    shapes_to_burn = [(geometry, 0) for geometry in edge_polygons]
    return features.rasterize(
        shapes=shapes_to_burn,
        out_shape=(height, width),
        transform=transform,
        fill=255,
        default_value=0,
        dtype=np.uint8
    )
