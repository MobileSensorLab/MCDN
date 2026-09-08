"""Geometry and spatial validation utilities for CRASAR processing."""

import json
import math
import rasterio

import geopandas as gpd
import pandas as pd

from pathlib import Path
from rasterio.transform import rowcol, xy
from shapely import make_valid
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.data.io import parse_crasar_json

# Sliver filter: thinness ratio = 4*pi*area/perimeter^2 (1=circle, 0=line). Keep parts >= threshold.
_MIN_THINNESS_RATIO = 0.025
_MIN_CLIP_PART_AREA = 1e-12

# Ordinal damage labels used for training; excludes "un-classified" and "obscured".
VALID_ORDINAL_LABELS: frozenset[str] = frozenset({
    "no damage",
    "minor damage",
    "major damage",
    "destroyed"
})


def raster_ground_gsd_m(raster_path: str | Path) -> float:
    """Geometric-mean ground sample distance of a raster in meters, CRS-aware.

    Projected rasters return their native grid step directly. Geographic-CRS rasters
    (the crewed NOAA products are gridded in degrees) convert the per-axis degree steps
    to meters at the raster's central latitude, where ground pixels are anisotropic by
    the cos(latitude) factor; the geometric mean collapses the two axes to one scalar.

    Args:
        raster_path: Path to the raster.

    Returns:
        Ground sample distance in meters per pixel (geometric mean of the two axes).
    """

    with rasterio.open(raster_path) as src:
        step_x, step_y = abs(src.transform.a), abs(src.transform.e)
        if src.crs is None or not src.crs.is_geographic:
            return math.sqrt(step_x * step_y)
        lat = math.radians((src.bounds.bottom + src.bounds.top) / 2.0)
        m_per_deg_lat = 111132.954 - 559.822 * math.cos(2.0 * lat) + 1.175 * math.cos(4.0 * lat)
        m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3.0 * lat)
        return math.sqrt((step_x * m_per_deg_lon) * (step_y * m_per_deg_lat))


def filter_polygons_valid_labels(gdf: gpd.GeoDataFrame, valid_labels: frozenset[str] | set[str] | None = None) -> gpd.GeoDataFrame:
    """Keep polygons whose label is in the valid ordinal set.

    Args:
        gdf: Input GeoDataFrame with a label column.
        valid_labels: Optional override set of allowed labels. Defaults to VALID_ORDINAL_LABELS.

    Returns:
        A filtered GeoDataFrame copy containing only rows with allowed non-null labels.
    """

    labels = VALID_ORDINAL_LABELS if valid_labels is None else valid_labels
    if "label" not in gdf.columns:
        return gdf.iloc[0:0].copy()
    return gdf[gdf["label"].notna() & gdf["label"].isin(labels)].copy()


def _safe_intersection(geom: BaseGeometry, clip_box: BaseGeometry) -> BaseGeometry:
    """Intersect a geometry with bounds, retrying after repair if needed.

    Args:
        geom: Input geometry to clip.
        clip_box: Clipping geometry, typically a bounding box polygon.

    Returns:
        The clipped geometry result.
    """

    try:
        return geom.intersection(clip_box)
    except GEOSException:
        return make_valid(geom).intersection(clip_box)


def _polygon_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Collect polygon parts from polygon-like geometry containers.

    Args:
        geom: Input geometry to decompose.

    Returns:
        A list of non-empty polygon geometries extracted from the input.
    """

    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [item for item in geom.geoms if item.geom_type == "Polygon" and not item.is_empty]
    return []


def _thinness_ratio(geom: BaseGeometry) -> float:
    """Compute a shape thinness ratio for sliver filtering.

    Args:
        geom: Polygon geometry to score.

    Returns:
        Thinness ratio in [0, 1], where values near 0 indicate sliver-like polygons.
    """

    if geom is None or geom.is_empty or geom.area <= 0:
        return 0.0
    perimeter = geom.length
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * geom.area / (perimeter * perimeter)


def _normalize_clipped(geom: BaseGeometry) -> BaseGeometry:
    """Normalize clipped geometry by removing tiny or sliver polygon parts.

    Args:
        geom: Geometry produced by clipping, potentially multipart.

    Returns:
        A Polygon or MultiPolygon containing only retained parts, or an empty polygon.
    """

    parts = _polygon_parts(geom)
    if not parts:
        return Polygon()
    kept_parts = [part for part in parts if part.area >= _MIN_CLIP_PART_AREA and _thinness_ratio(part) >= _MIN_THINNESS_RATIO]
    if not kept_parts:
        return Polygon()
    if len(kept_parts) == 1:
        return kept_parts[0]
    return MultiPolygon(kept_parts)


def clip_polygons_to_bounds(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Clip polygons to raster-like bounds and drop empty results.

    Args:
        gdf: Input polygons to clip.
        bounds: Clipping bounds as (minx, miny, maxx, maxy).

    Returns:
        A clipped GeoDataFrame with empty geometries removed.
    """

    if gdf.empty:
        return gdf.copy()

    clip_box = box(*bounds)
    clipped = gdf.copy()

    def clip_one(geometry: BaseGeometry) -> BaseGeometry:
        result = _safe_intersection(geometry, clip_box)
        if result.is_empty:
            return result
        if result.geom_type == "GeometryCollection":
            return _normalize_clipped(result)
        return result

    clipped["geometry"] = clipped.geometry.apply(clip_one)
    return clipped[~clipped.geometry.is_empty].reset_index(drop=True)


def verify_crs_match(raster_path: str | Path, vector_path: str | Path) -> tuple[bool, object, object]:
    """Check whether raster and vector coordinate reference systems match.

    Args:
        raster_path: Path to raster image.
        vector_path: Path to CRASAR annotation JSON.

    Returns:
        Tuple of (is_match, raster_crs, vector_crs).
    """

    with rasterio.open(raster_path) as raster:
        raster_crs = raster.crs
    gdf = parse_crasar_json(vector_path)
    vector_crs = gdf.crs if not gdf.empty else None
    is_match = raster_crs is not None and vector_crs is not None and raster_crs == vector_crs
    return is_match, raster_crs, vector_crs


def align_vector_to_raster(gdf: gpd.GeoDataFrame, raster_path: str | Path) -> gpd.GeoDataFrame:
    """Reproject vectors to match raster CRS when required.

    Args:
        gdf: Input GeoDataFrame.
        raster_path: Path to target raster defining the destination CRS.

    Returns:
        A GeoDataFrame in the raster CRS, or an unchanged copy when already aligned.
    """

    if gdf.empty:
        return gdf.copy()
    with rasterio.open(raster_path) as raster:
        target_crs = raster.crs
    if gdf.crs == target_crs:
        return gdf.copy()
    return gdf.to_crs(target_crs)


def validate_integrity(inventory_df: pd.DataFrame) -> dict[str, int | list[str]]:
    """Validate inventory readability and vector-raster CRS consistency.

    Args:
        inventory_df: Dataset inventory with image_path, label_path, and valid columns.

    Returns:
        Summary dictionary with totals, error counters, and detailed error messages.
    """

    results: dict[str, int | list[str]] = {
        "total": len(inventory_df),
        "valid_image": 0,
        "valid_vector": 0,
        "crs_mismatch": 0,
        "corrupt": 0,
        "errors": []
    }

    print(f"Validating integrity for {len(inventory_df)} records...")
    for index, (_, row) in enumerate(inventory_df.iterrows()):
        if index > 0 and index % 100 == 0:
            print(f"Processed {index}/{len(inventory_df)}...")

        image_path = Path(row["image_path"])
        vector_path = Path(row["label_path"])

        try:
            with rasterio.open(image_path) as source:
                image_crs = source.crs
                results["valid_image"] = int(results["valid_image"]) + 1

            if row["valid"] and vector_path.exists():
                vector_gdf = parse_crasar_json(vector_path)
                results["valid_vector"] = int(results["valid_vector"]) + 1
                if vector_gdf.crs != image_crs:
                    results["crs_mismatch"] = int(results["crs_mismatch"]) + 1
                    error_message = f"CRS Mismatch: {image_path.name} ({image_crs}) vs {vector_path.name} ({vector_gdf.crs})"
                    results["errors"].append(error_message)
        except Exception as error:
            results["corrupt"] = int(results["corrupt"]) + 1
            results["errors"].append(f"Corrupt/Error: {image_path.name} - {error!s}")

    return results


def _parse_one_segment(segment: list) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Parse one alignment segment into source and destination pixel tuples.

    Args:
        segment: Raw alignment segment shaped like [[x_from, y_from], [x_to, y_to]].

    Returns:
        Rounded integer pixel tuples ((x_from, y_from), (x_to, y_to)), or None when invalid.
    """

    if not isinstance(segment, list) or len(segment) != 2:
        return None
    from_pt, to_pt = segment[0], segment[1]
    if not isinstance(from_pt, list | tuple) or len(from_pt) != 2:
        return None
    if not isinstance(to_pt, list | tuple) or len(to_pt) != 2:
        return None

    try:
        x_from = round(float(from_pt[0]))
        y_from = round(float(from_pt[1]))
        x_to = round(float(to_pt[0]))
        y_to = round(float(to_pt[1]))
    except (TypeError, ValueError):
        return None
    return (x_from, y_from), (x_to, y_to)


def load_alignment_adjustments(alignment_path: Path | str | None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Load CRASAR alignment tie-point segments from JSON.

    Args:
        alignment_path: Path to optional alignment JSON file.

    Returns:
        List of parsed source/destination pixel segment pairs.
    """

    if alignment_path is None:
        return []

    path = Path(alignment_path)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    result: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for segment in payload:
        parsed = _parse_one_segment(segment)
        if parsed is not None:
            result.append(parsed)
    return result


def apply_alignment_adjustments(gdf: gpd.GeoDataFrame, raster_path: Path | str,
                                adjustments: list[tuple[tuple[int, int], tuple[int, int]]]) -> gpd.GeoDataFrame:
    """Apply nearest tie-point pixel shifts to each polygon geometry.

    Args:
        gdf: Input vector geometries to adjust.
        raster_path: Path to raster providing pixel-to-world transform.
        adjustments: Alignment segments as ((x_from, y_from), (x_to, y_to)).

    Returns:
        A GeoDataFrame copy with transformed geometries.
    """

    if not adjustments:
        return gdf.copy()

    with rasterio.open(raster_path) as source:
        transform = source.transform

    from_points = [segment[0] for segment in adjustments]
    to_points = [segment[1] for segment in adjustments]

    def transform_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
        if geometry is None or geometry.is_empty:
            return geometry

        centroid_x, centroid_y = geometry.centroid.x, geometry.centroid.y
        row_center, col_center = rowcol(transform, centroid_x, centroid_y)
        nearest_index = min(
            range(len(from_points)),
            key=lambda idx: math.hypot(col_center - from_points[idx][0], row_center - from_points[idx][1])
        )
        dx = to_points[nearest_index][0] - from_points[nearest_index][0]
        dy = to_points[nearest_index][1] - from_points[nearest_index][1]

        def vertex_transform(x_coord: float, y_coord: float) -> tuple[float, float]:
            row, col = rowcol(transform, x_coord, y_coord)
            x_geo, y_geo = xy(transform, row + dy, col + dx)
            return x_geo, y_geo

        return shapely_transform(vertex_transform, geometry)

    output = gdf.copy()
    output.geometry = output.geometry.apply(transform_geometry)
    return output


