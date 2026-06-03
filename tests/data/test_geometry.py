"""Tests for `src.data.geometry` utilities."""

import json

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin
from shapely import from_wkt
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from src.data.geometry import VALID_ORDINAL_LABELS
from src.data.geometry import _normalize_clipped, _parse_one_segment, _polygon_parts, _safe_intersection, _thinness_ratio
from src.data.geometry import align_vector_to_raster, apply_alignment_adjustments, clip_polygons_to_bounds
from src.data.geometry import filter_polygons_valid_labels, load_alignment_adjustments, validate_integrity, verify_crs_match


@pytest.fixture
def labeled_gdf() -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with valid and invalid labels."""

    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
    ]
    labels = ["no damage", "minor damage", "destroyed"]
    return gpd.GeoDataFrame(geometry=polygons, data={"label": labels}, crs="EPSG:4326")


def test_filter_polygons_valid_labels_behaviors(labeled_gdf: gpd.GeoDataFrame) -> None:
    """Filtering keeps valid labels and drops missing/invalid labels."""

    keep_all = filter_polygons_valid_labels(labeled_gdf)
    assert len(keep_all) == 3
    assert set(keep_all["label"]) == {"no damage", "minor damage", "destroyed"}

    custom = filter_polygons_valid_labels(labeled_gdf, valid_labels={"no damage"})
    assert len(custom) == 1
    assert custom["label"].iloc[0] == "no damage"

    modified = labeled_gdf.copy()
    modified.loc[1, "label"] = "un-classified"
    modified.loc[2, "label"] = pd.NA
    dropped = filter_polygons_valid_labels(modified)
    assert len(dropped) == 1
    assert dropped["label"].iloc[0] == "no damage"

    no_label = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
    assert filter_polygons_valid_labels(no_label).empty


def test_polygon_helpers_and_thinness() -> None:
    """Private geometry helper functions handle diverse geometry types."""

    assert _polygon_parts(None) == []
    assert _polygon_parts(Polygon()) == []

    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    assert len(_polygon_parts(square)) == 1

    multi = MultiPolygon([square, Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])])
    assert len(_polygon_parts(multi)) == 2

    gc = from_wkt("GEOMETRYCOLLECTION (POINT (0.5 0.5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))")
    parts = _polygon_parts(gc)
    assert len(parts) == 1
    assert parts[0].geom_type == "Polygon"

    line = LineString([(0, 0), (1, 1)])
    assert _polygon_parts(line) == []

    assert _thinness_ratio(None) == 0.0
    assert _thinness_ratio(Polygon()) == 0.0
    assert _thinness_ratio(Polygon([(0, 0), (1, 0), (0, 0)])) == 0.0
    assert _thinness_ratio(square) > 0.5

    class DummyGeometry:
        """Simple duck-typed geometry to hit zero-perimeter branch."""

        is_empty = False
        area = 1.0
        length = 0.0

    assert _thinness_ratio(DummyGeometry()) == 0.0


def test_normalize_clipped_and_safe_intersection() -> None:
    """Normalization removes non-polygon/sliver output from geometry collections."""

    non_poly_gc = from_wkt("GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))")
    normalized_empty = _normalize_clipped(non_poly_gc)
    assert normalized_empty.geom_type == "Polygon"
    assert normalized_empty.is_empty

    thin_gc = from_wkt("GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 0.001, 0 0.001, 0 0)))")
    assert _normalize_clipped(thin_gc).is_empty

    one_gc = from_wkt("GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))")
    one_out = _normalize_clipped(one_gc)
    assert one_out.geom_type == "Polygon"
    assert not one_out.is_empty

    two_gc = from_wkt(
        "GEOMETRYCOLLECTION ("
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), "
        "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2)))"
    )
    two_out = _normalize_clipped(two_gc)
    assert two_out.geom_type == "MultiPolygon"
    assert len(two_out.geoms) == 2

    clip_box = box(0, 0, 2, 2)
    valid_intersection = _safe_intersection(Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]), clip_box)
    assert valid_intersection.geom_type == "Polygon"

    bowtie = Polygon([(0.5, 0), (1.5, 0), (0.5, 1), (1.5, 1), (0.5, 0)])
    repaired_intersection = _safe_intersection(bowtie, clip_box)
    assert repaired_intersection is not None


def test_clip_polygons_to_bounds_variants() -> None:
    """Clipping handles empty, outside, invalid, and geometry-collection inputs."""

    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    assert clip_polygons_to_bounds(empty, bounds=(0, 0, 1, 1)).empty

    gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        ],
        crs="EPSG:4326"
    )
    clipped = clip_polygons_to_bounds(gdf, bounds=(0.0, 0.0, 2.0, 2.0))
    assert len(clipped) == 2

    outside = gpd.GeoDataFrame(geometry=[Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])], crs="EPSG:4326")
    assert clip_polygons_to_bounds(outside, bounds=(0, 0, 1, 1)).empty

    invalid = gpd.GeoDataFrame(geometry=[Polygon([(0.5, 0), (1.5, 0), (0.5, 1), (1.5, 1), (0.5, 0)])], crs="EPSG:4326")
    invalid_out = clip_polygons_to_bounds(invalid, bounds=(0.0, 0.0, 2.0, 2.0))
    assert len(invalid_out) <= 1

    gc = from_wkt("GEOMETRYCOLLECTION (POINT (0.5 0.5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))")
    gc_out = clip_polygons_to_bounds(gpd.GeoDataFrame(geometry=[gc], crs="EPSG:4326"), bounds=(0.0, 0.0, 2.0, 2.0))
    assert not gc_out.empty
    assert gc_out.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")


def test_clip_polygons_to_bounds_geometry_collection_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly cover GeometryCollection normalization branch in clipper."""

    forced_gc = from_wkt("GEOMETRYCOLLECTION (POINT (0.5 0.5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))")

    def fake_safe_intersection(_geom: BaseGeometry, _clip_box: BaseGeometry) -> BaseGeometry:
        return forced_gc

    monkeypatch.setattr("src.data.geometry._safe_intersection", fake_safe_intersection)
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], crs="EPSG:4326")
    clipped = clip_polygons_to_bounds(gdf, bounds=(0.0, 0.0, 2.0, 2.0))
    assert not clipped.empty
    assert clipped.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")


def test_verify_and_align_vector_crs(tmp_path: Path) -> None:
    """CRS verification and alignment work for match and mismatch cases."""

    raster_4326 = tmp_path / "raster_4326.tif"
    with rasterio.open(
        raster_4326,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=from_origin(0, 10, 1.0, 1.0)
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=rasterio.uint8))

    vector_4326 = tmp_path / "labels_4326.json"
    vector_4326.write_text(
        json.dumps([{"EPSG:4326": [{"lon": 1, "lat": 1}, {"lon": 2, "lat": 1}, {"lon": 2, "lat": 2}], "label": "no damage"}]),
        encoding="utf-8"
    )
    is_match, raster_crs, vector_crs = verify_crs_match(raster_4326, vector_4326)
    assert is_match is True
    assert raster_crs == vector_crs

    raster_utm = tmp_path / "raster_utm.tif"
    with rasterio.open(
        raster_utm,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:32616",
        transform=from_origin(500000, 2200000, 1.0, 1.0)
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=rasterio.uint8))

    is_match_mismatch, _, _ = verify_crs_match(raster_utm, vector_4326)
    assert is_match_mismatch is False

    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    assert align_vector_to_raster(empty_gdf, raster_4326).empty

    source_gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    aligned_same = align_vector_to_raster(source_gdf, raster_4326)
    assert aligned_same.crs is not None
    assert aligned_same.crs.to_epsg() == 4326

    aligned_utm = align_vector_to_raster(source_gdf, raster_utm)
    assert aligned_utm.crs is not None
    assert aligned_utm.crs.to_epsg() == 32616


def test_validate_integrity_paths(tmp_path: Path) -> None:
    """Integrity validation covers valid, mismatch, skipped-vector, and corrupt rows."""

    img_4326 = tmp_path / "img_4326.tif"
    with rasterio.open(
        img_4326,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=from_origin(0, 10, 1.0, 1.0)
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=rasterio.uint8))

    img_utm = tmp_path / "img_utm.tif"
    with rasterio.open(
        img_utm,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:32616",
        transform=from_origin(500000, 2200000, 1.0, 1.0)
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=rasterio.uint8))

    vec_4326 = tmp_path / "labels.json"
    vec_4326.write_text(
        json.dumps([{"EPSG:4326": [{"lon": 1, "lat": 1}, {"lon": 2, "lat": 1}, {"lon": 2, "lat": 2}], "label": "minor damage"}]),
        encoding="utf-8"
    )

    bad_image = tmp_path / "not_a_tif.txt"
    bad_image.write_text("not geotiff", encoding="utf-8")

    records = [
        {"image_path": str(img_4326), "label_path": str(vec_4326), "image_name": img_4326.name, "valid": True},
        {"image_path": str(img_utm), "label_path": str(vec_4326), "image_name": img_utm.name, "valid": True},
        {"image_path": str(img_4326), "label_path": str(vec_4326), "image_name": "skip_vector.tif", "valid": False},
        {"image_path": str(bad_image), "label_path": str(vec_4326), "image_name": bad_image.name, "valid": True}
    ]
    df = pd.DataFrame(records)
    result = validate_integrity(df)

    assert result["total"] == 4
    assert result["valid_image"] == 3
    assert result["valid_vector"] == 2
    assert result["crs_mismatch"] == 1
    assert result["corrupt"] == 1
    assert len(result["errors"]) >= 2


def test_validate_integrity_progress_print(tmp_path: Path) -> None:
    """Large inventory triggers periodic progress print branch."""

    image = tmp_path / "img.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1.0, 1.0)
    ) as dst:
        dst.write(np.zeros((1, 2, 2), dtype=rasterio.uint8))

    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([{"EPSG:4326": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}, {"lon": 1, "lat": 1}], "label": "no damage"}]), encoding="utf-8")

    records = [
        {"image_path": str(image), "label_path": str(labels), "image_name": f"img_{idx}.tif", "valid": True}
        for idx in range(101)
    ]
    result = validate_integrity(pd.DataFrame(records))
    assert result["total"] == 101
    assert result["valid_image"] == 101


def test_alignment_segment_parsing_and_loading(tmp_path: Path) -> None:
    """Segment parser and loader handle malformed and valid inputs."""

    assert _parse_one_segment("bad") is None
    assert _parse_one_segment([]) is None
    assert _parse_one_segment([[1, 2]]) is None
    assert _parse_one_segment([[1], [2, 3]]) is None
    assert _parse_one_segment([[1, 2], [3]]) is None
    assert _parse_one_segment([[1, "x"], [3, 4]]) is None
    assert _parse_one_segment([[1.1, 2.2], [3.4, 4.6]]) == ((1, 2), (3, 5))

    assert load_alignment_adjustments(None) == []
    assert load_alignment_adjustments(tmp_path / "missing.json") == []

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert load_alignment_adjustments(bad_json) == []

    not_list = tmp_path / "obj.json"
    not_list.write_text("{}", encoding="utf-8")
    assert load_alignment_adjustments(not_list) == []

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps([[[1, 2], [3, 4]], "bad", [[10, 20], [30, 40]]]), encoding="utf-8")
    loaded = load_alignment_adjustments(valid)
    assert loaded == [((1, 2), (3, 4)), ((10, 20), (30, 40))]


def test_apply_alignment_adjustments_behaviors(tmp_path: Path) -> None:
    """Alignment application handles empty adjustments and identity shifts."""

    width, height = 100, 100
    transform = from_bounds(0, 0, 100, 100, width, height)
    raster_path = tmp_path / "raster.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        count=1,
        width=width,
        height=height,
        dtype=np.uint8,
        transform=transform,
        crs="EPSG:32616"
    ) as dst:
        dst.write(np.zeros((1, height, width), dtype=np.uint8))

    poly = Polygon([(45, 45), (55, 45), (55, 55), (45, 55), (45, 45)])
    gdf = gpd.GeoDataFrame(geometry=[poly, None], crs="EPSG:32616")

    unchanged = apply_alignment_adjustments(gdf, raster_path, adjustments=[])
    assert unchanged.equals(gdf)

    identity_adjustments = [
        ((0, 0), (0, 0)),
        ((99, 0), (99, 0)),
        ((99, 99), (99, 99)),
        ((0, 99), (0, 99))
    ]
    adjusted = apply_alignment_adjustments(gdf, raster_path, adjustments=identity_adjustments)
    assert adjusted.geometry.iloc[0].area <= poly.area * 2
    assert adjusted.geometry.iloc[1] is None


def test_valid_ordinal_labels_constant() -> None:
    """Ordinal labels constant matches expected classes."""

    assert frozenset({"no damage", "minor damage", "major damage", "destroyed"}) == VALID_ORDINAL_LABELS


def _polygon_part_count(geom: BaseGeometry | None) -> int:
    """Count polygon parts in a geometry."""

    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return 1
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    return 0


def _max_perimeter_sq_over_area(geom: BaseGeometry | None) -> float:
    """Return max perimeter^2/area over polygon parts."""

    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type == "Polygon":
        area, perimeter = geom.area, geom.length
        return (perimeter * perimeter / area) if area > 0 else 0.0
    if geom.geom_type == "MultiPolygon":
        return max((part.length**2 / part.area if part.area > 0 else 0.0) for part in geom.geoms)
    return 0.0


def test_clip_bounds_preserves_valid_multi_parts_without_spaghetti() -> None:
    """Clip output should avoid GeometryCollection and sliver artifacts."""

    max_perimeter_sq_over_area = 502.0
    polygons = [
        Polygon([(offset, 0), (offset + 0.1, 0), (offset + 0.1, 0.1), (offset, 0.1), (offset, 0)])
        for offset in [0.1 + index * 0.15 for index in range(20)]
    ]
    multi = MultiPolygon(polygons)
    gdf = gpd.GeoDataFrame(geometry=[multi], crs="EPSG:4326")
    out = clip_polygons_to_bounds(gdf, bounds=(0.0, 0.0, 3.5, 1.0))
    assert not out.empty
    for _, row in out.iterrows():
        geom = row.geometry
        assert geom.geom_type in ("Polygon", "MultiPolygon")
        assert _max_perimeter_sq_over_area(geom) <= max_perimeter_sq_over_area
    assert _polygon_part_count(out.geometry.iloc[0]) == 20
