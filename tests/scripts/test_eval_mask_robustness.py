"""Unit tests for the footprint-robustness evaluation tooling (``scripts/eval_mask_robustness.py``)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import eval_mask_robustness as emr


PIXEL_DEG = 0.01  # test-raster ground sampling in degrees per pixel


@pytest.fixture
def robustness_world(tmp_path: Path) -> dict:
    """Tiny mosaic + raw-cache annotations + tie-point adjustments.

    Polygon 1 (Microsoft): 4x4 px square at cols 40-44 / rows 56-60, adjustment (+5, +10) px.
    Polygon 2 (custom):    4x4 px square at cols 70-74 / rows 26-30, adjustment (-3, 0) px.
    """

    image_path = tmp_path / "mosaic.geo.tif"
    profile = {
        "driver": "GTiff", "height": 100, "width": 100, "count": 3,
        "dtype": rasterio.uint8, "crs": "EPSG:4326",
        "transform": from_origin(10.0, 20.0, PIXEL_DEG, PIXEL_DEG),
    }
    with rasterio.open(image_path, "w", **profile) as dst:
        dst.write(np.zeros((3, 100, 100), dtype=np.uint8))

    poly1 = Polygon([(10.40, 19.40), (10.44, 19.40), (10.44, 19.44), (10.40, 19.44)])
    poly2 = Polygon([(10.70, 19.70), (10.74, 19.70), (10.74, 19.74), (10.70, 19.74)])
    entries = [
        {"EPSG:4326": [{"lon": x, "lat": y} for x, y in poly1.exterior.coords],
         "label": "minor damage", "source": "Microsoft"},
        {"EPSG:4326": [{"lon": x, "lat": y} for x, y in poly2.exterior.coords],
         "label": "destroyed", "source": "custom"},
    ]
    label_path = tmp_path / "mosaic.geo.tif.json"
    label_path.write_text(json.dumps(entries), encoding="utf-8")

    # Tie points at each polygon's pixel centroid: [[from_xy], [to_xy]] per the dataset card.
    alignment_path = tmp_path / "mosaic.geo.tif.align.json"
    alignment_path.write_text(json.dumps([[[42, 58], [47, 68]], [[72, 28], [69, 28]]]), encoding="utf-8")

    manifest = pd.DataFrame([{
        "image_path": str(image_path),
        "label_path": str(label_path),
        "alignment_path": str(alignment_path),
        "event": "Hurricane Ian",
    }])
    return {"manifest": manifest, "image_path": image_path, "label_path": label_path, "polygons": [poly1, poly2]}


def _dataset(world: dict, spec: emr.PerturbationSpec) -> emr.RobustnessDataset:
    return emr.RobustnessDataset(world["manifest"], perturbation=spec, chip_size=64, transform=None)


def _centroids(dataset: emr.RobustnessDataset, world: dict) -> list[tuple[float, float]]:
    gdf = dataset.gdf_cache[str(Path(world["manifest"]["image_path"].iloc[0]))]
    return [(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry]


# Condition parsing -----------------------------------------------------------

def test_parse_condition_all_modes() -> None:
    """Every documented token form parses into the expected spec."""

    assert emr.parse_condition("aligned").mode == "aligned"
    assert emr.parse_condition("raw_cache").mode == "raw_cache"

    offset = emr.parse_condition("offset:75", circular_variance=0.3, perturbation_seed=99)
    assert (offset.mode, offset.magnitude_px, offset.circular_variance, offset.seed) == ("offset", 75.0, 0.3, 99)

    buffer_spec = emr.parse_condition("buffer:-8")
    assert (buffer_spec.mode, buffer_spec.buffer_px) == ("buffer", -8.0)

    explicit = emr.parse_condition("mask_zero:0.05")
    assert (explicit.mode, explicit.rate) == ("mask_zero", 0.05)
    assert emr.parse_condition("mask_zero:auto").rate is None


@pytest.mark.parametrize("token", ["bogus", "offset:0", "offset:-5", "aligned:1", "mask_zero:1.5", "buffer:0"])
def test_parse_condition_rejects_invalid_tokens(token: str) -> None:
    """Unknown modes and out-of-range arguments raise ValueError."""

    with pytest.raises(ValueError, match=r"."):
        emr.parse_condition(token)


def test_condition_labels() -> None:
    """Labels are compact and filename-safe."""

    assert emr.parse_condition("offset:75").label == "offset_75px"
    assert emr.parse_condition("buffer:-8").label == "buffer_-8px"
    assert emr.parse_condition("mask_zero:0.05").label == "mask_zero_0.05"
    assert emr.parse_condition("mask_zero:auto").label == "mask_zero_auto"


# Offset-field sampling -------------------------------------------------------

def test_wrapped_normal_sigma_identity() -> None:
    """Sigma follows sqrt(-2 ln(1 - CV)) with CV=0 collapsing to a shared direction."""

    assert emr.wrapped_normal_sigma(0.0) == 0.0
    assert emr.wrapped_normal_sigma(0.28) == pytest.approx(math.sqrt(-2.0 * math.log(0.72)))
    with pytest.raises(ValueError, match="circular_variance"):
        emr.wrapped_normal_sigma(1.0)


def test_sample_offset_field_matches_calibration() -> None:
    """Sampled field has exact magnitudes and the requested empirical circular variance."""

    rng = np.random.default_rng(7)
    field = emr.sample_offset_field(num_polygons=20000, magnitude_px=75.0, circular_variance=0.28, rng=rng)

    magnitudes = np.hypot(field[:, 0], field[:, 1])
    assert np.allclose(magnitudes, 75.0)

    angles = np.arctan2(field[:, 1], field[:, 0])
    resultant = abs(np.exp(1j * angles).mean())
    assert 1.0 - resultant == pytest.approx(0.28, abs=0.02)


def test_mosaic_rng_is_deterministic_per_mosaic() -> None:
    """Same (seed, mosaic) reproduces draws; different mosaics diverge."""

    draw_a = emr.mosaic_rng(1, "a.geo.tif").uniform(size=4)
    draw_a_again = emr.mosaic_rng(1, "a.geo.tif").uniform(size=4)
    draw_b = emr.mosaic_rng(1, "b.geo.tif").uniform(size=4)
    assert np.array_equal(draw_a, draw_a_again)
    assert not np.array_equal(draw_a, draw_b)


# Geometry perturbations ------------------------------------------------------

def test_raw_cache_reproduces_shipped_geometry_exactly(robustness_world: dict) -> None:
    """raw_cache skips the tie-point adjustment, leaving shipped coordinates untouched."""

    dataset = _dataset(robustness_world, emr.PerturbationSpec(mode="raw_cache"))
    centroids = _centroids(dataset, robustness_world)
    for (x, y), polygon in zip(centroids, robustness_world["polygons"], strict=True):
        assert x == pytest.approx(polygon.centroid.x, abs=1e-9)
        assert y == pytest.approx(polygon.centroid.y, abs=1e-9)


def test_aligned_mode_applies_tie_point_shifts(robustness_world: dict) -> None:
    """The aligned mode shifts each polygon by its nearest adjustment (within pixel-center snap)."""

    raw = _centroids(_dataset(robustness_world, emr.PerturbationSpec(mode="raw_cache")), robustness_world)
    aligned = _centroids(_dataset(robustness_world, emr.PerturbationSpec(mode="aligned")), robustness_world)

    # Polygon 1: (+5, +10) px -> (+0.05 deg lon, -0.10 deg lat); polygon 2: (-3, 0) px.
    snap_tolerance = PIXEL_DEG * 1.1
    assert aligned[0][0] - raw[0][0] == pytest.approx(+5 * PIXEL_DEG, abs=snap_tolerance)
    assert aligned[0][1] - raw[0][1] == pytest.approx(-10 * PIXEL_DEG, abs=snap_tolerance)
    assert aligned[1][0] - raw[1][0] == pytest.approx(-3 * PIXEL_DEG, abs=snap_tolerance)
    assert aligned[1][1] - raw[1][1] == pytest.approx(0.0, abs=snap_tolerance)


def test_offset_mode_is_coherent_at_zero_variance(robustness_world: dict) -> None:
    """CV=0 gives one shared translation across the mosaic with exact pixel magnitude."""

    aligned = _centroids(_dataset(robustness_world, emr.PerturbationSpec(mode="aligned")), robustness_world)
    offset_spec = emr.PerturbationSpec(mode="offset", magnitude_px=10.0, circular_variance=0.0)
    shifted = _centroids(_dataset(robustness_world, offset_spec), robustness_world)

    deltas = [(sx - ax, sy - ay) for (sx, sy), (ax, ay) in zip(shifted, aligned, strict=True)]
    assert deltas[0][0] == pytest.approx(deltas[1][0], abs=1e-12)
    assert deltas[0][1] == pytest.approx(deltas[1][1], abs=1e-12)

    magnitude_px = math.hypot(deltas[0][0] / PIXEL_DEG, deltas[0][1] / PIXEL_DEG)
    assert magnitude_px == pytest.approx(10.0, abs=1e-9)


def test_buffer_mode_dilates_and_erodes(robustness_world: dict) -> None:
    """Positive buffers grow footprints; strong erosion collapses to a centroid marker."""

    key = str(Path(robustness_world["manifest"]["image_path"].iloc[0]))
    aligned_gdf = _dataset(robustness_world, emr.PerturbationSpec(mode="aligned")).gdf_cache[key]
    grown_gdf = _dataset(robustness_world, emr.PerturbationSpec(mode="buffer", buffer_px=2.0)).gdf_cache[key]
    eroded_gdf = _dataset(robustness_world, emr.PerturbationSpec(mode="buffer", buffer_px=-30.0)).gdf_cache[key]

    for aligned_geom, grown_geom, eroded_geom in zip(aligned_gdf.geometry, grown_gdf.geometry, eroded_gdf.geometry, strict=True):
        assert grown_geom.area > aligned_geom.area
        assert eroded_geom.area < 1e-12
        assert eroded_geom.centroid.x == pytest.approx(aligned_geom.centroid.x, abs=1e-6)
        assert eroded_geom.centroid.y == pytest.approx(aligned_geom.centroid.y, abs=1e-6)


# Mask deletion ---------------------------------------------------------------

def test_mask_zero_rate_one_empties_every_mask_channel(robustness_world: dict) -> None:
    """rate=1.0 zeroes the mask channel of every chip; the aligned reference is non-empty."""

    reference = _dataset(robustness_world, emr.PerturbationSpec(mode="aligned"))
    zeroed = _dataset(robustness_world, emr.PerturbationSpec(mode="mask_zero", rate=1.0))
    assert len(zeroed) == len(reference)

    for index in range(len(reference)):
        assert int(reference[index]["image"][emr.MASK_CHANNEL_INDEX].sum()) > 0
        assert int(zeroed[index]["image"][emr.MASK_CHANNEL_INDEX].sum()) == 0


def test_mask_zero_rate_zero_matches_reference(robustness_world: dict) -> None:
    """rate=0.0 leaves every chip identical to the aligned reference."""

    reference = _dataset(robustness_world, emr.PerturbationSpec(mode="aligned"))
    untouched = _dataset(robustness_world, emr.PerturbationSpec(mode="mask_zero", rate=0.0))
    for index in range(len(reference)):
        assert bool((reference[index]["image"] == untouched[index]["image"]).all())


def test_mask_zero_requires_resolved_rate(robustness_world: dict) -> None:
    """Constructing the dataset with an unresolved (auto) rate is rejected."""

    with pytest.raises(ValueError, match="rate must be resolved"):
        _dataset(robustness_world, emr.PerturbationSpec(mode="mask_zero", rate=None))


# Cross-condition invariants --------------------------------------------------

def test_instance_index_is_stable_across_conditions(robustness_world: dict) -> None:
    """Every condition preserves instance count, ordering, and labels (paired metrics)."""

    specs = [
        emr.PerturbationSpec(mode="aligned"),
        emr.PerturbationSpec(mode="raw_cache"),
        emr.PerturbationSpec(mode="offset", magnitude_px=60.0),
        emr.PerturbationSpec(mode="buffer", buffer_px=-2.0),
    ]
    label_sequences = []
    for spec in specs:
        dataset = _dataset(robustness_world, spec)
        label_sequences.append([instance["damage_label"] for instance in dataset.instances])

    assert all(sequence == label_sequences[0] for sequence in label_sequences[1:])
    assert label_sequences[0] == ["minor damage", "destroyed"]


# Custom-source rate ----------------------------------------------------------

def test_count_polygon_sources_and_auto_rate(robustness_world: dict) -> None:
    """Source tallies and the derived auto deletion rate match the annotation contents."""

    counts = emr.count_polygon_sources([str(robustness_world["label_path"])])
    assert counts == {"Microsoft": 1, "custom": 1}

    rate = emr.resolve_auto_rate(robustness_world["manifest"])
    assert rate == pytest.approx(0.5)


def test_deltas_vs_aligned_reference() -> None:
    """Delta computation subtracts the aligned ensemble metrics per rule and metric."""

    def fake_metrics(offset: float) -> dict:
        return {rule: {"qwk": 0.8 + offset, "macro_f1": 0.6 + offset, "accuracy": 0.7 + offset}
                for rule in emr.RULE_FNS}

    records = [
        {"condition": "aligned", "ensemble": fake_metrics(0.0)},
        {"condition": "raw_cache", "ensemble": fake_metrics(-0.05)},
    ]
    deltas = emr.compute_deltas_vs_aligned(records)
    assert set(deltas) == {"raw_cache"}
    for rule in emr.RULE_FNS:
        for metric in emr.REPORT_METRIC_KEYS:
            assert deltas["raw_cache"][rule][metric] == pytest.approx(-0.05)

    assert emr.compute_deltas_vs_aligned([records[1]]) == {}
