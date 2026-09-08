"""Unit tests for the footprint-robustness evaluation tooling (``scripts/eval_mask_robustness.py``)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import yaml
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


def test_count_polygon_sources_warns_on_unreadable_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Unreadable or malformed annotation files are skipped with a warning rather than aborting."""

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    counts = emr.count_polygon_sources([str(broken), str(tmp_path / "absent.json")])
    assert counts == {}
    assert capsys.readouterr().out.count("WARNING: skipping unreadable annotation file") == 2


# Inference orchestration -----------------------------------------------------

CHIP = 64


def _fold_cfg(*, num_workers: int = 0, persistent_workers: bool = False) -> dict:
    """Resolved-config snapshot sufficient for loader construction and model rebuilding."""

    return {
        "data": {"chip_size": CHIP, "dir": "data", "sensor_profile": "uas_5cm"},
        "runtime": {"num_workers": num_workers, "pin_memory": False, "persistent_workers": persistent_workers,
                    "prefetch_factor": 2, "val_batch_size_factor": 1.0, "seed": 3},
        "training": {"batch_size": 2},
        "model": {"name": "resnet18", "drop_path_rate": 0.0},
        "ablation": {"mask_enabled": True, "typology_enabled": True, "mask_weighted_pooling_enabled": True,
                     "mask_dilation_px": 0},
    }


def _write_fold(fold_dir: Path) -> Path:
    """Persist a resolved config and real resnet18 MCDN weights into one seed fold directory."""

    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "config_resolved.yaml").write_text(yaml.safe_dump(_fold_cfg()), encoding="utf-8")
    model = emr.build_model_from_config(cfg=_fold_cfg(), device="cpu")
    torch.save(model.state_dict(), fold_dir / "best_model.pt")
    return fold_dir


@pytest.fixture
def variant_root(tmp_path: Path) -> Path:
    """Two-seed variant layout ``<variant>/<split>/seed_NN`` with loadable checkpoints."""

    root = tmp_path / "outputs" / "all_features"
    for seed in (0, 1):
        _write_fold(root / "Hurricane_Ian" / f"seed_{seed:02d}")
    return root


def test_build_perturbed_val_loader_batches_the_holdout(robustness_world: dict) -> None:
    """The loader wraps a RobustnessDataset with the configured batch size and yields loader-shaped chips."""

    loader = emr.build_perturbed_val_loader(cfg=_fold_cfg(), val_df=robustness_world["manifest"],
                                            perturbation=emr.PerturbationSpec(mode="aligned"))
    assert isinstance(loader.dataset, emr.RobustnessDataset)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 4, CHIP, CHIP)
    assert batch["image"].dtype == torch.uint8
    assert batch["context"].shape == (2, 4)
    assert batch["label"].tolist() == [1, 3]


def test_build_perturbed_val_loader_rejects_persistent_workers_without_workers(robustness_world: dict) -> None:
    """persistent_workers=True with num_workers=0 is an invalid runtime combination."""

    with pytest.raises(ValueError, match="persistent_workers"):
        emr.build_perturbed_val_loader(cfg=_fold_cfg(persistent_workers=True), val_df=robustness_world["manifest"],
                                       perturbation=emr.PerturbationSpec(mode="aligned"))


def test_build_perturbed_val_loader_forwards_prefetch_with_workers(robustness_world: dict) -> None:
    """With workers enabled the prefetch factor is forwarded to the DataLoader."""

    loader = emr.build_perturbed_val_loader(cfg=_fold_cfg(num_workers=1, persistent_workers=True),
                                            val_df=robustness_world["manifest"],
                                            perturbation=emr.PerturbationSpec(mode="aligned"))
    assert loader.num_workers == 1
    assert loader.prefetch_factor == 2


def test_collect_tta_probs_returns_softmax_rows_and_targets(robustness_world: dict) -> None:
    """TTA collection concatenates per-batch probabilities and targets over the loader."""

    loader = emr.build_perturbed_val_loader(cfg=_fold_cfg(), val_df=robustness_world["manifest"],
                                            perturbation=emr.PerturbationSpec(mode="aligned"))
    model = emr.build_model_from_config(cfg=_fold_cfg(), device="cpu")
    probs, targets = emr.collect_tta_probs(model=model, val_loader=loader, device="cpu")
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert targets.tolist() == [1, 3]


def test_evaluate_condition_stacks_seeds_and_summarises(robustness_world: dict, variant_root: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """Every seed is evaluated once; the record carries per-seed, summary, ensemble metrics and tensors."""

    fold_dirs = {0: variant_root / "Hurricane_Ian" / "seed_00", 1: variant_root / "Hurricane_Ian" / "seed_01"}
    record = emr.evaluate_condition(fold_dirs=fold_dirs, cfg=_fold_cfg(), val_df=robustness_world["manifest"],
                                    spec=emr.PerturbationSpec(mode="raw_cache"), device="cpu")

    assert record["condition"] == "raw_cache"
    assert record["seeds"] == [0, 1]
    assert record["probs"].shape == (2, 2, 4)
    assert record["targets"].tolist() == [1, 3]
    assert [entry["seed"] for entry in record["per_seed"]] == [0, 1]
    assert set(record["ensemble"]) == set(emr.RULE_FNS)
    assert set(record["summary"]) == set(emr.RULE_FNS)
    assert "seed 1: probs (2, 4)" in capsys.readouterr().out


def test_evaluate_condition_detects_target_order_drift(robustness_world: dict, variant_root: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeds whose loaders disagree on target order raise SeedTargetMismatchError."""

    fold_dirs = {0: variant_root / "Hurricane_Ian" / "seed_00", 1: variant_root / "Hurricane_Ian" / "seed_01"}
    calls: list[int] = []

    def _drifting_collect(model: torch.nn.Module, val_loader: object, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = model, val_loader, device
        calls.append(1)
        return torch.full((2, 4), 0.25), torch.tensor([1, 3]) if len(calls) == 1 else torch.tensor([3, 1])

    monkeypatch.setattr(emr, "collect_tta_probs", _drifting_collect)
    with pytest.raises(emr.SeedTargetMismatchError, match="diverged between seeds 0 and 1"):
        emr.evaluate_condition(fold_dirs=fold_dirs, cfg=_fold_cfg(), val_df=robustness_world["manifest"],
                               spec=emr.PerturbationSpec(mode="aligned"), device="cpu")


def test_run_variant_robustness_resolves_auto_rate_and_writes_caches(robustness_world: dict, variant_root: Path,
                                                                     monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The variant driver discovers seeds, calibrates mask_zero:auto, and persists per-condition caches."""

    manifest = robustness_world["manifest"]
    monkeypatch.setattr(emr, "get_fold_dataframes_from_config",
                        lambda *_args, **_kwargs: (manifest.iloc[:0], manifest, "Hurricane Ian"))
    cache_dir = tmp_path / "caches"
    specs = [emr.parse_condition("aligned"), emr.parse_condition("mask_zero:auto")]

    result = emr.run_variant_robustness(variant_root=variant_root, seeds=[0, 1, 7], split_name="Hurricane_Ian",
                                        condition_specs=specs, data_dir=None, device="cpu", probs_cache_dir=cache_dir)

    assert result["holdout_event"] == "Hurricane Ian"
    assert [record["condition"] for record in result["conditions"]] == ["aligned", "mask_zero_0.5"]
    assert result["conditions"][1]["spec"]["rate"] == pytest.approx(0.5)
    cached = torch.load(cache_dir / "all_features_Hurricane_Ian_aligned.pt", weights_only=True)
    assert cached["probs"].shape == (2, 2, 4)
    assert cached["class_names"] == list(emr.ORDINAL_CLASS_DISPLAY_NAMES)


def test_run_variant_robustness_requires_seed_folds(tmp_path: Path) -> None:
    """A variant root with none of the requested seeds is rejected."""

    with pytest.raises(FileNotFoundError, match="No seed folds"):
        emr.run_variant_robustness(variant_root=tmp_path / "missing", seeds=[0], split_name="Hurricane_Ian",
                                   condition_specs=[emr.parse_condition("aligned")], data_dir=None, device="cpu")


# Reporting -------------------------------------------------------------------

def _metrics(offset: float) -> dict:
    return {rule: {"qwk": 0.8 + offset, "macro_f1": 0.6 + offset, "accuracy": 0.7 + offset,
                   "macro_precision": 0.65 + offset, "macro_recall": 0.62 + offset} for rule in emr.RULE_FNS}


def _condition_record(name: str, offset: float) -> dict:
    per_seed = [{"seed": seed, "metrics": _metrics(offset + 0.01 * seed)} for seed in (0, 1)]
    return {
        "condition": name,
        "spec": {"mode": name},
        "seeds": [0, 1],
        "per_seed": per_seed,
        "summary": emr._summarize_across_seeds([entry["metrics"] for entry in per_seed]),
        "ensemble": _metrics(offset),
        "probs": torch.zeros(2, 3, 4),
        "targets": torch.zeros(3, dtype=torch.long),
    }


def test_print_robustness_table_reports_deltas(capsys: pytest.CaptureFixture[str]) -> None:
    """The table shows a delta for perturbed conditions and a placeholder for the aligned row."""

    record = {"variant_root": "outputs/ablation/all_features", "holdout_event": "Hurricane Ian",
              "conditions": [_condition_record("aligned", 0.0), _condition_record("raw_cache", -0.05)]}
    emr.print_robustness_table(record)
    out = capsys.readouterr().out
    assert "FOOTPRINT ROBUSTNESS  variant=all_features  holdout=Hurricane Ian  seeds=2" in out
    assert " -0.0500" in out
    assert "--" in out
    for rule in emr.RULE_FNS:
        assert f"rule={rule}" in out


def test_write_robustness_artifact_excludes_tensors(tmp_path: Path) -> None:
    """The JSON artifact carries metrics and deltas for every variant but no probability tensors."""

    record = {"variant_root": "outputs/ablation/all_features", "holdout_event": "Hurricane Ian",
              "conditions": [_condition_record("aligned", 0.0), _condition_record("offset_60px", -0.1)]}
    path = tmp_path / "nested" / "mask_robustness.json"
    emr.write_robustness_artifact(path=path, variant_records=[record], split_name="Hurricane_Ian")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "Hurricane_Ian"
    assert payload["holdout_event"] == "Hurricane Ian"
    variant = payload["variants"][0]
    assert [condition["condition"] for condition in variant["conditions"]] == ["aligned", "offset_60px"]
    assert all("probs" not in condition for condition in variant["conditions"])
    assert variant["deltas_vs_aligned"]["offset_60px"]["EV"]["qwk"] == pytest.approx(-0.1)


# CLI -------------------------------------------------------------------------

def test_main_report_custom_rate_only(robustness_world: dict, variant_root: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    """--report-custom-rate prints the custom-source fraction per variant and exits before evaluation."""

    manifest = robustness_world["manifest"]
    monkeypatch.setattr(emr, "get_fold_dataframes_from_config",
                        lambda *_args, **_kwargs: (manifest.iloc[:0], manifest, "Hurricane Ian"))
    monkeypatch.setattr(emr, "run_variant_robustness", lambda **_kwargs: pytest.fail("evaluation must not run"))
    monkeypatch.setattr(sys, "argv", ["eval_mask_robustness", "--variant-roots", str(variant_root),
                                      "--split", "Hurricane_Ian", "--report-custom-rate"])
    emr.main()
    out = capsys.readouterr().out
    assert "=== all_features ===  holdout=Hurricane Ian" in out
    assert "custom-source polygons: 1/2 = 0.5000" in out


def test_main_report_custom_rate_requires_seed_folds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rate-only mode still validates that seed folds exist."""

    monkeypatch.setattr(sys, "argv", ["eval_mask_robustness", "--variant-roots", str(tmp_path / "none"),
                                      "--report-custom-rate"])
    with pytest.raises(FileNotFoundError, match="No seed folds"):
        emr.main()


def test_main_end_to_end_writes_artifact(robustness_world: dict, variant_root: Path, monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The full CLI evaluates the requested conditions, prints the table, and writes the JSON artifact."""

    manifest = robustness_world["manifest"]
    monkeypatch.setattr(emr, "get_fold_dataframes_from_config",
                        lambda *_args, **_kwargs: (manifest.iloc[:0], manifest, "Hurricane Ian"))
    out_json = tmp_path / "artifact" / "mask_robustness.json"
    monkeypatch.setattr(sys, "argv", ["eval_mask_robustness", "--variant-roots", str(variant_root),
                                      "--split", "Hurricane_Ian", "--seeds", "0", "--conditions", "aligned", "offset:5",
                                      "--device", "cpu", "--output-json", str(out_json)])
    emr.main()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    conditions = payload["variants"][0]["conditions"]
    assert [condition["condition"] for condition in conditions] == ["aligned", "offset_5px"]
    assert conditions[0]["seeds"] == [0]
    assert "offset_5px" in payload["variants"][0]["deltas_vs_aligned"]
    out = capsys.readouterr().out
    assert "Device: cpu" in out
    assert "Conditions: ['aligned', 'offset_5px']" in out
    assert "FOOTPRINT ROBUSTNESS" in out
