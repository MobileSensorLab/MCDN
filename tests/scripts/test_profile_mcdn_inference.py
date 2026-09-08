"""Unit tests for deployment-profile helpers in ``scripts/profile_mcdn_inference.py``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

import profile_mcdn_inference as pmi
from src.model.mcdn import MaskCenteredDamageNet


def _make_fold(split_root: Path, name: str, with_weights: bool = True, val_instances: int | None = None) -> Path:
    fold_dir = split_root / name
    fold_dir.mkdir(parents=True)
    if with_weights:
        (fold_dir / "best_model.pt").write_bytes(b"\x00")
    if val_instances is not None:
        (fold_dir / "split_summary.json").write_text(json.dumps({"val_instances": val_instances}), encoding="utf-8")
    return fold_dir


def test_discover_seed_checkpoints_sorts_filters_and_limits(tmp_path: Path) -> None:
    """Only seed_* folds containing weights are returned, name-sorted and capped."""

    split_root = tmp_path / "Spatial_Block_East"
    _make_fold(split_root, "seed_11")
    _make_fold(split_root, "seed_00")
    _make_fold(split_root, "seed_22", with_weights=False)
    (split_root / "not_a_fold").mkdir()

    folds = pmi.discover_seed_checkpoints(split_root)
    assert [fold.name for fold in folds] == ["seed_00", "seed_11"]
    assert [fold.name for fold in pmi.discover_seed_checkpoints(split_root, limit=1)] == ["seed_00"]


def test_read_event_chip_counts_reads_first_available_summary(tmp_path: Path) -> None:
    """Each split contributes its val_instances; splits without summaries are skipped."""

    variant_root = tmp_path / "baseline"
    _make_fold(variant_root / "Hurricane_Michael", "seed_00", val_instances=1110)
    _make_fold(variant_root / "Mayfield_Tornado", "seed_00", val_instances=1948)
    _make_fold(variant_root / "No_Summary_Split", "seed_00")

    counts = pmi.read_event_chip_counts(variant_root)
    assert counts == {"Hurricane_Michael": 1110, "Mayfield_Tornado": 1948}


def test_derive_operational_times_scales_linearly() -> None:
    """Per-event seconds are chip counts times per-chip latency, plus the 415-chip yardstick."""

    times = pmi.derive_operational_times(2.0, {"Hurricane_Michael": 1110, "Mayfield_Tornado": 1948})
    assert times["Hurricane_Michael_s"] == pytest.approx(2.22)
    assert times["Mayfield_Tornado_s"] == pytest.approx(3.896)
    assert times[f"yardstick_{pmi.YARDSTICK_CHIPS}_chips_s"] == pytest.approx(0.83)


def test_synthetic_uint8_batch_matches_loader_contract() -> None:
    """Synthetic chips are uint8 [B, 4, H, W] with a binary mask channel and one-hot context."""

    images_u8, context = pmi.synthetic_uint8_batch(batch_size=2, chip_size=32, device=torch.device("cpu"))
    assert images_u8.shape == (2, 4, 32, 32)
    assert images_u8.dtype == torch.uint8
    mask = images_u8[:, 3]
    assert set(torch.unique(mask).tolist()) <= {0, 1}
    assert int(mask.sum()) > 0
    assert context.shape == (2, 4)
    assert torch.equal(context.sum(dim=1), torch.ones(2))


def test_no_tta_softmax_probs_returns_normalized_distribution() -> None:
    """The no-TTA entry point emits [B, K] softmax rows through the deployed conventions."""

    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False)
    images_u8, context = pmi.synthetic_uint8_batch(batch_size=2, chip_size=64, device=torch.device("cpu"))
    probs = pmi.no_tta_softmax_probs(model=model, images_u8=images_u8, context=context, device="cpu")
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_benchmark_callable_ms_reports_stats_on_cpu() -> None:
    """The generic timer returns mean/std/percentile keys for a trivial step."""

    stats = pmi.benchmark_callable_ms(lambda: None, device=torch.device("cpu"), warmup=1, iterations=5)
    assert set(stats) == {"latency_batch_ms_mean", "latency_batch_ms_std", "latency_batch_ms_p50", "latency_batch_ms_p95"}
    assert stats["latency_batch_ms_mean"] >= 0.0


# Resolved-config parsing -----------------------------------------------------

CPU = torch.device("cpu")
CHIP = 32


def _config_yaml(**overrides: object) -> str:
    """Render a minimal ``config_resolved.yaml`` body; overrides replace top-level sections."""

    sections: dict[str, object] = {
        "data": {"chip_size": CHIP},
        "model": {"name": "resnet18", "pretrained": False},
        "ablation": {"mask_enabled": True, "typology_enabled": True, "mask_weighted_pooling_enabled": True},
    }
    sections.update(overrides)
    return yaml.safe_dump(sections)


def _write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_resolved_fold_config_reads_required_fields(tmp_path: Path) -> None:
    """A well-formed snapshot maps onto ``ResolvedFoldConfig`` with the pooling flag preserved."""

    cfg = pmi.load_resolved_fold_config(_write_config(tmp_path / "config_resolved.yaml", _config_yaml()))
    assert cfg == pmi.ResolvedFoldConfig(chip_size=CHIP, model_name="resnet18", model_pretrained=False,
                                         mask_enabled=True, typology_enabled=True, mask_weighted_pooling_enabled=True)


def test_load_resolved_fold_config_defaults_pooling_flag_false(tmp_path: Path) -> None:
    """Snapshots predating mask-weighted pooling resolve that flag to False."""

    text = _config_yaml(ablation={"mask_enabled": False, "typology_enabled": False})
    cfg = pmi.load_resolved_fold_config(_write_config(tmp_path / "c.yaml", text))
    assert cfg.mask_weighted_pooling_enabled is False
    assert cfg.mask_enabled is False


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("- just\n- a list\n", "Expected mapping"),
        (_config_yaml(ablation="oops"), "Missing or invalid"),
        (_config_yaml(data={"chip_size": 0}), "chip_size"),
        (_config_yaml(data={"chip_size": "512"}), "chip_size"),
        (_config_yaml(model={"name": "", "pretrained": False}), "model.name"),
        (_config_yaml(model={"name": "resnet18", "pretrained": "yes"}), "model.pretrained"),
        (_config_yaml(ablation={"mask_enabled": 1, "typology_enabled": True}), "must be bools")
    ]
)
def test_load_resolved_fold_config_rejects_malformed_snapshots(tmp_path: Path, text: str, match: str) -> None:
    """Each validation branch raises ``ProfileConfigError`` naming the offending key."""

    with pytest.raises(pmi.ProfileConfigError, match=match):
        pmi.load_resolved_fold_config(_write_config(tmp_path / "bad.yaml", text))


# Model construction and weights ----------------------------------------------

def _tiny_cfg(**kwargs: bool) -> pmi.ResolvedFoldConfig:
    flags = {"mask_enabled": True, "typology_enabled": True, "mask_weighted_pooling_enabled": False}
    flags.update(kwargs)
    return pmi.ResolvedFoldConfig(chip_size=CHIP, model_name="resnet18", model_pretrained=False, **flags)


def _write_checkpoint_dir(fold_dir: Path, *, val_instances: int | None = None) -> Path:
    """Write ``config_resolved.yaml`` plus real ``best_model.pt`` weights for a tiny resnet18 MCDN."""

    fold_dir.mkdir(parents=True, exist_ok=True)
    _write_config(fold_dir / "config_resolved.yaml", _config_yaml())
    model = pmi.build_model(_tiny_cfg(mask_weighted_pooling_enabled=True))
    torch.save(model.state_dict(), fold_dir / "best_model.pt")
    if val_instances is not None:
        (fold_dir / "split_summary.json").write_text(json.dumps({"val_instances": val_instances}), encoding="utf-8")
    return fold_dir


def test_build_model_and_load_state_dict_roundtrip(tmp_path: Path) -> None:
    """Weights saved from one build load strictly into a fresh build of the same config."""

    fold_dir = _write_checkpoint_dir(tmp_path / "seed_00")
    model = pmi.build_model(_tiny_cfg(mask_weighted_pooling_enabled=True))
    pmi.load_state_dict_into_model(model, fold_dir / "best_model.pt", device=CPU)
    assert isinstance(model, MaskCenteredDamageNet)


def test_load_state_dict_into_model_rejects_non_mapping_payload(tmp_path: Path) -> None:
    """A checkpoint that is not a state-dict mapping is refused."""

    weights = tmp_path / "best_model.pt"
    torch.save(torch.zeros(3), weights)
    with pytest.raises(TypeError, match="state_dict mapping"):
        pmi.load_state_dict_into_model(pmi.build_model(_tiny_cfg()), weights, device=CPU)


def test_synthetic_batch_honours_channel_and_typology_flags() -> None:
    """Mask disabled drops to three channels; typology disabled leaves an all-zero context."""

    x4, ctx_on = pmi.synthetic_batch(batch_size=2, chip_size=CHIP, mask_enabled=True, typology_enabled=True, device=CPU)
    x3, ctx_off = pmi.synthetic_batch(batch_size=2, chip_size=CHIP, mask_enabled=False, typology_enabled=False, device=CPU)
    assert x4.shape == (2, 4, CHIP, CHIP)
    assert x3.shape == (2, 3, CHIP, CHIP)
    assert torch.equal(ctx_on[:, 0], torch.ones(2))
    assert float(ctx_off.abs().sum()) == 0.0


def test_percentile_sorted_edge_cases() -> None:
    """Empty input yields NaN, singletons return themselves, interior points interpolate linearly."""

    assert math.isnan(pmi.percentile_sorted([], 50.0))
    assert pmi.percentile_sorted([7.0], 95.0) == 7.0
    assert pmi.percentile_sorted([0.0, 10.0], 50.0) == pytest.approx(5.0)
    assert pmi.percentile_sorted([0.0, 10.0, 20.0], 100.0) == pytest.approx(20.0)


# Latency, FLOPs, and profiler on CPU -----------------------------------------

def test_benchmark_latency_ms_reports_per_chip_and_throughput() -> None:
    """CPU timing reports batch and per-chip latencies that agree with the batch size."""

    model = pmi.build_model(_tiny_cfg())
    x, ctx = pmi.synthetic_batch(batch_size=2, chip_size=CHIP, mask_enabled=True, typology_enabled=True, device=CPU)
    lat = pmi.benchmark_latency_ms(model, x, ctx, device=CPU, warmup=1, iterations=3, use_amp=True)
    assert lat["latency_per_chip_ms_mean"] == pytest.approx(lat["latency_batch_ms_mean"] / 2)
    assert lat["throughput_chips_per_s_mean"] > 0.0
    assert lat["latency_batch_ms_p50"] <= lat["latency_batch_ms_p99"]


def test_total_flops_fvcore_is_positive_and_restores_device() -> None:
    """The fvcore analysis counts a positive FLOP total and leaves the model on its original device."""

    model = pmi.build_model(_tiny_cfg())
    x, ctx = pmi.synthetic_batch(batch_size=1, chip_size=CHIP, mask_enabled=True, typology_enabled=True, device=CPU)
    assert pmi.total_flops_fvcore(model, x, ctx) > 0.0
    assert next(model.parameters()).device.type == "cpu"


def test_profiler_table_string_lists_operators() -> None:
    """The torch profiler table is returned as text containing operator rows."""

    model = pmi.build_model(_tiny_cfg())
    x, ctx = pmi.synthetic_batch(batch_size=1, chip_size=CHIP, mask_enabled=True, typology_enabled=True, device=CPU)
    table = pmi.profiler_table_string(model, x, ctx, device=CPU, row_limit=5)
    assert "aten::" in table
    assert "Self CPU" in table


# Suites ----------------------------------------------------------------------

def test_run_profile_suite_end_to_end_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The forward suite loads a checkpoint, reports latency/FLOPs/profiler, and writes the JSON payload."""

    fold_dir = _write_checkpoint_dir(tmp_path / "seed_00")
    out_json = tmp_path / "reports" / "profile.json"
    payload = pmi.run_profile_suite(checkpoint_dir=fold_dir, batch_size=2, warmup=1, iterations=2, device_str="cpu",
                                    use_amp=False, profiler_rows=3, output_json=out_json)

    assert payload["chip_size"] == CHIP
    assert payload["parameter_count"] > 0
    assert payload["peak_cuda_memory_allocated_bytes"] is None
    assert json.loads(out_json.read_text(encoding="utf-8"))["batch_size"] == 2
    captured = capsys.readouterr().out
    assert "fvcore_total_flops" in captured
    assert "Wrote JSON report" in captured


def test_run_profile_suite_requires_config_and_weights(tmp_path: Path) -> None:
    """Missing config or weights fail fast with a path-bearing FileNotFoundError."""

    fold_dir = tmp_path / "seed_00"
    fold_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing config"):
        pmi.run_profile_suite(checkpoint_dir=fold_dir, batch_size=1, warmup=0, iterations=1, device_str="cpu",
                              use_amp=False, profiler_rows=1, output_json=None)
    _write_config(fold_dir / "config_resolved.yaml", _config_yaml())
    with pytest.raises(FileNotFoundError, match="Missing weights"):
        pmi.run_profile_suite(checkpoint_dir=fold_dir, batch_size=1, warmup=0, iterations=1, device_str="cpu",
                              use_amp=False, profiler_rows=1, output_json=None)


def test_run_deployment_suite_sweeps_configs_and_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Deployment mode times all three configurations per batch size and derives event timings."""

    split_root = tmp_path / "all_features" / "Hurricane_Ian"
    seed_00 = _write_checkpoint_dir(split_root / "seed_00", val_instances=7)
    _write_checkpoint_dir(split_root / "seed_01")
    out_json = tmp_path / "deploy.json"

    payload = pmi.run_deployment_suite(checkpoint_dir=seed_00, batch_sizes=[1, 2], warmup=0, iterations=1,
                                       device_str="cpu", ensemble_size=2, output_json=out_json)

    assert payload["mode"] == "deployment"
    assert payload["split"] == "Hurricane_Ian"
    assert len(payload["ensemble_seed_dirs"]) == 2
    assert payload["event_chip_counts"] == {"Hurricane_Ian": 7}
    assert [sweep["batch_size"] for sweep in payload["batch_sweeps"]] == [1, 2]
    configs = payload["batch_sweeps"][1]["configs"]
    assert set(configs) == {"single_no_tta", "single_tta8", "ensemble2_tta8"}
    assert configs["single_tta8"]["operational"]["Hurricane_Ian_s"] == pytest.approx(
        7 * configs["single_tta8"]["per_chip_ms"] / 1000.0)
    assert out_json.is_file()
    captured = capsys.readouterr().out
    assert "batch_size=2" in captured
    assert "Hurricane_Ian(7)s" in captured


def test_run_deployment_suite_without_summaries_skips_operational(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Absent split summaries disable time-to-event reporting without failing the sweep."""

    seed_00 = _write_checkpoint_dir(tmp_path / "arm" / "Split" / "seed_00")
    payload = pmi.run_deployment_suite(checkpoint_dir=seed_00, batch_sizes=[1], warmup=0, iterations=1,
                                       device_str="cpu", ensemble_size=1, output_json=None)
    assert payload["event_chip_counts"] == {}
    assert "operational" not in payload["batch_sweeps"][0]["configs"]["single_no_tta"]
    assert "time-to-event skipped" in capsys.readouterr().out


def test_run_deployment_suite_requires_seed_folds(tmp_path: Path) -> None:
    """A split root with no weight-bearing seed folds is rejected."""

    empty = tmp_path / "arm" / "Split" / "seed_00"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No seed_\\* folds"):
        pmi.run_deployment_suite(checkpoint_dir=empty, batch_sizes=[1], warmup=0, iterations=1,
                                 device_str="cpu", ensemble_size=1, output_json=None)


# CLI dispatch ----------------------------------------------------------------

def test_main_refuses_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Requesting CUDA on a CPU-only host exits non-zero with guidance."""

    monkeypatch.setattr(pmi.torch.cuda, "is_available", lambda: False)
    assert pmi.main(["--device", "cuda"]) == 1
    assert "CUDA is not available" in capsys.readouterr().err


def test_main_dispatches_forward_and_deployment_suites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Argument parsing routes to the forward suite by default and to deployment mode with --deployment."""

    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(pmi, "run_profile_suite", lambda **kwargs: calls.append(("profile", kwargs)) or {})
    monkeypatch.setattr(pmi, "run_deployment_suite", lambda **kwargs: calls.append(("deployment", kwargs)) or {})

    assert pmi.main(["--device", "cpu", "--checkpoint-dir", str(tmp_path), "--batch-size", "3", "--amp"]) == 0
    assert pmi.main(["--device", "cpu", "--deployment", "--deployment-batch-sizes", "1", "4", "--ensemble-size", "2"]) == 0

    assert [name for name, _ in calls] == ["profile", "deployment"]
    assert calls[0][1]["batch_size"] == 3
    assert calls[0][1]["use_amp"] is True
    assert calls[0][1]["checkpoint_dir"] == tmp_path
    assert calls[1][1]["batch_sizes"] == [1, 4]
    assert calls[1][1]["ensemble_size"] == 2
