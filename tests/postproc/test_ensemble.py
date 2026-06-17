"""Unit tests for ``src.postproc.ensemble`` prediction rules, metrics, and I/O helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.model.mcdn import MaskCenteredDamageNet
from src.model.trainer import _validation_qwk_and_classification
from src.postproc import ensemble as ens


class _OneBatchDataset(Dataset):
    """Two-sample val contract: ``image`` uint8 [B,4,H,W], ``context``, ``label``."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros((4, 8, 8), dtype=torch.uint8),
            "context": torch.tensor([0.0, 1.0] if idx == 0 else [1.0, 0.0], dtype=torch.float32),
            "label": torch.tensor(0 if idx == 0 else 3, dtype=torch.long),
        }


class _TinyLogitModel(nn.Module):
    """Deterministic logits for TTA averaging without a real backbone."""

    def forward(self, x: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        row0 = torch.tensor([4.0, 0.0, 0.0, 0.0], device=x.device, dtype=torch.float32)
        row_last = torch.tensor([0.0, 0.0, 0.0, 4.0], device=x.device, dtype=torch.float32)
        rows = [row0, row_last]
        return torch.stack([rows[min(i, 1)] for i in range(b)], dim=0)


def test_apply_ev_rounding_uniform_probs_mid_class() -> None:
    """Uniform probabilities yield EV ``sum i p_i = 1.5`` then ``floor(1.5 + 0.5) = 2`` for K=4."""

    probs = torch.full((1, 4), 0.25, dtype=torch.float32)
    out = ens.apply_ev_rounding(probs)
    assert out.tolist() == [2]


def test_apply_argmax_picks_peak() -> None:
    """Argmax follows the largest mass column."""

    probs = torch.tensor([[0.1, 0.7, 0.1, 0.1]], dtype=torch.float32)
    assert ens.apply_argmax(probs).tolist() == [1]


def test_apply_hybrid_uses_argmax_on_extremes() -> None:
    """When argmax is 0 or K-1, hybrid follows argmax; otherwise EV."""

    probs = torch.tensor([[0.9, 0.05, 0.03, 0.02]], dtype=torch.float32)
    assert ens.apply_hybrid(probs).tolist() == [0]

    probs2 = torch.tensor([[0.02, 0.03, 0.05, 0.9]], dtype=torch.float32)
    assert ens.apply_hybrid(probs2).tolist() == [3]

    probs3 = torch.tensor([[0.2, 0.5, 0.2, 0.1]], dtype=torch.float32)
    ev = ens.apply_ev_rounding(probs3)
    assert ens.apply_hybrid(probs3).tolist() == ev.tolist()


def test_rule_fns_cover_expected_keys() -> None:
    """Public rule registry stays aligned with ensemble reporting."""

    assert set(ens.RULE_FNS.keys()) == {"EV", "argmax", "hybrid"}


def test_metrics_under_all_rules_matches_validation_helper() -> None:
    """Each rule's QWK matches ``_validation_qwk_and_classification`` on the same preds."""

    probs = F.softmax(torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.5, 0.0]], dtype=torch.float32), dim=-1)
    targets = torch.tensor([0, 1], dtype=torch.long)
    all_m = ens.metrics_under_all_rules(probs=probs, targets=targets)
    for rule_name, rule_fn in ens.RULE_FNS.items():
        preds = rule_fn(probs)
        qwk, _ = _validation_qwk_and_classification(preds=preds, targets=targets)
        assert all_m[rule_name]["qwk"] == pytest.approx(float(qwk))


def test_compute_rule_metrics_includes_confusion_matrix() -> None:
    """``compute_rule_metrics`` surfaces a 4x4 integer confusion matrix."""

    probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    m = ens.compute_rule_metrics(probs=probs, targets=targets, rule_fn=ens.apply_argmax)
    assert len(m["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in m["confusion_matrix"])


def test_format_conf_matrix_and_metric_line_non_empty() -> None:
    """Formatting helpers emit header rows and a single summary line."""

    cm = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    text = ens.format_conf_matrix(cm)
    assert "No Damage" in text
    assert "Destroyed" in text

    m = {
        "qwk": 0.5,
        "macro_f1": 0.6,
        "accuracy": 0.7,
        "per_class_f1": dict.fromkeys(ens.ORDINAL_CLASS_DISPLAY_NAMES, 0.55),
    }
    line = ens.format_metric_line("test_label", m)
    assert "test_label" in line
    assert "QWK=" in line


def test_compute_cross_variant_ensemble_equal_weights() -> None:
    """Cross-variant pooling matches manual mean over variant-level ensemble probs."""

    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    k = 4
    logits_a = torch.zeros(4, k, dtype=torch.float32)
    logits_b = torch.zeros(4, k, dtype=torch.float32)
    for i in range(4):
        logits_a[i, i] = 5.0
        logits_b[i, (i + 1) % k] = 5.0
    pa = F.softmax(logits_a, dim=-1)
    pb = F.softmax(logits_b, dim=-1)
    replay_a = ens.metrics_under_all_rules(probs=pa, targets=targets)
    replay_b = ens.metrics_under_all_rules(probs=pb, targets=targets)

    def _vr(name: str, p: torch.Tensor, replay: dict[str, dict]) -> dict:
        return {
            "variant_root": str(Path("/tmp") / name),
            "holdout_event": "evt",
            "seeds": [0, 1],
            "per_seed": [],
            "ensemble_probs": p,
            "individual_probs": torch.stack([p, p], dim=0),
            "targets": targets,
            "ensemble_metrics": replay,
        }

    vr_a = _vr("a", pa, replay_a)
    vr_b = _vr("b", pb, replay_b)
    cross = ens.compute_cross_variant_ensemble([vr_a, vr_b])
    manual_mean = torch.stack([pa, pb], dim=0).mean(dim=0)
    manual_metrics = ens.metrics_under_all_rules(probs=manual_mean, targets=targets)
    assert cross["num_variants"] == 2
    assert cross["num_seeds_total"] == 4
    assert cross["equal_variant_metrics"]["EV"]["qwk"] == pytest.approx(manual_metrics["EV"]["qwk"])


def test_compute_cross_variant_ensemble_target_mismatch_raises() -> None:
    """Misaligned val targets across variants must abort before averaging."""

    t1 = torch.tensor([0, 1], dtype=torch.long)
    t2 = torch.tensor([1, 0], dtype=torch.long)
    k = 4
    p = torch.ones(2, k, dtype=torch.float32) / k
    replay = ens.metrics_under_all_rules(probs=p, targets=t1)
    vr1 = {
        "variant_root": "/a",
        "holdout_event": "e",
        "seeds": [0],
        "per_seed": [],
        "ensemble_probs": p,
        "individual_probs": p.unsqueeze(0),
        "targets": t1,
        "ensemble_metrics": replay,
    }
    vr2 = {**vr1, "variant_root": "/b", "targets": t2}
    with pytest.raises(RuntimeError, match="Cross-variant target mismatch"):
        ens.compute_cross_variant_ensemble([vr1, vr2])


def test_collect_averaged_probabilities_shapes_and_softmax() -> None:
    """TTA stack runs on CPU and returns concatenated [T,K] probabilities."""

    loader = DataLoader(_OneBatchDataset(), batch_size=2)
    model = _TinyLogitModel()
    model.eval()
    probs, targets = ens.collect_averaged_probabilities(model=model, val_loader=loader, device="cpu")
    assert probs.shape == (2, 4)
    assert targets.shape == (2,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_load_resolved_config_and_metrics_missing_files(tmp_path: Path) -> None:
    """Missing YAML / JSON raise ``FileNotFoundError`` with the expected stem."""

    fold = tmp_path / "seed_00"
    fold.mkdir()
    with pytest.raises(FileNotFoundError, match="config_resolved"):
        ens.load_resolved_config(fold)
    (fold / "config_resolved.yaml").write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metrics"):
        ens.load_stored_metrics(fold)


def test_load_resolved_config_and_metrics_roundtrip(tmp_path: Path) -> None:
    """Happy path parses YAML and JSON payloads."""

    fold = tmp_path / "seed_00"
    fold.mkdir()
    (fold / "config_resolved.yaml").write_text("model:\n  name: test\n", encoding="utf-8")
    metrics = {"best_val_qwk": 0.42}
    (fold / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert ens.load_resolved_config(fold)["model"]["name"] == "test"
    assert ens.load_stored_metrics(fold)["best_val_qwk"] == 0.42


def test_serialize_for_json_is_identity_on_pure_dicts() -> None:
    """``serialize_for_json`` currently passes through JSON-safe metric dicts."""

    d = {"a": 1, "b": [1, 2]}
    assert ens.serialize_for_json(d) is d


def test_write_prob_cache_roundtrip(tmp_path: Path) -> None:
    """Probability cache preserves tensors and metadata for downstream tools."""

    targets = torch.tensor([0, 1], dtype=torch.long)
    k = 4
    p = F.softmax(torch.randn(2, k), dim=-1)
    replay = ens.metrics_under_all_rules(probs=p, targets=targets)
    vr = {
        "variant_root": str(tmp_path / "variant_a"),
        "holdout_event": "H1",
        "seeds": [0, 11],
        "per_seed": [],
        "ensemble_probs": p,
        "individual_probs": torch.stack([p, p], dim=0),
        "targets": targets,
        "ensemble_metrics": replay,
    }
    out = tmp_path / "cache.pt"
    ens.write_prob_cache(path=out, variant_results=[vr], split_name="S1", holdout_event="H1")
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert loaded["split"] == "S1"
    assert loaded["holdout_event"] == "H1"
    assert torch.equal(loaded["targets"], targets)
    assert len(loaded["variants"]) == 1
    assert loaded["variants"][0]["ensemble_probs"].shape == (2, 4)


def test_write_artifact_json_structure(tmp_path: Path) -> None:
    """JSON artifact lists variants and cross-variant metric blocks."""

    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    k = 4
    p = F.softmax(torch.randn(4, k), dim=-1)
    replay = ens.metrics_under_all_rules(probs=p, targets=targets)
    per_seed = [
        {
            "seed": 0,
            "stored_qwk": 0.1,
            "stored_macro_f1": 0.2,
            "replay_delta_vs_stored": 0.0,
            "replay_ok": True,
            "replay_metrics": replay,
        }
    ]
    vr = {
        "variant_root": str(tmp_path / "v"),
        "holdout_event": "evt",
        "seeds": [0],
        "per_seed": per_seed,
        "ensemble_probs": p,
        "individual_probs": p.unsqueeze(0),
        "targets": targets,
        "ensemble_metrics": replay,
    }
    cross = ens.compute_cross_variant_ensemble([vr])
    path = tmp_path / "out.json"
    ens.write_artifact(
        path=path,
        variant_results=[vr],
        cross_results=cross,
        split_name="S",
        holdout_event="evt",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"] == "S"
    assert len(payload["variants"]) == 1
    assert "replay_metrics" in payload["variants"][0]["per_seed"][0]
    assert "equal_variant_metrics" in payload["cross_variant"]


def test_print_summary_table_prints_header(capsys: pytest.CaptureFixture[str]) -> None:
    """Summary table includes the fixed banner line."""

    targets = torch.tensor([0, 1], dtype=torch.long)
    k = 4
    p = torch.ones(2, k, dtype=torch.float32) / k
    replay = ens.metrics_under_all_rules(probs=p, targets=targets)
    vr = {
        "variant_root": str(Path("/tmp") / "variant_x"),
        "holdout_event": "e",
        "seeds": [0],
        "per_seed": [],
        "ensemble_probs": p,
        "individual_probs": p.unsqueeze(0),
        "targets": targets,
        "ensemble_metrics": replay,
    }
    cross = ens.compute_cross_variant_ensemble([vr])
    ens.print_summary_table([vr], cross)
    captured = capsys.readouterr().out
    assert "SUMMARY" in captured


def _ensemble_ablation_cfg() -> dict[str, object]:
    """Minimal ``ablation`` block matching ``build_model_from_config`` / ``run_variant`` prints."""

    return {
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": False,
        "mask_dilation_px": 2,
    }


def _ensemble_model_cfg(*, drop_path: float | None = None) -> dict[str, object]:
    """Minimal ``model`` block for ``build_model_from_config``."""

    d: dict[str, object] = {"name": "resnet18"}
    if drop_path is not None:
        d["drop_path_rate"] = drop_path
    return d


def test_build_model_from_config_resnet18_eval_mode_cpu() -> None:
    """``build_model_from_config`` wires ablation flags and leaves the net in eval mode on CPU."""

    cfg = {"ablation": _ensemble_ablation_cfg(), "model": _ensemble_model_cfg()}
    model = ens.build_model_from_config(cfg=cfg, device="cpu")
    assert isinstance(model, MaskCenteredDamageNet)
    assert not model.training
    x = torch.randn(1, 4, 64, 64, dtype=torch.float32)
    ctx = torch.randn(1, 4, dtype=torch.float32)
    logits = model(x, ctx)
    assert logits.shape == (1, 4)


def test_build_model_from_config_explicit_drop_path_still_builds() -> None:
    """YAML may set ``drop_path_rate``; construction stays in ``[0, 1)``."""

    cfg = {"ablation": _ensemble_ablation_cfg(), "model": _ensemble_model_cfg(drop_path=0.05)}
    model = ens.build_model_from_config(cfg=cfg, device="cpu")
    x = torch.randn(1, 4, 32, 32, dtype=torch.float32)
    ctx = torch.randn(1, 4, dtype=torch.float32)
    assert model(x, ctx).shape == (1, 4)


class _FiveSampleDataset(Dataset):
    """Five val samples so ``collect_averaged_probabilities`` iterates more than one batch."""

    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros((4, 8, 8), dtype=torch.uint8),
            "context": torch.zeros(4, dtype=torch.float32),
            "label": torch.tensor(idx % 4, dtype=torch.long),
        }


def test_collect_averaged_probabilities_concatenates_multiple_batches() -> None:
    """Multiple val batches are concatenated along the sample dimension."""

    loader = DataLoader(_FiveSampleDataset(), batch_size=2)
    model = _TinyLogitModel()
    model.eval()
    probs, targets = ens.collect_averaged_probabilities(model=model, val_loader=loader, device="cpu")
    assert probs.shape == (5, 4)
    assert targets.shape == (5,)


def _write_seed_fold(
    path: Path,
    *,
    metrics: dict[str, object],
    state: dict[str, torch.Tensor],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cfg_yaml = "\n".join(
        [
            "ablation:",
            "  mask_enabled: true",
            "  typology_enabled: true",
            "  mask_weighted_pooling_enabled: false",
            "  mask_dilation_px: 0",
            "model:",
            "  name: resnet18",
            "  drop_path_rate: 0.0",
        ]
    )
    (path / "config_resolved.yaml").write_text(cfg_yaml, encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    torch.save(state, path / "best_model.pt")


def test_run_variant_raises_when_no_seed_directories(tmp_path: Path) -> None:
    """``run_variant`` requires at least one ``seed_<nn>`` directory under the split."""

    variant = tmp_path / "arm_a"
    (variant / "Spatial_Block_East").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No seed folds"):
        ens.run_variant(
            variant_root=variant,
            seeds=[0, 11],
            split_name="Spatial_Block_East",
            data_dir=None,
            device="cpu",
        )


def test_run_variant_single_seed_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``run_variant`` stitches loader, checkpoint replay, and within-variant ensemble (mocked I/O)."""

    split = "Spatial_Block_East"
    variant = tmp_path / "baseline"
    seed_dir = variant / split / "seed_00"
    tiny = _TinyLogitModel()
    _write_seed_fold(
        seed_dir,
        metrics={"best_val_qwk": 1.0, "best_classification": {"macro_f1": 0.25}},
        state=tiny.state_dict(),
    )

    loader = DataLoader(_OneBatchDataset(), batch_size=2)

    def _fake_build_val_loader(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
        _ = cfg
        _ = data_dir_override
        return loader, "SyntheticHoldout"

    monkeypatch.setattr(ens, "build_val_loader_from_config", _fake_build_val_loader)

    def _stub_build_model(cfg: dict, device: str) -> _TinyLogitModel:
        _ = cfg
        return tiny.to(device)

    monkeypatch.setattr(ens, "build_model_from_config", _stub_build_model)

    out = ens.run_variant(
        variant_root=variant,
        seeds=[0, 99],
        split_name=split,
        data_dir=None,
        device="cpu",
    )
    assert out["holdout_event"] == "SyntheticHoldout"
    assert out["seeds"] == [0]
    assert out["ensemble_probs"].shape == (2, 4)
    assert out["individual_probs"].shape == (1, 2, 4)
    assert "ensemble_metrics" in out
    assert len(out["per_seed"]) == 1
    captured = capsys.readouterr().out
    assert "baseline" in captured
    assert "SyntheticHoldout" in captured


def test_run_variant_stored_metrics_without_macro_f1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When ``metrics.json`` omits macro F1, the seed log line skips the stored-F1 suffix."""

    split = "S"
    variant = tmp_path / "v"
    seed_dir = variant / split / "seed_00"
    tiny = _TinyLogitModel()
    _write_seed_fold(seed_dir, metrics={"best_val_qwk": 1.0}, state=tiny.state_dict())
    loader = DataLoader(_OneBatchDataset(), batch_size=2)

    def _stub_val_loader(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
        _ = cfg
        _ = data_dir_override
        return loader, "H"

    def _stub_build_model(cfg: dict, device: str) -> _TinyLogitModel:
        _ = cfg
        return tiny.to(device)

    monkeypatch.setattr(ens, "build_val_loader_from_config", _stub_val_loader)
    monkeypatch.setattr(ens, "build_model_from_config", _stub_build_model)
    ens.run_variant(variant_root=variant, seeds=[0], split_name=split, data_dir=None, device="cpu")
    out = capsys.readouterr().out
    assert "stored_F1=" not in out


def test_run_variant_replay_not_ok_when_stored_qwk_far_from_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Large ``|replay QWK - stored|`` yields ``replay_ok`` False and ``!!`` in the log."""

    split = "S"
    variant = tmp_path / "v"
    seed_dir = variant / split / "seed_00"
    tiny = _TinyLogitModel()
    _write_seed_fold(seed_dir, metrics={"best_val_qwk": -1.0}, state=tiny.state_dict())
    loader = DataLoader(_OneBatchDataset(), batch_size=2)

    def _stub_val_loader(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
        _ = cfg
        _ = data_dir_override
        return loader, "H"

    def _stub_build_model(cfg: dict, device: str) -> _TinyLogitModel:
        _ = cfg
        return tiny.to(device)

    monkeypatch.setattr(ens, "build_val_loader_from_config", _stub_val_loader)
    monkeypatch.setattr(ens, "build_model_from_config", _stub_build_model)
    out = ens.run_variant(variant_root=variant, seeds=[0], split_name=split, data_dir=None, device="cpu")
    assert out["per_seed"][0]["replay_ok"] is False
    assert "[!!]" in capsys.readouterr().out


def test_run_variant_raises_target_mismatch_between_seeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Second seed whose TTA targets differ triggers the alignment ``RuntimeError``."""

    split = "S"
    variant = tmp_path / "v"
    for seed in (0, 11):
        d = variant / split / f"seed_{seed:02d}"
        tiny = _TinyLogitModel()
        _write_seed_fold(d, metrics={"best_val_qwk": 1.0}, state=tiny.state_dict())

    loader = DataLoader(_OneBatchDataset(), batch_size=2)

    def _stub_val_loader(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
        _ = cfg
        _ = data_dir_override
        return loader, "H"

    def _stub_build_model(cfg: dict, device: str) -> _TinyLogitModel:
        _ = cfg
        return _TinyLogitModel().to(device)

    monkeypatch.setattr(ens, "build_val_loader_from_config", _stub_val_loader)
    monkeypatch.setattr(ens, "build_model_from_config", _stub_build_model)

    calls = {"n": 0}

    def _fake_collect(model: nn.Module, val_loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        _ = model
        _ = val_loader
        _ = device
        calls["n"] += 1
        p = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        t = torch.tensor([0, 1], dtype=torch.long) if calls["n"] == 1 else torch.tensor([1, 0], dtype=torch.long)
        return p, t

    monkeypatch.setattr(ens, "collect_averaged_probabilities", _fake_collect)
    with pytest.raises(RuntimeError, match="Target mismatch"):
        ens.run_variant(variant_root=variant, seeds=[0, 11], split_name=split, data_dir=None, device="cpu")


def test_run_variant_calls_cuda_empty_cache_when_gpu_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """After each seed, ``torch.cuda.empty_cache`` runs when CUDA is reported available."""

    split = "S"
    variant = tmp_path / "v"
    seed_dir = variant / split / "seed_00"
    tiny = _TinyLogitModel()
    _write_seed_fold(seed_dir, metrics={"best_val_qwk": 1.0}, state=tiny.state_dict())
    loader = DataLoader(_OneBatchDataset(), batch_size=2)

    def _stub_val_loader(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
        _ = cfg
        _ = data_dir_override
        return loader, "H"

    def _stub_build_model(cfg: dict, device: str) -> _TinyLogitModel:
        _ = cfg
        return tiny.to(device)

    monkeypatch.setattr(ens, "build_val_loader_from_config", _stub_val_loader)
    monkeypatch.setattr(ens, "build_model_from_config", _stub_build_model)
    mock_empty = MagicMock()
    monkeypatch.setattr(ens.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ens.torch.cuda, "empty_cache", mock_empty)
    ens.run_variant(variant_root=variant, seeds=[0], split_name=split, data_dir=None, device="cpu")
    mock_empty.assert_called()


def _fake_variant_result(variant_root: Path) -> dict[str, object]:
    targets = torch.tensor([0, 1], dtype=torch.long)
    k = 4
    p = F.softmax(torch.randn(2, k), dim=-1)
    replay = ens.metrics_under_all_rules(probs=p, targets=targets)
    return {
        "variant_root": str(variant_root),
        "holdout_event": "MainHoldout",
        "seeds": [0],
        "per_seed": [
            {
                "seed": 0,
                "stored_qwk": 0.5,
                "stored_macro_f1": 0.5,
                "replay_delta_vs_stored": 0.0,
                "replay_ok": True,
                "replay_metrics": replay,
            }
        ],
        "ensemble_probs": p,
        "individual_probs": p.unsqueeze(0),
        "targets": targets,
        "ensemble_metrics": replay,
    }


def test_main_end_to_end_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``main`` resolves device from YAML, runs variants, writes JSON and optional prob cache."""

    v1 = tmp_path / "variant_a"
    v2 = tmp_path / "variant_b"
    for v in (v1, v2):
        seed = v / "FoldX" / "seed_00"
        seed.mkdir(parents=True)
        (seed / "config_resolved.yaml").write_text("runtime:\n  device: cpu\n  seed: 7\n", encoding="utf-8")

    json_out = tmp_path / "ensemble.json"
    cache_out = tmp_path / "probs.pt"

    def _stub_run_variant(variant_root: Path, *_args: object, **_kwargs: object) -> dict[str, object]:
        return _fake_variant_result(Path(variant_root))

    monkeypatch.setattr(ens, "run_variant", _stub_run_variant)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ensemble",
            "--variant-roots",
            str(v1),
            str(v2),
            "--split",
            "FoldX",
            "--device",
            "cpu",
            "--seeds",
            "0",
            "--output-json",
            str(json_out),
            "--prob-cache",
            str(cache_out),
        ],
    )
    ens.main()
    assert json_out.is_file()
    assert cache_out.is_file()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["holdout_event"] == "MainHoldout"
    assert len(payload["variants"]) == 2


def test_main_raises_when_variant_has_no_seed_directories(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI fails fast when the first variant root has no matching ``seed_*`` folds."""

    empty = tmp_path / "no_seeds"
    empty.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ensemble",
            "--variant-roots",
            str(empty),
            "--split",
            "MissingSplit",
            "--device",
            "cpu",
            "--output-json",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="No usable seed"):
        ens.main()


def test_default_seeds_tuple_covers_expected_values() -> None:
    """Module-level seed list stays stable for CLI defaults and layout discovery."""

    assert ens.DEFAULT_SEEDS[0] == 0
    assert ens.DEFAULT_SEEDS[-1] == 99
    assert len(ens.DEFAULT_SEEDS) == 10


def test_replay_tolerance_is_strict_enough_for_float_noise() -> None:
    """Documented replay slack matches a small float window."""

    assert pytest.approx(5e-3) == ens.REPLAY_TOLERANCE
