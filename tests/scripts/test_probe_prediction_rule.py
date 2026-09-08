"""Unit tests for prediction-rule helpers in ``scripts/probe_prediction_rule.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import probe_prediction_rule as ppr
from src.model.trainer import ORDINAL_CLASS_DISPLAY_NAMES


def test_apply_ev_rounding_deterministic() -> None:
    """EV rounding matches ``floor(sum i p_i + 0.5)``."""

    probs = torch.tensor([[0.9, 0.033, 0.033, 0.034]], dtype=torch.float32)
    ev = float((probs * torch.arange(4, dtype=torch.float32)).sum())
    assert ev < 0.5
    assert int(ppr.apply_ev_rounding(probs).item()) == 0
    probs2 = torch.tensor([[0.05, 0.1, 0.35, 0.5]], dtype=torch.float32)
    ev = (0 * 0.05 + 1 * 0.1 + 2 * 0.35 + 3 * 0.5)
    assert int(ppr.apply_ev_rounding(probs2).item()) == int(torch.floor(torch.tensor(ev + 0.5)).item())


def test_apply_argmax() -> None:
    """Argmax picks the largest mass index."""

    probs = torch.tensor([[0.1, 0.45, 0.44, 0.01]], dtype=torch.float32)
    assert int(ppr.apply_argmax(probs).item()) == 1


def test_apply_hybrid_uses_ev_for_interior_argmax() -> None:
    """When argmax is an interior class, hybrid follows EV rounding."""

    probs = torch.tensor([[0.05, 0.4, 0.4, 0.15]], dtype=torch.float32)
    ev = ppr.apply_ev_rounding(probs)
    hy = ppr.apply_hybrid(probs)
    assert int(ev.item()) == 2
    assert int(hy.item()) == 2
    assert int(ppr.apply_argmax(probs).item()) == 1


def test_apply_hybrid_trusts_argmax_on_extremes() -> None:
    """When argmax is class 0 or K-1, hybrid equals argmax."""

    probs = torch.tensor([[0.85, 0.05, 0.05, 0.05]], dtype=torch.float32)
    assert torch.equal(ppr.apply_hybrid(probs), ppr.apply_argmax(probs))


def test_compute_rule_metrics_keys() -> None:
    """``compute_rule_metrics`` returns QWK and flat per-class recall keys."""

    probs = torch.eye(4, dtype=torch.float32)
    targets = torch.arange(4, dtype=torch.long)
    metrics = ppr.compute_rule_metrics(probs=probs, targets=targets, rule_fn=ppr.apply_argmax)
    assert "qwk" in metrics
    assert "macro_f1" in metrics
    assert set(metrics["per_class_recall"]) == set(ORDINAL_CLASS_DISPLAY_NAMES)
    assert len(metrics["confusion_matrix"]) == 4


def test_format_per_class_row_and_confusion() -> None:
    """Formatters emit non-empty strings."""

    per_class = dict.fromkeys(ORDINAL_CLASS_DISPLAY_NAMES, 0.1234)
    row = ppr.format_per_class_row("hybrid", per_class)
    assert "hybrid" in row
    cm = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
    block = ppr.format_confusion(cm)
    assert "No Damage" in block
    assert "1" in block


def test_load_resolved_config_and_metrics(tmp_path: Path) -> None:
    """YAML / JSON loaders read fold artifacts."""

    fold = tmp_path / "seed_00"
    fold.mkdir()
    (fold / "config_resolved.yaml").write_text(
        "data:\n  dir: data/\n  chip_size: 256\n  sensor_profile: uas_5cm\n",
        encoding="utf-8",
    )
    (fold / "metrics.json").write_text(json.dumps({"best_val_qwk": 0.5}), encoding="utf-8")
    cfg = ppr.load_resolved_config(fold)
    assert cfg["data"]["chip_size"] == 256
    m = ppr.load_stored_metrics(fold)
    assert m["best_val_qwk"] == 0.5


def test_load_resolved_config_missing_raises(tmp_path: Path) -> None:
    """Missing ``config_resolved.yaml`` raises ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError, match="config_resolved"):
        ppr.load_resolved_config(tmp_path)


def test_load_stored_metrics_missing_raises(tmp_path: Path) -> None:
    """Missing ``metrics.json`` raises ``FileNotFoundError``."""

    with pytest.raises(FileNotFoundError, match=r"metrics\.json"):
        ppr.load_stored_metrics(tmp_path)


# Model and loader reconstruction -----------------------------------------------

def _replay_cfg(*, holdout_event: str | None, holdout_selection: str | None = None) -> dict:
    """Resolved-config snapshot with every section the replay helpers read."""

    cfg: dict = {
        "data": {"dir": "data", "sensor_profile": "uas_5cm", "chip_size": 64, "holdout_event": holdout_event},
        "runtime": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2,
                    "drop_last": False, "val_batch_size_factor": 2.0, "seed": 5},
        "training": {"batch_size": 1},
        "model": {"name": "resnet18", "drop_path_rate": 0.0},
        "ablation": {"mask_enabled": True, "typology_enabled": True, "mask_weighted_pooling_enabled": True},
    }
    if holdout_selection is not None:
        cfg["metadata"] = {"holdout_selection": holdout_selection}
    return cfg


def test_build_model_from_config_matches_snapshot_flags() -> None:
    """The rebuilt network is in eval mode on the requested device and accepts a 4-channel chip."""

    model = ppr.build_model_from_config(cfg=_replay_cfg(holdout_event=None), device="cpu")
    assert isinstance(model, ppr.MaskCenteredDamageNet)
    assert not model.training
    assert model(torch.randn(1, 4, 64, 64), torch.zeros(1, 4)).shape == (1, 4)


def _two_event_manifest(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    """Duplicate the conftest mosaic under a second event so LOEO selection has a choice."""

    other = sample_manifest.copy()
    other["event"] = "Hurricane Ida"
    manifest = pd.concat([sample_manifest, other], ignore_index=True)
    manifest["image_name"] = [f"{event.replace(' ', '_')}_mosaic" for event in manifest["event"]]
    manifest["valid"] = True
    return manifest


def test_build_val_loader_from_config_explicit_holdout(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit holdout reconstructs that fold's validation loader from the scanned manifest."""

    seen: dict[str, str] = {}

    def _fake_scan(data_dir: str, sensor_profile: str) -> pd.DataFrame:
        seen.update(data_dir=data_dir, sensor_profile=sensor_profile)
        return _two_event_manifest(sample_manifest)

    monkeypatch.setattr(ppr, "build_valid_manifest", _fake_scan)
    val_loader, holdout = ppr.build_val_loader_from_config(cfg=_replay_cfg(holdout_event="Hurricane Ida"),
                                                           data_dir_override="/override")
    assert seen == {"data_dir": "/override", "sensor_profile": "uas_5cm"}
    assert holdout == "Hurricane Ida"
    assert val_loader.batch_size == 2
    batch = next(iter(val_loader))
    assert batch["image"].shape == (2, 4, 64, 64)
    assert batch["label"].tolist() == [1, 3]


def test_build_val_loader_from_config_default_spatial(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """default_spatial snapshots replay through the spatial-block fold rather than an event holdout."""

    monkeypatch.setattr(ppr, "build_valid_manifest", lambda *_args, **_kwargs: _two_event_manifest(sample_manifest))
    val_loader, holdout = ppr.build_val_loader_from_config(
        cfg=_replay_cfg(holdout_event=None, holdout_selection="default_spatial"))
    assert holdout == "Spatial_Block_East"
    assert len(val_loader.dataset) == 2


def test_collect_averaged_probabilities_rotation_tta(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four-rotation TTA yields one normalized probability row per validation instance."""

    monkeypatch.setattr(ppr, "build_valid_manifest", lambda *_args, **_kwargs: _two_event_manifest(sample_manifest))
    cfg = _replay_cfg(holdout_event="Hurricane Ian")
    val_loader, _ = ppr.build_val_loader_from_config(cfg=cfg)
    model = ppr.build_model_from_config(cfg=cfg, device="cpu")
    probs, targets = ppr.collect_averaged_probabilities(model=model, val_loader=val_loader, device="cpu")
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert targets.tolist() == [1, 3]
