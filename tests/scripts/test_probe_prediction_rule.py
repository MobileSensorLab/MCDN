"""Unit tests for prediction-rule helpers in ``scripts/probe_prediction_rule.py``."""

from __future__ import annotations

import json
from pathlib import Path

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
