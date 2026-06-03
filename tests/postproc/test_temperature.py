"""Unit tests for ``src.postproc.temperature`` (per-seed temperature scaling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.postproc import temperature as temp


def test_apply_temperature_rows_sum_to_one() -> None:
    """Temperature-scaled softmax rows are non-negative and sum to 1."""

    log_p = np.log(np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]], dtype=np.float64) + 1e-12)
    log_p = np.log(np.exp(log_p) / np.exp(log_p).sum(axis=1, keepdims=True))
    out = temp.apply_temperature(log_probs=log_p, temperature=2.0)
    assert out.shape == log_p.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0.0).all()


def test_nll_of_temperature_non_positive_is_inf() -> None:
    """Non-positive ``T`` yields ``+inf`` NLL (invalid distribution)."""

    log_p = np.zeros((2, 4), dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)
    assert temp.nll_of_temperature(0.0, log_p, targets) == float("inf")
    assert temp.nll_of_temperature(-1.0, log_p, targets) == float("inf")


def test_fit_temperature_returns_bounded_scalar() -> None:
    """``minimize_scalar`` returns ``T`` in ``(0.1, 10)`` with finite train NLL."""

    rng = np.random.default_rng(1)
    n, k = 16, 4
    logits = rng.standard_normal((n, k)).astype(np.float64) * 0.3
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    targets = rng.integers(0, k, size=n, dtype=np.int64)
    t_opt, nll = temp.fit_temperature(log_probs=log_probs, targets=targets)
    assert 0.1 <= t_opt <= 10.0
    assert np.isfinite(nll)


def test_ev_preds_uniform_four_class() -> None:
    """Uniform probabilities yield EV class 2 for K=4 (same convention as ensemble)."""

    p = np.full((1, 4), 0.25, dtype=np.float64)
    assert temp.ev_preds(p).tolist() == [2]


def test_hybrid_extremes_follow_argmax() -> None:
    """Hybrid uses argmax when argmax is class 0 or K-1."""

    p = np.array([[0.95, 0.02, 0.02, 0.01]], dtype=np.float64)
    assert temp.hybrid_preds(p).tolist() == [0]
    p2 = np.array([[0.01, 0.02, 0.02, 0.95]], dtype=np.float64)
    assert temp.hybrid_preds(p2).tolist() == [3]


def test_macro_recall_diagonal_prefers_correct() -> None:
    """Perfect predictions yield macro recall 1.0."""

    preds = np.array([0, 1, 2, 3], dtype=np.int64)
    targets = preds.copy()
    assert temp.macro_recall(preds=preds, targets=targets, num_classes=4) == pytest.approx(1.0)


def test_classification_breakdown_includes_macro_recall() -> None:
    """Breakdown bundles QWK, F1, accuracy, recall, and confusion matrix."""

    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    targets = np.array([0, 1], dtype=np.int64)
    m = temp.classification_breakdown(preds=temp.argmax_preds(probs), targets=targets, num_classes=4)
    assert "macro_recall" in m
    assert m["accuracy"] == pytest.approx(1.0)


def test_print_rule_row_format(capsys: pytest.CaptureFixture[str]) -> None:
    """``print_rule_row`` emits QWK / F1 / acc / recall labels."""

    m = {
        "qwk": 0.5,
        "macro_f1": 0.6,
        "accuracy": 0.7,
        "macro_recall": 0.55,
        "per_class_f1": {},
        "confusion_matrix": [],
    }
    temp.print_rule_row(label="argmax", m=m)
    out = capsys.readouterr().out
    assert "argmax" in out
    assert "QWK=" in out


def test_rule_fns_cover_three_rules() -> None:
    """Registry matches ensemble-style reporting."""

    assert set(temp.RULE_FNS.keys()) == {"EV", "argmax", "hybrid"}


def _minimal_prob_payload() -> dict[str, object]:
    """Cache shape ``[num_seeds, N, K]`` as loaded in ``temperature.main``."""

    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    k = 4
    logits = torch.zeros(8, k, dtype=torch.float32)
    for i in range(8):
        logits[i, int(targets[i])] = 2.0
    p = torch.softmax(logits, dim=-1)
    individual = torch.stack([p, p.clone()], dim=0)
    return {
        "targets": targets,
        "variants": [
            {
                "variant_root": "/tmp/temp_variant",
                "seeds": [0, 11],
                "individual_probs": individual,
            }
        ],
    }


def test_main_smoke_writes_json(tmp_path: Path) -> None:
    """``main`` completes on a tiny probability cache with reduced fold count."""

    cache = tmp_path / "probs.pt"
    torch.save(_minimal_prob_payload(), cache)
    out_json = tmp_path / "temperature.json"

    argv = [
        "temperature",
        "--prob-cache",
        str(cache),
        "--num-folds",
        "2",
        "--seed",
        "0",
        "--output-json",
        str(out_json),
    ]
    with patch.object(sys, "argv", argv):
        temp.main()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["num_folds"] == 2
    assert "uncalibrated" in data
    assert "oof_calibrated" in data
    assert len(data["per_fold_temperatures"]) == 2
