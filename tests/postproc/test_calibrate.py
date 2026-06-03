"""Unit tests for ``src.postproc.calibrate`` (logit-bias ensemble calibration)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.postproc import calibrate as cal


def test_build_quadratic_weights_caches_by_num_classes() -> None:
    """Repeated ``K`` returns the identical cached weight matrix."""

    cal.QWK_WEIGHTS_CACHE.clear()
    w1 = cal.build_quadratic_weights(num_classes=4)
    w2 = cal.build_quadratic_weights(num_classes=4)
    assert w1.shape == (4, 4)
    assert w1 is w2


def test_quadratic_weighted_kappa_perfect_agreement() -> None:
    """Identical preds and targets yield QWK 1.0."""

    t = np.array([0, 1, 2, 3], dtype=np.int64)
    assert cal.quadratic_weighted_kappa(preds=t, targets=t, num_classes=4) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_denominator_zero_returns_one() -> None:
    """Degenerate expected matrix yields QWK 1.0 (library contract)."""

    preds = np.array([0, 0], dtype=np.int64)
    targets = np.array([0, 0], dtype=np.int64)
    out = cal.quadratic_weighted_kappa(preds=preds, targets=targets, num_classes=2)
    assert out == pytest.approx(1.0)


def test_classification_breakdown_keys_and_shapes() -> None:
    """``classification_breakdown`` returns macro F1, per-class dict, and 4x4 CM."""

    preds = np.array([0, 1, 2, 3], dtype=np.int64)
    targets = np.array([0, 1, 2, 2], dtype=np.int64)
    m = cal.classification_breakdown(preds=preds, targets=targets, num_classes=4)
    assert set(m.keys()) == {"macro_f1", "per_class_f1", "confusion_matrix", "qwk", "accuracy"}
    assert len(m["per_class_f1"]) == 4
    assert len(m["confusion_matrix"]) == 4
    assert all(len(row) == 4 for row in m["confusion_matrix"])


def test_ensemble_log_probs_mean_then_log() -> None:
    """``ensemble_log_probs`` matches ``log(mean(softmax, dim=0))`` with clamp."""

    probs = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    lp = cal.ensemble_log_probs(individual_probs=probs)
    assert lp.shape == (2, 4)
    manual = torch.log(probs.mean(dim=0).clamp(min=1e-12)).numpy()
    assert np.allclose(lp, manual)


def test_apply_bias_argmax_matches_manual_argmax() -> None:
    """Additive bias shifts logits before ``argmax``."""

    probs = np.array([[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05]], dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_probs = np.log(probs + 1e-12)
    bias = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float64)
    preds = cal.apply_bias_argmax(log_probs=log_probs, bias_vec=bias)
    assert preds.tolist() == [0, 0]


def test_format_bias_and_per_class_non_empty() -> None:
    """Formatting helpers emit short class prefixes."""

    b = np.array([0.1, -0.2, 0.0, 0.3], dtype=np.float64)
    text = cal.format_bias(b)
    assert "NoD" in text or "no" in text.lower()
    per = dict.fromkeys(cal.ORDINAL_CLASS_DISPLAY_NAMES, 0.5)
    line = cal.format_per_class(per)
    assert "0.5000" in line


def test_fit_bias_for_f1_runs_on_toy_logits() -> None:
    """``differential_evolution`` returns a bias vector in bounds with finite F1."""

    rng = np.random.default_rng(0)
    n, k = 24, 4
    logits = rng.standard_normal((n, k)).astype(np.float64) * 0.5
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    targets = rng.integers(0, k, size=n, dtype=np.int64)
    bias, f1 = cal.fit_bias_for_f1(
        log_probs=log_probs,
        targets=targets,
        num_classes=k,
        bounds_half_width=1.5,
        seed=1,
        max_iter=8,
    )
    assert bias.shape == (k,)
    assert np.all(bias >= -1.5)
    assert np.all(bias <= 1.5)
    assert f1 >= 0.0
    assert f1 <= 1.0


def _minimal_prob_payload() -> dict[str, object]:
    """Torch-serializable cache matching ``src.postproc.ensemble.write_prob_cache``."""

    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    k = 4
    logits = torch.zeros(8, k, dtype=torch.float32)
    for i in range(8):
        logits[i, int(targets[i])] = 2.0
    p = torch.softmax(logits, dim=-1)
    individual = torch.stack([p, p.clone()], dim=0)
    return {
        "split": "TestSplit",
        "holdout_event": "TestHoldout",
        "targets": targets,
        "variants": [
            {
                "variant_root": "/tmp/variant_unit",
                "seeds": [0, 1],
                "individual_probs": individual,
                "ensemble_probs": p,
            }
        ],
    }


def test_main_smoke_writes_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``main`` runs end-to-end with a tiny cache and stubbed DE for CI wall time."""

    cache = tmp_path / "probs.pt"
    torch.save(_minimal_prob_payload(), cache)
    out_json = tmp_path / "calibration.json"

    def _fake_de(
        func: object,
        bounds: object,
        *,
        seed: int,
        maxiter: int,
        **_kwargs: object,
    ) -> SimpleNamespace:
        _ = seed
        _ = maxiter
        n_dim = len(bounds)
        x0 = np.zeros(n_dim, dtype=np.float64)

        class _R(SimpleNamespace):
            pass

        fval = float(func(x0))
        return _R(x=x0, fun=fval)

    monkeypatch.setattr(cal, "differential_evolution", _fake_de)

    argv = [
        "calibrate",
        "--prob-cache",
        str(cache),
        "--num-folds",
        "2",
        "--max-iter",
        "1",
        "--seed",
        "0",
        "--output-json",
        str(out_json),
    ]
    with patch.object(sys, "argv", argv):
        cal.main()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["split"] == "TestSplit"
    assert data["num_folds"] == 2
    assert "uncalibrated" in data
    assert "oof_calibrated" in data
    assert len(data["per_fold"]) == 2
