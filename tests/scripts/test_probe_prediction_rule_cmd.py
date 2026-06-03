"""Smoke tests for ``scripts/probe_prediction_rule.py::main`` with heavy dependencies mocked."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import probe_prediction_rule as ppr


class _OneBatchLoader:
    """Minimal loader with ``__len__`` (``main`` logs batch count)."""

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        batch = {
            "image": torch.randint(0, 255, (2, 4, 8, 8), dtype=torch.uint8),
            "context": torch.zeros(2, 4, dtype=torch.float32),
            "label": torch.tensor([0, 1], dtype=torch.long),
        }
        yield batch


def _minimal_cfg() -> dict[str, Any]:
    return {
        "data": {
            "dir": "data/",
            "chip_size": 8,
            "sensor_profile": "uas_5cm",
            "holdout_event": "Spatial_Block_East",
        },
        "model": {"name": "convnext_tiny", "pretrained": False, "drop_path_rate": 0.0},
        "ablation": {
            "mask_enabled": True,
            "typology_enabled": True,
            "mask_weighted_pooling_enabled": False,
        },
        "runtime": {
            "device": "cpu",
            "seed": 0,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "drop_last": False,
            "batch_size": 2,
            "val_batch_size_factor": 1.0,
        },
        "training": {"batch_size": 2, "sampler_mode": "uniform"},
        "metadata": {"holdout_selection": "explicit"},
    }


def test_main_writes_probe_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` runs one seed, mocked inference, and writes ``prediction_rule_probe.json``."""

    variant = tmp_path / "baseline"
    fold = variant / "seed_11" / "fold_Spatial_Block_Spatial_Block_East"
    fold.mkdir(parents=True)
    (fold / "config_resolved.yaml").write_text("x: 1\n", encoding="utf-8")

    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    targets = torch.tensor([0, 1], dtype=torch.long)
    ev_metrics = ppr.compute_rule_metrics(probs=probs, targets=targets, rule_fn=ppr.apply_ev_rounding)
    (fold / "metrics.json").write_text(json.dumps({"best_val_qwk": ev_metrics["qwk"]}), encoding="utf-8")
    (fold / "best_model.pt").write_bytes(b"")

    tiny = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 8 * 8 + 4, 4),
    )
    st = tiny.state_dict()

    def _fake_collect(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return probs, targets

    def _fake_load(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return st

    monkeypatch.setattr(ppr, "build_val_loader_from_config", lambda **_k: (_OneBatchLoader(), "Spatial_Block_East"))
    def _fake_build_model(**kwargs: object) -> torch.nn.Module:
        device = str(kwargs["device"])
        return tiny.to(device)

    monkeypatch.setattr(ppr, "build_model_from_config", _fake_build_model)
    monkeypatch.setattr(ppr, "load_resolved_config", lambda _p: _minimal_cfg())
    monkeypatch.setattr(ppr.torch, "load", _fake_load)
    monkeypatch.setattr(ppr, "collect_averaged_probabilities", _fake_collect)
    monkeypatch.setattr(ppr, "set_seeds", lambda **_k: None)

    out_json = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_",
            "--variant-root",
            str(variant),
            "--seeds",
            "11",
            "--device",
            "cpu",
            "--output-json",
            str(out_json),
        ],
    )
    ppr.main()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["seeds"] == [11]
    assert len(payload["per_seed"]) == 1
    assert "aggregate" in payload
    assert payload["per_seed"][0]["replay_ok"] is True
