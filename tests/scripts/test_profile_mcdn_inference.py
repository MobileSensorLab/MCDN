"""Unit tests for deployment-profile helpers in ``scripts/profile_mcdn_inference.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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
