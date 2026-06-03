"""Tests for small trainer utilities not covered elsewhere (time rollover, params, class weights, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Never
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.config.settings import (
    AblationConfig,
    AppConfig,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from src.model.trainer import (
    _build_dataloaders,
    _compute_class_weights,
    _count_model_params,
    _emit_run_header,
    _format_grad_norm,
    _fmt_time_long,
    _print_end_banner,
    _print_lr_drop_block,
    _print_shadow_note,
    _resolve_device_label,
    _resolve_run_label,
    batches_in_accum_window,
    set_seeds,
)


def test_fmt_time_long_minute_and_hour_rollover() -> None:
    """Rounding can push ``secs`` and ``minutes`` past 59; the formatter carries into the next unit."""

    assert _fmt_time_long(3599.9) == "1h 00m 00s"


def test_count_model_params_non_callable_parameters() -> None:
    """Models whose ``parameters`` is not callable yield zero counts (defensive path)."""

    class Weird(nn.Module):
        """``timm``-like object where ``parameters`` is not a method."""

        parameters = "not-callable"

    total, trainable = _count_model_params(Weird())
    assert total == 0
    assert trainable == 0


def test_count_model_params_iter_raises_returns_zero() -> None:
    """If iterating parameters raises, the helper returns ``(0, 0)`` instead of propagating."""

    class Broken(nn.Module):
        def parameters(self) -> Never:
            raise RuntimeError("simulated worker failure")

    assert _count_model_params(Broken()) == (0, 0)


def test_resolve_device_label_cpu_string() -> None:
    """Non-CUDA device strings pass through unchanged."""

    assert _resolve_device_label("cpu") == "cpu"


def test_resolve_device_label_cuda_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA builds a ``cuda:i (name, GiB)`` label when device properties are available."""

    fake_props = MagicMock()
    fake_props.name = "TestGPU"
    fake_props.total_memory = 32 * (1024**3)

    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.model.trainer.torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("src.model.trainer.torch.cuda.get_device_properties", lambda _i: fake_props)

    label = _resolve_device_label("cuda")
    assert "cuda:0" in label
    assert "TestGPU" in label
    assert "32.0 GiB" in label


def test_resolve_device_label_cuda_properties_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``get_device_properties`` fails, the label falls back to ``cuda:{index}``."""

    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.model.trainer.torch.cuda.current_device", lambda: 1)

    def _boom(_idx: int) -> None:
        raise RuntimeError("no device")

    monkeypatch.setattr("src.model.trainer.torch.cuda.get_device_properties", _boom)
    assert _resolve_device_label("cuda") == "cuda:1"


def test_format_grad_norm_non_finite() -> None:
    """Non-finite or missing norms render the verbose-table sentinel."""

    assert _format_grad_norm(None) == "    --"
    assert _format_grad_norm(float("nan")) == "    --"


def test_print_shadow_note_only_when_ema_enabled(capsys: pytest.CaptureFixture[str]) -> None:
    """EMA shadow line prints only when the trainer is using a shadow model."""

    _print_shadow_note(ema_enabled=False, ema_decay=0.99)
    assert capsys.readouterr().out == ""

    _print_shadow_note(ema_enabled=True, ema_decay=0.995)
    out = capsys.readouterr().out
    assert "shadow" in out
    assert "0.995" in out


def test_print_lr_drop_block_without_warmstart_note(capsys: pytest.CaptureFixture[str]) -> None:
    """LR-drop block omits the warm-start line when ``warmstart_note`` is ``None``."""

    _print_lr_drop_block(
        epoch=3,
        previous_lr=1e-3,
        updated_lr=5e-4,
        first_reduction=False,
        warmstart_note=None,
    )
    text = capsys.readouterr().out
    assert "LR reduced" in text
    assert "early-stopping active" in text
    assert "warm-started" not in text


def test_print_end_banner_skips_top3_when_history_empty(capsys: pytest.CaptureFixture[str]) -> None:
    """With no epoch history, the end banner still prints best/final lines without a top-3 table."""

    best_classification = {
        "accuracy": 1.0,
        "macro_recall": 1.0,
        "macro_precision": 1.0,
        "macro_f1": 0.5,
        "num_samples": 0,
        "confusion_matrix": [],
        "per_class": {},
    }
    final_classification = {**best_classification, "macro_f1": 0.4}

    _print_end_banner(
        epochs_completed=1,
        wall_seconds=10.0,
        images_seen=0,
        history=[],
        best_classification=best_classification,
        final_classification=final_classification,
        best_selection_score=0.2,
        best_qwk=0.3,
        final_selection_score=0.2,
        final_qwk=0.3,
    )
    out = capsys.readouterr().out
    assert "Training complete" in out
    assert "images seen --" in out
    assert "#1" not in out


def test_batches_in_accum_window_last_partial_block() -> None:
    """The final accumulation window may be shorter than ``accum_steps``."""

    assert batches_in_accum_window(batch_idx=8, accum_steps=4, total_batches=9) == 1
    assert batches_in_accum_window(batch_idx=0, accum_steps=4, total_batches=10) == 4


def test_set_seeds_calls_cuda_manual_seed_when_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a seed is set and CUDA is available, ``torch.cuda.manual_seed_all`` is invoked."""

    mock_manual = MagicMock()
    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.model.trainer.torch.cuda.manual_seed_all", mock_manual)
    set_seeds(seed=123)
    # PyTorch may route ``torch.manual_seed`` through the same CUDA hook on some builds.
    assert mock_manual.call_count >= 1
    assert all(call.args == (123,) for call in mock_manual.call_args_list)


def test_compute_class_weights_empty_counts_returns_none() -> None:
    """All-zero training counts yield ``None`` (caller disables weighting)."""

    counts = {"no damage": 0, "minor damage": 0, "major damage": 0, "destroyed": 0}
    assert _compute_class_weights(train_label_counts=counts, power=1.0, eps=1.0, max_ratio=None) is None


def test_compute_class_weights_power_and_max_ratio_clip() -> None:
    """``power`` reweights frequencies; ``max_ratio`` clips outliers relative to the median."""

    counts = {"no damage": 100, "minor damage": 10, "major damage": 10, "destroyed": 10}
    w = _compute_class_weights(train_label_counts=counts, power=2.0, eps=1.0, max_ratio=1.5)
    assert w is not None
    assert w.shape == (4,)
    assert torch.all(torch.isfinite(w))


def test_resolve_run_label_prefers_ablation_parent(tmp_path: Path) -> None:
    """Paths under ``.../ablation/<variant>/...`` resolve to ``variant``."""

    root = tmp_path / "outputs" / "ablation" / "no_mask" / "Spatial_Block_East" / "seed_00"
    root.mkdir(parents=True)
    assert _resolve_run_label(root) == "no_mask"


def test_build_dataloaders_persistent_workers_without_workers_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``persistent_workers=True`` with ``num_workers=0`` is rejected (PyTorch constraint)."""

    class _StubDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.instances = [{"damage_label": "no damage"}]

    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", _StubDataset)
    train_df = pd.DataFrame({"a": [1]})
    val_df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="persistent_workers"):
        _build_dataloaders(
            train_df=train_df,
            val_df=val_df,
            chip_size=64,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False,
            val_batch_size_factor=1.0,
            seed=0,
            sampler_mode="uniform",
            mask_dilation_px=0,
        )


def test_emit_run_header_prints_class_weight_line(capsys: pytest.CaptureFixture[str]) -> None:
    """When class weights are enabled, the timeline includes the weight vector."""

    config = AppConfig(
        data=DataConfig(dir=Path("dataset-root"), chip_size=64),
        training=TrainingConfig(epochs=1, batch_size=2, accum_steps=1, lr=1e-4),
        model=ModelConfig(name="resnet18", pretrained=False),
        ablation=AblationConfig(),
        runtime=RuntimeConfig(num_workers=0, persistent_workers=False),
    )
    split_summary = {
        "train_instances": 4,
        "val_instances": 2,
        "train_label_counts": {"no damage": 1, "minor damage": 1, "major damage": 1, "destroyed": 1},
        "val_label_counts": {"no damage": 1, "minor damage": 0, "major damage": 0, "destroyed": 0},
    }
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

    _emit_run_header(
        config=config,
        config_hash_str="deadbeef",
        run_label="weighted",
        effective_seed=0,
        fold_name="F",
        resolved_device="cpu",
        param_total=1000,
        param_trainable=1000,
        sensor_profile="uas_5cm",
        valid_instance_count=10,
        split_summary=split_summary,
        class_weights=weights,
        data_root=Path("dataset-root"),
    )
    out = capsys.readouterr().out
    assert "class weights" in out
    assert "0.25" in out
