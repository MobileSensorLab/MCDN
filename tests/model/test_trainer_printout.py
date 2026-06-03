"""Tests for the trainer's fixed-width printout helpers and verbose/non-verbose gating."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model.trainer import (
    UnitemporalTrainer,
    _emit_run_header,
    _format_param_count,
    _format_patience_str,
    _format_sel_delta,
    _format_split_counts,
    _fmt_gpu_mem_gib,
    _fmt_time_long,
    _fmt_time_short,
    _print_end_banner,
    _print_lr_drop_block,
    _resolve_run_label,
)


def test_fmt_time_short_done_sentinel() -> None:
    """Negative / None / non-finite durations render the ``'   done'`` sentinel."""

    assert _fmt_time_short(None) == "   done"
    assert _fmt_time_short(-1.0) == "   done"
    assert _fmt_time_short(float("nan")) == "   done"


def test_fmt_time_short_wraps_rollover() -> None:
    """Sub-second rounding-up rolls minutes forward rather than emitting ``60s``."""

    assert _fmt_time_short(59.6) == " 1m 00s"
    assert _fmt_time_short(0.0) == " 0m 00s"
    assert _fmt_time_short(125.0) == " 2m 05s"


def test_fmt_time_long_formats_hms() -> None:
    """Wall-time formatter matches the end-banner grammar."""

    assert _fmt_time_long(3_600) == "1h 00m 00s"
    assert _fmt_time_long(4_934.0) == "1h 22m 14s"


def test_fmt_gpu_mem_gib_handles_zero_and_bytes() -> None:
    """GPU memory formatter handles CPU/unavailable (-> ``'   --'``) and GiB conversion."""

    assert _fmt_gpu_mem_gib(0) == "   --"
    assert _fmt_gpu_mem_gib(None) == "   --"
    assert _fmt_gpu_mem_gib(17_400_000_000).strip().endswith("G")


def test_format_patience_str_armed_vs_paused() -> None:
    """Patience formatter distinguishes ``'a'`` (armed) from ``'p'`` (paused)."""

    assert _format_patience_str(counter=3, patience=10, armed=True).endswith(" a")
    assert _format_patience_str(counter=0, patience=10, armed=False).endswith(" p")


def test_format_sel_delta_none_renders_dashes() -> None:
    """Missing delta renders as the ``'   ----'`` seven-char sentinel."""

    assert _format_sel_delta(None) == "   ----"
    assert _format_sel_delta(0.013) == "+0.0130"
    assert _format_sel_delta(-0.003) == "-0.0030"


def test_format_param_count_units() -> None:
    """Parameter count formatter applies ``M``/``K`` suffixes at the canonical thresholds."""

    assert _format_param_count(17_400_000) == "17.4M"
    assert _format_param_count(1_024) == "1.0K"
    assert _format_param_count(42) == "42"


def test_format_split_counts_uses_short_names() -> None:
    """Split counts render with the short ordinal names in canonical order."""

    result = _format_split_counts({
        "no damage": 4320,
        "minor damage": 1210,
        "major damage": 780,
        "destroyed": 421,
    })
    assert result == "{noD 4320, min 1210, maj 780, des 421}"


def test_resolve_run_label_strips_seed_suffix(tmp_path: Path) -> None:
    """Run-label helper drops ``seed_*`` leaves and falls back to ``'training'``."""

    variant_root = tmp_path / "ablation" / "baseline"
    seed_root = variant_root / "seed_11"
    seed_root.mkdir(parents=True)
    assert _resolve_run_label(variant_root) == "baseline"
    assert _resolve_run_label(seed_root) == "baseline"
    assert _resolve_run_label(Path("")) == "training"


def test_resolve_run_label_handles_new_split_layout(tmp_path: Path) -> None:
    """New layout ``ablation/<variant>/<split>/seed_<n>`` surfaces the variant name."""

    seed_root = tmp_path / "ablation" / "baseline" / "Spatial_Block_East" / "seed_11"
    seed_root.mkdir(parents=True)
    assert _resolve_run_label(seed_root) == "baseline"


def test_print_lr_drop_block_wraps_in_dividers(capsys: pytest.CaptureFixture[str]) -> None:
    """LR-drop block prints an ``===`` divider before and after the event lines."""

    _print_lr_drop_block(
        epoch=12,
        previous_lr=1.0e-4,
        updated_lr=3.16e-5,
        first_reduction=True,
        warmstart_note="warm-started training model from best checkpoint after LR drop",
    )
    captured_stdout = capsys.readouterr().out
    lines = [line for line in captured_stdout.splitlines() if line.strip()]
    assert lines[0] == "=" * 80
    assert lines[-1] == "=" * 80
    assert any("LR reduced 1.00e-04 -> 3.16e-05" in line and "early-stopping armed" in line for line in lines)
    assert any("warm-started" in line for line in lines)


def test_print_end_banner_emits_top3_when_history_present(capsys: pytest.CaptureFixture[str]) -> None:
    """End banner lists up to three epochs sorted by selection score."""

    history = [
        {"epoch": 5, "train_loss": 0.3, "val_loss": 0.3, "val_qwk": 0.80, "val_selection_score": 0.80,
         "best_val_qwk": 0.80, "best_val_selection_score": 0.80, "lr": 1e-4, "ema_active": False},
        {"epoch": 8, "train_loss": 0.2, "val_loss": 0.25, "val_qwk": 0.87, "val_selection_score": 0.85,
         "best_val_qwk": 0.87, "best_val_selection_score": 0.85, "lr": 1e-4, "ema_active": False},
        {"epoch": 12, "train_loss": 0.15, "val_loss": 0.22, "val_qwk": 0.88, "val_selection_score": 0.86,
         "best_val_qwk": 0.88, "best_val_selection_score": 0.86, "lr": 1e-4, "ema_active": False},
    ]
    best_classification = {"accuracy": 0.9, "macro_recall": 0.85, "macro_precision": 0.84, "macro_f1": 0.84,
                           "num_samples": 100, "confusion_matrix": [], "per_class": {}}
    final_classification = {"accuracy": 0.82, "macro_recall": 0.78, "macro_precision": 0.77, "macro_f1": 0.77,
                            "num_samples": 100, "confusion_matrix": [], "per_class": {}}

    _print_end_banner(
        epochs_completed=12,
        wall_seconds=4_934.0,
        images_seen=229_154,
        history=history,
        best_classification=best_classification,
        final_classification=final_classification,
        best_selection_score=0.86,
        best_qwk=0.88,
        final_selection_score=0.86,
        final_qwk=0.88,
    )
    stdout_text = capsys.readouterr().out
    assert "Training complete" in stdout_text
    assert "wall 1h 22m 14s" in stdout_text
    assert "images seen 229,154" in stdout_text
    assert "#1" in stdout_text
    assert "#2" in stdout_text
    assert "#3" in stdout_text
    assert "best    epoch 12" in stdout_text
    assert "final   epoch 12" in stdout_text


class _DummyDataset(Dataset):
    """Tiny dataset that satisfies the trainer's image/label/context batch contract."""

    def __len__(self) -> int:
        """Two-sample fixture is sufficient to exercise the verbose print path."""

        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a synthetic sample with deterministic shape (uint8 image per dataset contract)."""

        _ = idx
        return {
            "image": torch.randint(low=0, high=256, size=(4, 8, 8), dtype=torch.uint8),
            "label": torch.tensor(1, dtype=torch.long),
            "context": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }


class _DummyModel(nn.Module):
    """Minimal classifier covering the trainer's image/context forward signature."""

    def __init__(self) -> None:
        """Linear head wide enough for the 8x8 dummy chip."""

        super().__init__()
        self.flatten_size = 4 * 8 * 8 + 4
        self.fc = nn.Linear(self.flatten_size, 4)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Flatten the image chip and concatenate context vector before linear projection."""

        fused = torch.cat([image.reshape(image.size(0), -1), context], dim=1)
        return self.fc(fused)


@patch("torch.compile", new=lambda module: module)
def test_verbose_mode_emits_batch_table_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """verbose=True triggers the per-batch table header when an epoch runs."""

    loader = DataLoader(_DummyDataset(), batch_size=1)
    criterion = nn.CrossEntropyLoss()
    trainer = UnitemporalTrainer(
        model=_DummyModel(),
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        accum_steps=1,
        checkpoint_dir=tmp_path,
        verbose=True,
        print_every_n_batches=1,
    )
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer._train_epoch(1)
    stdout_text = capsys.readouterr().out
    assert "batch" in stdout_text
    assert "==========+" in stdout_text


@patch("torch.compile", new=lambda module: module)
def test_non_verbose_mode_emits_heartbeats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """verbose=False fires heartbeat lines (and no batch-table header)."""

    class _LongDataset(Dataset):
        """Loader with enough batches that heartbeat fractions fire interior to the epoch."""

        def __len__(self) -> int:
            """Twenty samples yields ten batches at batch_size=2."""

            return 20

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            """Return a synthetic sample matching the image/label/context contract (uint8 image)."""

            _ = idx
            return {
                "image": torch.randint(low=0, high=256, size=(4, 8, 8), dtype=torch.uint8),
                "label": torch.tensor(1, dtype=torch.long),
                "context": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            }

    loader = DataLoader(_LongDataset(), batch_size=2)
    criterion = nn.CrossEntropyLoss()
    trainer = UnitemporalTrainer(
        model=_DummyModel(),
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        accum_steps=1,
        checkpoint_dir=tmp_path,
        verbose=False,
    )
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer._train_epoch(1)
    stdout_text = capsys.readouterr().out
    assert "heartbeat" in stdout_text
    assert "==========+" not in stdout_text


def test_emit_run_header_renders_identity_box_and_timeline(capsys: pytest.CaptureFixture[str]) -> None:
    """The run-header helper emits the identity box followed by the timeline markers."""

    from src.config.settings import (
        AblationConfig,
        AppConfig,
        DataConfig,
        ModelConfig,
        RuntimeConfig,
        TrainingConfig,
    )

    config = AppConfig(
        data=DataConfig(dir=Path("dataset-root"), chip_size=512),
        training=TrainingConfig(epochs=5, batch_size=4, accum_steps=2, lr=1e-4, max_grad_norm=1.0),
        model=ModelConfig(name="convnextv2_nano", pretrained=True),
        ablation=AblationConfig(mask_enabled=True, typology_enabled=True),
        runtime=RuntimeConfig(num_workers=0, persistent_workers=False),
    )
    split_summary = {
        "train_instances": 120,
        "val_instances": 30,
        "train_label_counts": {"no damage": 80, "minor damage": 20, "major damage": 15, "destroyed": 5},
        "val_label_counts": {"no damage": 18, "minor damage": 6, "major damage": 4, "destroyed": 2},
    }

    _emit_run_header(
        config=config,
        config_hash_str="abcd1234",
        run_label="baseline",
        effective_seed=11,
        fold_name="Event_B",
        resolved_device="cpu",
        param_total=17_400_000,
        param_trainable=17_400_000,
        sensor_profile="uas_5cm",
        valid_instance_count=150,
        split_summary=split_summary,
        class_weights=None,
        data_root=Path("dataset-root"),
    )
    out = capsys.readouterr().out
    assert "=" * 80 in out
    assert "baseline (hash abcd1234)" in out
    assert "seed 11" in out
    assert "fold Event_B" in out
    assert "[data]" in out
    assert "[split]" in out
    assert "[loaders]" in out
    assert "[opt]" in out
    assert "Training   5 epochs" in out
