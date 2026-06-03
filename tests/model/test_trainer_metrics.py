"""Targeted tests for trainer confusion-matrix / QWK helpers and device resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model.trainer import (
    ORDINAL_CLASS_DISPLAY_NAMES,
    UnitemporalTrainer,
    _classification_metrics_from_confusion,
    _confusion_matrix,
    _dataset_label_summary,
    _qwk_from_confusion,
    _resolve_device,
    _validation_qwk_and_classification,
    compute_qwk,
)


def test_confusion_matrix_accumulates_off_diagonal() -> None:
    """``_confusion_matrix`` places counts at (target, pred) indices."""

    preds = torch.tensor([1, 2, 0], dtype=torch.long)
    targets = torch.tensor([0, 2, 0], dtype=torch.long)
    cm = _confusion_matrix(preds=preds, targets=targets, num_classes=4)
    assert int(cm[0, 1].item()) == 1
    assert int(cm[0, 0].item()) == 1
    assert int(cm[2, 2].item()) == 1


def test_qwk_from_confusion_perfect_diagonal() -> None:
    """Perfect agreement yields QWK 1.0."""

    cm = torch.eye(4, dtype=torch.float64)
    cm[0, 0] = 5.0
    assert _qwk_from_confusion(conf_mat=cm, num_classes=4) == pytest.approx(1.0)


def test_qwk_from_confusion_den_zero_returns_one() -> None:
    """When ``(expected * weights).sum()`` is zero, QWK defaults to 1.0 (degenerate agreement)."""

    cm = torch.zeros((2, 2), dtype=torch.float64)
    cm[0, 0] = 1.0
    assert _qwk_from_confusion(conf_mat=cm, num_classes=2) == pytest.approx(1.0)


def test_classification_metrics_from_confusion_bad_class_names() -> None:
    """``class_names`` length must match the confusion matrix order."""

    cm = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="class_names length"):
        _classification_metrics_from_confusion(conf_mat=cm, class_names=("a", "b", "c"))


def test_validation_qwk_matches_compute_qwk() -> None:
    """``_validation_qwk_and_classification`` QWK matches ``compute_qwk`` on the same preds."""

    preds = torch.tensor([0, 1, 2, 3, 1, 0], dtype=torch.long)
    targets = torch.tensor([0, 1, 2, 3, 2, 3], dtype=torch.long)
    qwk_val, cls = _validation_qwk_and_classification(preds=preds, targets=targets)
    assert float(qwk_val) == pytest.approx(compute_qwk(preds=preds, targets=targets))
    assert cls["num_samples"] == preds.numel()
    assert len(cls["per_class"]) == len(ORDINAL_CLASS_DISPLAY_NAMES)


def test_resolve_device_cpu_and_auto() -> None:
    """Concrete ``cpu`` passes through; ``auto`` resolves without raising."""

    assert _resolve_device("cpu") == "cpu"
    out = _resolve_device("auto")
    assert out in {"cpu", "cuda"}


def test_resolve_device_invalid_raises() -> None:
    """Unsupported device strings are rejected before touching hardware."""

    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_device("mps")


def test_resolve_device_cuda_without_gpu_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting CUDA when unavailable surfaces a clear error."""

    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA availability"):
        _resolve_device("cuda")


def test_dataset_label_summary_counts_known_labels() -> None:
    """``_dataset_label_summary`` tallies normalized damage_label strings."""

    instances = [
        {"damage_label": "No Damage"},
        {"damage_label": "minor damage"},
        {"damage_label": "DESTROYED"},
        {"damage_label": "unknown"},
        {"damage_label": ""},
    ]
    ds = SimpleNamespace(instances=instances)
    summary = _dataset_label_summary(ds)
    assert summary["no damage"] == 1
    assert summary["minor damage"] == 1
    assert summary["destroyed"] == 1
    assert summary["major damage"] == 0


class _MiniModel(nn.Module):
    """Tiny conv stack matching ``test_trainer_engine.DummyModel`` input contract."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3)
        self.fc = nn.Linear(4 * 62 * 62 + 4, 4)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        v = torch.relu(self.conv(x))
        v = v.view(v.size(0), -1)
        return self.fc(torch.cat([v, context], dim=1))


class _MiniDataset(Dataset):
    """One sample with uint8 four-channel imagery (RGB random; mask channel 0/255 only)."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _ = idx
        rgb = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
        mask = torch.full((1, 64, 64), 255, dtype=torch.uint8)
        return {
            "image": torch.cat([rgb, mask], dim=0),
            "label": torch.tensor(0, dtype=torch.long),
            "context": torch.zeros(4, dtype=torch.float32),
        }


def test_prepare_batch_normalizes_rgb_leaves_mask_unscaled(tmp_path: Path) -> None:
    """``_prepare_batch`` applies ImageNet stats to RGB only; mask stays raw uint8-as-float."""

    model = _MiniModel()
    ds = _MiniDataset()
    loader = DataLoader(ds, batch_size=1)
    trainer = UnitemporalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        accum_steps=1,
        checkpoint_dir=tmp_path,
    )
    batch = next(iter(loader))
    images, labels, context = trainer._prepare_batch(batch)
    assert images.dtype == torch.float32
    assert labels.shape == (1,)
    assert context.shape == (1, 4)
    assert torch.allclose(images[:, 3, :, :], torch.full((1, 64, 64), 255.0))
    assert images[:, :3, :, :].abs().max() > 1.0
