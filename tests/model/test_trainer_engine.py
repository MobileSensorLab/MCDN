"""Engine-level tests for trainer loop behavior and deterministic controls."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from unittest.mock import MagicMock, patch

from src.model.trainer import UnitemporalTrainer, compute_qwk, set_seeds


class DummyDataset(Dataset):
    """Generates random data matching trainer input schema."""

    def __len__(self) -> int:
        """Return a tiny fixed dataset size."""

        return 8

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one synthetic sample. Mirrors the production dataset contract: uint8 [4, H, W]."""

        _ = idx
        return {
            "image": torch.randint(low=0, high=256, size=(4, 64, 64), dtype=torch.uint8),
            "label": torch.tensor(1, dtype=torch.long),
            "context": torch.tensor([1.0, 0.0, 0.0, 0.0])
        }


class DummyModel(nn.Module):
    """Simple linear model to bypass heavy backbone setup."""

    def __init__(self) -> None:
        """Initialize a compact conv + linear classifier."""

        super().__init__()
        self.conv = nn.Conv2d(4, 16, 3)
        self.fc = nn.Linear(16 * 62 * 62 + 4, 4)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute logits from image and context tensors."""

        v = torch.relu(self.conv(x))
        v = v.view(v.size(0), -1)
        fused = torch.cat([v, context], dim=1)
        return self.fc(fused)


def _build_dummy_trainer(tmp_path: Path) -> UnitemporalTrainer:
    """Build a lightweight trainer for control-flow tests."""

    model = DummyModel()
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()
    return UnitemporalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        accum_steps=2,
        checkpoint_dir=tmp_path
    )


def test_unitemporal_trainer_ema_enables_shadow_model(tmp_path: Path) -> None:
    """EMA builds an AveragedModel shadow when ema_enabled=True."""

    model = DummyModel()
    loader = DataLoader(DummyDataset(), batch_size=2)
    criterion = nn.CrossEntropyLoss()
    trainer = UnitemporalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        accum_steps=2,
        checkpoint_dir=tmp_path,
        ema_enabled=True,
        ema_decay=0.99,
    )
    assert trainer.ema_model is not None


def test_train_epoch_updates_ema_after_optimizer_step(tmp_path: Path) -> None:
    """Each optimizer step invokes EMA update_parameters when a shadow exists."""

    trainer = _build_dummy_trainer(tmp_path)
    ema = MagicMock()
    trainer.ema_model = ema
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer._train_epoch(1)
    ema.update_parameters.assert_called()
    assert all(call.args[0] is trainer.model for call in ema.update_parameters.call_args_list)


def test_compute_qwk_perfect_score() -> None:
    """Identical predictions yield QWK == 1.0."""

    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3])
    assert compute_qwk(preds, targets) == 1.0


def test_compute_qwk_catastrophic_score() -> None:
    """Extreme inverse errors produce negative QWK."""

    preds = torch.tensor([0, 0, 3, 3])
    targets = torch.tensor([3, 3, 0, 0])
    assert compute_qwk(preds, targets) < 0.0


def test_set_seeds_enables_determinism_with_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Providing a seed enables deterministic backend mode."""

    recorded: dict[str, object] = {}
    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.model.trainer.torch.use_deterministic_algorithms",
        lambda enabled: recorded.setdefault("enabled", enabled)
    )

    set_seeds(seed=7)
    assert recorded["enabled"] is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seeds_disables_determinism_without_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No seed keeps stochastic backend mode enabled."""

    recorded: dict[str, object] = {}
    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.model.trainer.torch.use_deterministic_algorithms",
        lambda enabled: recorded.setdefault("enabled", enabled)
    )

    set_seeds(seed=None)
    assert recorded["enabled"] is False
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


@patch("torch.compile")
def test_trainer_initialization_and_forward_pass(mock_compile: MagicMock) -> None:
    """Trainer initializes and runs one train/validate cycle."""

    mock_compile.side_effect = lambda x: x
    model = DummyModel()
    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2)
    criterion = nn.CrossEntropyLoss()

    trainer = UnitemporalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        device="cpu",
        accum_steps=2
    )
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    assert trainer._train_epoch(1) > 0.0
    metrics = trainer._validate(eval_model=trainer.model)
    assert "val_loss" in metrics
    assert "val_qwk" in metrics


def test_trainer_fit_saves_checkpoints_on_improvement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() saves a best checkpoint for each QWK improvement."""

    trainer = _build_dummy_trainer(tmp_path)

    def mock_train_epoch(_epoch: int) -> float:
        trainer.optimizer.step()
        return 0.1

    monkeypatch.setattr(trainer, "_train_epoch", mock_train_epoch)
    qwk_values = iter([0.10, 0.20, 0.30])
    monkeypatch.setattr(trainer, "_validate", lambda **_kwargs: {"val_loss": 0.5, "val_qwk": next(qwk_values)})
    saved_paths: list[str] = []
    monkeypatch.setattr("src.model.trainer.torch.save", lambda _state_dict, checkpoint_path: saved_paths.append(str(checkpoint_path)))

    trainer.fit(epochs=3, early_stopping_patience=10)
    assert trainer.best_qwk == pytest.approx(0.30)
    assert len(saved_paths) == 3
    assert all(path.endswith("best_model.pt") for path in saved_paths)


def test_trainer_fit_covers_no_improvement_and_early_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """fit() increments no-improvement counter and early-stops."""

    trainer = _build_dummy_trainer(tmp_path)
    monkeypatch.setattr(trainer, "_train_epoch", lambda _epoch: 0.1)
    monkeypatch.setattr(trainer, "_validate", lambda **_kwargs: {"val_loss": 1.0, "val_qwk": -1.0})
    monkeypatch.setattr("src.model.trainer.torch.save", lambda _state_dict, _checkpoint_path: None)

    trainer.fit(epochs=3, early_stopping_patience=1)
    assert trainer.best_qwk == 0.0


