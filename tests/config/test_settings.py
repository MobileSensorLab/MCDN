"""Tests for runtime configuration parsing and validation."""

import json

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import load_config


def test_load_config_yaml_success(tmp_path: Path) -> None:
    """YAML config is parsed into validated runtime model."""

    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  chip_size: 512",
            "training:",
            "  epochs: 10",
            "  batch_size: 16",
            "  accum_steps: 2",
            "  lr: 0.0002",
            "model:",
            "  name: convnext_tiny",
            "  pretrained: true"
        ]),
        encoding="utf-8"
    )

    config = load_config(path)
    assert config.data.dir == Path("data/")
    assert config.data.sensor_profile == "uas_5cm"
    assert config.training.epochs == 10
    assert config.training.sampler_mode == "weighted"
    assert config.training.class_weighting_enabled is False
    assert config.model.pretrained is True
    assert config.ablation.mask_enabled is True
    assert config.ablation.typology_enabled is True
    assert config.runtime.num_workers == 8
    assert config.runtime.device == "auto"
    assert config.runtime.seed is None


def test_load_config_json_success(tmp_path: Path) -> None:
    """JSON config is parsed into validated runtime model."""

    path = tmp_path / "config.json"
    payload = {
        "data": {"dir": "data/", "chip_size": 256, "holdout_event": "Hurricane Ian", "sensor_profile": "manned_15cm"},
        "training": {"epochs": 2, "batch_size": 8, "accum_steps": 1, "lr": 1e-4, "sampler_mode": "uniform", "class_weighting_enabled": True, "class_weighting_power": 0.5},
        "model": {"name": "convnext_nano", "pretrained": False},
        "ablation": {"mask_enabled": False, "typology_enabled": False},
        "runtime": {
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "drop_last": False,
            "val_batch_size_factor": 0.75,
            "device": "cpu",
            "seed": 123,
            "early_stopping_patience": 6,
            "checkpoint_root": "outputs/checkpoints"
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_config(path)
    assert config.data.chip_size == 256
    assert config.data.holdout_event == "Hurricane Ian"
    assert config.data.sensor_profile == "manned_15cm"
    assert config.training.sampler_mode == "uniform"
    assert config.training.class_weighting_enabled is True
    assert config.training.class_weighting_power == 0.5
    assert config.model.name == "convnext_nano"
    assert config.ablation.mask_enabled is False
    assert config.ablation.typology_enabled is False
    assert config.runtime.num_workers == 4
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 123
    assert config.runtime.checkpoint_root == Path("outputs/checkpoints")


def test_load_config_missing_file() -> None:
    """Missing config path raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_config("missing.yaml")


def test_load_config_validation_error(tmp_path: Path) -> None:
    """Invalid config shape raises Pydantic ValidationError."""

    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  chip_size: -1",
            "training:",
            "  epochs: 5",
            "  batch_size: 16",
            "  accum_steps: 2",
            "  lr: 0.0001"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_runtime_worker_constraint_error(tmp_path: Path) -> None:
    """persistent_workers requires num_workers > 0."""

    path = tmp_path / "bad_runtime.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "runtime:",
            "  num_workers: 0",
            "  persistent_workers: true"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_invalid_sensor_profile_error(tmp_path: Path) -> None:
    """Invalid sensor profile enum value raises ValidationError."""

    path = tmp_path / "bad_profile.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  sensor_profile: satellite_30cm"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_weighted_sampler_and_class_weights_allowed(tmp_path: Path) -> None:
    """Weighted sampler plus loss class weights is allowed (may compound rare-class emphasis)."""

    path = tmp_path / "weighting_combo.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "training:",
            "  sampler_mode: weighted",
            "  class_weighting_enabled: true",
            "model:",
            "  name: convnext_tiny",
            "  pretrained: true"
        ]),
        encoding="utf-8"
    )

    config = load_config(path)
    assert config.training.sampler_mode == "weighted"
    assert config.training.class_weighting_enabled is True


def test_runtime_lr_drop_warmstart_from_best_yaml_key(tmp_path: Path) -> None:
    """runtime.lr_drop_warmstart_from_best is the canonical YAML key."""

    path = tmp_path / "warm.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "runtime:",
            "  lr_drop_warmstart_from_best: true",
            "model:",
            "  name: convnext_tiny",
            "  pretrained: true"
        ]),
        encoding="utf-8"
    )

    config = load_config(path)
    assert config.runtime.lr_drop_warmstart_from_best is True


