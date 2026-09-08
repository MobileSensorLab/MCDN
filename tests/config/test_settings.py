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


def test_load_config_loss_selector_default_and_ce(tmp_path: Path) -> None:
    """training.loss defaults to emd and accepts the ce ablation selector."""

    default_path = tmp_path / "loss_default.yaml"
    default_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/"
        ]),
        encoding="utf-8"
    )
    assert load_config(default_path).training.loss == "emd"

    ce_path = tmp_path / "loss_ce.yaml"
    ce_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "training:",
            "  loss: ce"
        ]),
        encoding="utf-8"
    )
    assert load_config(ce_path).training.loss == "ce"


def test_load_config_invalid_loss_selector_error(tmp_path: Path) -> None:
    """Loss selector outside the emd/ce pair raises ValidationError."""

    path = tmp_path / "bad_loss.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "training:",
            "  loss: focal"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_synthetic_gsd_factor_default_and_explicit(tmp_path: Path) -> None:
    """data.synthetic_gsd_factor defaults to 1.0 (disabled) and accepts degradation factors."""

    default_path = tmp_path / "gsd_default.yaml"
    default_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/"
        ]),
        encoding="utf-8"
    )
    assert load_config(default_path).data.synthetic_gsd_factor == 1.0

    degraded_path = tmp_path / "gsd_3x.yaml"
    degraded_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 3.0"
        ]),
        encoding="utf-8"
    )
    assert load_config(degraded_path).data.synthetic_gsd_factor == 3.0


def test_load_config_synthetic_gsd_factor_below_one_error(tmp_path: Path) -> None:
    """Factors below 1.0 (upsampling) raise ValidationError."""

    path = tmp_path / "bad_gsd.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 0.5"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_synthetic_gsd_mtf_default_and_explicit(tmp_path: Path) -> None:
    """data.synthetic_gsd_mtf_at_nyquist defaults to None (sampling-only) and accepts targets in (0, 1)."""

    default_path = tmp_path / "mtf_default.yaml"
    default_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 3.0"
        ]),
        encoding="utf-8"
    )
    assert load_config(default_path).data.synthetic_gsd_mtf_at_nyquist is None

    mtf_path = tmp_path / "mtf_03.yaml"
    mtf_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 3.0",
            "  synthetic_gsd_mtf_at_nyquist: 0.3"
        ]),
        encoding="utf-8"
    )
    assert load_config(mtf_path).data.synthetic_gsd_mtf_at_nyquist == 0.3


def test_load_config_synthetic_gsd_mtf_requires_degradation(tmp_path: Path) -> None:
    """An MTF target without an active degradation factor is a configuration error."""

    path = tmp_path / "mtf_no_factor.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_mtf_at_nyquist: 0.3"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_synthetic_gsd_mtf_rejects_out_of_range(tmp_path: Path) -> None:
    """MTF targets at or beyond the (0, 1) open interval raise ValidationError."""

    for bad_value in ["0.0", "1.0"]:
        path = tmp_path / f"mtf_bad_{bad_value.replace('.', '_')}.yaml"
        path.write_text(
            "\n".join([
                "data:",
                "  dir: data/",
                "  synthetic_gsd_factor: 3.0",
                f"  synthetic_gsd_mtf_at_nyquist: {bad_value}"
            ]),
            encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            load_config(path)


def test_load_config_synthetic_gsd_post_sharpen_default_and_explicit(tmp_path: Path) -> None:
    """data.synthetic_gsd_post_sharpen defaults to None and accepts positive amounts with a factor."""

    default_path = tmp_path / "sharpen_default.yaml"
    default_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 7.0"
        ]),
        encoding="utf-8"
    )
    assert load_config(default_path).data.synthetic_gsd_post_sharpen is None

    sharpen_path = tmp_path / "sharpen_02.yaml"
    sharpen_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_factor: 7.0",
            "  synthetic_gsd_post_sharpen: 0.2"
        ]),
        encoding="utf-8"
    )
    assert load_config(sharpen_path).data.synthetic_gsd_post_sharpen == 0.2


def test_load_config_synthetic_gsd_post_sharpen_requires_degradation(tmp_path: Path) -> None:
    """An unsharp amount without an active degradation factor is a configuration error."""

    path = tmp_path / "sharpen_no_factor.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  synthetic_gsd_post_sharpen: 0.2"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_load_config_synthetic_gsd_post_sharpen_rejects_non_positive(tmp_path: Path) -> None:
    """Zero or negative unsharp amounts raise ValidationError."""

    for bad_value in ["0.0", "-0.2"]:
        path = tmp_path / f"sharpen_bad_{bad_value.replace('.', '_').replace('-', 'neg')}.yaml"
        path.write_text(
            "\n".join([
                "data:",
                "  dir: data/",
                "  synthetic_gsd_factor: 7.0",
                f"  synthetic_gsd_post_sharpen: {bad_value}"
            ]),
            encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            load_config(path)


def test_load_config_chip_window_defaults_and_explicit(tmp_path: Path) -> None:
    """chip_window_scale defaults to 1.0 / chip_window_ground_m to None; explicit values load."""

    default_path = tmp_path / "window_default.yaml"
    default_path.write_text("\n".join(["data:", "  dir: data/"]), encoding="utf-8")
    default_cfg = load_config(default_path).data
    assert default_cfg.chip_window_scale == 1.0
    assert default_cfg.chip_window_ground_m is None

    fov_path = tmp_path / "window_fov.yaml"
    fov_path.write_text(
        "\n".join(["data:", "  dir: data/", "  chip_window_scale: 7.0"]),
        encoding="utf-8"
    )
    assert load_config(fov_path).data.chip_window_scale == 7.0

    ground_path = tmp_path / "window_ground.yaml"
    ground_path.write_text(
        "\n".join(["data:", "  dir: data/", "  chip_window_ground_m: 17.4"]),
        encoding="utf-8"
    )
    assert load_config(ground_path).data.chip_window_ground_m == 17.4


def test_load_config_chip_window_rejects_non_positive(tmp_path: Path) -> None:
    """Zero or negative window parameters raise ValidationError."""

    for field, bad_value in [("chip_window_scale", "0.0"), ("chip_window_scale", "-2.0"),
                             ("chip_window_ground_m", "0.0"), ("chip_window_ground_m", "-17.4")]:
        path = tmp_path / f"window_bad_{field}_{bad_value.replace('.', '_').replace('-', 'neg')}.yaml"
        path.write_text(
            "\n".join(["data:", "  dir: data/", f"  {field}: {bad_value}"]),
            encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            load_config(path)


def test_load_config_chip_window_excludes_synthetic_gsd(tmp_path: Path) -> None:
    """Window-geometry knobs and synthetic GSD degradation cannot combine."""

    for window_line in ["  chip_window_scale: 7.0", "  chip_window_ground_m: 122.0"]:
        path = tmp_path / f"window_conflict_{'scale' if 'scale' in window_line else 'ground'}.yaml"
        path.write_text(
            "\n".join(["data:", "  dir: data/", "  synthetic_gsd_factor: 7.0", window_line]),
            encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            load_config(path)


def test_load_config_holdout_event_accepts_string_list_and_null(tmp_path: Path) -> None:
    """data.holdout_event supports a single event name, a multi-event list, and null."""

    single_path = tmp_path / "holdout_single.yaml"
    single_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  holdout_event: Hurricane Michael"
        ]),
        encoding="utf-8"
    )
    assert load_config(single_path).data.holdout_event == "Hurricane Michael"

    multi_path = tmp_path / "holdout_multi.yaml"
    multi_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  holdout_event:",
            "    - Hurricane Michael",
            "    - '  Mayfield Tornado '",
            "    - Hurricane Michael"
        ]),
        encoding="utf-8"
    )
    # The list form strips whitespace and drops duplicates while preserving order.
    assert load_config(multi_path).data.holdout_event == ["Hurricane Michael", "Mayfield Tornado"]

    null_path = tmp_path / "holdout_null.yaml"
    null_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  holdout_event: null"
        ]),
        encoding="utf-8"
    )
    assert load_config(null_path).data.holdout_event is None


def test_load_config_holdout_event_rejects_empty_list(tmp_path: Path) -> None:
    """An empty (or all-blank) holdout_event list raises ValidationError."""

    path = tmp_path / "holdout_empty.yaml"
    path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  holdout_event: []"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(path)

    blank_path = tmp_path / "holdout_blank.yaml"
    blank_path.write_text(
        "\n".join([
            "data:",
            "  dir: data/",
            "  holdout_event:",
            "    - '   '"
        ]),
        encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        load_config(blank_path)


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
    """runtime.lr_drop_warmstart_from_best is the standard YAML key."""

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


