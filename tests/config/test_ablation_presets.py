"""Tests for ablation experiment matrix config presets."""

from pathlib import Path

import pytest
import yaml

from src.config.settings import load_config


PRESET_EXPECTATIONS = {
    "ablation_baseline.yaml": {
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_mask.yaml": {
        "mask_enabled": False,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": False,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_typology.yaml": {
        "mask_enabled": True,
        "typology_enabled": False,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_resolution.yaml": {
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "manned_15cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_rgb_only.yaml": {
        "mask_enabled": False,
        "typology_enabled": False,
        "mask_weighted_pooling_enabled": False,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_no_smoothing.yaml": {
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.0,
        "loss": "emd"
    },
    "ablation_mask_channel_only.yaml": {
        "mask_enabled": True,
        "typology_enabled": False,
        "mask_weighted_pooling_enabled": False,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_pooling_only.yaml": {
        "mask_enabled": False,
        "typology_enabled": False,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "emd"
    },
    "ablation_ce_loss.yaml": {
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": True,
        "sensor_profile": "uas_5cm",
        "label_smoothing": 0.02,
        "loss": "ce"
    }
}


def _presets_dir() -> Path:
    """Return the ablation presets directory path."""

    return Path("config") / "presets"


@pytest.mark.parametrize("preset_name", sorted(PRESET_EXPECTATIONS))
def test_ablation_preset_files_exist(preset_name: str) -> None:
    """Each matrix preset file exists in config/presets."""

    preset_path = _presets_dir() / preset_name
    assert preset_path.exists()


@pytest.mark.parametrize(("preset_name", "expected"), sorted(PRESET_EXPECTATIONS.items()))
def test_ablation_presets_define_expected_matrix_values(preset_name: str, expected: dict[str, bool | str | float]) -> None:
    """Preset files define the expected mask/typology/pooling/sensor/smoothing/loss matrix."""

    preset_path = _presets_dir() / preset_name
    raw_payload = yaml.safe_load(preset_path.read_text(encoding="utf-8"))

    assert raw_payload["ablation"]["mask_enabled"] is expected["mask_enabled"]
    assert raw_payload["ablation"]["typology_enabled"] is expected["typology_enabled"]
    assert raw_payload["ablation"]["mask_weighted_pooling_enabled"] is expected["mask_weighted_pooling_enabled"]
    assert raw_payload["data"]["sensor_profile"] == expected["sensor_profile"]
    assert raw_payload["training"]["label_smoothing"] == expected["label_smoothing"]
    assert raw_payload["training"].get("loss", "emd") == expected["loss"]


@pytest.mark.parametrize("preset_name", sorted(PRESET_EXPECTATIONS))
def test_ablation_presets_are_loadable_by_runtime_config_parser(preset_name: str) -> None:
    """Each preset can be parsed by the shared runtime config loader."""

    preset_path = _presets_dir() / preset_name
    config = load_config(preset_path)
    assert config.data.dir == Path("data/")
