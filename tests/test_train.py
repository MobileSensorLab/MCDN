"""Tests for train.py config override and forwarding behavior."""

import argparse

import pytest

from pathlib import Path

from src.config.settings import AppConfig
from train import _apply_overrides
from train import _aggregate_ablation_metrics
from train import _compute_seed_macro_qwk
from train import _seed_run_is_complete
from train import main


def _build_minimal_config() -> AppConfig:
    """Create a complete config object with runtime defaults for entrypoint tests."""

    return AppConfig.model_validate({
        "data": {
            "dir": "data/",
            "chip_size": 512,
            "holdout_event": None,
            "sensor_profile": "uas_5cm"
        },
        "training": {
            "epochs": 50,
            "batch_size": 64,
            "accum_steps": 4,
            "lr": 1e-4
        },
        "model": {
            "name": "convnext_tiny",
            "pretrained": True
        },
        "ablation": {
            "mask_enabled": True,
            "typology_enabled": True
        },
        "runtime": {
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 3,
            "drop_last": True,
            "val_batch_size_factor": 0.5,
            "device": "auto",
            "seed": None,
            "early_stopping_patience": 10,
            "checkpoint_root": "outputs/checkpoints"
        }
    })


def test_apply_overrides_updates_supported_cli_fields() -> None:
    """CLI-supported data/training overrides are applied and validated."""

    config = _build_minimal_config()
    args = argparse.Namespace(
        config=Path("config/config.yaml"),
        epochs=12,
        batch_size=16,
        accum_steps=2,
        lr=0.0003,
        chip_size=256,
        holdout_event="Event X",
        data_dir=Path("alt-data")
    )

    updated = _apply_overrides(config=config, args=args)
    assert updated.training.epochs == 12
    assert updated.training.batch_size == 16
    assert updated.training.accum_steps == 2
    assert updated.training.lr == 0.0003
    assert updated.data.chip_size == 256
    assert updated.data.holdout_event == "Event X"
    assert updated.data.dir == Path("alt-data")


def test_main_forwards_full_config_to_training_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() forwards data, training, model, and runtime config to pipeline."""

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(
                config=Path("config/config.yaml"),
                epochs=None,
                batch_size=None,
                accum_steps=None,
                lr=None,
                chip_size=None,
                holdout_event=None,
                data_dir=None,
                seeds=None,
                fixed_seeds=False,
                ablation_preset=None
            )

    captured: dict[str, object] = {}

    def _capture_pipeline_kwargs(**kwargs: object) -> None:
        captured.update(kwargs)

    config = AppConfig.model_validate({
        "data": {
            "dir": "dataset-root",
            "chip_size": 320,
            "holdout_event": "Event Z",
            "sensor_profile": "manned_15cm"
        },
        "training": {
            "epochs": 9,
            "batch_size": 10,
            "accum_steps": 3,
            "lr": 0.0002
        },
        "model": {
            "name": "convnext_nano",
            "pretrained": False
        },
        "ablation": {
            "mask_enabled": False,
            "typology_enabled": False
        },
        "runtime": {
            "num_workers": 5,
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
    })

    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", lambda _path: config)
    monkeypatch.setattr("train.run_training_pipeline", _capture_pipeline_kwargs)

    main()

    forwarded_config = captured["config"]
    assert isinstance(forwarded_config, AppConfig)
    assert str(forwarded_config.data.dir) == "dataset-root"
    assert forwarded_config.training.epochs == 9
    assert forwarded_config.training.batch_size == 10
    assert forwarded_config.training.accum_steps == 3
    assert forwarded_config.training.lr == 0.0002
    assert forwarded_config.data.chip_size == 320
    assert forwarded_config.data.holdout_event == "Event Z"
    assert forwarded_config.data.sensor_profile == "manned_15cm"
    assert forwarded_config.runtime.num_workers == 5
    assert forwarded_config.runtime.pin_memory is False
    assert forwarded_config.runtime.persistent_workers is False
    assert forwarded_config.runtime.prefetch_factor == 2
    assert forwarded_config.runtime.drop_last is False
    assert forwarded_config.runtime.val_batch_size_factor == 0.75
    assert forwarded_config.runtime.device == "cpu"
    assert forwarded_config.runtime.early_stopping_patience == 6
    assert forwarded_config.model.name == "convnext_nano"
    assert forwarded_config.model.pretrained is False
    assert forwarded_config.ablation.mask_enabled is False
    assert forwarded_config.ablation.typology_enabled is False
    assert captured["seed"] == 123
    assert Path(str(captured["checkpoint_root"])) == Path("outputs/checkpoints")


def test_main_executes_multi_seed_runs_from_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() executes one training pipeline call per provided CLI seed."""

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(
                config=Path("config/config.yaml"),
                epochs=None,
                batch_size=None,
                accum_steps=None,
                lr=None,
                chip_size=None,
                holdout_event=None,
                data_dir=None,
                seeds=[11, 22, 11],
                fixed_seeds=False,
                ablation_preset=None
            )

    config = _build_minimal_config()
    calls: list[dict[str, object]] = []

    def _capture_pipeline_kwargs(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", lambda _path: config)
    monkeypatch.setattr("train.run_training_pipeline", _capture_pipeline_kwargs)

    main()

    assert len(calls) == 2
    assert calls[0]["seed"] == 11
    assert calls[1]["seed"] == 22
    assert str(calls[0]["checkpoint_root"]).replace("\\", "/").endswith("outputs/checkpoints/seed_11")
    assert str(calls[1]["checkpoint_root"]).replace("\\", "/").endswith("outputs/checkpoints/seed_22")


def test_main_executes_fixed_seed_protocol_for_ablation_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ablation preset mode expands fixed seed protocol and routes output roots."""

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(
                config=Path("config/config.yaml"),
                epochs=None,
                batch_size=None,
                accum_steps=None,
                lr=None,
                chip_size=None,
                holdout_event=None,
                data_dir=None,
                seeds=None,
                fixed_seeds=True,
                ablation_preset="baseline"
            )

    config = _build_minimal_config()
    loaded_paths: list[Path] = []
    calls: list[dict[str, object]] = []

    def _load_config(path: Path) -> AppConfig:
        loaded_paths.append(path)
        return config

    def _capture_pipeline_kwargs(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", _load_config)
    monkeypatch.setattr("train.run_training_pipeline", _capture_pipeline_kwargs)
    monkeypatch.setattr("train._aggregate_ablation_metrics", lambda variant_root, expected_seeds: {"variant_root": variant_root, "expected_seeds": expected_seeds})

    main()

    assert loaded_paths[0].as_posix().endswith("config/presets/ablation_baseline.yaml")
    assert [call["seed"] for call in calls] == [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    expected_base = "outputs/ablation/baseline/Spatial_Block_East"
    for call, seed in zip(calls, [0, 11, 22, 33, 44, 55, 66, 77, 88, 99], strict=True):
        assert str(call["checkpoint_root"]).replace("\\", "/").endswith(f"{expected_base}/seed_{seed:02d}")


def _write_split_artifacts(variant_root: Path, split: str, seed: int, qwk: float) -> Path:
    """Scaffold ``<variant>/<split>/seed_<n>/`` with the three files ``_compute_seed_macro_qwk`` expects."""

    seed_dir = variant_root / split / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "metrics.json").write_text(f'{{"best_val_qwk": {qwk}}}', encoding="utf-8")
    (seed_dir / "config_resolved.yaml").write_text(f"runtime:\n  seed: {seed}\n", encoding="utf-8")
    (seed_dir / "best_model.pt").write_text("placeholder", encoding="utf-8")
    return seed_dir


def test_compute_seed_macro_qwk_averages_split_metrics(tmp_path: Path) -> None:
    """Seed macro-QWK is mean of ``best_val_qwk`` across every split under one variant."""

    variant_root = tmp_path / "baseline"
    _write_split_artifacts(variant_root=variant_root, split="Hurricane_Michael", seed=11, qwk=0.80)
    _write_split_artifacts(variant_root=variant_root, split="Spatial_Block_East", seed=11, qwk=0.70)

    macro_qwk, split_count = _compute_seed_macro_qwk(variant_root=variant_root, seed=11)
    assert macro_qwk == pytest.approx(0.75)
    assert split_count == 2


def test_aggregate_ablation_metrics_writes_summary_payload(tmp_path: Path) -> None:
    """Ablation aggregator writes seed metrics and mean/std summary under the new layout."""

    variant_root = tmp_path / "outputs" / "ablation" / "baseline"
    expected_seeds = [11, 22, 33]
    seed_values = {
        11: {"Hurricane_Michael": 0.80, "Spatial_Block_East": 0.70},
        22: {"Hurricane_Michael": 0.60, "Spatial_Block_East": 0.70},
        33: {"Hurricane_Michael": 0.90, "Spatial_Block_East": 0.80},
    }
    for seed, split_scores in seed_values.items():
        for split, score in split_scores.items():
            _write_split_artifacts(variant_root=variant_root, split=split, seed=seed, qwk=score)

    payload = _aggregate_ablation_metrics(variant_root=variant_root, expected_seeds=expected_seeds)
    summary_path = variant_root / "summary_metrics.json"
    assert summary_path.exists()
    assert payload["variant"] == "baseline"
    assert payload["seed_protocol"] == expected_seeds
    assert payload["macro_qwk_mean"] == pytest.approx(0.75)
    assert payload["macro_qwk_std"] == pytest.approx(0.1)


def test_aggregate_ablation_metrics_requires_expected_seeds(tmp_path: Path) -> None:
    """Aggregator fails fast when an expected seed has no split entries."""

    variant_root = tmp_path / "outputs" / "ablation" / "baseline"
    _write_split_artifacts(variant_root=variant_root, split="Spatial_Block_East", seed=11, qwk=0.80)

    with pytest.raises(ValueError, match="Missing expected seed outputs"):
        _aggregate_ablation_metrics(variant_root=variant_root, expected_seeds=[11, 22, 33])


def test_compute_seed_macro_qwk_requires_all_split_artifacts(tmp_path: Path) -> None:
    """Seed aggregation fails when a present split is missing required files."""

    variant_root = tmp_path / "baseline"
    seed_dir = variant_root / "Spatial_Block_East" / "seed_11"
    seed_dir.mkdir(parents=True)
    (seed_dir / "metrics.json").write_text('{"best_val_qwk": 0.80}', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required fold artifacts"):
        _compute_seed_macro_qwk(variant_root=variant_root, seed=11)


def _write_seed_run_artifacts(checkpoint_root: Path) -> None:
    """Scaffold the per-seed artifact triplet that marks a run complete."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "metrics.json").write_text('{"best_val_qwk": 0.80}', encoding="utf-8")
    (checkpoint_root / "config_resolved.yaml").write_text("runtime:\n  seed: 11\n", encoding="utf-8")
    (checkpoint_root / "best_model.pt").write_text("placeholder", encoding="utf-8")


def _multi_seed_namespace(*, skip_if_complete: bool) -> argparse.Namespace:
    """CLI namespace for a two-seed run with the skip-if-complete guard toggled."""

    return argparse.Namespace(
        config=Path("config/config.yaml"),
        epochs=None,
        batch_size=None,
        accum_steps=None,
        lr=None,
        chip_size=None,
        holdout_event=None,
        data_dir=None,
        seeds=[11, 22],
        fixed_seeds=False,
        ablation_preset=None,
        skip_if_complete=skip_if_complete
    )


def test_seed_run_is_complete_requires_full_artifact_triplet(tmp_path: Path) -> None:
    """Completion requires metrics.json, config_resolved.yaml, and best_model.pt together."""

    assert _seed_run_is_complete(tmp_path) is False

    _write_seed_run_artifacts(tmp_path)
    assert _seed_run_is_complete(tmp_path) is True

    (tmp_path / "best_model.pt").unlink()
    assert _seed_run_is_complete(tmp_path) is False


def test_main_skip_if_complete_backfills_only_missing_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-if-complete skips seeds whose artifact triplet exists and runs the rest."""

    config = _build_minimal_config()
    payload = config.model_dump(mode="python")
    payload["runtime"]["checkpoint_root"] = str(tmp_path)
    config = AppConfig.model_validate(payload)

    _write_seed_run_artifacts(tmp_path / "seed_11")

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return _multi_seed_namespace(skip_if_complete=True)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", lambda _path: config)
    monkeypatch.setattr("train.run_training_pipeline", lambda **kwargs: calls.append(kwargs))

    main()

    assert [call["seed"] for call in calls] == [22]
    assert str(calls[0]["checkpoint_root"]).replace("\\", "/").endswith("seed_22")


def test_main_skip_if_complete_reruns_partial_seed_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed directory missing any triplet member (e.g. a preempted run) is not skipped."""

    config = _build_minimal_config()
    payload = config.model_dump(mode="python")
    payload["runtime"]["checkpoint_root"] = str(tmp_path)
    config = AppConfig.model_validate(payload)

    _write_seed_run_artifacts(tmp_path / "seed_11")
    (tmp_path / "seed_11" / "best_model.pt").unlink()

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return _multi_seed_namespace(skip_if_complete=True)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", lambda _path: config)
    monkeypatch.setattr("train.run_training_pipeline", lambda **kwargs: calls.append(kwargs))

    main()

    assert [call["seed"] for call in calls] == [11, 22]


def test_main_without_skip_flag_reruns_completed_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default behavior is unchanged: completed seed directories are retrained when the flag is off."""

    config = _build_minimal_config()
    payload = config.model_dump(mode="python")
    payload["runtime"]["checkpoint_root"] = str(tmp_path)
    config = AppConfig.model_validate(payload)

    _write_seed_run_artifacts(tmp_path / "seed_11")

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return _multi_seed_namespace(skip_if_complete=False)

    calls: list[dict[str, object]] = []
    monkeypatch.setattr("train._build_parser", lambda: _Parser())
    monkeypatch.setattr("train.load_config", lambda _path: config)
    monkeypatch.setattr("train.run_training_pipeline", lambda **kwargs: calls.append(kwargs))

    main()

    assert [call["seed"] for call in calls] == [11, 22]


def test_main_rejects_fixed_seed_without_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI rejects fixed seed protocol unless a preset is selected."""

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(
                config=Path("config/config.yaml"),
                epochs=None,
                batch_size=None,
                accum_steps=None,
                lr=None,
                chip_size=None,
                holdout_event=None,
                data_dir=None,
                seeds=None,
                fixed_seeds=True,
                ablation_preset=None
            )

        @staticmethod
        def error(message: str) -> None:
            raise ValueError(message)

    monkeypatch.setattr("train._build_parser", lambda: _Parser())

    with pytest.raises(ValueError, match="requires --ablation-preset"):
        main()


def test_main_rejects_conflicting_seed_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI rejects simultaneous --seeds and --fixed-seeds."""

    class _Parser:
        @staticmethod
        def parse_args() -> argparse.Namespace:
            return argparse.Namespace(
                config=Path("config/config.yaml"),
                epochs=None,
                batch_size=None,
                accum_steps=None,
                lr=None,
                chip_size=None,
                holdout_event=None,
                data_dir=None,
                seeds=[11],
                fixed_seeds=True,
                ablation_preset="baseline"
            )

        @staticmethod
        def error(message: str) -> None:
            raise ValueError(message)

    monkeypatch.setattr("train._build_parser", lambda: _Parser())

    with pytest.raises(ValueError, match="Use either --seeds or --fixed-seeds"):
        main()
