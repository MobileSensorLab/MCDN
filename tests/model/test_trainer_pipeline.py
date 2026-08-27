"""Pipeline orchestration tests for run_training_pipeline.

All tests exercise the AppConfig-first signature of ``run_training_pipeline``. Heavy
dependencies are patched at the module boundary (``src.data.sampling.*`` for manifest
scanning / integrity / source metadata, and ``src.model.trainer.*`` for rebound helpers
such as ``generate_loeo_splits`` and the dataset / dataloader / trainer constructors).
"""


import json
import pandas as pd
import pytest
import yaml

from pathlib import Path
from typing import Literal

from src.config.settings import (
    AblationConfig,
    AppConfig,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from src.data.sampling import select_fold
from src.model.trainer import run_training_pipeline


def _make_config(
    *,
    data_dir: str = "dataset-root",
    holdout_event: str | list[str] | None = None,
    sensor_profile: str = "uas_5cm",
    chip_size: int = 512,
    batch_size: int = 64,
    epochs: int = 50,
    accum_steps: int = 4,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 3,
    drop_last: bool = True,
    val_batch_size_factor: float = 0.5,
    device: Literal["auto", "cpu", "cuda"] = "cpu",
    seed: int | None = None,
    early_stopping_patience: int = 10,
    checkpoint_root: str | Path = "outputs/checkpoints",
    model_name: str = "convnext_tiny",
    model_pretrained: bool = True,
    drop_path_rate: float = 0.1,
    mask_enabled: bool = True,
    typology_enabled: bool = True,
    mask_weighted_pooling_enabled: bool = False,
) -> AppConfig:
    """Build an AppConfig with test-safe defaults.

    Defaults run on CPU with zero workers and no persistence so callers don't need to
    repeat the same kwargs; override only the fields a test actually cares about.
    """

    return AppConfig(
        data=DataConfig(
            dir=Path(data_dir),
            chip_size=chip_size,
            holdout_event=holdout_event,
            sensor_profile=sensor_profile,
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            accum_steps=accum_steps,
        ),
        model=ModelConfig(
            name=model_name,
            pretrained=model_pretrained,
            drop_path_rate=drop_path_rate,
        ),
        ablation=AblationConfig(
            mask_enabled=mask_enabled,
            typology_enabled=typology_enabled,
            mask_weighted_pooling_enabled=mask_weighted_pooling_enabled,
        ),
        runtime=RuntimeConfig(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            val_batch_size_factor=val_batch_size_factor,
            device=device,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            checkpoint_root=Path(checkpoint_root),
        ),
    )


def _patch_manifest_scan(monkeypatch: pytest.MonkeyPatch, manifest: pd.DataFrame, *, corrupt: int = 0) -> None:
    """Patch ``scan_dataset`` and ``validate_integrity`` at the sampling module boundary."""

    monkeypatch.setattr("src.data.sampling.scan_dataset", lambda _data_dir: manifest)
    monkeypatch.setattr("src.data.sampling.validate_integrity", lambda _inventory_df: {"corrupt": corrupt})


def _patch_heavy_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy runtime collaborators so the orchestrator runs without real deps."""

    class _Dataset:
        def __init__(self) -> None:
            self.instances = [{"damage_label": "no damage"}]

    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_args, **_kwargs: _Dataset())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", lambda **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())
    monkeypatch.setattr(
        "src.model.trainer.UnitemporalTrainer",
        lambda **_kwargs: type("T", (), {"fit": staticmethod(lambda **_k: {})})(),
    )


def _patch_pipeline_components_for_split_capture(
    monkeypatch: pytest.MonkeyPatch,
    manifest: pd.DataFrame,
    captured: dict[str, object],
) -> None:
    """Patch dependencies and capture the train/val frames passed into the dataset ctor."""

    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    def _dataset_ctor(df: pd.DataFrame, **_kwargs: object) -> object:
        split_key = "train_df" if "train_df" not in captured else "val_df"
        captured[split_key] = df.copy()

        class _Dataset:
            def __init__(self) -> None:
                self.instances = [{"damage_label": "no damage"}]

        return _Dataset()

    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", _dataset_ctor)
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", lambda **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())

    def _trainer_ctor(**kwargs: object) -> object:
        captured["checkpoint_dir"] = kwargs["checkpoint_dir"]

        class _TrainerInstance:
            @staticmethod
            def fit(**_k: object) -> dict[str, object]:
                return {}

        return _TrainerInstance()

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", _trainer_ctor)


def _build_event_manifest() -> pd.DataFrame:
    """Construct a small multi-event manifest."""

    return pd.DataFrame({
        "image_path": ["a_1.tif", "a_2.tif", "b_1.tif", "b_2.tif", "c_1.tif", "c_2.tif"],
        "image_name": ["a_1.tif", "a_2.tif", "b_1.tif", "b_2.tif", "c_1.tif", "c_2.tif"],
        "valid": [True, True, True, True, True, True],
        "event": ["Event A", "Event A", "Event B", "Event B", "Event C", "Event C"],
    })


def test_run_training_pipeline_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Orchestrator initializes components and executes fit."""

    manifest = pd.DataFrame({
        "image_path": ["a.tif", "b.tif"],
        "image_name": ["EventA_tile_01.tif", "EventB_tile_01.tif"],
        "valid": [True, True],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Hurricane Dummy", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )

    classifier_kwargs: dict[str, object] = {}

    def _classifier_ctor(**kwargs: object) -> object:
        classifier_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", _classifier_ctor)

    trainer_calls: dict[str, object] = {}

    def _trainer_ctor(**kwargs: object) -> object:
        trainer_calls["kwargs"] = kwargs

        class _TrainerInstance:
            @staticmethod
            def fit(**fit_kwargs: object) -> dict[str, object]:
                trainer_calls["fit"] = fit_kwargs
                return {}

        return _TrainerInstance()

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", _trainer_ctor)
    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_a, **_k: type("D", (), {"instances": [{"damage_label": "no damage"}]})())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())

    config = _make_config(data_dir="dummy/path", epochs=3, batch_size=4, accum_steps=2, chip_size=128, holdout_event="Hurricane Dummy")
    run_training_pipeline(config=config)

    assert classifier_kwargs == {
        "backbone_name": "convnext_tiny",
        "pretrained": True,
        "mask_enabled": True,
        "typology_enabled": True,
        "mask_weighted_pooling_enabled": False,
        "drop_path_rate": 0.1,
    }
    assert trainer_calls["fit"] == {
        "epochs": 3,
        "early_stopping_patience": 10,
        "lr_drop_warmstart_from_best": False,
        "reset_optimizer_on_lr_drop": True,
    }


def test_run_training_pipeline_specific_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Orchestrator can target an explicit holdout event."""

    manifest = pd.DataFrame({
        "image_path": ["a.tif", "b.tif", "c.tif"],
        "image_name": ["EventA_tile_01.tif", "TargetEvent_tile_01.tif", "EventC_tile_01.tif"],
        "valid": [True, True, True],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([
            ("Event A", pd.DataFrame([1]), pd.DataFrame([2])),
            ("Target Event", pd.DataFrame([3]), pd.DataFrame([4])),
            ("Event C", pd.DataFrame([5]), pd.DataFrame([6])),
        ]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    captured: dict[str, object] = {}

    def _trainer_ctor(**kwargs: object) -> object:
        captured["kwargs"] = kwargs
        return type("T", (), {"fit": staticmethod(lambda **_k: {})})()

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", _trainer_ctor)

    config = _make_config(holdout_event="Target Event", checkpoint_root=str(tmp_path))
    run_training_pipeline(config=config)

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert Path(str(kwargs["checkpoint_dir"])) == tmp_path

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["holdout_event"] == "Target Event"
    assert metrics_payload["holdout_selection"] == "explicit"


def test_select_fold_defaults_to_spatial_block_event() -> None:
    """Default fold selection targets spatial block holdout when present."""

    splits = [
        ("Hurricane Ian", pd.DataFrame([1]), pd.DataFrame([2])),
        ("Test_Event_Spatial_Block_East", pd.DataFrame([3]), pd.DataFrame([4])),
        ("Hurricane Michael", pd.DataFrame([5]), pd.DataFrame([6])),
    ]
    holdout, _train_df, _val_df, holdout_selection = select_fold(splits=splits, holdout_event=None)
    assert holdout == "Test_Event_Spatial_Block_East"
    assert holdout_selection == "default_spatial"


def test_select_fold_explicit_event_overrides_default_spatial() -> None:
    """Explicit holdout event remains authoritative over default spatial policy."""

    splits = [
        ("Hurricane Ian", pd.DataFrame([1]), pd.DataFrame([2])),
        ("Test_Event_Spatial_Block_East", pd.DataFrame([3]), pd.DataFrame([4])),
    ]
    holdout, _train_df, _val_df, holdout_selection = select_fold(splits=splits, holdout_event="Hurricane Ian")
    assert holdout == "Hurricane Ian"
    assert holdout_selection == "explicit"


def test_select_fold_raises_when_default_spatial_event_missing() -> None:
    """Missing spatial default triggers a hard error to preserve protocol consistency."""

    splits = [("Hurricane Ian", pd.DataFrame([1]), pd.DataFrame([2]))]
    with pytest.raises(ValueError, match="No default spatial holdout event found"):
        select_fold(splits=splits, holdout_event=None)


def test_run_training_pipeline_defaults_to_spatial_block_east_segmentation(monkeypatch: pytest.MonkeyPatch) -> None:
    """When holdout is omitted, pipeline derives spatial-block events from image names."""

    manifest = pd.DataFrame({
        "image_path": ["e1.tif", "e2.tif", "w1.tif", "w2.tif"],
        "image_name": [
            "tile_Test_Event_Spatial_Block_East_01.tif",
            "tile_Test_Event_Spatial_Block_East_02.tif",
            "tile_Test_Event_Spatial_Block_West_01.tif",
            "tile_Test_Event_Spatial_Block_West_02.tif",
        ],
        "valid": [True, True, True, True],
        "event": ["Hurricane Ian", "Hurricane Ian", "Hurricane Ian", "Hurricane Ian"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    def _capture_splits(valid_manifest: pd.DataFrame) -> object:
        assert set(valid_manifest["event"]) == {"spatial_block_east", "spatial_block_west"}
        east_df = valid_manifest[valid_manifest["event"] == "spatial_block_east"].copy()
        west_df = valid_manifest[valid_manifest["event"] == "spatial_block_west"].copy()
        return iter([
            ("spatial_block_west", east_df, west_df),
            ("spatial_block_east", west_df, east_df),
        ])

    monkeypatch.setattr("src.model.trainer.generate_loeo_splits", _capture_splits)
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    captured: dict[str, object] = {}

    def _trainer_ctor(**kwargs: object) -> object:
        captured["checkpoint_dir"] = kwargs["checkpoint_dir"]
        return type("T", (), {"fit": staticmethod(lambda **_k: {"best_val_qwk": 0.1})})()

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", _trainer_ctor)

    config = _make_config(holdout_event=None)
    run_training_pipeline(config=config)
    assert str(captured["checkpoint_dir"]).replace("\\", "/").endswith("outputs/checkpoints")


def test_run_training_pipeline_spatial_default_falls_back_without_name_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default spatial mode falls back to synthetic event when names lack spatial tokens."""

    manifest = pd.DataFrame({
        "image_path": ["a.tif", "b.tif"],
        "image_name": ["0827-A-01.geo.tif", "0827-B-02.geo.tif"],
        "valid": [True, True],
        "event": ["Hurricane Ian", "Hurricane Ian"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    def _capture_splits(valid_manifest: pd.DataFrame) -> object:
        assert set(valid_manifest["event"]) == {"Spatial_Block"}
        return iter([("Spatial_Block_East", pd.DataFrame([1]), pd.DataFrame([2]))])

    monkeypatch.setattr("src.model.trainer.generate_loeo_splits", _capture_splits)
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    config = _make_config(holdout_event=None)
    run_training_pipeline(config=config)


def test_run_training_pipeline_model_and_ablation_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_training_pipeline forwards model and ablation fields to constructor."""

    manifest = pd.DataFrame({
        "image_path": ["a.tif", "b.tif"],
        "image_name": ["EventA_tile_01.tif", "EventB_tile_01.tif"],
        "valid": [True, True],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Hurricane Dummy", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )

    classifier_kwargs: dict[str, object] = {}

    def _classifier_ctor(**kwargs: object) -> object:
        classifier_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", _classifier_ctor)
    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_a, **_k: type("D", (), {"instances": [{"damage_label": "no damage"}]})())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())
    monkeypatch.setattr(
        "src.model.trainer.UnitemporalTrainer",
        lambda **_kwargs: type("T", (), {"fit": staticmethod(lambda **_k: {})})(),
    )

    config = _make_config(
        holdout_event="Hurricane Dummy",
        model_name="convnext_nano",
        model_pretrained=False,
        mask_enabled=False,
        typology_enabled=False,
    )
    run_training_pipeline(config=config)
    assert classifier_kwargs == {
        "backbone_name": "convnext_nano",
        "pretrained": False,
        "mask_enabled": False,
        "typology_enabled": False,
        "mask_weighted_pooling_enabled": False,
        "drop_path_rate": 0.1,
    }


def test_run_training_pipeline_invalid_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline halts if requested holdout event is missing."""

    manifest = pd.DataFrame({
        "image_path": ["a.tif", "b.tif"],
        "image_name": ["EventA_tile_01.tif", "EventB_tile_01.tif"],
        "valid": [True, True],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event A", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )

    config = _make_config(holdout_event="Missing Event")
    with pytest.raises(ValueError, match="not found in manifest"):
        run_training_pipeline(config=config)


def test_run_training_pipeline_warns_for_corrupt_and_skips_event_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline prints corrupt warnings and uses provided event labels."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest, corrupt=2)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *args, **_kwargs: printed.append(" ".join(str(a) for a in args)))
    config = _make_config(holdout_event="Event B")
    run_training_pipeline(config=config)
    assert any("WARNING: Found 2 corrupt images during scan." in line for line in printed)


def test_run_training_pipeline_loeo_has_zero_event_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOEO split has no event overlap between train and validation."""

    captured: dict[str, object] = {}
    manifest = _build_event_manifest()
    _patch_pipeline_components_for_split_capture(monkeypatch=monkeypatch, manifest=manifest, captured=captured)

    config = _make_config(holdout_event="Event B")
    run_training_pipeline(config=config)
    train_df = captured["train_df"]
    val_df = captured["val_df"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert set(train_df["event"]).isdisjoint(set(val_df["event"]))


def test_run_training_pipeline_holdout_event_absent_from_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requested holdout event is fully excluded from train rows."""

    captured: dict[str, object] = {}
    manifest = _build_event_manifest()
    _patch_pipeline_components_for_split_capture(monkeypatch=monkeypatch, manifest=manifest, captured=captured)

    config = _make_config(holdout_event="Event C")
    run_training_pipeline(config=config)
    train_df = captured["train_df"]
    val_df = captured["val_df"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert set(val_df["event"]) == {"Event C"}
    assert "Event C" not in set(train_df["event"])


def test_run_training_pipeline_invalid_holdout_with_multi_event_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing holdout remains an explicit error in multi-event manifests."""

    manifest = _build_event_manifest()
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    config = _make_config(holdout_event="Missing Event")
    with pytest.raises(ValueError, match=r"Event 'Missing Event' not found in manifest\."):
        run_training_pipeline(config=config)


def test_run_training_pipeline_derived_event_column_preserves_fold_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived event labels from image names still produce disjoint folds."""

    captured: dict[str, object] = {}
    manifest = pd.DataFrame({
        "image_path": ["ea_1.tif", "ea_2.tif", "eb_1.tif", "eb_2.tif", "ec_1.tif", "ec_2.tif"],
        "image_name": [
            "EventA_tile_01.tif",
            "EventA_tile_02.tif",
            "EventB_tile_01.tif",
            "EventB_tile_02.tif",
            "EventC_tile_01.tif",
            "EventC_tile_02.tif",
        ],
        "valid": [True, True, True, True, True, True],
    })
    _patch_pipeline_components_for_split_capture(monkeypatch=monkeypatch, manifest=manifest, captured=captured)

    config = _make_config(holdout_event="EventB")
    run_training_pipeline(config=config)
    train_df = captured["train_df"]
    val_df = captured["val_df"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert set(val_df["event"]) == {"EventB"}
    assert set(train_df["event"]).isdisjoint(set(val_df["event"]))


def test_run_training_pipeline_rejects_inconsistent_event_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline fails fast when one image_name maps to multiple events."""

    manifest = pd.DataFrame({
        "image_path": ["dup_1.tif", "dup_1.tif", "stable_1.tif"],
        "image_name": ["dup_1.tif", "dup_1.tif", "stable_1.tif"],
        "valid": [True, True, True],
        "event": ["Event A", "Event B", "Event C"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    config = _make_config(holdout_event=None)
    with pytest.raises(ValueError, match="Inconsistent event mapping detected"):
        run_training_pipeline(config=config)


def test_run_training_pipeline_applies_runtime_config_to_dataloaders_and_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Externalized runtime settings are consumed in dataloaders and fit control."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_a, **_k: type("D", (), {"instances": [{"damage_label": "no damage"}]})())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())

    dataloader_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *args, **kwargs: dataloader_calls.append((args, kwargs)) or object())
    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", lambda **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())
    determinism_calls: list[int | None] = []
    monkeypatch.setattr("src.model.trainer.set_seeds", lambda seed: determinism_calls.append(seed))

    captured: dict[str, object] = {}

    def _trainer_ctor(**kwargs: object) -> object:
        captured["trainer_kwargs"] = kwargs

        class _TrainerInstance:
            @staticmethod
            def fit(**fit_kwargs: object) -> dict[str, object]:
                captured["fit_args"] = fit_kwargs
                return {}

        return _TrainerInstance()

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", _trainer_ctor)

    config = _make_config(
        holdout_event="Event B",
        batch_size=12,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,
        val_batch_size_factor=0.75,
        device="cpu",
        seed=123,
        early_stopping_patience=6,
        checkpoint_root="outputs/checkpoints",
    )
    run_training_pipeline(config=config)

    assert len(dataloader_calls) == 2
    train_kwargs = dataloader_calls[0][1]
    val_kwargs = dataloader_calls[1][1]
    assert train_kwargs["batch_size"] == 12
    assert train_kwargs["num_workers"] == 4
    assert train_kwargs["pin_memory"] is False
    assert train_kwargs["persistent_workers"] is False
    assert train_kwargs["prefetch_factor"] == 2
    assert train_kwargs["drop_last"] is False
    assert val_kwargs["batch_size"] == 9
    assert val_kwargs["num_workers"] == 4
    assert val_kwargs["pin_memory"] is False
    assert val_kwargs["persistent_workers"] is False
    assert determinism_calls == [123]

    trainer_kwargs = captured["trainer_kwargs"]
    assert isinstance(trainer_kwargs, dict)
    assert trainer_kwargs["device"] == "cpu"
    assert trainer_kwargs["ema_enabled"] is False
    assert trainer_kwargs["ema_decay"] == pytest.approx(0.995)
    assert str(trainer_kwargs["checkpoint_dir"]).replace("\\", "/").endswith("outputs/checkpoints")
    fit_args = captured["fit_args"]
    assert isinstance(fit_args, dict)
    assert fit_args["epochs"] == 50
    assert fit_args["early_stopping_patience"] == 6
    assert fit_args["lr_drop_warmstart_from_best"] is False


def test_run_training_pipeline_ignores_prefetch_when_num_workers_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """prefetch_factor is omitted when num_workers=0."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_a, **_k: type("D", (), {"instances": [{"damage_label": "no damage"}]})())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())

    dataloader_calls: list[dict[str, object]] = []
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **kwargs: dataloader_calls.append(kwargs) or object())
    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", lambda **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())
    monkeypatch.setattr(
        "src.model.trainer.UnitemporalTrainer",
        lambda **_kwargs: type("T", (), {"fit": staticmethod(lambda **_k: {})})(),
    )

    config = _make_config(holdout_event="Event B", num_workers=0, persistent_workers=False, prefetch_factor=4)
    run_training_pipeline(config=config)
    assert len(dataloader_calls) == 2
    assert "prefetch_factor" not in dataloader_calls[0]


def test_run_training_pipeline_rejects_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit cuda device fails fast without CUDA."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)
    monkeypatch.setattr("src.model.trainer.torch.cuda.is_available", lambda: False)

    config = _make_config(holdout_event="Event B", device="cuda")
    with pytest.raises(ValueError, match="requires CUDA availability"):
        run_training_pipeline(config=config)


def test_run_training_pipeline_resolves_data_root_from_sensor_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """sensor_profile keeps scan rooted at configured base data directory."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    scan_calls: list[object] = []

    def _scan_dataset(data_dir: object) -> pd.DataFrame:
        scan_calls.append(data_dir)
        return manifest

    monkeypatch.setattr("src.data.sampling.scan_dataset", _scan_dataset)
    monkeypatch.setattr("src.data.sampling.validate_integrity", lambda _inventory_df: {"corrupt": 0})
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    config = _make_config(holdout_event="Event B", data_dir="dataset-root", sensor_profile="manned_15cm")
    run_training_pipeline(config=config)

    assert len(scan_calls) == 1
    assert str(scan_calls[0]).replace("\\", "/").endswith("dataset-root")


def test_run_training_pipeline_uses_base_root_for_default_sensor_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default uas_5cm profile resolves directly to configured base data root."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    scan_calls: list[object] = []

    def _scan_dataset(data_dir: object) -> pd.DataFrame:
        scan_calls.append(data_dir)
        return manifest

    monkeypatch.setattr("src.data.sampling.scan_dataset", _scan_dataset)
    monkeypatch.setattr("src.data.sampling.validate_integrity", lambda _inventory_df: {"corrupt": 0})
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    config = _make_config(holdout_event="Event B", data_dir="dataset-root", sensor_profile="uas_5cm")
    run_training_pipeline(config=config)

    assert len(scan_calls) == 1
    assert str(scan_calls[0]).replace("\\", "/").endswith("dataset-root")


def test_run_training_pipeline_filters_manifest_by_source_for_manned_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """manned_15cm profile keeps only crewed-source rows when source metadata exists."""

    manifest = pd.DataFrame({
        "image_path": ["u1.tif", "m1.tif", "u2.tif", "m2.tif"],
        "image_name": ["u1.tif", "m1.tif", "u2.tif", "m2.tif"],
        "valid": [True, True, True, True],
        "event": ["Event A", "Event B", "Event C", "Event D"],
        "source": ["sUAS", "Crewed Aircraft", "sUAS", "Crewed Aircraft"],
    })
    captured: dict[str, object] = {}

    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)

    def _capture_splits(valid_manifest: pd.DataFrame) -> object:
        captured["filtered_manifest"] = valid_manifest.copy()
        return iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))])

    monkeypatch.setattr("src.model.trainer.generate_loeo_splits", _capture_splits)

    def _attach_source_metadata_passthrough(manifest: pd.DataFrame, data_dir: str) -> pd.DataFrame:
        _ = data_dir
        return manifest

    monkeypatch.setattr("src.data.sampling._attach_source_metadata", _attach_source_metadata_passthrough)
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    config = _make_config(holdout_event="Event B", sensor_profile="manned_15cm")
    run_training_pipeline(config=config)
    filtered_manifest = captured["filtered_manifest"]
    assert isinstance(filtered_manifest, pd.DataFrame)
    assert set(filtered_manifest["source"]) == {"Crewed Aircraft"}


def test_sensor_profile_uses_statistics_event_mapping_for_uas_and_manned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """statistics.csv Event mapping drives fold events for both source profiles."""

    statistics_path = tmp_path / "statistics.csv"
    statistics_path.write_text(
        "\n".join([
            "Orthomosaic,Source,Event",
            "tile_uas.tif,sUAS,Hurricane Ian",
            "tile_manned.tif,Crewed Aircraft,Hurricane Harvey",
        ]),
        encoding="utf-8",
    )

    manifest = pd.DataFrame({
        "image_path": ["u1.tif", "m1.tif"],
        "image_name": ["tile_uas.tif", "tile_manned.tif"],
        "valid": [True, True],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    captured: dict[str, pd.DataFrame] = {}

    def _run_and_capture(profile: str) -> pd.DataFrame:
        def _capture_splits(valid_manifest: pd.DataFrame) -> object:
            captured[profile] = valid_manifest.copy()
            return iter([("Captured", pd.DataFrame([1]), pd.DataFrame([2]))])

        monkeypatch.setattr("src.model.trainer.generate_loeo_splits", _capture_splits)
        config = _make_config(
            holdout_event="Captured",
            data_dir=str(tmp_path),
            sensor_profile=profile,
        )
        run_training_pipeline(config=config)
        return captured[profile]

    uas_manifest = _run_and_capture("uas_5cm")
    manned_manifest = _run_and_capture("manned_15cm")
    assert set(uas_manifest["source"]) == {"sUAS"}
    assert set(manned_manifest["source"]) == {"Crewed Aircraft"}
    assert set(uas_manifest["event"]) == {"Hurricane Ian"}
    assert set(manned_manifest["event"]) == {"Hurricane Harvey"}


def test_run_training_pipeline_fails_fast_on_empty_manifest_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline raises clear error when scan result lacks required columns."""

    empty_manifest = pd.DataFrame()
    monkeypatch.setattr("src.data.sampling.scan_dataset", lambda _data_dir: empty_manifest)

    config = _make_config(holdout_event=None)
    with pytest.raises(ValueError, match="required manifest columns"):
        run_training_pipeline(config=config)


def test_run_training_pipeline_tees_stdout_when_log_to_file_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime.log_to_file routes stdout into ``<checkpoint_root>/train_log.txt``."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    _patch_heavy_runtime(monkeypatch=monkeypatch)

    config = _make_config(
        holdout_event="Event B",
        checkpoint_root=str(tmp_path),
    )
    config_dict = config.model_dump(mode="python")
    config_dict["runtime"]["log_to_file"] = True
    config = AppConfig.model_validate(config_dict)

    run_training_pipeline(config=config)

    log_path = tmp_path / "train_log.txt"
    assert log_path.exists()
    log_contents = log_path.read_text(encoding="utf-8")
    assert "# MCDN training run" in log_contents
    assert "[data]" in log_contents
    assert "[split]" in log_contents


def test_run_training_pipeline_persists_reproducibility_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline writes metrics and resolved config snapshot per fold."""

    manifest = pd.DataFrame({
        "image_path": ["one.tif", "two.tif"],
        "image_name": ["one.tif", "two.tif"],
        "valid": [True, True],
        "event": ["Event A", "Event B"],
    })
    _patch_manifest_scan(monkeypatch=monkeypatch, manifest=manifest)
    monkeypatch.setattr(
        "src.model.trainer.generate_loeo_splits",
        lambda _valid_manifest: iter([("Event B", pd.DataFrame([1]), pd.DataFrame([2]))]),
    )
    monkeypatch.setattr("src.model.trainer.CRASARUnitemporalDataset", lambda *_a, **_k: type("D", (), {"instances": [{"damage_label": "no damage"}]})())
    monkeypatch.setattr("src.model.trainer.get_train_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.get_val_transforms", lambda **_kwargs: None)
    monkeypatch.setattr("src.model.trainer.create_weighted_sampler", lambda _dataset: object())
    monkeypatch.setattr("src.model.trainer.DataLoader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.MaskCenteredDamageNet", lambda **_kwargs: object())
    monkeypatch.setattr("src.model.trainer.create_criterion", lambda **_kwargs: object())

    class _Trainer:
        @staticmethod
        def fit(**kwargs: object) -> dict[str, object]:
            _ = kwargs
            return {
                "best_val_qwk": 0.85,
                "final_val_qwk": 0.84,
                "final_val_loss": 0.40,
                "epochs_completed": 5,
                "history": [{"epoch": 1, "val_qwk": 0.84}],
                "final_classification": {
                    "accuracy": 0.5,
                    "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                },
                "best_classification": {
                    "accuracy": 0.6,
                    "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                },
            }

    monkeypatch.setattr("src.model.trainer.UnitemporalTrainer", lambda **_kwargs: _Trainer())

    config = _make_config(
        holdout_event="Event B",
        data_dir="dataset-root",
        sensor_profile="uas_5cm",
        checkpoint_root=str(tmp_path),
        seed=22,
    )
    run_training_pipeline(config=config)

    run_dir = tmp_path
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config_resolved.yaml"
    history_path = run_dir / "history.json"
    split_summary_path = run_dir / "split_summary.json"
    classification_path = run_dir / "classification_metrics.json"
    assert metrics_path.exists()
    assert config_path.exists()
    assert history_path.exists()
    assert split_summary_path.exists()
    assert classification_path.exists()

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["holdout_event"] == "Event B"
    assert metrics_payload["holdout_selection"] == "explicit"
    assert metrics_payload["seed"] == 22
    assert metrics_payload["best_val_qwk"] == 0.85
    assert metrics_payload["final_classification"]["accuracy"] == 0.5
    assert metrics_payload["best_classification"]["accuracy"] == 0.6
    assert "train_class_counts_ordinal" in metrics_payload
    assert "class_weights_ordinal" in metrics_payload

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config_payload["data"]["sensor_profile"] == "uas_5cm"
    assert config_payload["runtime"]["seed"] == 22
    assert config_payload["data"]["holdout_event"] == "Event B"
    assert config_payload["metadata"]["holdout_selection"] == "explicit"

    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert history_payload[0]["epoch"] == 1

    split_summary_payload = json.loads(split_summary_path.read_text(encoding="utf-8"))
    assert split_summary_payload["train_instances"] == 1
    assert split_summary_payload["val_instances"] == 1

    classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
    assert classification_payload["final"]["accuracy"] == 0.5
    assert classification_payload["best"]["accuracy"] == 0.6


def test_run_training_pipeline_multi_event_holdout_builds_composite_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A list-valued holdout trains on the remaining events and round-trips through the snapshot.

    Exercises the real ``build_multi_event_fold`` path (no LOEO split patching): the listed
    events form the validation pool, the composite fold name lands in ``metrics.json``, and
    ``config_resolved.yaml`` retains the original event list so postproc reconstruction can
    rebuild the same split.
    """

    manifest = _build_event_manifest()
    captured: dict[str, object] = {}
    _patch_pipeline_components_for_split_capture(monkeypatch=monkeypatch, manifest=manifest, captured=captured)

    config = _make_config(
        holdout_event=["Event B", "Event C"],
        data_dir="dataset-root",
        checkpoint_root=str(tmp_path),
        seed=22,
    )
    run_training_pipeline(config=config)

    train_df = captured["train_df"]
    val_df = captured["val_df"]
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert set(train_df["event"]) == {"Event A"}
    assert set(val_df["event"]) == {"Event B", "Event C"}

    metrics_payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["holdout_event"] == "Event B+Event C"
    assert metrics_payload["holdout_selection"] == "explicit_multi"

    config_payload = yaml.safe_load((tmp_path / "config_resolved.yaml").read_text(encoding="utf-8"))
    assert config_payload["data"]["holdout_event"] == ["Event B", "Event C"]
    assert config_payload["metadata"]["holdout_event"] == ["Event B", "Event C"]
    assert config_payload["metadata"]["holdout_selection"] == "explicit_multi"
