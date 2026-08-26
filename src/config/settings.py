"""Configuration parsing for training and ablations."""

import json
import yaml

from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from typing import Any, Literal, Self


class DataConfig(BaseModel):
    """Data-related settings.

    Args:
        dir: Dataset root directory.
        chip_size: Structure-centered chip size in pixels.
        holdout_event: Optional LOEO holdout event name.
        sensor_profile: Imagery source profile selector for resolution ablations.
        synthetic_gsd_factor: Linear ground-sample-distance degradation factor for the controlled
            resolution ablation. Chips are anti-alias downsampled by this factor and restored to
            the original pixel grid (RGB only; the vector-derived mask channel is untouched), so
            the chip inventory is identical to the source-profile arm. ``1.0`` (default) disables
            the degradation; ``3.0`` simulates 15 cm GSD from the 5 cm sUAS imagery.
    """

    dir: Path
    chip_size: int = Field(default=512, gt=0)
    holdout_event: str | None = None
    sensor_profile: Literal["uas_5cm", "manned_15cm"] = "uas_5cm"
    synthetic_gsd_factor: float = Field(default=1.0, ge=1.0)

    @field_validator("dir", mode="before")
    @classmethod
    def validate_dir_path(cls, value: object) -> Path:
        """Convert supported path-like inputs to Path."""

        return Path(value)


class TrainingConfig(BaseModel):
    """Training hyperparameters.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size per step.
        accum_steps: Gradient accumulation steps.
        lr: Optimizer learning rate.
        max_grad_norm: Optional L2 global-norm clip applied (AMP-safe) after each accumulation
            window, immediately before ``scaler.step(optimizer)``. ``None`` disables clipping.
            Common values: 1.0 (transformer / AdamW fine-tune standard), 5.0 (permissive, catches
            only pathological spikes). Relevant mainly for (a) rare-class batch gradient outliers
            and (b) the first few steps after ``optimizer.state.clear()`` on LR-drop warmstart,
            where absent momenta can produce abnormally large effective steps.
        loss: Training criterion selector. "emd" (default) is the squared Earth Mover's Distance
            with adjacency-aware ordinal label smoothing; "ce" is categorical cross-entropy with
            torch-standard uniform label smoothing (the EMD-vs-CE ablation arm). Both receive the
            same per-class weight tensor; ``label_smoothing`` semantics follow the selected loss.
    """

    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=64, gt=0)
    accum_steps: int = Field(default=4, gt=0)
    lr: float = Field(default=1e-4, gt=0.0)
    max_grad_norm: float | None = Field(default=None, gt=0.0)
    sampler_mode: Literal["weighted", "uniform"] = "weighted"
    class_weighting_enabled: bool = False
    class_weighting_strategy: Literal["inverse_freq"] = "inverse_freq"
    class_weighting_power: float = Field(default=1.0, ge=0.0)
    class_weighting_eps: float = Field(default=1e-6, gt=0.0)
    class_weighting_max_ratio: float | None = Field(default=None, gt=1.0)
    lr_plateau_patience: int = Field(default=2, ge=1)
    label_smoothing: float = Field(default=0.025, ge=0.0, lt=1.0)
    loss: Literal["emd", "ce"] = "emd"


class ModelConfig(BaseModel):
    """Model architecture settings.

    Args:
        name: Backbone model name.
        pretrained: Whether to use pretrained model weights.
        drop_path_rate: Stochastic-depth rate applied uniformly to the timm backbone via
            ``drop_path_rate`` at construction time. Exposed for ablation; prior behavior
            hardcoded this at 0.3. Values in ``[0.0, 1.0)``.
    """

    name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    drop_path_rate: float = Field(default=0.1, ge=0.0, lt=1.0)


class AblationConfig(BaseModel):
    """Feature toggles for controlled ablation runs.

    Args:
        mask_enabled: Use the 4th-channel structural mask when True.
        typology_enabled: Inject disaster typology via FiLM modulation over stage-3/4 feature maps
            when True. When False, the FiLM MLP and per-stage affine layers are not constructed and
            the forward pass skips conditioning entirely. The legacy late-concat path
            (``head_input_dim = visual_feature_dim + embedding_dim``) has been retired; the head
            always operates on ``visual_feature_dim`` regardless of typology routing.
        mask_weighted_pooling_enabled: When True, concatenate global avg/max with mask-weighted
            avg/max on backbone feature maps. Independent of ``mask_enabled``: pooling keys on the
            dataset's footprint channel directly, so enabling it without the 4th input channel
            yields the pooling-only ablation arm (footprint steers readout but never enters the
            backbone representation).
        mask_dilation_px: Morphological dilation applied to the footprint mask itself (in final-chip
            pixel units) to compensate for systematic label-source under-segmentation - polygon
            annotations typically exclude roof overhangs, eaves, and small attached structures.
            Dilation is applied post-augmentation. At 5 cm GSD a value of 10 corresponds to a 50 cm
            ground buffer. A value of ``0`` preserves legacy behavior byte-for-byte.
    """

    mask_enabled: bool = True
    typology_enabled: bool = True
    mask_weighted_pooling_enabled: bool = False
    mask_dilation_px: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_ablation_constraints(self) -> Self:
        if self.mask_dilation_px > 0 and not self.mask_enabled:
            raise ValueError("ablation.mask_dilation_px > 0 requires ablation.mask_enabled=true.")
        return self


class RuntimeConfig(BaseModel):
    """Runtime execution settings for dataloading and trainer control flow.

    Args:
        num_workers: DataLoader worker process count.
        pin_memory: Whether to pin host memory during transfer.
        persistent_workers: Keep workers alive across epochs.
        prefetch_factor: Number of batches prefetched per worker.
        drop_last: Drop trailing incomplete training batch.
        val_batch_size_factor: Validation batch size multiplier relative to training batch size.
        device: Device strategy: auto-select, force cpu, or force cuda.
        seed: Optional global random seed used for reproducibility.
        early_stopping_patience: Epochs without improvement before stopping.
        lr_drop_warmstart_from_best: On each ReduceLROnPlateau LR reduction, reload the best
            weights into the training model (and EMA shadow when active).
        reset_optimizer_on_lr_drop: On the EMA warmstart path, whether to clear AdamW's first /
            second moments and step counter after reloading best weights. False preserves Adam
            moments across the reload, eliminating the bias-correction transient that otherwise
            produces the first-post-drop-epoch QWK cliff.
        ema_enabled: Exponential moving average shadow weights for eval/checkpoint.
        ema_decay: EMA decay in ``[0, 1]`` (torch ``get_ema_multi_avg_fn``).
        checkpoint_root: Root directory for fold checkpoint outputs.
        cuda_launch_blocking: When True, train.py sets CUDA_LAUNCH_BLOCKING before importing torch.
        verbose: When True, per-epoch output renders as tabular per-batch + val/recall tables;
            when False (default), emits labeled-prose summaries with mid-epoch heartbeats.
        print_every_n_batches: Override the verbose per-batch print cadence. None (default)
            derives cadence per epoch as ``max(1, total_batches // 8)``.
        log_to_file: When True, the trainer tees stdout into ``<checkpoint_dir>/train_log.txt``.
            Auto-enabled by ``train.py`` when invoked with ``--ablation-preset``; leave False for
            bare training invocations.
    """

    num_workers: int = Field(default=8, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = Field(default=3, ge=1)
    drop_last: bool = True
    val_batch_size_factor: float = Field(default=0.5, gt=0.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int | None = Field(default=None, ge=0)
    early_stopping_patience: int = Field(default=10, ge=1)
    lr_drop_warmstart_from_best: bool = False
    reset_optimizer_on_lr_drop: bool = True
    ema_enabled: bool = False
    ema_decay: float = Field(default=0.995, ge=0.0, le=1.0)
    checkpoint_root: Path = Path("outputs/checkpoints")
    cuda_launch_blocking: bool = False
    verbose: bool = False
    print_every_n_batches: int | None = Field(default=None, ge=1)
    log_to_file: bool = False

    @model_validator(mode="after")
    def validate_worker_constraints(self) -> "RuntimeConfig":
        """Ensure worker-related settings are internally consistent."""

        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("runtime.persistent_workers=true requires runtime.num_workers > 0.")
        return self


class AppConfig(BaseModel):
    """Root configuration used by the training entrypoint.

    Args:
        data: Data settings.
        training: Training settings.
        model: Model settings.
        ablation: Ablation feature toggles.
        runtime: Runtime execution settings.
    """

    data: DataConfig
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    ablation: AblationConfig = AblationConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def _load_raw_config(path: Path) -> dict[str, Any]:
    """Load a raw config mapping from YAML or JSON."""

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use .yaml/.yml/.json.")

    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping/object.")
    return payload


def load_config(config_path: str | Path) -> AppConfig:
    """Parse and validate runtime config.

    Args:
        config_path: Path to config file.

    Returns:
        Validated AppConfig instance.
    """

    path = Path(config_path)
    raw_payload = _load_raw_config(path)
    try:
        return AppConfig.model_validate(raw_payload)
    except ValidationError:
        print("Configuration validation failed.")
        raise
