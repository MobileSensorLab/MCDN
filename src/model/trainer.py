"""Training loop engine with high-performance hardware optimizations."""

import cv2
import json
import math
import random
import time
import torch
import yaml

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime, UTC
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import TypedDict

from src.common.logging import config_hash, tee_stdout_to_file
from src.config.settings import AppConfig
from src.data.dataset import CRASARUnitemporalDataset, to_normalized_float
from src.data.sampling import (
    build_multi_event_fold,
    build_spatial_split_manifest,
    build_valid_manifest,
    create_weighted_sampler,
    generate_loeo_splits,
    select_fold,
)
from src.data.transform import get_train_transforms, get_val_transforms
from src.model.loss import create_criterion
from src.model.mcdn import MaskCenteredDamageNet

_NUMPY_RNG = np.random.default_rng()
ORDINAL_CLASS_DISPLAY_NAMES: tuple[str, ...] = ("No Damage", "Minor", "Major", "Destroyed")
ORDINAL_CLASS_SHORT_NAMES: tuple[str, ...] = ("noD", "min", "maj", "des")
ORDINAL_LABEL_KEYS: tuple[str, ...] = ("no damage", "minor damage", "major damage", "destroyed")
SELECTION_QWK_WEIGHT = 0.5
SELECTION_MACRO_F1_WEIGHT = 0.5
SELECTION_QWK_GUARDRAIL = 0.005
_BANNER_WIDTH = 80
_HEARTBEAT_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75)


class PerClassMetrics(TypedDict):
    """Per-class diagnostics emitted from a confusion matrix."""

    support: int
    predicted: int
    correct: int
    recall: float
    precision: float
    f1: float


class ClassificationMetrics(TypedDict):
    """Classification diagnostics rolled up across all ordinal classes."""

    accuracy: float
    macro_recall: float
    macro_precision: float
    macro_f1: float
    num_samples: int
    confusion_matrix: list[list[int]]
    per_class: dict[str, PerClassMetrics]


class ValidationMetrics(TypedDict):
    """Validation-pass output consumed by the trainer loop."""

    val_loss: float
    val_qwk: float
    val_selection_score: float
    classification: ClassificationMetrics


class EpochRecord(TypedDict):
    """One row of per-epoch telemetry persisted to ``history.json``."""

    epoch: int
    train_loss: float
    val_loss: float
    val_qwk: float
    val_selection_score: float
    best_val_qwk: float
    best_val_selection_score: float
    lr: float
    ema_active: bool


class FitMetrics(TypedDict):
    """Summary payload returned by :meth:`UnitemporalTrainer.fit`."""

    best_val_qwk: float
    final_val_qwk: float
    final_val_loss: float
    epochs_completed: int
    history: list[EpochRecord]
    final_classification: ClassificationMetrics
    best_classification: ClassificationMetrics


def _empty_classification_metrics() -> ClassificationMetrics:
    """Return a zero-populated ``ClassificationMetrics`` for trainers that never validated."""

    return {
        "accuracy": 0.0,
        "macro_recall": 0.0,
        "macro_precision": 0.0,
        "macro_f1": 0.0,
        "num_samples": 0,
        "confusion_matrix": [],
        "per_class": {},
    }


# Printout helpers.
# Every train-time print routes through one of these so fixed-width specs live in one place.
# Three block conventions exist across the whole run: the identity box and end banner reuse
# the same `===` rule, the training header banner reuses it, and LR-drop events wrap their
# event lines in the same rule to mark warm-up / fine-tune transitions.


def _hr() -> str:
    """Full-width '=' divider used for identity box, training banner, LR drops, and end banner."""

    return "=" * _BANNER_WIDTH


def _fmt_time_short(seconds: float | None) -> str:
    """Format a duration as ``' 1m 42s'``. ``None`` / negative renders ``'   done'`` sentinel."""

    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "   done"
    minutes = int(seconds // 60)
    remaining = round(seconds - minutes * 60)
    if remaining >= 60:
        minutes += 1
        remaining = 0
    return f"{minutes:>2d}m {remaining:02d}s"


def _fmt_time_long(seconds: float) -> str:
    """Format wall time as ``'Xh YYm ZZs'`` for the end banner."""

    hours = int(seconds // 3600)
    remainder = seconds - hours * 3600
    minutes = int(remainder // 60)
    secs = round(remainder - minutes * 60)
    if secs >= 60:
        minutes += 1
        secs = 0
    if minutes >= 60:
        hours += 1
        minutes = 0
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def _fmt_gpu_mem_gib(bytes_count: int | None) -> str:
    """Format CUDA memory bytes as ``'X.YG'`` (CPU / unavailable -> ``'   --'``)."""

    if not bytes_count:
        return "   --"
    return f"{bytes_count / (1024 ** 3):4.1f}G"


def _count_model_params(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_params, trainable_params)`` with defensive handling for mocked models."""

    params_fn = getattr(model, "parameters", None)
    if not callable(params_fn):
        return 0, 0
    total = 0
    trainable = 0
    try:
        for parameter in params_fn():
            count = int(parameter.numel()) if hasattr(parameter, "numel") else 0
            total += count
            if getattr(parameter, "requires_grad", False):
                trainable += count
    except (TypeError, RuntimeError):
        return 0, 0
    return total, trainable


def _format_param_count(n: int) -> str:
    """Format parameter count as ``'17.4M'`` / ``'320K'`` / raw integer."""

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _resolve_device_label(resolved_device: str) -> str:
    """Build the ``device`` identity-box line (``'cuda:0 (RTX 5090, 32.0 GiB)'`` on CUDA)."""

    if resolved_device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        try:
            props = torch.cuda.get_device_properties(index)
            gib = props.total_memory / (1024 ** 3)
            return f"cuda:{index} ({props.name}, {gib:.1f} GiB)"
        except (AssertionError, RuntimeError):
            return f"cuda:{index}"
    return resolved_device


def _build_arch_lines(*, config: AppConfig) -> list[str]:
    """Compose the identity-box ``arch`` lines from model / ablation / runtime config."""

    toggles: list[str] = []
    if config.ablation.mask_enabled:
        toggles.append("mask")
    if config.ablation.mask_weighted_pooling_enabled:
        toggles.append("mask_weighted_pooling")
    if config.ablation.typology_enabled:
        toggles.append("typology (FiLM)")
    if config.runtime.ema_enabled:
        toggles.append(f"EMA (decay {config.runtime.ema_decay:.3f})")

    lines = [config.model.name]
    if toggles:
        lines.append("   ".join(toggles))
    return lines


def _print_identity_box(*, config: AppConfig, config_hash_str: str, config_label: str,
                        effective_seed: int | None, fold_name: str, resolved_device: str,
                        param_total: int, param_trainable: int,
                        timestamp: str | None = None) -> None:
    """Emit the startup identity box with run metadata, device, arch toggles, and param counts."""

    stamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seed_str = str(effective_seed) if effective_seed is not None else "none"
    device_label = _resolve_device_label(resolved_device)
    arch_lines = _build_arch_lines(config=config)

    print(_hr())
    print(f"  started  {stamp}")
    print(f"  config   {config_label} (hash {config_hash_str})   seed {seed_str}   fold {fold_name}")
    print(f"  device   {device_label}")
    first_arch, *continuations = arch_lines
    print(f"  arch     {first_arch}")
    for line in continuations:
        print(f"           {line}")
    print(f"  params   total {_format_param_count(param_total)}   trainable {_format_param_count(param_trainable)}")
    print(_hr())
    print()


def _print_timeline(marker: str, body: str, *, continuation: bool = False) -> None:
    """Emit a ``[marker]  body`` timeline line (or an indented continuation)."""

    prefix = "         " if continuation else f"[{marker}]".ljust(9)
    print(f"{prefix} {body}")


def _print_training_header(epochs: int) -> None:
    """Emit the full-width banner between the timeline and the first epoch block."""

    print()
    print(_hr())
    print(f"  Training   {epochs} epochs")
    print(_hr())


def _print_epoch_banner(epoch: int, total_epochs: int) -> None:
    """Emit the ``[E NN / MM  HH:MM:SS]`` epoch banner."""

    stamp = datetime.now().strftime("%H:%M:%S")
    print()
    print(f"[E {epoch:02d} / {total_epochs:02d}  {stamp}]")


def _format_patience_str(*, counter: int, patience: int, armed: bool) -> str:
    """Render patience as ``'NN/MM p'`` (paused) or ``'NN/MM a'`` (armed)."""

    state = "a" if armed else "p"
    return f"{counter:>2d}/{patience:<2d} {state}"


def _format_sel_delta(delta: float | None) -> str:
    """Format selection-score delta as ``'+0.0130'`` / ``'-0.0030'`` / ``'   ----'`` sentinel."""

    if delta is None:
        return "   ----"
    return f"{delta:+.4f}"


def _print_heartbeat(*, done: int, total: int, loss: float, run_loss: float, eta_s: float) -> None:
    """Emit a non-verbose mid-epoch heartbeat at 25/50/75% of total batches."""

    pct = round(100 * done / total) if total > 0 else 0
    print(f"  heartbeat {done:>3d}/{total:<3d} ({pct:>2d}%)   loss {loss:.3f}  run {run_loss:.3f}   eta {_fmt_time_short(eta_s)}")


def _print_batch_table_header() -> None:
    """Emit the verbose per-batch table header + ``=`` underline."""

    print()
    print("  batch   |  loss  |  run   |    lr    | img/s |  gpu   |   grad  |   eta  ")
    print("==========+========+========+==========+=======+========+=========+=========")


def _format_grad_norm(grad_norm: float | None) -> str:
    """Format pre-clip gradient norm as fixed-width ``'  1.842'`` or ``'    --'`` sentinel."""

    if grad_norm is None or not math.isfinite(grad_norm):
        return "    --"
    return f"{grad_norm:>6.3f}"


def _print_batch_row(*, done: int, total: int, loss: float, run_loss: float, lr: float,
                     imgs_per_sec: int, gpu_mem_bytes: int, grad_norm: float | None,
                     eta_s: float) -> None:
    """Emit one verbose per-batch row with fixed widths for every scalar."""

    print(
        f"  {done:>3d}/{total:<3d} | {loss:.3f}  | {run_loss:.3f}  | {lr:.2e} | "
        f"{imgs_per_sec:>5d} | {_fmt_gpu_mem_gib(gpu_mem_bytes)}  | {_format_grad_norm(grad_norm)}  | {_fmt_time_short(eta_s)}"
    )


def _print_train_summary(*, loss: float, lr: float, imgs_per_sec: int,
                         gpu_mem_bytes: int, elapsed_s: float) -> None:
    """Emit the non-verbose labeled-prose train summary."""

    print(
        f"  train   loss {loss:.3f}  lr {lr:.2e}  imgs/s {imgs_per_sec:>4d}  "
        f"gpu {_fmt_gpu_mem_gib(gpu_mem_bytes)}  time {_fmt_time_short(elapsed_s)}"
    )


def _print_val_summary(*, loss: float, qwk: float, mf1: float, sel: float,
                       sel_delta: float | None, patience_str: str) -> None:
    """Emit the non-verbose labeled-prose val summary."""

    print(
        f"   val    loss {loss:.3f}  qwk {qwk:.4f}   mf1 {mf1:.4f}  sel {sel:.4f}  "
        f"+sel {_format_sel_delta(sel_delta)}   patience {patience_str}"
    )


def _print_recall_summary(recalls: list[float]) -> None:
    """Emit the non-verbose per-class recall summary."""

    cells = [f"{short} {pct:5.1f}%" for short, pct in zip(ORDINAL_CLASS_SHORT_NAMES, recalls, strict=False)]
    print("  recall  " + "   ".join(cells))


def _print_val_table(*, loss: float, qwk: float, mf1: float, sel: float,
                     sel_delta: float | None, patience_str: str, recalls: list[float]) -> None:
    """Emit the verbose wide val/recall table."""

    print()
    print("   val    |  loss  |   qwk   |   mf1   |   sel   |  +sel   | patience | rec noD | rec min | rec maj | rec des")
    print("==========+========+=========+=========+=========+=========+==========+=========+=========+=========+=========")
    rec_cells = "".join(f" |  {pct:5.1f}% " for pct in recalls)
    print(
        f"  TTA x8  | {loss:.3f}  | {qwk:.4f}  | {mf1:.4f}  | {sel:.4f}  | {_format_sel_delta(sel_delta)} | "
        f"{patience_str} |"
        + rec_cells.replace(" | ", " | ", 1)
    )


def _print_shadow_note(*, ema_enabled: bool, ema_decay: float) -> None:
    """Emit the shadow-weights note beneath the val summary."""

    if ema_enabled:
        print(f"  shadow: EMA (decay {ema_decay:.3f})")


def _print_new_best(checkpoint_path: Path) -> None:
    """Emit the ``*** new best -> ...`` one-liner."""

    print(f"  *** new best -> {checkpoint_path}")


def _print_event(*, epoch: int, body: str) -> None:
    """Emit a run-level ``[event]`` one-liner."""

    print(f"[event] epoch {epoch:02d}  {body}")


def _print_lr_drop_block(*, epoch: int, previous_lr: float, updated_lr: float,
                         first_reduction: bool, warmstart_note: str | None) -> None:
    """Emit the LR-drop event cluster wrapped in ``===`` dividers."""

    print()
    print(_hr())
    armed_note = "early-stopping armed" if first_reduction else "early-stopping active"
    _print_event(epoch=epoch, body=f"LR reduced {previous_lr:.2e} -> {updated_lr:.2e}  ({armed_note})")
    if warmstart_note is not None:
        _print_event(epoch=epoch, body=warmstart_note)
    print(_hr())


def _print_end_banner(*, epochs_completed: int, wall_seconds: float, images_seen: int,
                      history: list["EpochRecord"], best_classification: ClassificationMetrics,
                      final_classification: ClassificationMetrics,
                      best_selection_score: float, best_qwk: float,
                      final_selection_score: float, final_qwk: float) -> None:
    """Emit the run end banner with wall time, images seen, top-3 epoch table, and best/final."""

    print()
    print(_hr())
    images_label = f"{images_seen:,}" if images_seen > 0 else "--"
    print(
        f"  Training complete   {epochs_completed} epochs   wall {_fmt_time_long(wall_seconds)}   "
        f"images seen {images_label}"
    )
    print()

    ranked = sorted(
        history,
        key=lambda record: float(record.get("val_selection_score", 0.0)),
        reverse=True,
    )[:3]
    if ranked:
        print("  top 3   |  epoch  |   sel   |   qwk   |   mf1  ")
        print("==========+=========+=========+=========+=========")
        # Macro F1 is not carried in EpochRecord; back it out from the classification summary
        # only when the top row matches the best snapshot (keeps the table grounded in real data
        # without synthesizing per-epoch macro-F1 we do not have).
        for rank, record in enumerate(ranked, start=1):
            epoch_number = int(record.get("epoch", 0))
            sel_score = float(record.get("val_selection_score", 0.0))
            qwk_score = float(record.get("val_qwk", 0.0))
            if rank == 1 and best_classification.get("num_samples", 0) > 0:
                mf1_score = float(best_classification.get("macro_f1", 0.0))
            else:
                mf1_score = max(0.0, 2.0 * sel_score - qwk_score)
            print(
                f"   #{rank}     |   {epoch_number:>3d}   | {sel_score:.4f}  | {qwk_score:.4f}  | {mf1_score:.4f}"
            )
        print()

    best_mf1 = float(best_classification.get("macro_f1", 0.0))
    final_mf1 = float(final_classification.get("macro_f1", 0.0))
    best_epoch = 0
    if ranked:
        best_epoch = int(ranked[0].get("epoch", 0))
    final_epoch = int(history[-1].get("epoch", 0)) if history else 0
    print(
        f"  best    epoch {best_epoch:<3d}  sel {best_selection_score:.4f}   qwk {best_qwk:.4f}   mf1 {best_mf1:.4f}"
    )
    print(
        f"  final   epoch {final_epoch:<3d}  sel {final_selection_score:.4f}   qwk {final_qwk:.4f}   mf1 {final_mf1:.4f}"
    )
    print(_hr())


def batches_in_accum_window(batch_idx: int, accum_steps: int, total_batches: int) -> int:
    """Return how many ``DataLoader`` batches belong to the current accumulation window.

    Windows are aligned to ``accum_steps`` along ``batch_idx`` (0-based). The last window is
    shorter when ``total_batches`` is not a multiple of ``accum_steps``; loss should be divided
    by this value before ``backward`` so the combined gradient matches the mean over that window.

    Args:
        batch_idx: Index of the current batch within the epoch.
        accum_steps: Target micro-batches per optimizer step (>= 1).
        total_batches: ``len(train_loader)`` for the epoch.

    Returns:
        Window size in ``[1, accum_steps]``.
    """

    block_start = (batch_idx // accum_steps) * accum_steps
    return min(accum_steps, total_batches - block_start)


def set_seeds(seed: int | None) -> None:
    """Set global random seeds and deterministic backend behavior.

    Args:
        seed: Optional random seed for Python, NumPy, and PyTorch RNGs.
            Deterministic mode is enabled only when seed is provided.
    """

    cudnn_deterministic = seed is not None
    global _NUMPY_RNG
    if seed is not None:
        random.seed(seed)
        _NUMPY_RNG = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        _NUMPY_RNG = np.random.default_rng()
    torch.use_deterministic_algorithms(cudnn_deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = not cudnn_deterministic


def _seed_worker(_worker_id: int) -> None:
    """Seed numpy and python RNGs per dataloader worker process."""

    # OpenCV worker pools oversubscribe the cores when num_workers > 1; pinning to
    # one thread per worker avoids contention with the main-process GPU launches.
    cv2.setNumThreads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002 - legacy global RNG is what albumentations/other libs consume inside workers
    random.seed(worker_seed)


def _confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Accumulate a single [K, K] confusion matrix for integer class predictions."""

    flat_p = preds.view(-1).long()
    flat_t = targets.view(-1).long()
    conf = torch.zeros((num_classes, num_classes), device=preds.device, dtype=torch.float64)
    ones = torch.ones(flat_p.shape[0], device=preds.device, dtype=torch.float64)
    conf.index_put_((flat_t, flat_p), ones, accumulate=True)
    return conf


def _qwk_from_confusion(conf_mat: torch.Tensor, num_classes: int) -> float:
    """Compute Quadratic Weighted Kappa from a confusion matrix."""

    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat.sum(dim=0)
    expected = torch.outer(row_sum, col_sum) / conf_mat.sum()

    weights = torch.zeros((num_classes, num_classes), device=conf_mat.device, dtype=torch.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = (conf_mat * weights).sum()
    den = (expected * weights).sum()

    if den == 0:
        return 1.0
    return 1.0 - (num / den).item()


def compute_qwk(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> float:
    """Computes the Quadratic Weighted Kappa (QWK) directly in PyTorch.

    Args:
        preds: Integer predictions [B].
        targets: Integer ground truth labels [B].
        num_classes: Total number of ordinal categories.

    Returns:
        The QWK score as a float [-1.0, 1.0].
    """

    conf_mat = _confusion_matrix(preds=preds, targets=targets, num_classes=num_classes)
    return _qwk_from_confusion(conf_mat=conf_mat, num_classes=num_classes)


def _classification_metrics_from_confusion(conf_mat: torch.Tensor, class_names: tuple[str, ...]) -> ClassificationMetrics:
    """Build per-class classification diagnostics from a confusion matrix."""

    num_classes = conf_mat.shape[0]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match confusion matrix order.")

    per_class: dict[str, PerClassMetrics] = {}
    recall_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []
    total_correct = int(torch.diag(conf_mat).sum().item())
    total_samples = int(conf_mat.sum().item())

    for class_index, class_name in enumerate(class_names):
        true_count = int(conf_mat[class_index, :].sum().item())
        pred_count = int(conf_mat[:, class_index].sum().item())
        correct_count = int(conf_mat[class_index, class_index].item())
        recall = (correct_count / true_count) if true_count > 0 else 0.0
        precision = (correct_count / pred_count) if pred_count > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recall_values.append(recall)
        precision_values.append(precision)
        f1_values.append(f1)
        per_class[class_name] = {
            "support": true_count,
            "predicted": pred_count,
            "correct": correct_count,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    return {
        "accuracy": (total_correct / total_samples) if total_samples > 0 else 0.0,
        "macro_recall": float(sum(recall_values) / len(recall_values)),
        "macro_precision": float(sum(precision_values) / len(precision_values)),
        "macro_f1": float(sum(f1_values) / len(f1_values)),
        "num_samples": total_samples,
        "confusion_matrix": conf_mat.cpu().long().tolist(),
        "per_class": per_class,
    }


def _validation_qwk_and_classification(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> tuple[float, ClassificationMetrics]:
    """Compute QWK and classification metrics from a single confusion-matrix pass."""

    conf_mat = _confusion_matrix(preds=preds, targets=targets, num_classes=num_classes)
    qwk = _qwk_from_confusion(conf_mat=conf_mat, num_classes=num_classes)
    class_metrics = _classification_metrics_from_confusion(
        conf_mat=conf_mat,
        class_names=ORDINAL_CLASS_DISPLAY_NAMES,
    )
    return qwk, class_metrics


class UnitemporalTrainer:
    """Orchestrates the high-performance training loop for the damage classifier.

    Args:
        model: The PyTorch neural network to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: The ordinal loss function.
        device: Target compute device (e.g., "cuda").
        learning_rate: Base learning rate for the optimizer.
        accum_steps: Number of forward passes to accumulate before stepping.
        checkpoint_dir: Directory to save the best model weights.
        max_grad_norm: When set, L2 global norm clip after each accumulation window (AMP-safe).
        ema_enabled: When True, maintain an EMA shadow updated each optimizer step; validation and
            best checkpoint use the shadow.
        ema_decay: Decay for ``get_ema_multi_avg_fn`` in ``[0, 1]``.
        teacher_probs_cpu: Optional ``[N, K]`` float tensor of ensemble teacher probabilities
            on CPU, row-aligned with ``train_loader`` iteration order (deterministic loader).
            When set, training loss mixes ordinal EMD with temperature-scaled KL to this teacher.
        distill_alpha: Weight in ``[0, 1]`` on the KD term; ignored when ``teacher_probs_cpu`` is None.
        distill_temperature: Softmax temperature for KD (Hinton ``T``); ignored without a teacher.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, device: str = "cuda",
                 learning_rate: float = 1e-4, accum_steps: int = 4, checkpoint_dir: str | Path = "outputs",
                 lr_plateau_patience: int = 2, max_grad_norm: float | None = None,
                 ema_enabled: bool = False, ema_decay: float = 0.995,
                 verbose: bool = False, print_every_n_batches: int | None = None,
                 teacher_probs_cpu: torch.Tensor | None = None, distill_alpha: float = 0.0,
                 distill_temperature: float = 2.0) -> None:
        """Initialize trainer state, optimizer, schedulers, and scaling tools."""

        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.accum_steps = accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.print_every_n_batches = print_every_n_batches
        self.ema_enabled = ema_enabled
        self.ema_decay = ema_decay

        if teacher_probs_cpu is not None:
            if teacher_probs_cpu.dim() != 2:
                raise ValueError("teacher_probs_cpu must be 2D [N, K].")
            alpha = float(distill_alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("distill_alpha must lie in [0, 1].")
            temp = float(distill_temperature)
            if temp <= 0.0:
                raise ValueError("distill_temperature must be positive.")
            self._teacher_probs_cpu: torch.Tensor | None = teacher_probs_cpu.detach().cpu().float().contiguous()
            self._distill_alpha = alpha
            self._distill_temperature = temp
        else:
            self._teacher_probs_cpu = None
            self._distill_alpha = 0.0
            self._distill_temperature = float(distill_temperature)

        self.model = model.to(self.device)
        self.ema_model: AveragedModel | None = None
        if ema_enabled:
            self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.02,
            fused=self.device.type == "cuda"
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=lr_plateau_patience
        )

        self.scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
        self.best_qwk = 0.0
        self.best_selection_score = 0.0
        self.total_images_seen = 0
        self._last_epoch_telemetry: dict[str, float] = {
            "elapsed_s": 0.0,
            "imgs_per_sec": 0.0,
            "gpu_mem_bytes": 0.0,
        }
        self._last_val_recalls: list[float] = [0.0] * len(ORDINAL_CLASS_DISPLAY_NAMES)

        # ImageNet normalization constants pre-allocated on-device. The dataset emits a uint8
        # [B, 4, H, W] tensor; _prepare_batch casts to float32 and applies these to the RGB
        # channels GPU-side, leaving the mask channel as 0/1.
        self._rgb_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._rgb_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a batch to device and apply ImageNet normalization to RGB channels.

        Args:
            batch: Dict with ``image`` (uint8 [B, 4, H, W]), ``label``, and ``context`` tensors.

        Returns:
            Tuple of (images, labels, context) on ``self.device``. ``images`` is float32
            with the first three channels ImageNet-normalized in-place; the fourth (mask)
            channel is preserved at 0/1.
        """

        images_u8 = batch["image"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        context = batch["context"].to(self.device, non_blocking=True)

        images = to_normalized_float(images_u8, mean=self._rgb_mean, std=self._rgb_std)
        return images, labels, context

    def _train_epoch(self, _epoch: int) -> float:
        """Executes a single training epoch with gradient accumulation.

        Returns:
            Mean loss over processed batches. Side effect: wall-clock elapsed,
            imgs/s, and peak GPU-memory bytes land in ``self._last_epoch_telemetry``
            so the caller can compose the train-summary line without refactoring
            the return type.
        """

        self.model.train()
        cuda_available = torch.cuda.is_available()

        # Loss accumulators stay on-device. Synced to host only at heartbeat / verbose-row
        # boundaries (typically 3 prints per epoch), not per batch. NaN/Inf detection is
        # delegated to GradScaler, which silently skips bad steps and reduces the scale.
        total_loss_tensor = torch.zeros((), device=self.device, dtype=torch.float32)
        last_batch_loss_tensor = torch.zeros((), device=self.device, dtype=torch.float32)
        grad_norm_tensor: torch.Tensor | None = None
        processed_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        # Reset the CUDA peak-memory counter once per epoch; the train summary line then
        # reports the high-water mark across the whole epoch instead of the last batch only.
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()

        total_batches = len(self.train_loader)
        epoch_start = time.time()
        batch_wall_history: list[float] = []
        last_time = epoch_start

        print_cadence = self.print_every_n_batches or max(1, total_batches // 8)
        heartbeat_targets: set[int] = set()
        if not self.verbose and total_batches > 0:
            for fraction in _HEARTBEAT_FRACTIONS:
                target = max(1, round(total_batches * fraction))
                heartbeat_targets.add(target)

        if self.verbose and total_batches > 0:
            _print_batch_table_header()

        last_imgs_per_sec = 0
        last_gpu_mem = 0
        teacher_row = 0

        for batch_idx, batch in enumerate(self.train_loader):

            if batch is None or len(batch) == 0:
                continue

            images, labels, context = self._prepare_batch(batch)

            # Scale loss by the number of micro-batches in this accumulation window so the final
            # partial window (when len(loader) is not a multiple of accum_steps) matches mean loss.
            batches_in_block = batches_in_accum_window(
                batch_idx=batch_idx,
                accum_steps=self.accum_steps,
                total_batches=total_batches,
            )

            with torch.amp.autocast("cuda" if cuda_available else "cpu"):
                logits = self.model(images, context)
                if self._teacher_probs_cpu is None:
                    loss = self.criterion(logits, labels)
                else:
                    loss_emd = self.criterion(logits, labels)
                    batch_rows = int(labels.shape[0])
                    teacher_slice = self._teacher_probs_cpu[teacher_row:teacher_row + batch_rows].to(
                        self.device, non_blocking=True
                    ).clamp_min(1e-8)
                    teacher_slice = teacher_slice / teacher_slice.sum(dim=1, keepdim=True)
                    log_probs = F.log_softmax(logits / self._distill_temperature, dim=1)
                    loss_kd = F.kl_div(log_probs, teacher_slice, reduction="batchmean") * (
                        self._distill_temperature ** 2
                    )
                    loss = (1.0 - self._distill_alpha) * loss_emd + self._distill_alpha * loss_kd
                    teacher_row += batch_rows
                loss = loss / float(batches_in_block)

            is_last_batch = (batch_idx + 1) == total_batches
            is_step_boundary = (batch_idx + 1) % self.accum_steps == 0 or is_last_batch

            self.scaler.scale(loss).backward()

            if is_step_boundary:
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm_tensor = nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema_model is not None:
                    self.ema_model.update_parameters(self.model)

            with torch.no_grad():
                raw_batch_loss = loss.detach() * float(batches_in_block)
                total_loss_tensor += raw_batch_loss
                last_batch_loss_tensor = raw_batch_loss
            processed_batches += 1

            now = time.time()
            batch_time = now - last_time
            last_time = now
            batch_wall_history.append(batch_time)

            images_in_batch = int(images.shape[0])
            self.total_images_seen += images_in_batch
            last_imgs_per_sec = int(images_in_batch / batch_time) if batch_time > 0 else 0

            done = batch_idx + 1
            # ETA: running mean of per-batch wall time skipping the first 5 batches (warm-up).
            timing_sample = batch_wall_history[5:] if len(batch_wall_history) > 5 else batch_wall_history
            mean_batch_time = sum(timing_sample) / len(timing_sample) if timing_sample else 0.0
            remaining_batches = max(0, total_batches - done)
            eta_seconds = -1.0 if is_last_batch else remaining_batches * mean_batch_time

            will_print_verbose_row = self.verbose and (done % print_cadence == 0 or is_last_batch)
            will_print_heartbeat = (not self.verbose) and done in heartbeat_targets and not is_last_batch
            if will_print_verbose_row or will_print_heartbeat:
                # Single sync point per print: pull the running aggregates and current peak
                # memory off the device. Skipped batches advance no further than this.
                running_mean_loss = (
                    float(total_loss_tensor) / processed_batches if processed_batches > 0 else 0.0
                )
                last_batch_loss_value = float(last_batch_loss_tensor)
                if cuda_available:
                    last_gpu_mem = int(torch.cuda.max_memory_allocated())
                if will_print_verbose_row:
                    grad_norm_print = (
                        float(grad_norm_tensor) if grad_norm_tensor is not None else None
                    )
                    _print_batch_row(
                        done=done,
                        total=total_batches,
                        loss=last_batch_loss_value,
                        run_loss=running_mean_loss,
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                        imgs_per_sec=last_imgs_per_sec,
                        gpu_mem_bytes=last_gpu_mem,
                        grad_norm=grad_norm_print,
                        eta_s=eta_seconds,
                    )
                else:
                    _print_heartbeat(
                        done=done,
                        total=total_batches,
                        loss=last_batch_loss_value,
                        run_loss=running_mean_loss,
                        eta_s=eta_seconds,
                    )

        if self._teacher_probs_cpu is not None:
            expected_rows = int(self._teacher_probs_cpu.shape[0])
            if teacher_row != expected_rows:
                raise RuntimeError(
                    f"Distillation teacher row mismatch: consumed {teacher_row} rows, "
                    f"expected {expected_rows}; train_loader order or batching drifted vs cache."
                )

        elapsed_total = time.time() - epoch_start
        mean_loss = float(total_loss_tensor) / processed_batches if processed_batches > 0 else 0.0
        final_gpu_mem = int(torch.cuda.max_memory_allocated()) if cuda_available else 0

        if self.verbose:
            print()

        self._last_epoch_telemetry = {
            "elapsed_s": elapsed_total,
            "imgs_per_sec": float(last_imgs_per_sec),
            "gpu_mem_bytes": float(final_gpu_mem),
        }
        return mean_loss

    @torch.no_grad()
    def _validate(self, eval_model: nn.Module) -> ValidationMetrics:
        """Evaluates the model on the validation set using 8x D4 TTA (4 rotations x horizontal flip).

        Training uses Horizontal + Vertical Flip (p=0.5) augmentation, so the model is
        D4-invariance-trained; validation averages softmax probabilities over all 8 D4
        symmetries (4 rotations crossed with identity/hflip) before EV rounding for
        prediction and log-prob conversion for EMD loss. Vertical flip is implicit
        through rot180 + hflip composition; adding it explicitly would be redundant.
        """

        eval_model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            images, labels, context = self._prepare_batch(batch)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                # D4 TTA: 4 rotations x {identity, hflip} = 8 views; the mask channel (index 3)
                # transforms with the image tensor.
                rot_views = [
                    images,
                    torch.rot90(images, k=1, dims=[2, 3]),
                    torch.rot90(images, k=2, dims=[2, 3]),
                    torch.rot90(images, k=3, dims=[2, 3])
                ]
                tta_views = rot_views + [torch.flip(v, dims=[3]) for v in rot_views]

                # Arithmetic ensembling in probability space (upcast to float32 to prevent fp16 underflow).
                probs_stack = torch.stack(
                    [F.softmax(eval_model(view, context).float(), dim=1) for view in tta_views],
                    dim=0
                )
                avg_probs = probs_stack.mean(dim=0)

                # Convert back to pseudo-logits for EMD loss (eps=1e-7 safe in fp32).
                logits = torch.log(avg_probs + 1e-7)
                loss = self.criterion(logits.to(probs_stack.dtype), labels)

            total_loss += loss.item()

            # Expected Value Inference (replaces argmax)
            classes = torch.arange(4, device=self.device, dtype=torch.float32)
            expected_value = torch.sum(avg_probs * classes, dim=1)
            preds = torch.floor(expected_value + 0.5).long()

            all_preds.append(preds)
            all_targets.append(labels)

        flat_preds = torch.cat(all_preds)
        flat_targets = torch.cat(all_targets)

        avg_loss = total_loss / len(self.val_loader)
        qwk, class_metrics = _validation_qwk_and_classification(preds=flat_preds, targets=flat_targets)
        macro_f1 = class_metrics["macro_f1"]
        selection_score = (SELECTION_QWK_WEIGHT * qwk) + (SELECTION_MACRO_F1_WEIGHT * macro_f1)

        # Per-class recall percentages are stashed for the caller to fold into the per-epoch
        # block (no inline print -- the standalone "Per-Class Recall (Diagnostic)" stanza was
        # a detached trailing block; it now rides with the rest of the epoch summary).
        self._last_val_recalls: list[float] = []
        for class_index in range(len(ORDINAL_CLASS_DISPLAY_NAMES)):
            mask = (flat_targets == class_index)
            total = int(mask.sum().item())
            if total > 0:
                correct = int((flat_preds[mask] == class_index).sum().item())
                self._last_val_recalls.append(100.0 * correct / total)
            else:
                self._last_val_recalls.append(0.0)

        return {
            "val_loss": avg_loss,
            "val_qwk": qwk,
            "val_selection_score": selection_score,
            "classification": class_metrics,
        }

    def fit(self, epochs: int, early_stopping_patience: int = 10,
            lr_drop_warmstart_from_best: bool = False,
            reset_optimizer_on_lr_drop: bool = True) -> FitMetrics:
        """Main execution loop for training the network with optional EMA shadow weights.

        Early-stopping patience applies only after the first ReduceLROnPlateau LR reduction; if the
        schedule never reduces LR, the counter never arms. When ``lr_drop_warmstart_from_best`` is
        True, each ReduceLROnPlateau LR reduction loads the best checkpoint into the training model
        and, when EMA is enabled, re-aligns the EMA shadow from the same best state. The optimizer's
        internal state is cleared immediately after reloading weights (when
        ``reset_optimizer_on_lr_drop`` is True) so AdamW moments match the new parameters.

        Args:
            epochs: Maximum training epochs.
            early_stopping_patience: Epochs without selection-score improvement after the first LR drop.
            lr_drop_warmstart_from_best: On every LR reduction, warm-start from best weights when available.
            reset_optimizer_on_lr_drop: EMA-path only. When True (legacy), clear AdamW state on every warmstart.
                When False, preserve Adam moments across the reload (removes the first-post-drop bias-correction
                transient; per-parameter variance estimates stay informative since best-checkpoint EMA weights
                are a smoothed average of the same trajectory the moments were accumulated on).
        """

        epochs_without_improvement = 0
        post_lr_drop_phase = False
        lr_reduction_ever_seen = False

        best_state_for_warmstart: dict[str, torch.Tensor] | None = None
        previous_selection_score: float | None = None

        last_epoch = 0
        final_val_loss = 0.0
        final_val_qwk = 0.0
        final_selection_score = 0.0
        final_classification: ClassificationMetrics = _empty_classification_metrics()
        best_classification: ClassificationMetrics = _empty_classification_metrics()
        epoch_history: list[EpochRecord] = []
        fit_wall_start = time.time()

        for epoch in range(1, epochs + 1):
            last_epoch = epoch
            _print_epoch_banner(epoch=epoch, total_epochs=epochs)
            train_loss = self._train_epoch(epoch)

            metrics = self._validate(eval_model=self.ema_model if self.ema_model is not None else self.model)

            previous_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step(metrics.get("val_selection_score", metrics["val_qwk"]))
            updated_lr = float(self.optimizer.param_groups[0]["lr"])

            val_loss = float(metrics["val_loss"])
            val_qwk = float(metrics["val_qwk"])
            val_selection_score = float(metrics.get("val_selection_score", val_qwk))
            final_classification = metrics.get("classification", final_classification)
            final_val_loss = val_loss
            final_val_qwk = val_qwk
            final_selection_score = val_selection_score
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            telemetry = self._last_epoch_telemetry
            train_summary_kwargs = {
                "loss": float(train_loss),
                "lr": current_lr,
                "imgs_per_sec": int(telemetry.get("imgs_per_sec", 0.0)),
                "gpu_mem_bytes": int(telemetry.get("gpu_mem_bytes", 0.0)),
                "elapsed_s": float(telemetry.get("elapsed_s", 0.0)),
            }

            sel_delta = None if previous_selection_score is None else val_selection_score - previous_selection_score
            previous_selection_score = val_selection_score
            patience_str = _format_patience_str(
                counter=epochs_without_improvement,
                patience=early_stopping_patience,
                armed=post_lr_drop_phase,
            )
            classification_payload = metrics.get("classification", {}) or {}
            mf1 = float(classification_payload.get("macro_f1", 0.0))
            recalls = list(self._last_val_recalls) if self._last_val_recalls else [0.0] * 4

            if self.verbose:
                _print_train_summary(**train_summary_kwargs)
                _print_val_table(
                    loss=val_loss, qwk=val_qwk, mf1=mf1, sel=val_selection_score,
                    sel_delta=sel_delta, patience_str=patience_str, recalls=recalls,
                )
            else:
                _print_train_summary(**train_summary_kwargs)
                _print_val_summary(
                    loss=val_loss, qwk=val_qwk, mf1=mf1, sel=val_selection_score,
                    sel_delta=sel_delta, patience_str=patience_str,
                )
                _print_recall_summary(recalls=recalls)

            _print_shadow_note(ema_enabled=self.ema_model is not None, ema_decay=self.ema_decay)

            epoch_record: EpochRecord = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_qwk": float(val_qwk),
                "val_selection_score": float(val_selection_score),
                "best_val_qwk": float(max(self.best_qwk, val_qwk)),
                "best_val_selection_score": float(max(self.best_selection_score, val_selection_score)),
                "lr": current_lr,
                "ema_active": self.ema_model is not None,
            }
            epoch_history.append(epoch_record)

            qwk_guardrail_floor = self.best_qwk - SELECTION_QWK_GUARDRAIL
            passes_qwk_guardrail = val_qwk >= qwk_guardrail_floor
            if val_selection_score > self.best_selection_score and passes_qwk_guardrail:
                self.best_selection_score = val_selection_score
                self.best_qwk = val_qwk
                epochs_without_improvement = 0
                best_classification = metrics.get("classification", best_classification)

                checkpoint_path = self.checkpoint_dir / "best_model.pt"
                if self.ema_model is not None:
                    torch.save(self.ema_model.module.state_dict(), checkpoint_path)
                    source_state = self.ema_model.module.state_dict()
                else:
                    torch.save(self.model.state_dict(), checkpoint_path)
                    source_state = self.model.state_dict()
                best_state_for_warmstart = {key: value.detach().clone() for key, value in source_state.items()}
                _print_new_best(checkpoint_path)
            elif post_lr_drop_phase:
                epochs_without_improvement += 1

            if updated_lr + 1e-12 < previous_lr:
                post_lr_drop_phase = True
                first_lr_reduction = not lr_reduction_ever_seen
                if first_lr_reduction:
                    lr_reduction_ever_seen = True

                warmstart_note: str | None = None
                if lr_drop_warmstart_from_best and best_state_for_warmstart is not None:
                    self.model.load_state_dict(best_state_for_warmstart)
                    if self.ema_model is not None:
                        self.ema_model.module.load_state_dict(best_state_for_warmstart)
                        if reset_optimizer_on_lr_drop:
                            self.optimizer.state.clear()
                            warmstart_note = "warm-started training model + EMA shadow from best (optimizer state cleared)"
                        else:
                            warmstart_note = "warm-started training model + EMA shadow from best (optimizer state preserved)"
                    else:
                        warmstart_note = "warm-started training model from best checkpoint after LR drop"
                _print_lr_drop_block(
                    epoch=epoch,
                    previous_lr=previous_lr,
                    updated_lr=updated_lr,
                    first_reduction=first_lr_reduction,
                    warmstart_note=warmstart_note,
                )

            if post_lr_drop_phase and epochs_without_improvement >= early_stopping_patience:
                _print_event(
                    epoch=epoch,
                    body=f"early stopping  ({epochs_without_improvement} epochs without improvement post-LR-drop)",
                )
                break

        wall_seconds = time.time() - fit_wall_start
        _print_end_banner(
            epochs_completed=last_epoch,
            wall_seconds=wall_seconds,
            images_seen=self.total_images_seen,
            history=epoch_history,
            best_classification=best_classification,
            final_classification=final_classification,
            best_selection_score=self.best_selection_score,
            best_qwk=self.best_qwk,
            final_selection_score=final_selection_score,
            final_qwk=final_val_qwk,
        )

        return {
            "best_val_qwk": float(self.best_qwk),
            "final_val_qwk": float(final_val_qwk),
            "final_val_loss": float(final_val_loss),
            "epochs_completed": int(last_epoch),
            "history": epoch_history,
            "final_classification": final_classification,
            "best_classification": best_classification,
        }


def _resolve_device(device: str) -> str:
    """Resolve runtime device string to a concrete torch device name."""

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("runtime.device='cuda' requires CUDA availability.")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported runtime.device value: {device}")
    return device


def _dataset_label_summary(dataset: CRASARUnitemporalDataset) -> dict[str, int]:
    """Count ordinal labels from dataset instances for split diagnostics."""

    labels = ["no damage", "minor damage", "major damage", "destroyed"]
    counts = dict.fromkeys(labels, 0)
    instances = getattr(dataset, "instances", [])
    for instance in instances:
        damage_label = str(instance.get("damage_label", "")).strip().lower()
        if damage_label in counts:
            counts[damage_label] += 1
    return counts


class SplitSummary(TypedDict):
    """Fold-level dataset composition captured after dataloader construction."""

    train_instances: int
    val_instances: int
    train_label_counts: dict[str, int]
    val_label_counts: dict[str, int]


def _build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, chip_size: int, batch_size: int, num_workers: int,
                       pin_memory: bool, persistent_workers: bool, prefetch_factor: int, drop_last: bool,
                       val_batch_size_factor: float, seed: int | None, sampler_mode: str,
                       mask_dilation_px: int = 0, synthetic_gsd_factor: float = 1.0,
                       synthetic_gsd_mtf_at_nyquist: float | None = None,
                       synthetic_gsd_post_sharpen: float | None = None,
                       chip_window_scale: float = 1.0,
                       chip_window_ground_m: float | None = None) -> tuple[DataLoader, DataLoader, SplitSummary]:
    """Build train/validation datasets and dataloaders from split manifests."""

    train_dataset = CRASARUnitemporalDataset(
        train_df,
        chip_size=chip_size,
        transform=get_train_transforms(synthetic_gsd_factor=synthetic_gsd_factor,
                                       synthetic_gsd_mtf_at_nyquist=synthetic_gsd_mtf_at_nyquist,
                                       synthetic_gsd_post_sharpen=synthetic_gsd_post_sharpen),
        is_train=True,
        mask_dilation_px=mask_dilation_px,
        window_scale=chip_window_scale,
        window_ground_m=chip_window_ground_m,
    )
    # cache_validation_tensors: speeds validation from epoch 2 onward (~4MB per val instance in RAM).
    # The GSD degradation models the sensor, so it applies to validation too (deterministic, cache-safe).
    val_dataset = CRASARUnitemporalDataset(
        val_df,
        chip_size=chip_size,
        transform=get_val_transforms(synthetic_gsd_factor=synthetic_gsd_factor,
                                     synthetic_gsd_mtf_at_nyquist=synthetic_gsd_mtf_at_nyquist,
                                     synthetic_gsd_post_sharpen=synthetic_gsd_post_sharpen),
        is_train=False,
        cache_validation_tensors=True,
        mask_dilation_px=mask_dilation_px,
        window_scale=chip_window_scale,
        window_ground_m=chip_window_ground_m,
    )

    train_sampler = create_weighted_sampler(train_dataset) if sampler_mode == "weighted" else None

    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")
    val_batch_size = max(1, int(batch_size * val_batch_size_factor))

    data_loader_generator = torch.Generator()
    if seed is not None:
        data_loader_generator.manual_seed(seed)

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker,
        "generator": data_loader_generator
    }
    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler
    else:
        train_loader_kwargs["shuffle"] = True

    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    split_summary: SplitSummary = {
        "train_instances": len(getattr(train_dataset, "instances", [])),
        "val_instances": len(getattr(val_dataset, "instances", [])),
        "train_label_counts": _dataset_label_summary(train_dataset),
        "val_label_counts": _dataset_label_summary(val_dataset),
    }
    return train_loader, val_loader, split_summary


def _build_model_and_loss(config: AppConfig, class_weights: torch.Tensor | None = None) -> tuple[nn.Module, nn.Module]:
    """Build model and criterion from the validated AppConfig.

    Args:
        config: Fully-validated root config. Model architecture is pulled from ``config.model``
            and ``config.ablation``; the loss selector and smoothing rate are pulled from
            ``config.training``.
        class_weights: Optional per-class weighting tensor computed per fold.
    """

    model = MaskCenteredDamageNet(
        backbone_name=config.model.name,
        pretrained=config.model.pretrained,
        mask_enabled=config.ablation.mask_enabled,
        typology_enabled=config.ablation.typology_enabled,
        mask_weighted_pooling_enabled=config.ablation.mask_weighted_pooling_enabled,
        drop_path_rate=config.model.drop_path_rate,
    )
    criterion = create_criterion(
        loss_name=config.training.loss,
        weight=class_weights,
        label_smoothing=config.training.label_smoothing,
    )
    return model, criterion


def _build_resolved_config_snapshot(
    config: AppConfig,
    *,
    resolved_data_root: Path,
    resolved_holdout: str | list[str],
    resolved_device: str,
    resolved_seed: int | None,
    checkpoint_root: str,
    holdout_selection: str,
) -> dict[str, object]:
    """Merge the validated AppConfig with pipeline-resolved values for the run snapshot.

    The snapshot feeds ``config_resolved.yaml`` alongside fold metrics. It starts from
    ``config.model_dump(mode="json")`` so every validated field round-trips, and then
    overrides the small number of keys whose final values are only known after
    manifest scanning, device resolution, and holdout selection. ``resolved_holdout`` is
    the resolved fold name for single-event runs and the original event list for
    composite multi-event runs, keeping the snapshot reloadable for split reconstruction.
    """

    snapshot = config.model_dump(mode="json")
    snapshot["data"]["resolved_data_root"] = str(resolved_data_root)
    snapshot["data"]["holdout_event"] = resolved_holdout
    snapshot["runtime"]["device"] = resolved_device
    snapshot["runtime"]["seed"] = resolved_seed
    snapshot["runtime"]["checkpoint_root"] = str(checkpoint_root)
    snapshot["metadata"] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "holdout_event": resolved_holdout,
        "holdout_selection": holdout_selection,
    }
    return snapshot


def _write_run_metadata(checkpoint_dir: Path, resolved_config: dict[str, object], metrics: dict[str, object]) -> None:
    """Persist resolved run configuration and summary metrics for replay."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = checkpoint_dir / "config_resolved.yaml"
    metrics_path = checkpoint_dir / "metrics.json"
    config_snapshot_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _write_json_artifact(checkpoint_dir: Path, filename: str, payload: dict[str, object] | list[dict[str, object]]) -> None:
    """Write structured JSON artifact into the fold checkpoint directory."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = checkpoint_dir / filename
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fold_output_dir(checkpoint_root: str) -> Path:
    """Return the output directory for a single training run.

    Under the new ablation layout (``outputs/ablation/<variant>/<split>/seed_<n>``), the
    split name is already encoded by the caller, so artifacts are written directly under
    ``checkpoint_root`` with no ``fold_<event>`` suffix. The trainer writes at most one
    fold per run.
    """

    return Path(checkpoint_root)


def _compute_class_weights(
    *,
    train_label_counts: dict[str, int],
    power: float,
    eps: float,
    max_ratio: float | None,
) -> torch.Tensor | None:
    """Compute a per-fold class weighting tensor following the inverse-frequency recipe."""

    counts = [
        train_label_counts.get("no damage", 0),
        train_label_counts.get("minor damage", 0),
        train_label_counts.get("major damage", 0),
        train_label_counts.get("destroyed", 0),
    ]
    n_c = np.array(counts, dtype=np.float32)
    total_samples = float(np.sum(n_c))
    if total_samples <= 0:
        return None

    num_classes = len(counts)
    w_c = total_samples / (num_classes * (n_c + eps))
    if power != 1.0:
        w_c = w_c ** power
    if max_ratio is not None:
        baseline = float(np.median(w_c))
        if baseline > 0:
            w_c = np.clip(w_c, a_min=0.0, a_max=baseline * max_ratio)

    w_c_mean = float(np.mean(w_c))
    if w_c_mean > 0:
        w_c = w_c / w_c_mean

    return torch.tensor(w_c, dtype=torch.float32)


def _resolve_run_label(checkpoint_root: Path) -> str:
    """Derive a human-readable run label from the checkpoint-root directory name.

    Walks upward from the checkpoint root and returns the first ancestor nested directly
    inside an ``ablation/`` directory. This matches both the standard layout
    (``outputs/ablation/<variant>/<split>/seed_<n>``) and the legacy layout
    (``outputs/ablation/<variant>/seed_<n>``) because the variant folder is always the
    immediate child of ``ablation``. If no ``ablation`` ancestor is found, the ``seed_<n>``
    leaf is stripped and the next directory name is used; otherwise the directory name
    itself (or ``'training'``) is returned.
    """

    parts = checkpoint_root.parts
    if "ablation" in parts:
        idx = parts.index("ablation")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    candidate = checkpoint_root.name
    if candidate.startswith("seed_") and checkpoint_root.parent.name:
        return checkpoint_root.parent.name
    return candidate or "training"


def _format_split_counts(label_counts: dict[str, int]) -> str:
    """Format the per-class count summary used in the ``[split]`` timeline line."""

    cells = [f"{short} {label_counts.get(key, 0)}" for short, key in zip(ORDINAL_CLASS_SHORT_NAMES, ORDINAL_LABEL_KEYS, strict=True)]
    return "{" + ", ".join(cells) + "}"


def _emit_run_header(*, config: AppConfig, config_hash_str: str, run_label: str,
                     effective_seed: int | None, fold_name: str, resolved_device: str,
                     param_total: int, param_trainable: int, sensor_profile: str,
                     valid_instance_count: int, split_summary: SplitSummary,
                     class_weights: torch.Tensor | None, data_root: Path) -> None:
    """Emit the identity box followed by the ``[data]/[split]/[loaders]/[opt]`` timeline."""

    _print_identity_box(
        config=config,
        config_hash_str=config_hash_str,
        config_label=run_label,
        effective_seed=effective_seed,
        fold_name=fold_name,
        resolved_device=resolved_device,
        param_total=param_total,
        param_trainable=param_trainable,
    )

    _print_timeline(
        "data",
        f"scanning {data_root}/   ->   {valid_instance_count:,} valid instances (sensor_profile={sensor_profile})",
    )
    _print_timeline(
        "split",
        f"train {split_summary['train_instances']:,}   val {split_summary['val_instances']:,}   "
        f"counts {_format_split_counts(split_summary['train_label_counts'])}",
    )
    effective_batch = config.training.batch_size * config.training.accum_steps
    _print_timeline(
        "loaders",
        f"workers {config.runtime.num_workers}   pin_memory {config.runtime.pin_memory}   "
        f"persistent {config.runtime.persistent_workers}   prefetch {config.runtime.prefetch_factor}   "
        f"drop_last {config.runtime.drop_last}",
    )
    sampler_label = "weighted-inv-freq" if config.training.sampler_mode == "weighted" else "uniform-random"
    _print_timeline(
        "loaders",
        f"sampler {sampler_label}   batch {config.training.batch_size} x{config.training.accum_steps} accum -> "
        f"eff {effective_batch}   chip {config.data.chip_size}",
        continuation=True,
    )
    grad_clip_label = f"{config.training.max_grad_norm:.1f}" if config.training.max_grad_norm is not None else "off"
    _print_timeline(
        "opt",
        f"AdamW lr {config.training.lr:.1e} fused   epochs {config.training.epochs}   "
        f"plateau patience {config.training.lr_plateau_patience}   "
        f"early-stop patience {config.runtime.early_stopping_patience}   "
        f"grad_clip {grad_clip_label}",
    )
    if class_weights is not None:
        weights_list = [round(float(value), 4) for value in class_weights.tolist()]
        _print_timeline("opt", f"class weights {weights_list}", continuation=True)
    _print_training_header(epochs=config.training.epochs)


def _execute_training_pipeline(
    config: AppConfig,
    *,
    effective_seed: int | None,
    effective_checkpoint_root: str,
) -> None:
    """Inner pipeline body, invoked beneath the optional ``tee_stdout_to_file`` wrapper."""

    set_seeds(seed=effective_seed)
    sensor_profile = config.data.sensor_profile
    resolved_data_root = Path(config.data.dir)
    valid_manifest = build_valid_manifest(data_dir=str(resolved_data_root), sensor_profile=sensor_profile)

    holdout_event = config.data.holdout_event
    if isinstance(holdout_event, list):
        # Composite multi-event holdout (e.g. deployed-baseline split replication): the listed
        # events form the validation pool and every remaining event trains.
        holdout, train_df, val_df, holdout_selection = build_multi_event_fold(manifest=valid_manifest, holdout_events=holdout_event)
    else:
        split_manifest = valid_manifest if holdout_event is not None else build_spatial_split_manifest(valid_manifest)
        splits = list(generate_loeo_splits(split_manifest))
        holdout, train_df, val_df, holdout_selection = select_fold(splits=splits, holdout_event=holdout_event)

    resolved_device = _resolve_device(device=config.runtime.device)
    train_loader, val_loader, split_summary = _build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        chip_size=config.data.chip_size,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        persistent_workers=config.runtime.persistent_workers,
        prefetch_factor=config.runtime.prefetch_factor,
        drop_last=config.runtime.drop_last,
        val_batch_size_factor=config.runtime.val_batch_size_factor,
        seed=effective_seed,
        sampler_mode=config.training.sampler_mode,
        mask_dilation_px=config.ablation.mask_dilation_px,
        synthetic_gsd_factor=config.data.synthetic_gsd_factor,
        synthetic_gsd_mtf_at_nyquist=config.data.synthetic_gsd_mtf_at_nyquist,
        synthetic_gsd_post_sharpen=config.data.synthetic_gsd_post_sharpen,
        chip_window_scale=config.data.chip_window_scale,
        chip_window_ground_m=config.data.chip_window_ground_m,
    )

    class_weights_tensor: torch.Tensor | None = None
    if config.training.class_weighting_enabled:
        class_weights_tensor = _compute_class_weights(
            train_label_counts=split_summary["train_label_counts"],
            power=config.training.class_weighting_power,
            eps=config.training.class_weighting_eps,
            max_ratio=config.training.class_weighting_max_ratio,
        )

    model, criterion = _build_model_and_loss(config=config, class_weights=class_weights_tensor)
    fold_checkpoint_dir = _fold_output_dir(checkpoint_root=effective_checkpoint_root)

    param_total, param_trainable = _count_model_params(model)
    run_label = _resolve_run_label(Path(effective_checkpoint_root))
    config_hash_str = config_hash(config)
    fold_name = holdout.replace(" ", "_")

    _emit_run_header(
        config=config,
        config_hash_str=config_hash_str,
        run_label=run_label,
        effective_seed=effective_seed,
        fold_name=fold_name,
        resolved_device=resolved_device,
        param_total=param_total,
        param_trainable=param_trainable,
        sensor_profile=sensor_profile,
        valid_instance_count=len(valid_manifest),
        split_summary=split_summary,
        class_weights=class_weights_tensor,
        data_root=resolved_data_root,
    )

    trainer = UnitemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=resolved_device,
        learning_rate=config.training.lr,
        accum_steps=config.training.accum_steps,
        checkpoint_dir=fold_checkpoint_dir,
        lr_plateau_patience=config.training.lr_plateau_patience,
        max_grad_norm=config.training.max_grad_norm,
        ema_enabled=config.runtime.ema_enabled,
        ema_decay=config.runtime.ema_decay,
        verbose=config.runtime.verbose,
        print_every_n_batches=config.runtime.print_every_n_batches,
    )

    training_metrics = trainer.fit(
        epochs=config.training.epochs,
        early_stopping_patience=config.runtime.early_stopping_patience,
        lr_drop_warmstart_from_best=config.runtime.lr_drop_warmstart_from_best,
        reset_optimizer_on_lr_drop=config.runtime.reset_optimizer_on_lr_drop,
    )

    # Multi-event runs round-trip the original list through the snapshot so downstream
    # reconstruction (postproc ensemble / probe tooling) can rebuild the composite split;
    # single-event runs record the resolved fold name as before.
    resolved_config = _build_resolved_config_snapshot(
        config=config,
        resolved_data_root=resolved_data_root,
        resolved_holdout=holdout_event if isinstance(holdout_event, list) else holdout,
        resolved_device=resolved_device,
        resolved_seed=effective_seed,
        checkpoint_root=effective_checkpoint_root,
        holdout_selection=holdout_selection,
    )
    best_model_path = fold_checkpoint_dir / "best_model.pt"
    if not best_model_path.exists():
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fallback_state: dict[str, object] = {}
        if hasattr(model, "state_dict"):
            candidate_state = model.state_dict()
            if isinstance(candidate_state, dict):
                fallback_state = candidate_state
        torch.save(fallback_state, best_model_path)

    history_payload: list[EpochRecord] = training_metrics.get("history", [])
    final_classification: ClassificationMetrics = training_metrics.get(
        "final_classification", _empty_classification_metrics()
    )
    best_classification: ClassificationMetrics = training_metrics.get(
        "best_classification", _empty_classification_metrics()
    )
    compact_training_metrics: dict[str, object] = {
        key: value
        for key, value in training_metrics.items()
        if key not in {"history", "final_classification", "best_classification"}
    }

    metrics_payload = {
        "holdout_event": holdout,
        "holdout_selection": holdout_selection,
        "seed": effective_seed,
        "train_class_counts_ordinal": [
            split_summary["train_label_counts"].get("no damage", 0),
            split_summary["train_label_counts"].get("minor damage", 0),
            split_summary["train_label_counts"].get("major damage", 0),
            split_summary["train_label_counts"].get("destroyed", 0),
        ],
        "class_weights_ordinal": class_weights_tensor.tolist() if class_weights_tensor is not None else None,
        "best_model_path": str(best_model_path),
        "final_classification": final_classification,
        "best_classification": best_classification,
        **compact_training_metrics,
    }
    _write_run_metadata(checkpoint_dir=fold_checkpoint_dir, resolved_config=resolved_config, metrics=metrics_payload)
    _write_json_artifact(checkpoint_dir=fold_checkpoint_dir, filename="history.json", payload=list(history_payload))
    _write_json_artifact(checkpoint_dir=fold_checkpoint_dir, filename="split_summary.json", payload=dict(split_summary))
    _write_json_artifact(
        checkpoint_dir=fold_checkpoint_dir,
        filename="classification_metrics.json",
        payload={"final": final_classification, "best": best_classification},
    )


def run_training_pipeline(
    config: AppConfig,
    *,
    seed: int | None = None,
    checkpoint_root: str | Path | None = None,
) -> None:
    """Execute the full training pipeline for a single fold.

    Args:
        config: Validated ``AppConfig`` carrying data, training, model, ablation, and runtime
            settings.
        seed: Optional override for ``config.runtime.seed`` (one run per seed in ablation sweeps).
        checkpoint_root: Optional override for ``config.runtime.checkpoint_root`` (used by the CLI
            to route multi-seed output into ``.../seed_<seed>`` subdirectories).

    When ``config.runtime.log_to_file`` is enabled, stdout is tee'd to
    ``<fold_checkpoint_dir>/train_log.txt`` for the duration of the pipeline so ablation runs
    always produce a permanent log next to their metric artifacts.
    """

    effective_seed = seed if seed is not None else config.runtime.seed
    effective_checkpoint_root = str(
        checkpoint_root if checkpoint_root is not None else config.runtime.checkpoint_root
    )

    if not config.runtime.log_to_file:
        _execute_training_pipeline(
            config=config,
            effective_seed=effective_seed,
            effective_checkpoint_root=effective_checkpoint_root,
        )
        return

    # Route the entire pipeline body through a Tee context. The log path is anchored at the
    # checkpoint root rather than the fold directory so it captures the identity box and the
    # data-scan warnings that precede fold selection.
    log_path = Path(effective_checkpoint_root) / "train_log.txt"
    header_lines = [
        f"# MCDN training run - label={_resolve_run_label(Path(effective_checkpoint_root))}   seed={effective_seed}",
        f"# started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# config_hash {config_hash(config)}",
    ]
    with tee_stdout_to_file(log_path, header_lines=header_lines):
        _execute_training_pipeline(
            config=config,
            effective_seed=effective_seed,
            effective_checkpoint_root=effective_checkpoint_root,
        )
