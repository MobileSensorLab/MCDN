"""Softmax-ensemble trained checkpoints across seeds and variants.

Loads every seed's ``best_model.pt`` under one or more variant roots, runs the
same 8x D4 TTA (4 rotations x {identity, hflip}) the trainer uses post-overnight
baseline, and reports QWK / macro F1 / per-class metrics under multiple
ensemble strategies and prediction rules.

Ensemble strategies reported:
    - per_seed            : individual seed (sanity check vs. stored metrics)
    - within_variant      : mean softmax across the seeds of a single variant
    - cross_variant_var   : mean of the within-variant ensemble softmaxes (equal
                            weight per variant, irrespective of seed count)
    - cross_variant_seed  : mean of every individual seed's softmax across all
                            variants (equal weight per seed; variants with more
                            seeds contribute more)

Prediction rules applied to each ensemble probability tensor:
    - EV:     ``floor(sum_i i * p(i) + 0.5)`` -- ordinal-aware, optimizes QWK.
    - argmax: ``argmax_i p(i)`` -- categorical, optimizes macro F1 / accuracy;
              promoted to the primary rule after the per-arm sweep landed because it
              dominates hybrid on F1/accuracy across all 11 arm/split directories
              while staying within ~0.005 QWK of EV.
    - hybrid: argmax on extreme classes (0, K-1), else EV rounding -- retained
              alongside EV for the paper's ablation discussion (see
              ``doc/4-modeling.qmd`` §Inference pipeline).

Alignment contract:
    Each variant may differ in ``mask_dilation_px``, ``mask_weighted_pooling_enabled``,
    etc. The underlying val samples (buildings, chip ordering, labels) are identical
    across variants because the fold is identical and the val loader uses
    ``shuffle=False``. We verify alignment by requiring that targets from every
    (variant, seed) pair match exactly.

Outputs:
    - printed per-variant / cross-variant / summary tables
    - JSON artifact at ``outputs/ablation/ensemble_inference.json`` (overridable)

Layout:
    Operates on the standard ablation layout
    ``<variant>/<split>/seed_<n>/{best_model.pt, config_resolved.yaml}``. The
    ``--split`` argument selects the split directory name (default:
    ``Spatial_Block_East``).

Usage:
    uv run python -m src.postproc.ensemble \
        --variant-roots outputs/ablation/all_features outputs/ablation/no-mask \
                         outputs/ablation/context outputs/ablation/resolution \
        --split Spatial_Block_East
"""

import argparse
import json
import time

import pandas as pd
import torch
import yaml

import torch.nn.functional as F

from collections.abc import Callable
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import CRASARUnitemporalDataset, to_normalized_float
from src.data.transform import get_val_transforms
from src.data.sampling import (
    build_multi_event_fold,
    build_spatial_split_manifest,
    build_valid_manifest,
    generate_loeo_splits,
    select_fold,
)
from src.model.mcdn import MaskCenteredDamageNet
from src.model.trainer import (
    ORDINAL_CLASS_DISPLAY_NAMES,
    _build_dataloaders,
    _resolve_device,
    _seed_worker,
    _validation_qwk_and_classification,
    set_seeds,
)


DEFAULT_SEEDS: tuple[int, ...] = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)
REPLAY_TOLERANCE: float = 5e-3

# Prediction rules ---------------------------------------------------------

def apply_ev_rounding(probs: torch.Tensor) -> torch.Tensor:
    """Expected-value rounding: ``floor(sum_i i * p(i) + 0.5)``."""

    num_classes = probs.shape[1]
    classes = torch.arange(num_classes, dtype=torch.float32)
    expected_value = torch.sum(probs * classes, dim=1)
    return torch.floor(expected_value + 0.5).long()


def apply_argmax(probs: torch.Tensor) -> torch.Tensor:
    """Standard categorical argmax."""

    return torch.argmax(probs, dim=1)


def apply_hybrid(probs: torch.Tensor) -> torch.Tensor:
    """Argmax on extreme classes (0 or K-1), EV rounding elsewhere."""

    ev_preds = apply_ev_rounding(probs=probs)
    argmax_preds = apply_argmax(probs=probs)
    num_classes = probs.shape[1]
    extreme_mask = (argmax_preds == 0) | (argmax_preds == (num_classes - 1))
    return torch.where(extreme_mask, argmax_preds, ev_preds)


RULE_FNS = {
    "EV": apply_ev_rounding,
    "argmax": apply_argmax,
    "hybrid": apply_hybrid,
}


# Config / model / loader plumbing ----------------------------------------

def load_resolved_config(fold_dir: Path) -> dict:
    """Parse a fold's ``config_resolved.yaml``."""

    config_path = fold_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_stored_metrics(fold_dir: Path) -> dict:
    """Parse a fold's ``metrics.json``."""

    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_model_from_config(cfg: dict, device: str) -> MaskCenteredDamageNet:
    """Construct the model architecture matching a resolved-config snapshot."""

    ablation = cfg["ablation"]
    model_cfg = cfg["model"]
    model = MaskCenteredDamageNet(
        backbone_name=model_cfg["name"],
        pretrained=False,
        mask_enabled=ablation["mask_enabled"],
        typology_enabled=ablation["typology_enabled"],
        mask_weighted_pooling_enabled=ablation["mask_weighted_pooling_enabled"],
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )
    model = model.to(device)
    model.eval()
    return model


def build_student_model_from_config(cfg: dict, device: str) -> MaskCenteredDamageNet:
    """Construct a trainable MCDN from resolved config (pretrained backbone per yaml)."""

    ablation = cfg["ablation"]
    model_cfg = cfg["model"]
    model = MaskCenteredDamageNet(
        backbone_name=model_cfg["name"],
        pretrained=model_cfg.get("pretrained", True),
        mask_enabled=ablation["mask_enabled"],
        typology_enabled=ablation["typology_enabled"],
        mask_weighted_pooling_enabled=ablation["mask_weighted_pooling_enabled"],
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
    )
    model = model.to(device)
    model.train()
    return model


def get_fold_dataframes_from_config(cfg: dict, data_dir_override: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return train/val manifest slices and holdout label for one resolved fold.

    Mirrors the split semantics in the overnight trainer and in
    ``build_val_loader_from_config``.
    """

    data_cfg = cfg["data"]
    data_dir = data_dir_override if data_dir_override is not None else data_cfg["dir"]
    sensor_profile = data_cfg["sensor_profile"]

    valid_manifest = build_valid_manifest(data_dir=data_dir, sensor_profile=sensor_profile)

    holdout_selection = cfg.get("metadata", {}).get("holdout_selection") or data_cfg.get("holdout_selection", "explicit")
    if holdout_selection == "default_spatial":
        split_manifest = build_spatial_split_manifest(valid_manifest)
        holdout_event_arg: str | list[str] | None = None
    else:
        split_manifest = valid_manifest
        holdout_event_arg = data_cfg.get("holdout_event")

    # Multi-event snapshots persist the original event list, so the composite fold is
    # rebuilt directly rather than routed through single-event LOEO selection.
    if isinstance(holdout_event_arg, list):
        holdout, train_df, val_df, _selection = build_multi_event_fold(manifest=valid_manifest, holdout_events=holdout_event_arg)
        return train_df, val_df, holdout

    splits = list(generate_loeo_splits(split_manifest))
    holdout, train_df, val_df, _selection = select_fold(splits=splits, holdout_event=holdout_event_arg)
    return train_df, val_df, holdout


def build_val_loader_from_config(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
    """Reconstruct the validation loader the trainer used for this config.

    Forwards ``mask_dilation_px`` so baselines trained with a dilated mask (b5)
    see the same input distribution at inference.
    """

    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]
    training_cfg = cfg["training"]
    ablation = cfg["ablation"]

    train_df, val_df, holdout = get_fold_dataframes_from_config(cfg, data_dir_override)

    _, val_loader, _ = _build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        chip_size=data_cfg["chip_size"],
        batch_size=training_cfg["batch_size"],
        num_workers=runtime_cfg["num_workers"],
        pin_memory=runtime_cfg["pin_memory"],
        persistent_workers=runtime_cfg["persistent_workers"],
        prefetch_factor=runtime_cfg["prefetch_factor"],
        drop_last=runtime_cfg["drop_last"],
        val_batch_size_factor=runtime_cfg["val_batch_size_factor"],
        seed=runtime_cfg.get("seed"),
        sampler_mode=training_cfg.get("sampler_mode", "uniform"),
        mask_dilation_px=ablation.get("mask_dilation_px", 0),
        synthetic_gsd_factor=data_cfg.get("synthetic_gsd_factor", 1.0),
        synthetic_gsd_mtf_at_nyquist=data_cfg.get("synthetic_gsd_mtf_at_nyquist"),
        synthetic_gsd_post_sharpen=data_cfg.get("synthetic_gsd_post_sharpen"),
        chip_window_scale=data_cfg.get("chip_window_scale", 1.0),
        chip_window_ground_m=data_cfg.get("chip_window_ground_m"),
    )
    return val_loader, holdout


def build_deterministic_train_loader_from_config(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
    """Train loader with fixed sample order and no random jitter (matches teacher cache).

    Uses validation-style transforms (identity) and ``is_train=False`` so centroid
    windows are deterministic. ``shuffle=False`` and ``drop_last=False`` keep batching
    aligned row-for-row with a cached teacher probability tensor.
    """

    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]
    training_cfg = cfg["training"]
    ablation = cfg["ablation"]

    train_df, val_df, holdout = get_fold_dataframes_from_config(cfg, data_dir_override)
    del val_df

    data_loader_generator = torch.Generator()
    seed = runtime_cfg.get("seed")
    if seed is not None:
        data_loader_generator.manual_seed(int(seed))

    train_dataset = CRASARUnitemporalDataset(
        train_df,
        chip_size=data_cfg["chip_size"],
        transform=get_val_transforms(synthetic_gsd_factor=data_cfg.get("synthetic_gsd_factor", 1.0),
                                     synthetic_gsd_mtf_at_nyquist=data_cfg.get("synthetic_gsd_mtf_at_nyquist"),
                                     synthetic_gsd_post_sharpen=data_cfg.get("synthetic_gsd_post_sharpen")),
        is_train=False,
        mask_dilation_px=ablation.get("mask_dilation_px", 0),
        window_scale=data_cfg.get("chip_window_scale", 1.0),
        window_ground_m=data_cfg.get("chip_window_ground_m"),
    )

    if runtime_cfg["persistent_workers"] and runtime_cfg["num_workers"] == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    loader_kwargs: dict[str, object] = {
        "dataset": train_dataset,
        "batch_size": training_cfg["batch_size"],
        "shuffle": False,
        "num_workers": runtime_cfg["num_workers"],
        "pin_memory": runtime_cfg["pin_memory"],
        "persistent_workers": runtime_cfg["persistent_workers"],
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "generator": data_loader_generator,
    }
    if runtime_cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = runtime_cfg["prefetch_factor"]

    train_loader = DataLoader(**loader_kwargs)
    return train_loader, holdout


# Inference core ----------------------------------------------------------

# D4 view geometry shared by TTA averaging and the seed x TTA decomposition tooling.
# Index 0 is the untransformed identity view, which defines the no-TTA configuration.
D4_VIEW_ORDER: tuple[str, ...] = ("rot0", "rot90", "rot180", "rot270", "rot0_hflip", "rot90_hflip", "rot180_hflip", "rot270_hflip")
IDENTITY_VIEW_INDEX: int = 0


@torch.no_grad()
def tta_view_softmax_probs(model: MaskCenteredDamageNet, images_u8: torch.Tensor, context: torch.Tensor,
                           device: str) -> torch.Tensor:
    """Return per-view softmax probabilities [V, B, K] over the 8 D4 views.

    View order follows ``D4_VIEW_ORDER``: 4 rotations, then the hflip of each.
    ``images_u8`` and ``context`` may live on CPU or ``device``; views are evaluated on
    ``device``. Retaining the view axis lets downstream tooling score no-TTA (identity
    view only) configurations without re-running the model.
    """

    images_u8 = images_u8.to(device, non_blocking=True)
    context = context.to(device, non_blocking=True)
    images = to_normalized_float(images_u8)
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.amp.autocast(autocast_device):
        rot_views = [
            images,
            torch.rot90(images, k=1, dims=[2, 3]),
            torch.rot90(images, k=2, dims=[2, 3]),
            torch.rot90(images, k=3, dims=[2, 3])
        ]
        tta_views = rot_views + [torch.flip(v, dims=[3]) for v in rot_views]

        return torch.stack(
            [F.softmax(model(view, context).float(), dim=1) for view in tta_views],
            dim=0
        )


@torch.no_grad()
def tta_mean_softmax_probs(model: MaskCenteredDamageNet, images_u8: torch.Tensor, context: torch.Tensor,
                           device: str) -> torch.Tensor:
    """Apply 8x D4 TTA and return mean softmax probabilities [B, K].

    ``images_u8`` and ``context`` may live on CPU or ``device``; views are evaluated on
    ``device``. Matches :meth:`UnitemporalTrainer._validate` and legacy
    ``collect_averaged_probabilities`` probability geometry.
    """

    return tta_view_softmax_probs(model=model, images_u8=images_u8, context=context, device=device).mean(dim=0)


@torch.no_grad()
def ensemble_mean_tta_probs(models: list[MaskCenteredDamageNet], images_u8: torch.Tensor, context: torch.Tensor,
                            device: str) -> torch.Tensor:
    """Mean softmax probabilities across seeds after per-seed 8x D4 TTA."""

    stacked = torch.stack(
        [tta_mean_softmax_probs(model=m, images_u8=images_u8, context=context, device=device) for m in models],
        dim=0,
    )
    return stacked.mean(dim=0)


@torch.no_grad()
def collect_averaged_probabilities(model: MaskCenteredDamageNet, val_loader: DataLoader,
                                    device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 8x D4 TTA and return softmax-averaged probabilities + targets.

    D4 group: 4 rotations (0/90/180/270 deg) x {identity, hflip} = 8 views.
    Vertical flip is implicit via rot180 + hflip composition. The mask channel
    (3) transforms with the image tensor. Matches the trainer's post-overnight-baseline
    validation pathway so per-seed replay matches stored metrics.
    """

    probs_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []

    for batch in val_loader:
        labels = batch["label"]
        avg_probs = tta_mean_softmax_probs(
            model=model,
            images_u8=batch["image"],
            context=batch["context"],
            device=device,
        )
        probs_chunks.append(avg_probs.cpu())
        targets_chunks.append(labels.cpu())

    return torch.cat(probs_chunks, dim=0), torch.cat(targets_chunks, dim=0)


# Metric / formatting helpers ---------------------------------------------

def compute_rule_metrics(probs: torch.Tensor, targets: torch.Tensor, rule_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
    """Compute QWK, macro F1, per-class F1, confusion matrix for a rule."""

    preds = rule_fn(probs)
    qwk, class_metrics = _validation_qwk_and_classification(preds=preds, targets=targets)
    return {
        "qwk": float(qwk),
        "accuracy": float(class_metrics["accuracy"]),
        "macro_f1": float(class_metrics["macro_f1"]),
        "macro_recall": float(class_metrics["macro_recall"]),
        "macro_precision": float(class_metrics["macro_precision"]),
        "per_class_f1": {
            name: float(class_metrics["per_class"][name]["f1"])
            for name in ORDINAL_CLASS_DISPLAY_NAMES
        },
        "per_class_recall": {
            name: float(class_metrics["per_class"][name]["recall"])
            for name in ORDINAL_CLASS_DISPLAY_NAMES
        },
        "per_class_precision": {
            name: float(class_metrics["per_class"][name]["precision"])
            for name in ORDINAL_CLASS_DISPLAY_NAMES
        },
        "confusion_matrix": class_metrics["confusion_matrix"],
    }


def metrics_under_all_rules(probs: torch.Tensor, targets: torch.Tensor) -> dict[str, dict]:
    """Compute metrics under EV, argmax, hybrid for one probability tensor."""

    return {rule_name: compute_rule_metrics(probs=probs, targets=targets, rule_fn=rule_fn)
            for rule_name, rule_fn in RULE_FNS.items()}


def format_conf_matrix(matrix: list[list[int]]) -> str:
    """Render a 4x4 confusion matrix with class labels."""

    header = "         " + "  ".join(f"{n[:9]:>9}" for n in ORDINAL_CLASS_DISPLAY_NAMES)
    lines = [header]
    for row_idx, row in enumerate(matrix):
        row_str = "  ".join(f"{v:>9d}" for v in row)
        lines.append(f"  {ORDINAL_CLASS_DISPLAY_NAMES[row_idx]:>7s}  {row_str}")
    return "\n".join(lines)


def format_metric_line(label: str, m: dict) -> str:
    """Single-line summary: QWK / F1 / acc / per-class F1."""

    per_class = m["per_class_f1"]
    per_class_str = "  ".join(f"{n[:3]}={per_class[n]:.3f}" for n in ORDINAL_CLASS_DISPLAY_NAMES)
    return (f"  {label:<32s} QWK={m['qwk']:.4f}  F1={m['macro_f1']:.4f}  "
            f"acc={m['accuracy']:.4f}  [{per_class_str}]")


# Orchestration -----------------------------------------------------------

def run_variant(variant_root: Path, seeds: list[int], split_name: str, data_dir: str | None,
                device: str, data_override: dict | None = None) -> dict:
    """Build val loader from first seed's config; run TTA over every seed.

    The standard ablation layout places each seed at
    ``<variant_root>/<split_name>/seed_<n>/`` with ``best_model.pt`` and
    ``config_resolved.yaml`` written directly into the seed directory (no ``fold_*``
    subdir).

    Args:
        variant_root: ``outputs/ablation/<variant>`` directory.
        seeds: Seeds to attempt (missing seed directories are skipped).
        split_name: Fold directory name used to locate checkpoints.
        data_dir: Optional data-root override.
        device: Resolved torch device string.
        data_override: Optional replacement for the seeds' resolved ``data`` config
            (transfer evaluation on a corpus the models were not trained on). The val
            loader is built from this section instead; the replay-vs-stored check is
            reported as not applicable since stored metrics describe a different pool.

    Returns a dict with:
        - per_seed: list of {seed, probs [T,K], replay under each rule}
        - ensemble_probs: mean over seeds [T, K]
        - targets: [T]
        - holdout_event: str
        - seed_stored_metrics: per-seed stored f1 / qwk (for replay check)
    """

    fold_dirs: dict[int, Path] = {}
    for seed in seeds:
        candidate = variant_root / split_name / f"seed_{seed:02d}"
        if candidate.is_dir():
            fold_dirs[seed] = candidate
    if not fold_dirs:
        raise FileNotFoundError(f"No seed folds found under {variant_root / split_name}")

    print(f"\n=== {variant_root.name} ===  seeds present: {sorted(fold_dirs.keys())}")

    first_seed = sorted(fold_dirs.keys())[0]
    first_cfg = load_resolved_config(fold_dirs[first_seed])
    if data_override is not None:
        # Replace the data section wholesale and force explicit holdout selection so the
        # override's holdout_event governs the fold, not the checkpoints' split metadata.
        first_cfg = {**first_cfg, "data": dict(data_override)}
        first_cfg["metadata"] = {**first_cfg.get("metadata", {}), "holdout_selection": "explicit"}
        print(f"  data override active: sensor_profile={data_override.get('sensor_profile')}  "
              f"holdout_event={data_override.get('holdout_event')}  "
              f"chip_window_ground_m={data_override.get('chip_window_ground_m')}")

    t0 = time.time()
    val_loader, holdout_event = build_val_loader_from_config(cfg=first_cfg, data_dir_override=data_dir)
    print(f"  val loader ready ({time.time() - t0:.1f}s)  holdout={holdout_event}  batches={len(val_loader)}"
          f"  mask_dilation_px={first_cfg['ablation'].get('mask_dilation_px', 0)}"
          f"  mask_weighted_pool={first_cfg['ablation']['mask_weighted_pooling_enabled']}")

    all_probs: list[torch.Tensor] = []
    reference_targets: torch.Tensor | None = None
    per_seed_records: list[dict] = []

    for seed in sorted(fold_dirs.keys()):
        fold_dir = fold_dirs[seed]
        cfg = load_resolved_config(fold_dir)
        stored = load_stored_metrics(fold_dir)
        stored_qwk = stored["best_val_qwk"]
        stored_f1 = stored.get("best_classification", {}).get("macro_f1")

        model = build_model_from_config(cfg=cfg, device=device)
        checkpoint_path = fold_dir / "best_model.pt"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        ti = time.time()
        probs, targets = collect_averaged_probabilities(model=model, val_loader=val_loader, device=device)
        inf_seconds = time.time() - ti

        if reference_targets is None:
            reference_targets = targets
        else:
            if not torch.equal(reference_targets, targets):
                raise RuntimeError(
                    f"Target mismatch between seed {first_seed} and seed {seed} in variant "
                    f"{variant_root.name}; val loader produced different sample ordering."
                )

        replay = metrics_under_all_rules(probs=probs, targets=targets)
        if data_override is not None:
            # Stored metrics describe the training-time val pool, not the override pool;
            # the delta is meaningless here so the check is reported as not applicable.
            replay_delta = None
            replay_ok = None
            print(f"  seed {seed}: transfer EV QWK={replay['EV']['qwk']:.4f} "
                  f"(stored={stored_qwk:.4f} on training-time pool [n/a])  "
                  f"F1={replay['EV']['macro_f1']:.4f}  ({inf_seconds:.1f}s)")
        else:
            replay_delta = replay["EV"]["qwk"] - stored_qwk
            replay_ok = abs(replay_delta) <= REPLAY_TOLERANCE
            tag = "OK" if replay_ok else "!!"
            f1_str = f"  stored_F1={stored_f1:.4f}" if stored_f1 is not None else ""
            print(f"  seed {seed}: replay EV QWK={replay['EV']['qwk']:.4f} "
                  f"(stored={stored_qwk:.4f} delta={replay_delta:+.4f} [{tag}])  "
                  f"F1={replay['EV']['macro_f1']:.4f}{f1_str}  ({inf_seconds:.1f}s)")

        all_probs.append(probs)
        per_seed_records.append({
            "seed": seed,
            "stored_qwk": stored_qwk,
            "stored_macro_f1": stored_f1,
            "replay_delta_vs_stored": replay_delta,
            "replay_ok": replay_ok,
            "replay_metrics": replay,
        })

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probs_stack = torch.stack(all_probs, dim=0)
    ensemble_probs = probs_stack.mean(dim=0)
    ensemble_metrics = metrics_under_all_rules(probs=ensemble_probs, targets=reference_targets)
    print(format_metric_line(f"within_variant ensemble (N={len(all_probs)}):", ensemble_metrics["EV"]))
    print(format_metric_line("within_variant ensemble argmax:", ensemble_metrics["argmax"]))
    print(format_metric_line("within_variant ensemble hybrid:", ensemble_metrics["hybrid"]))

    return {
        "variant_root": str(variant_root),
        "holdout_event": holdout_event,
        "seeds": sorted(fold_dirs.keys()),
        "per_seed": per_seed_records,
        "ensemble_probs": ensemble_probs,
        "individual_probs": probs_stack,
        "targets": reference_targets,
        "ensemble_metrics": ensemble_metrics,
    }


def compute_cross_variant_ensemble(variant_results: list[dict]) -> dict:
    """Compute cross-variant ensemble probs under both weighting schemes.

    Requires all variants to have produced identical target tensors (verified).
    """

    reference_targets = variant_results[0]["targets"]
    for vr in variant_results[1:]:
        if not torch.equal(vr["targets"], reference_targets):
            raise RuntimeError("Cross-variant target mismatch; val loaders produced different ordering.")

    # equal weight per variant
    per_variant_probs = torch.stack([vr["ensemble_probs"] for vr in variant_results], dim=0)
    equal_variant_probs = per_variant_probs.mean(dim=0)

    # equal weight per seed (pool all individual seeds across variants)
    all_seed_probs = torch.cat([vr["individual_probs"] for vr in variant_results], dim=0)
    equal_seed_probs = all_seed_probs.mean(dim=0)

    return {
        "equal_variant_metrics": metrics_under_all_rules(probs=equal_variant_probs, targets=reference_targets),
        "equal_seed_metrics": metrics_under_all_rules(probs=equal_seed_probs, targets=reference_targets),
        "num_variants": len(variant_results),
        "num_seeds_total": int(all_seed_probs.shape[0]),
    }


# Reporting ---------------------------------------------------------------

def print_summary_table(variant_results: list[dict], cross_results: dict) -> None:
    """Print one clean summary table across variants and cross-variant ensembles."""

    print("\n" + "=" * 100)
    print("SUMMARY (argmax = primary ensemble rule; EV + hybrid retained as comparative references)")
    print("=" * 100)
    header = (f"{'variant':<32s}  {'N':>3s}  {'EV-F1':>7s}  {'arg-F1':>7s}  {'hyb-F1':>7s}"
              f"  {'EV-QWK':>7s}  {'arg-QWK':>7s}  {'hyb-QWK':>7s}  {'EV-acc':>7s}")
    print(header)
    print("-" * len(header))

    for vr in variant_results:
        em = vr["ensemble_metrics"]
        row = (f"{Path(vr['variant_root']).name:<32s}  {len(vr['seeds']):>3d}  "
               f"{em['EV']['macro_f1']:>7.4f}  {em['argmax']['macro_f1']:>7.4f}  {em['hybrid']['macro_f1']:>7.4f}  "
               f"{em['EV']['qwk']:>7.4f}  {em['argmax']['qwk']:>7.4f}  {em['hybrid']['qwk']:>7.4f}  "
               f"{em['EV']['accuracy']:>7.4f}")
        print(row)

    ev_m = cross_results["equal_variant_metrics"]
    es_m = cross_results["equal_seed_metrics"]
    nv = cross_results["num_variants"]
    ns = cross_results["num_seeds_total"]
    print("-" * len(header))
    print(f"{'cross-variant (equal per var)':<32s}  {nv:>3d}  "
          f"{ev_m['EV']['macro_f1']:>7.4f}  {ev_m['argmax']['macro_f1']:>7.4f}  {ev_m['hybrid']['macro_f1']:>7.4f}  "
          f"{ev_m['EV']['qwk']:>7.4f}  {ev_m['argmax']['qwk']:>7.4f}  {ev_m['hybrid']['qwk']:>7.4f}  "
          f"{ev_m['EV']['accuracy']:>7.4f}")
    print(f"{'cross-variant (equal per seed)':<32s}  {ns:>3d}  "
          f"{es_m['EV']['macro_f1']:>7.4f}  {es_m['argmax']['macro_f1']:>7.4f}  {es_m['hybrid']['macro_f1']:>7.4f}  "
          f"{es_m['EV']['qwk']:>7.4f}  {es_m['argmax']['qwk']:>7.4f}  {es_m['hybrid']['qwk']:>7.4f}  "
          f"{es_m['EV']['accuracy']:>7.4f}")

    print("\nHeadline reporting line = argmax rule (post-sweep selection: dominates hybrid on F1/"
          "accuracy across all 11 arm/split arms; EV / hybrid retained above for ablation context).")

    # per-class F1 breakdown on the winning cross-variant ensemble (best F1 across rules)
    print("\nPer-class F1 (cross-variant equal-per-seed, all rules):")
    for rule in ("EV", "argmax", "hybrid"):
        per_class = es_m[rule]["per_class_f1"]
        line = "  ".join(f"{n}={per_class[n]:.4f}" for n in ORDINAL_CLASS_DISPLAY_NAMES)
        print(f"  {rule:<8s}  {line}")

    print("\nConfusion matrix (cross-variant equal-per-seed, EV rule):")
    print(format_conf_matrix(es_m["EV"]["confusion_matrix"]))


def serialize_for_json(metrics: dict) -> dict:
    """Strip tensors; keep plain types for JSON export."""

    return metrics  # already pure python types in metrics_under_all_rules


def write_prob_cache(path: Path, variant_results: list[dict], split_name: str,
                      holdout_event: str) -> None:
    """Serialize per-seed + within-variant + cross-variant probability tensors.

    Downstream consumers (calibration, custom prediction rules) can load this
    file without re-running inference. Shape reference: each variant payload
    contains ``individual_probs`` [num_seeds, num_samples, K] and
    ``ensemble_probs`` [num_samples, K].
    """

    payload = {
        "split": split_name,
        "holdout_event": holdout_event,
        "class_names": list(ORDINAL_CLASS_DISPLAY_NAMES),
        "targets": variant_results[0]["targets"],
        "variants": [
            {
                "variant_root": vr["variant_root"],
                "seeds": vr["seeds"],
                "individual_probs": vr["individual_probs"],
                "ensemble_probs": vr["ensemble_probs"],
            }
            for vr in variant_results
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Probability cache written: {path}")


def write_artifact(path: Path, variant_results: list[dict], cross_results: dict,
                   split_name: str, holdout_event: str) -> None:
    """Write a single JSON summary covering all variants + cross ensembles."""

    variants_payload = [
        {
            "variant_root": vr["variant_root"],
            "seeds": vr["seeds"],
            "per_seed": [
                {k: v for k, v in rec.items() if k != "replay_metrics"}
                | {"replay_metrics": serialize_for_json(rec["replay_metrics"])}
                for rec in vr["per_seed"]
            ],
            "ensemble_metrics": serialize_for_json(vr["ensemble_metrics"]),
        }
        for vr in variant_results
    ]

    payload = {
        "split": split_name,
        "holdout_event": holdout_event,
        "variants": variants_payload,
        "cross_variant": {
            "num_variants": cross_results["num_variants"],
            "num_seeds_total": cross_results["num_seeds_total"],
            "equal_variant_metrics": serialize_for_json(cross_results["equal_variant_metrics"]),
            "equal_seed_metrics": serialize_for_json(cross_results["equal_seed_metrics"]),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nArtifact written: {path}")


def main() -> None:
    """CLI entry point for cross-seed / cross-variant ensemble inference."""

    parser = argparse.ArgumentParser(description="Cross-seed / cross-variant softmax ensemble inference.")
    parser.add_argument("--variant-roots", type=Path, nargs="+", required=True,
                        help="One or more ``outputs/ablation/<variant>`` directories.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to attempt per variant (missing ones are silently skipped).")
    parser.add_argument("--split", type=str, default="Spatial_Block_East",
                        help="Split/fold directory name under each variant root "
                             "(``<variant>/<split>/seed_<n>/``). Defaults to the "
                             "east/west spatial-block holdout.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the data root recorded in each seed's config_resolved.yaml.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path,
                        default=Path("outputs/ablation/ensemble_inference.json"))
    parser.add_argument("--prob-cache", type=Path, default=None,
                        help="Optional path to dump per-seed + ensemble probability tensors (.pt).")
    parser.add_argument("--data-config", type=Path, default=None,
                        help="Optional YAML whose data: section replaces every seed's resolved data "
                             "config (transfer evaluation on a corpus the models were not trained on). "
                             "Disables the replay-vs-stored check.")
    args = parser.parse_args()

    data_override: dict | None = None
    if args.data_config is not None:
        override_doc = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
        if not isinstance(override_doc, dict) or "data" not in override_doc:
            raise ValueError(f"--data-config file must contain a top-level 'data:' section: {args.data_config}")
        data_override = override_doc["data"]

    first_variant = args.variant_roots[0]
    first_seed_dir = next((first_variant / args.split / f"seed_{s:02d}" for s in args.seeds
                           if (first_variant / args.split / f"seed_{s:02d}").is_dir()), None)
    if first_seed_dir is None:
        raise FileNotFoundError(f"No usable seed under {first_variant / args.split}")
    first_cfg = load_resolved_config(first_seed_dir)
    resolved_device = _resolve_device(
        device=args.device if args.device != "auto" else first_cfg["runtime"].get("device", "auto"))
    print(f"Device: {resolved_device}")
    print(f"Variants: {[p.name for p in args.variant_roots]}")
    print(f"Seeds requested: {args.seeds}")
    print(f"Split: {args.split}")

    set_seeds(seed=first_cfg["runtime"].get("seed", args.seeds[0]))

    variant_results: list[dict] = []
    for variant_root in args.variant_roots:
        result = run_variant(
            variant_root=variant_root,
            seeds=args.seeds,
            split_name=args.split,
            data_dir=args.data_dir,
            device=resolved_device,
            data_override=data_override,
        )
        variant_results.append(result)

    cross_results = compute_cross_variant_ensemble(variant_results=variant_results)
    print_summary_table(variant_results=variant_results, cross_results=cross_results)

    holdout_event = variant_results[0]["holdout_event"]
    write_artifact(
        path=args.output_json,
        variant_results=variant_results,
        cross_results=cross_results,
        split_name=args.split,
        holdout_event=holdout_event,
    )

    if args.prob_cache is not None:
        write_prob_cache(
            path=args.prob_cache,
            variant_results=variant_results,
            split_name=args.split,
            holdout_event=holdout_event,
        )


if __name__ == "__main__":
    main()
