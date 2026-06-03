"""Probe EV-rounding vs argmax vs hybrid prediction rules on trained checkpoints.

Loads each seed's ``best_model.pt`` under a variant root, runs the same 4x
rotation TTA used at training time, then computes QWK and per-class recall
under three prediction rules applied to the identical averaged probabilities:

    - EV rounding: ``floor(sum_i i * p(i) + 0.5)`` (current default in trainer)
    - Argmax:      ``argmax_i p(i)`` (standard categorical rule)
    - Hybrid:      argmax where argmax predicts an extreme class (0 or K-1),
                   EV rounding otherwise. Motivation: EV rounding is a central
                   estimator biased toward mid-classes, which dampens confident
                   extreme-class probability mass. Argmax recovers that mass
                   but over-commits on middle-class samples (Minor/Major) whose
                   distributions are genuinely bimodal, producing non-adjacent
                   QWK-costly errors. The hybrid rule trusts argmax only when
                   it lands on an extreme — where the distribution is
                   already mass-concentrated — and falls back to EV for
                   mid-class predictions.

Emits a per-seed comparison table (including replay sanity check against the
stored ``best_val_qwk``) and a summary across seeds. A JSON artifact is written
next to the variant root under ``prediction_rule_probe.json``.

This script does not train anything, does not mutate any existing files, and
only touches data loaders / model forward passes. Run from the repo root.

Status: diagnostic-only. The findings produced here motivated the canonical
choice of the hybrid rule at ensemble time (see ``src/postproc/ensemble.py``
and ``doc/4-modeling.qmd`` §Inference pipeline). Retained as a reproducibility
artifact and for per-seed probing under new training configurations.

Usage:
    uv run python -m scripts.probe_prediction_rule \
        --variant-root outputs/ablation/baseline \
        --seeds 11 22 33 44 55 \
        --data-dir data \
        --device auto
"""

import argparse
import json
import time
import torch
import yaml

import torch.nn.functional as F

from collections.abc import Callable
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import to_normalized_float
from src.data.sampling import (
    build_spatial_split_manifest,
    build_valid_manifest,
    generate_loeo_splits,
    select_fold,
)
from src.model.mcdn import MaskConditionedDamageNet
from src.model.trainer import (
    ORDINAL_CLASS_DISPLAY_NAMES,
    _build_dataloaders,
    _resolve_device,
    _validation_qwk_and_classification,
    set_seeds,
)


DEFAULT_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
REPLAY_TOLERANCE: float = 5e-3


def load_resolved_config(fold_dir: Path) -> dict:
    """Parse a fold's ``config_resolved.yaml`` into a dict."""

    config_path = fold_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_stored_metrics(fold_dir: Path) -> dict:
    """Parse a fold's ``metrics.json`` into a dict."""

    metrics_path = fold_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_model_from_config(cfg: dict, device: str) -> MaskConditionedDamageNet:
    """Construct the model architecture matching a resolved-config snapshot."""

    ablation = cfg["ablation"]
    model_cfg = cfg["model"]
    model = MaskConditionedDamageNet(
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


def build_val_loader_from_config(cfg: dict, data_dir_override: str | None = None) -> tuple[DataLoader, str]:
    """Reconstruct the validation loader that produced the stored checkpoint.

    Returns:
        ``(val_loader, holdout_event)`` so the caller can log the fold.
    """

    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]
    training_cfg = cfg["training"]

    data_dir = data_dir_override if data_dir_override is not None else data_cfg["dir"]
    sensor_profile = data_cfg["sensor_profile"]

    valid_manifest = build_valid_manifest(data_dir=data_dir, sensor_profile=sensor_profile)

    # Honor the same manifest/fold-resolution path used at training time. When the run used
    # `holdout_selection: default_spatial`, the CLI passed no explicit holdout and the trainer
    # built the spatial-split manifest internally; replay must mirror that so fold names match.
    holdout_selection = cfg.get("metadata", {}).get("holdout_selection") or data_cfg.get("holdout_selection", "explicit")
    if holdout_selection == "default_spatial":
        split_manifest = build_spatial_split_manifest(valid_manifest)
        holdout_event_arg: str | None = None
    else:
        split_manifest = valid_manifest
        holdout_event_arg = data_cfg.get("holdout_event")

    splits = list(generate_loeo_splits(split_manifest))
    holdout, train_df, val_df, _selection = select_fold(splits=splits, holdout_event=holdout_event_arg)

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
    )
    return val_loader, holdout


@torch.no_grad()
def collect_averaged_probabilities(model: MaskConditionedDamageNet, val_loader: DataLoader,
                                    device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 4x rotation TTA and return softmax-averaged probabilities + targets.

    Matches the trainer's ``_validate`` pathway at ``src/model/trainer.py:495-522``.
    """

    probs_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []

    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch in val_loader:
        images = to_normalized_float(batch["image"].to(device, non_blocking=True))
        context = batch["context"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(autocast_device):
            logits_0 = model(images, context)
            logits_90 = model(torch.rot90(images, k=1, dims=[2, 3]), context)
            logits_180 = model(torch.rot90(images, k=2, dims=[2, 3]), context)
            logits_270 = model(torch.rot90(images, k=3, dims=[2, 3]), context)

            probs_0 = F.softmax(logits_0.float(), dim=1)
            probs_90 = F.softmax(logits_90.float(), dim=1)
            probs_180 = F.softmax(logits_180.float(), dim=1)
            probs_270 = F.softmax(logits_270.float(), dim=1)

            avg_probs = (probs_0 + probs_90 + probs_180 + probs_270) / 4.0

        probs_chunks.append(avg_probs.cpu())
        targets_chunks.append(labels.cpu())

    return torch.cat(probs_chunks, dim=0), torch.cat(targets_chunks, dim=0)


def apply_ev_rounding(probs: torch.Tensor) -> torch.Tensor:
    """Expected-value rounding: ``round(sum_i i * p(i))``."""

    num_classes = probs.shape[1]
    classes = torch.arange(num_classes, dtype=torch.float32)
    expected_value = torch.sum(probs * classes, dim=1)
    return torch.floor(expected_value + 0.5).long()


def apply_argmax(probs: torch.Tensor) -> torch.Tensor:
    """Standard categorical argmax."""

    return torch.argmax(probs, dim=1)


def apply_hybrid(probs: torch.Tensor) -> torch.Tensor:
    """Argmax where argmax predicts an extreme class; EV rounding otherwise.

    Extreme classes are the endpoints of the ordinal scale: ``0`` and ``K-1``.
    When argmax commits to an endpoint, the probability distribution is
    already mass-concentrated there and EV rounding's central-estimator bias
    would dampen the signal; we trust argmax. For interior classes (1..K-2)
    the EV rounding inherits its adjacency-smoothing benefit for QWK under
    bimodal mid-class distributions.
    """

    ev_preds = apply_ev_rounding(probs=probs)
    argmax_preds = apply_argmax(probs=probs)
    num_classes = probs.shape[1]
    extreme_mask = (argmax_preds == 0) | (argmax_preds == (num_classes - 1))
    return torch.where(extreme_mask, argmax_preds, ev_preds)


def compute_rule_metrics(probs: torch.Tensor, targets: torch.Tensor,
                          rule_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict:
    """Compute QWK and classification metrics under a given prediction rule."""

    preds = rule_fn(probs)
    qwk, class_metrics = _validation_qwk_and_classification(preds=preds, targets=targets)
    return {
        "qwk": float(qwk),
        "accuracy": class_metrics["accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "per_class_recall": {
            name: class_metrics["per_class"][name]["recall"]
            for name in ORDINAL_CLASS_DISPLAY_NAMES
        },
        "confusion_matrix": class_metrics["confusion_matrix"],
    }


def format_per_class_row(rule_label: str, per_class: dict[str, float]) -> str:
    """Render a labelled row of per-class metrics for terminal output."""

    parts = [f"  {rule_label:8s}"]
    parts.extend(f"{name}={per_class[name]:.4f}" for name in ORDINAL_CLASS_DISPLAY_NAMES)
    return "  ".join(parts)


def format_confusion(matrix: list[list[int]]) -> str:
    """Format a confusion matrix as a multi-line aligned text block."""

    header = "             " + "  ".join(f"{name[:10]:>10}" for name in ORDINAL_CLASS_DISPLAY_NAMES)
    lines = [header]
    for row_idx, row in enumerate(matrix):
        row_str = "  ".join(f"{val:>10d}" for val in row)
        lines.append(f"  {ORDINAL_CLASS_DISPLAY_NAMES[row_idx]:>10s}  {row_str}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for probing EV vs argmax prediction rules on trained checkpoints."""

    parser = argparse.ArgumentParser(description="Probe EV vs argmax prediction rules on trained checkpoints.")
    parser.add_argument("--variant-root", type=Path, default=Path("outputs/ablation/baseline"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the data root recorded in each seed's config_resolved.yaml.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fold-glob", type=str, default="fold_Spatial_Block_Spatial_Block_East",
                        help="Specific fold subdirectory name under each seed_{N}.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Override output location; defaults to <variant-root>/prediction_rule_probe.json.")
    args = parser.parse_args()

    variant_root: Path = args.variant_root
    seeds: list[int] = args.seeds
    fold_name: str = args.fold_glob

    fold_dirs_by_seed: dict[int, Path] = {}
    for seed in seeds:
        candidate = variant_root / f"seed_{seed:02d}" / fold_name
        if not candidate.is_dir():
            raise FileNotFoundError(f"Expected fold directory not found for seed {seed}: {candidate}")
        fold_dirs_by_seed[seed] = candidate

    first_seed = seeds[0]
    first_cfg = load_resolved_config(fold_dirs_by_seed[first_seed])
    resolved_device = _resolve_device(device=args.device if args.device != "auto" else first_cfg["runtime"].get("device", "auto"))
    print(f"Device: {resolved_device}")
    print(f"Variant root: {variant_root}")
    print(f"Seeds: {seeds}")
    print(f"Fold: {fold_name}")

    set_seeds(seed=first_cfg["runtime"].get("seed", first_seed))

    print("\nBuilding validation loader (shared across seeds — identical fold/config expected)...")
    t0 = time.time()
    val_loader, holdout_event = build_val_loader_from_config(cfg=first_cfg, data_dir_override=args.data_dir)
    print(f"Validation loader ready in {time.time() - t0:.1f}s. Holdout: {holdout_event}. "
          f"Batches: {len(val_loader)}")

    per_seed_results: list[dict] = []
    replay_mismatches: list[dict] = []

    for seed in seeds:
        fold_dir = fold_dirs_by_seed[seed]
        cfg = load_resolved_config(fold_dir)
        stored = load_stored_metrics(fold_dir)
        stored_qwk = stored["best_val_qwk"]

        print(f"\n--- seed {seed} ---")
        print(f"  Stored best_val_qwk: {stored_qwk:.6f}")

        model = build_model_from_config(cfg=cfg, device=resolved_device)
        checkpoint_path = fold_dir / "best_model.pt"
        state_dict = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            print(f"  WARNING: state_dict keys mismatch (missing={len(missing)}, unexpected={len(unexpected)})")

        t0 = time.time()
        probs, targets = collect_averaged_probabilities(model=model, val_loader=val_loader, device=resolved_device)
        inference_seconds = time.time() - t0
        print(f"  Inference: {inference_seconds:.1f}s  samples={probs.shape[0]}")

        ev_metrics = compute_rule_metrics(probs=probs, targets=targets, rule_fn=apply_ev_rounding)
        argmax_metrics = compute_rule_metrics(probs=probs, targets=targets, rule_fn=apply_argmax)
        hybrid_metrics = compute_rule_metrics(probs=probs, targets=targets, rule_fn=apply_hybrid)

        # Rule-agreement diagnostics: how often does hybrid defer to argmax vs to EV, and
        # when it uses argmax, how often does that change the prediction vs EV?
        ev_preds = apply_ev_rounding(probs=probs)
        argmax_preds = apply_argmax(probs=probs)
        hybrid_preds = apply_hybrid(probs=probs)
        num_classes = int(probs.shape[1])
        extreme_mask = (argmax_preds == 0) | (argmax_preds == (num_classes - 1))
        hybrid_used_argmax = int(extreme_mask.sum().item())
        hybrid_diff_from_ev = int((hybrid_preds != ev_preds).sum().item())
        hybrid_diff_from_argmax = int((hybrid_preds != argmax_preds).sum().item())

        replay_delta = ev_metrics["qwk"] - stored_qwk
        replay_ok = abs(replay_delta) <= REPLAY_TOLERANCE
        tag = "OK" if replay_ok else "!!"
        print(f"  Replay EV QWK:       {ev_metrics['qwk']:.6f}  (delta={replay_delta:+.6f}) [{tag}]")
        if not replay_ok:
            replay_mismatches.append({
                "seed": seed,
                "stored": stored_qwk,
                "replayed": ev_metrics["qwk"],
                "delta": replay_delta,
            })

        print(f"  Argmax QWK:          {argmax_metrics['qwk']:.6f}  "
              f"(delta vs EV: {argmax_metrics['qwk'] - ev_metrics['qwk']:+.6f})")
        print(f"  Hybrid QWK:          {hybrid_metrics['qwk']:.6f}  "
              f"(delta vs EV: {hybrid_metrics['qwk'] - ev_metrics['qwk']:+.6f})")
        total = int(probs.shape[0])
        print(f"  Hybrid used argmax on {hybrid_used_argmax}/{total} samples "
              f"({hybrid_used_argmax / total * 100:.1f}%); differed from EV on "
              f"{hybrid_diff_from_ev} samples.")
        print(format_per_class_row("EV", ev_metrics["per_class_recall"]))
        print(format_per_class_row("argmax", argmax_metrics["per_class_recall"]))
        print(format_per_class_row("hybrid", hybrid_metrics["per_class_recall"]))

        per_seed_results.append({
            "seed": seed,
            "stored_qwk": stored_qwk,
            "replay_ev": ev_metrics,
            "argmax": argmax_metrics,
            "hybrid": hybrid_metrics,
            "replay_ok": replay_ok,
            "replay_delta_vs_stored": replay_delta,
            "argmax_minus_ev_qwk": argmax_metrics["qwk"] - ev_metrics["qwk"],
            "hybrid_minus_ev_qwk": hybrid_metrics["qwk"] - ev_metrics["qwk"],
            "hybrid_used_argmax": hybrid_used_argmax,
            "hybrid_diff_from_ev": hybrid_diff_from_ev,
            "hybrid_diff_from_argmax": hybrid_diff_from_argmax,
            "total_samples": total,
        })

        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ev_qwks = [r["replay_ev"]["qwk"] for r in per_seed_results]
    argmax_qwks = [r["argmax"]["qwk"] for r in per_seed_results]
    hybrid_qwks = [r["hybrid"]["qwk"] for r in per_seed_results]
    argmax_deltas = [a - e for a, e in zip(argmax_qwks, ev_qwks, strict=True)]
    hybrid_deltas = [h - e for h, e in zip(hybrid_qwks, ev_qwks, strict=True)]
    hybrid_vs_argmax = [h - a for h, a in zip(hybrid_qwks, argmax_qwks, strict=True)]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mu = mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    summary = {
        "variant_root": str(variant_root),
        "fold": fold_name,
        "holdout_event": holdout_event,
        "seeds": seeds,
        "per_seed": per_seed_results,
        "aggregate": {
            "ev_qwk_mean": mean(ev_qwks),
            "ev_qwk_std": std(ev_qwks),
            "argmax_qwk_mean": mean(argmax_qwks),
            "argmax_qwk_std": std(argmax_qwks),
            "hybrid_qwk_mean": mean(hybrid_qwks),
            "hybrid_qwk_std": std(hybrid_qwks),
            "argmax_minus_ev_mean": mean(argmax_deltas),
            "argmax_minus_ev_std": std(argmax_deltas),
            "argmax_minus_ev_min": min(argmax_deltas) if argmax_deltas else 0.0,
            "argmax_minus_ev_max": max(argmax_deltas) if argmax_deltas else 0.0,
            "hybrid_minus_ev_mean": mean(hybrid_deltas),
            "hybrid_minus_ev_std": std(hybrid_deltas),
            "hybrid_minus_ev_min": min(hybrid_deltas) if hybrid_deltas else 0.0,
            "hybrid_minus_ev_max": max(hybrid_deltas) if hybrid_deltas else 0.0,
            "hybrid_minus_argmax_mean": mean(hybrid_vs_argmax),
            "num_seeds_argmax_beats_ev": sum(1 for d in argmax_deltas if d > 0),
            "num_seeds_hybrid_beats_ev": sum(1 for d in hybrid_deltas if d > 0),
            "num_seeds_hybrid_beats_argmax": sum(1 for d in hybrid_vs_argmax if d > 0),
        },
        "replay_mismatches": replay_mismatches,
    }

    print("\n========== AGGREGATE ==========")
    agg = summary["aggregate"]
    print(f"  EV-rounding QWK:  mean={agg['ev_qwk_mean']:.6f}  std={agg['ev_qwk_std']:.6f}")
    print(f"  Argmax     QWK:  mean={agg['argmax_qwk_mean']:.6f}  std={agg['argmax_qwk_std']:.6f}")
    print(f"  Hybrid     QWK:  mean={agg['hybrid_qwk_mean']:.6f}  std={agg['hybrid_qwk_std']:.6f}")
    print(f"  Delta (argmax - EV): mean={agg['argmax_minus_ev_mean']:+.6f}  "
          f"std={agg['argmax_minus_ev_std']:.6f}  "
          f"min={agg['argmax_minus_ev_min']:+.6f}  max={agg['argmax_minus_ev_max']:+.6f}")
    print(f"  Delta (hybrid - EV): mean={agg['hybrid_minus_ev_mean']:+.6f}  "
          f"std={agg['hybrid_minus_ev_std']:.6f}  "
          f"min={agg['hybrid_minus_ev_min']:+.6f}  max={agg['hybrid_minus_ev_max']:+.6f}")
    print(f"  Delta (hybrid - argmax): mean={agg['hybrid_minus_argmax_mean']:+.6f}")
    print(f"  Argmax beats EV on {agg['num_seeds_argmax_beats_ev']}/{len(seeds)} seeds.")
    print(f"  Hybrid beats EV on {agg['num_seeds_hybrid_beats_ev']}/{len(seeds)} seeds.")
    print(f"  Hybrid beats argmax on {agg['num_seeds_hybrid_beats_argmax']}/{len(seeds)} seeds.")

    print("\n========== PER-CLASS RECALL (MEAN ACROSS SEEDS) ==========")
    for rule_key in ("replay_ev", "argmax", "hybrid"):
        rule_label = {"replay_ev": "EV", "argmax": "argmax", "hybrid": "hybrid"}[rule_key]
        recalls = {name: [] for name in ORDINAL_CLASS_DISPLAY_NAMES}
        for r in per_seed_results:
            for name in ORDINAL_CLASS_DISPLAY_NAMES:
                recalls[name].append(r[rule_key]["per_class_recall"][name])
        cells = [f"{name}={mean(recalls[name]):.4f}" for name in ORDINAL_CLASS_DISPLAY_NAMES]
        print(f"  {rule_label:8s}  " + "  ".join(cells))

    if replay_mismatches:
        print("\nWARNING: replay mismatches above REPLAY_TOLERANCE threshold:")
        for entry in replay_mismatches:
            print(f"  seed {entry['seed']}: stored={entry['stored']:.6f}  "
                  f"replayed={entry['replayed']:.6f}  delta={entry['delta']:+.6f}")

    output_path = args.output_json if args.output_json is not None else variant_root / "prediction_rule_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nArtifact written: {output_path}")


if __name__ == "__main__":
    main()
