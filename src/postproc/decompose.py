"""Seed x TTA decomposition of the ensemble inference configuration.

Emits the 2x2 table — (single seed vs. multi-seed ensemble) x (no-TTA identity view vs.
8-view D4 TTA) — per variant and holdout, quantifying how much of the reported ensemble
headline comes from seed averaging versus test-time augmentation. Single-seed cells
report mean +/- std across the seed pool, matching the manuscript's per-seed dispersion
convention; ensemble cells are point values of the pooled softmax.

Reuses the checkpoint / val-loader plumbing from ``src.postproc.ensemble`` but retains
the per-view softmax axis that the standard ensemble path averages away, so the no-TTA
cells come from the same forward passes as the TTA cells rather than a separate run.

Outputs:
    - printed 2x2 table per variant (all three prediction rules) with gain lines
    - JSON artifact at ``outputs/ablation/decomposition.json`` (overridable)
    - optional view-resolved probability cache ([seeds, views, samples, classes])
      via ``--view-cache``; reuse it later with ``--from-view-cache`` to recompute
      tables without any model execution

Usage:
    uv run python -m src.postproc.decompose \
        --variant-roots outputs/ablation/baseline \
        --split Spatial_Block_East \
        --view-cache outputs/ablation/baseline/Spatial_Block_East/view_probs.pt
"""

import argparse
import json
import statistics
import time

import torch

from pathlib import Path
from torch.utils.data import DataLoader

from src.model.trainer import ORDINAL_CLASS_DISPLAY_NAMES, _resolve_device
from src.postproc.ensemble import (
    D4_VIEW_ORDER,
    DEFAULT_SEEDS,
    IDENTITY_VIEW_INDEX,
    RULE_FNS,
    build_model_from_config,
    build_val_loader_from_config,
    load_resolved_config,
    metrics_under_all_rules,
    tta_view_softmax_probs,
)


# Headline metrics summarized with mean/std across seeds and diffed in the gain lines.
SUMMARY_METRIC_KEYS = ("qwk", "macro_f1", "accuracy", "macro_precision", "macro_recall")
DELTA_METRIC_KEYS = ("qwk", "macro_f1", "accuracy")


@torch.no_grad()
def collect_view_probabilities(model: torch.nn.Module, val_loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 8x D4 TTA over a val loader preserving the view axis.

    Returns:
        Tuple of per-view softmax probabilities [V, T, K] and integer targets [T],
        where V follows ``D4_VIEW_ORDER`` and T is the full validation set.
    """

    view_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for batch in val_loader:
        view_probs = tta_view_softmax_probs(model=model, images_u8=batch["image"], context=batch["context"], device=device)
        view_chunks.append(view_probs.cpu())
        target_chunks.append(batch["label"].cpu())
    return torch.cat(view_chunks, dim=1), torch.cat(target_chunks, dim=0)


def cell_probability_tensors(view_probs: torch.Tensor, identity_view_index: int = IDENTITY_VIEW_INDEX) -> dict[str, torch.Tensor]:
    """Build the four 2x2 cell probability tensors from a view-resolved stack.

    Args:
        view_probs: Per-seed, per-view softmax stack [S, V, T, K].
        identity_view_index: Index of the untransformed view along V.

    Returns:
        Mapping with ``single_seed_no_tta`` / ``single_seed_tta`` of shape [S, T, K]
        and ``ensemble_no_tta`` / ``ensemble_tta`` of shape [T, K]. Both ensembling
        and TTA are plain softmax means, so the reduction order is immaterial.
    """

    single_no_tta = view_probs[:, identity_view_index]  # [S, V, T, K] -> [S, T, K]
    single_tta = view_probs.mean(dim=1)                 # [S, V, T, K] -> [S, T, K]
    return {
        "single_seed_no_tta": single_no_tta,
        "single_seed_tta": single_tta,
        "ensemble_no_tta": single_no_tta.mean(dim=0),
        "ensemble_tta": single_tta.mean(dim=0)
    }


def _summarize_across_seeds(per_seed_metrics: list[dict[str, dict]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean/std across seeds for every rule and headline metric.

    Std follows the repo convention (sample std, 0.0 for a single seed), matching
    ``train._aggregate_ablation_metrics``.
    """

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for rule_name in RULE_FNS:
        summary[rule_name] = {}
        for metric_key in SUMMARY_METRIC_KEYS:
            values = [seed_metrics[rule_name][metric_key] for seed_metrics in per_seed_metrics]
            summary[rule_name][metric_key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0
            }
    return summary


def _compute_deltas(single_no_tta_summary: dict, single_tta_summary: dict,
                    ensemble_no_tta: dict, ensemble_tta: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Pairwise gains between the 2x2 cells, per rule and headline metric.

    Single-seed cells enter as their across-seed means, so each gain reads as "what
    the average seed would gain" rather than being anchored to one lucky seed.
    """

    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for rule_name in RULE_FNS:
        deltas[rule_name] = {}
        for metric_key in DELTA_METRIC_KEYS:
            single_no = single_no_tta_summary[rule_name][metric_key]["mean"]
            single_tta = single_tta_summary[rule_name][metric_key]["mean"]
            ens_no = ensemble_no_tta[rule_name][metric_key]
            ens_tta = ensemble_tta[rule_name][metric_key]
            deltas[rule_name][metric_key] = {
                "tta_gain_single_seed": single_tta - single_no,
                "tta_gain_ensemble": ens_tta - ens_no,
                "ensemble_gain_no_tta": ens_no - single_no,
                "ensemble_gain_tta": ens_tta - single_tta,
                "total_gain": ens_tta - single_no
            }
    return deltas


def decompose_variant(view_probs: torch.Tensor, targets: torch.Tensor, seeds: list[int],
                      identity_view_index: int = IDENTITY_VIEW_INDEX) -> dict:
    """Compute the full 2x2 (seed x TTA) decomposition from a view-resolved stack.

    Args:
        view_probs: Per-seed, per-view softmax stack [S, V, T, K].
        targets: Integer targets [T] shared by every seed (alignment is the caller's
            contract, verified during collection).
        seeds: Seed values labeling axis S, in stack order.
        identity_view_index: Index of the untransformed view along V.

    Returns:
        JSON-safe dict with per-seed metrics and mean/std summaries for the two
        single-seed cells, point metrics for the two ensemble cells, and pairwise
        gains between cells per prediction rule.
    """

    cells = cell_probability_tensors(view_probs=view_probs, identity_view_index=identity_view_index)

    per_seed_no_tta = [
        {"seed": seed, "metrics": metrics_under_all_rules(probs=cells["single_seed_no_tta"][idx], targets=targets)}
        for idx, seed in enumerate(seeds)
    ]
    per_seed_tta = [
        {"seed": seed, "metrics": metrics_under_all_rules(probs=cells["single_seed_tta"][idx], targets=targets)}
        for idx, seed in enumerate(seeds)
    ]
    single_no_tta_summary = _summarize_across_seeds([record["metrics"] for record in per_seed_no_tta])
    single_tta_summary = _summarize_across_seeds([record["metrics"] for record in per_seed_tta])
    ensemble_no_tta = metrics_under_all_rules(probs=cells["ensemble_no_tta"], targets=targets)
    ensemble_tta = metrics_under_all_rules(probs=cells["ensemble_tta"], targets=targets)

    return {
        "seeds": list(seeds),
        "num_views": int(view_probs.shape[1]),
        "identity_view_index": identity_view_index,
        "cells": {
            "single_seed_no_tta": {"per_seed": per_seed_no_tta, "summary": single_no_tta_summary},
            "single_seed_tta": {"per_seed": per_seed_tta, "summary": single_tta_summary},
            "ensemble_no_tta": ensemble_no_tta,
            "ensemble_tta": ensemble_tta
        },
        "deltas": _compute_deltas(
            single_no_tta_summary=single_no_tta_summary,
            single_tta_summary=single_tta_summary,
            ensemble_no_tta=ensemble_no_tta,
            ensemble_tta=ensemble_tta
        )
    }


# Orchestration -------------------------------------------------------------

def run_variant_decomposition(variant_root: Path, seeds: list[int], split_name: str,
                              data_dir: str | None, device: str) -> dict:
    """Collect view-resolved probabilities for every seed of one variant and decompose.

    Mirrors ``ensemble.run_variant`` layout conventions
    (``<variant_root>/<split_name>/seed_<n>/``) and its target-alignment contract.
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

    t0 = time.time()
    val_loader, holdout_event = build_val_loader_from_config(cfg=first_cfg, data_dir_override=data_dir)
    print(f"  val loader ready ({time.time() - t0:.1f}s)  holdout={holdout_event}  batches={len(val_loader)}")

    seed_stacks: list[torch.Tensor] = []
    reference_targets: torch.Tensor | None = None
    ordered_seeds = sorted(fold_dirs.keys())

    for seed in ordered_seeds:
        fold_dir = fold_dirs[seed]
        cfg = load_resolved_config(fold_dir)
        model = build_model_from_config(cfg=cfg, device=device)
        state_dict = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        ti = time.time()
        view_probs, targets = collect_view_probabilities(model=model, val_loader=val_loader, device=device)
        print(f"  seed {seed}: collected view stack {tuple(view_probs.shape)}  ({time.time() - ti:.1f}s)")

        if reference_targets is None:
            reference_targets = targets
        elif not torch.equal(reference_targets, targets):
            raise RuntimeError(
                f"Target mismatch between seed {first_seed} and seed {seed} in variant "
                f"{variant_root.name}; val loader produced different sample ordering."
            )

        seed_stacks.append(view_probs)
        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked = torch.stack(seed_stacks, dim=0)  # [S, V, T, K]
    decomposition = decompose_variant(view_probs=stacked, targets=reference_targets, seeds=ordered_seeds)

    return {
        "variant_root": str(variant_root),
        "holdout_event": holdout_event,
        "seeds": ordered_seeds,
        "view_probs": stacked,
        "targets": reference_targets,
        "decomposition": decomposition
    }


# Reporting -----------------------------------------------------------------

def print_decomposition_table(record: dict) -> None:
    """Print the 2x2 table (all rules) with mean +/- std single-seed rows and gain lines."""

    decomposition = record["decomposition"]
    num_seeds = len(decomposition["seeds"])
    cells = decomposition["cells"]

    print(f"\n{'=' * 100}")
    print(f"SEED x TTA DECOMPOSITION  variant={Path(record['variant_root']).name}  "
          f"holdout={record['holdout_event']}  seeds={num_seeds}  views={decomposition['num_views']}")
    print("=" * 100)

    for rule_name in RULE_FNS:
        print(f"\n  rule={rule_name}")
        print(f"    {'configuration':<26s}  {'QWK':>18s}  {'macro-F1':>18s}  {'accuracy':>18s}")

        for cell_key, label in (("single_seed_no_tta", "single seed / no TTA"),
                                ("single_seed_tta", "single seed / 8-view TTA")):
            summary = cells[cell_key]["summary"][rule_name]
            values = "  ".join(
                f"{summary[metric]['mean']:>8.4f} +/- {summary[metric]['std']:.4f}"
                for metric in ("qwk", "macro_f1", "accuracy")
            )
            print(f"    {label:<26s}  {values}")

        for cell_key, label in (("ensemble_no_tta", f"{num_seeds}-seed ens / no TTA"),
                                ("ensemble_tta", f"{num_seeds}-seed ens / 8-view TTA")):
            metrics = cells[cell_key][rule_name]
            values = "  ".join(f"{metrics[metric]:>8.4f}           " for metric in ("qwk", "macro_f1", "accuracy"))
            print(f"    {label:<26s}  {values}")

        gains = decomposition["deltas"][rule_name]["qwk"]
        print(f"    QWK gains: TTA@single {gains['tta_gain_single_seed']:+.4f}  "
              f"ensemble@TTA {gains['ensemble_gain_tta']:+.4f}  total {gains['total_gain']:+.4f}")


def write_decomposition_artifact(path: Path, records: list[dict], split_name: str) -> None:
    """Write one JSON artifact covering every variant's 2x2 decomposition."""

    payload = {
        "split": split_name,
        "holdout_event": records[0]["holdout_event"],
        "view_order": list(D4_VIEW_ORDER),
        "class_names": list(ORDINAL_CLASS_DISPLAY_NAMES),
        "variants": [
            {"variant_root": record["variant_root"], "decomposition": record["decomposition"]}
            for record in records
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nArtifact written: {path}")


def write_view_cache(path: Path, records: list[dict], split_name: str) -> None:
    """Serialize view-resolved probability stacks for model-free recomputation.

    Each variant payload holds ``view_probs`` [S, V, T, K]; reload with
    ``--from-view-cache`` to rebuild every table and artifact without checkpoints,
    dataset, or GPU.
    """

    payload = {
        "split": split_name,
        "holdout_event": records[0]["holdout_event"],
        "class_names": list(ORDINAL_CLASS_DISPLAY_NAMES),
        "view_order": list(D4_VIEW_ORDER),
        "identity_view_index": IDENTITY_VIEW_INDEX,
        "targets": records[0]["targets"],
        "variants": [
            {"variant_root": record["variant_root"], "seeds": record["seeds"], "view_probs": record["view_probs"]}
            for record in records
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"View cache written: {path}")


def load_view_cache(path: Path) -> dict:
    """Load a view-resolved probability cache written by ``write_view_cache``."""

    if not path.exists():
        raise FileNotFoundError(f"View cache not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    """CLI entry point for the seed x TTA decomposition."""

    parser = argparse.ArgumentParser(description="2x2 (seed x TTA) decomposition of ensemble inference.")
    parser.add_argument("--variant-roots", type=Path, nargs="+", default=None,
                        help="One or more ``outputs/ablation/<variant>`` directories (required unless --from-view-cache).")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to attempt per variant (missing ones are silently skipped).")
    parser.add_argument("--split", type=str, default="Spatial_Block_East",
                        help="Split/fold directory name under each variant root.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the data root recorded in each seed's config_resolved.yaml.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ablation/decomposition.json"))
    parser.add_argument("--view-cache", type=Path, default=None,
                        help="Optional path to dump the view-resolved probability stacks (.pt).")
    parser.add_argument("--from-view-cache", type=Path, default=None,
                        help="Recompute tables from a previously written view cache (no model execution).")
    args = parser.parse_args()

    records: list[dict] = []
    if args.from_view_cache is not None:
        payload = load_view_cache(args.from_view_cache)
        split_name = payload["split"]
        identity_view_index = payload.get("identity_view_index", IDENTITY_VIEW_INDEX)
        for entry in payload["variants"]:
            decomposition = decompose_variant(
                view_probs=entry["view_probs"],
                targets=payload["targets"],
                seeds=list(entry["seeds"]),
                identity_view_index=identity_view_index
            )
            records.append({
                "variant_root": entry["variant_root"],
                "holdout_event": payload["holdout_event"],
                "seeds": list(entry["seeds"]),
                "view_probs": entry["view_probs"],
                "targets": payload["targets"],
                "decomposition": decomposition
            })
    else:
        if not args.variant_roots:
            parser.error("--variant-roots is required unless --from-view-cache is given.")
        resolved_device = _resolve_device(device=args.device)
        print(f"Device: {resolved_device}")
        print(f"Variants: {[p.name for p in args.variant_roots]}")
        print(f"Split: {args.split}")
        split_name = args.split
        records = [
            run_variant_decomposition(
                variant_root=variant_root,
                seeds=args.seeds,
                split_name=split_name,
                data_dir=args.data_dir,
                device=resolved_device
            )
            for variant_root in args.variant_roots
        ]

    for record in records:
        print_decomposition_table(record)

    write_decomposition_artifact(path=args.output_json, records=records, split_name=split_name)

    if args.view_cache is not None:
        write_view_cache(path=args.view_cache, records=records, split_name=split_name)


if __name__ == "__main__":
    main()
