"""Per-event decomposition of a multi-event holdout column from a cached probability tensor.

The dataset-default DROIDs split pools four test events (Michael, Idalia, Mussett Bayou,
Mayfield) into one column. Reviewers asked for evaluation on more events; this script
recovers the per-event view without any inference by rebuilding the evaluation loader's
instance sequence (verified against the cached target vector), mapping each instance's
source mosaic to its event through ``data/statistics.csv``, and scoring the cached
ensemble and per-seed probabilities on each event's subset under every decision rule.

Idalia and Mussett Bayou are class-degenerate or tiny as standalone LOEO holdouts, so
this is the only per-event view of them the protocol admits; their subset metrics are
reported with support so readers can weigh them accordingly.

Usage::

    python -m scripts.per_event_metrics \
        --variant-root outputs/ablation/all_features \
        --split "Hurricane_Idalia+Hurricane_Michael+Mayfield_Tornado+Mussett_Bayou_Fire" \
        --probs-cache "outputs/ablation/_ensembles/all_features__<split>_probs.pt" \
        --data-dir data \
        --output-json outputs/ablation/_ensembles/all_features__<split>_per_event.json
"""

import argparse
import json
import statistics

import torch

import pandas as pd

from pathlib import Path

from src.postproc.ensemble import build_val_loader_from_config, load_resolved_config, metrics_under_all_rules, serialize_for_json


def rebuild_instance_events(variant_root: Path, split: str, data_dir: str | None, statistics_csv: Path) -> tuple[list[str], torch.Tensor]:
    """Rebuild the evaluation loader's instance order and label each instance with its event.

    Args:
        variant_root: ``outputs/ablation/<variant>`` directory holding the seed folds.
        split: Split directory name under the variant root.
        data_dir: Optional data-root override passed to the loader builder.
        statistics_csv: CRASAR-U-DROIDs ``statistics.csv`` (``Orthomosaic`` -> ``Event``).

    Returns:
        Tuple of (event name per instance, label tensor per instance in loader order).
    """

    seed_dir = next(p for p in sorted((variant_root / split).glob("seed_*")) if p.is_dir())
    cfg = load_resolved_config(seed_dir)
    val_loader, _holdout = build_val_loader_from_config(cfg=cfg, data_dir_override=data_dir)
    dataset = val_loader.dataset

    mosaic_to_event = dict(zip(*(pd.read_csv(statistics_csv)[c] for c in ["Orthomosaic", "Event"]), strict=True))
    events = [mosaic_to_event[Path(instance["image_path"]).name] for instance in dataset.instances]
    labels = torch.tensor([int(instance["label_tensor"].item()) for instance in dataset.instances])
    return events, labels


def summarize_seeds(values: list[float]) -> dict[str, float]:
    """Mean and sample standard deviation across seeds (std 0 for a single seed)."""

    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def main() -> None:
    """CLI entry point: score each event subset and write the artifact."""

    parser = argparse.ArgumentParser(description="Per-event decomposition of a multi-event holdout column.")
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--probs-cache", type=Path, required=True, help="Probability cache written by src.postproc.ensemble.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--statistics-csv", type=Path, default=Path("data/statistics.csv"))
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.probs_cache, weights_only=False)
    targets: torch.Tensor = payload["targets"]
    variant = next(v for v in payload["variants"] if Path(v["variant_root"]).name == args.variant_root.name)
    ensemble_probs: torch.Tensor = variant["ensemble_probs"]
    individual = variant["individual_probs"]
    seed_probs = list(individual) if not isinstance(individual, dict) else list(individual.values())

    events, rebuilt = rebuild_instance_events(variant_root=args.variant_root, split=args.split,
                                              data_dir=args.data_dir, statistics_csv=args.statistics_csv)
    if len(events) != len(targets) or not torch.equal(rebuilt, targets):
        raise RuntimeError("Rebuilt instance sequence does not reproduce the cached targets; loader order changed.")
    print(f"instance order verified: {len(events)} buildings across {len(set(events))} events")

    # Score the pooled column first so the per-event rows have their reference in the same artifact.
    results: dict[str, dict] = {"__pooled__": {
        "n": len(targets),
        "ensemble": serialize_for_json(metrics_under_all_rules(probs=ensemble_probs, targets=targets)),
    }}
    for event in sorted(set(events)):
        idx = torch.tensor([i for i, e in enumerate(events) if e == event])
        sub_targets = targets[idx]
        support = torch.bincount(sub_targets, minlength=4).tolist()
        ensemble_metrics = metrics_under_all_rules(probs=ensemble_probs[idx], targets=sub_targets)
        per_seed = [metrics_under_all_rules(probs=p[idx], targets=sub_targets)["argmax"] for p in seed_probs]
        results[event] = {
            "n": len(idx),
            "class_support": support,
            "ensemble": serialize_for_json(ensemble_metrics),
            "single_seed_argmax": {
                "qwk": summarize_seeds([m["qwk"] for m in per_seed]),
                "macro_f1": summarize_seeds([m["macro_f1"] for m in per_seed]),
                "accuracy": summarize_seeds([m["accuracy"] for m in per_seed]),
            },
        }

    print(f"\n{'event':<28}{'n':>6}  {'support (No/Min/Maj/Des)':<26}{'QWK':>8}{'F1':>8}{'acc':>8}   single-seed QWK / F1 (mean ± std)")
    for name, r in results.items():
        m = r["ensemble"]["argmax"]
        label = "pooled column" if name == "__pooled__" else name
        support = "/".join(str(s) for s in r.get("class_support", [])) if name != "__pooled__" else ""
        seed_txt = ""
        if "single_seed_argmax" in r:
            s = r["single_seed_argmax"]
            seed_txt = f"{s['qwk']['mean']:.3f} ± {s['qwk']['std']:.3f} / {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f}"
        print(f"{label:<28}{r['n']:>6}  {support:<26}{m['qwk']:>8.4f}{m['macro_f1']:>8.4f}{m['accuracy']:>8.4f}   {seed_txt}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "variant_root": str(args.variant_root),
        "split": args.split,
        "probs_cache": str(args.probs_cache),
        "rule_note": "ensemble metrics under all rules; single-seed summaries under the argmax headline rule",
        "events": results,
    }, indent=2), encoding="utf-8")
    print(f"\nartifact written: {args.output_json}")


if __name__ == "__main__":
    main()
