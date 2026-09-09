"""Mine the highest-confidence ensemble errors of a holdout column and render exemplar sheets.

Reviewers asked for a discussion of *why* the model fails, not only how often. This script
works entirely from a cached probability tensor (no inference): it rebuilds the evaluation
loader's instance sequence (verified against the cached targets), ranks every
misclassified building by ensemble confidence and ordinal severity, and attaches
diagnostics that separate systematic failures from seed variance and flag mechanical
causes automatically:

- ``seed_agreement``: how many of the pooled seeds vote with the ensemble's wrong answer
  (10/10 marks a systematic error, not an unlucky seed);
- ``nodata_frac``: fraction of chip pixels with no imagery (mosaic edge / nodata fill);
- ``mask_frac``: footprint area as a fraction of the chip;
- ``mask_at_border``: whether the rasterized footprint touches the chip boundary
  (building larger than the chip window or badly off-center);
- ``mask_empty``: footprint rasterized to nothing (registration or geometry failure).

Two contact sheets are written — the most confident *severe* errors (|pred - target| >= 2)
and the most confident *adjacent* errors — each panel showing the RGB chip the model saw
with the footprint mask outlined, for manual categorization (pre-existing debris,
occlusion, ordinal-boundary ambiguity, nodata edges, footprint registration).

Usage::

    python -m scripts.mine_errors \
        --variant-root outputs/ablation/all_features \
        --split "Hurricane_Idalia+Hurricane_Michael+Mayfield_Tornado+Mussett_Bayou_Fire" \
        --probs-cache "outputs/ablation/_ensembles/all_features__<split>_probs.pt" \
        --data-dir data --output-dir outputs/error_mining/all_features__default_split
"""

import argparse
import json

import cv2
import torch

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from src.model.trainer import ORDINAL_CLASS_DISPLAY_NAMES
from src.postproc.ensemble import build_val_loader_from_config, load_resolved_config

SHORT_NAMES = ["No", "Min", "Maj", "Des"]


def chip_diagnostics(sample: torch.Tensor) -> dict[str, float | bool]:
    """Mechanical diagnostics from one uint8 [4, H, W] chip (RGB + footprint mask)."""

    rgb = sample[:3].numpy()
    mask = sample[3].numpy().astype(bool)
    nodata = ~np.any(rgb > 0, axis=0)
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return {
        "nodata_frac": float(nodata.mean()),
        "mask_frac": float(mask.mean()),
        "mask_at_border": bool(np.any(mask & border)),
        "mask_empty": bool(not mask.any()),
    }


def render_sheet(records: list[dict], dataset: torch.utils.data.Dataset, output_path: Path, title: str, cols: int = 4) -> None:
    """Contact sheet of chips with the footprint outlined; one panel per error record."""

    rows = max(1, int(np.ceil(len(records) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.9))
    for ax, record in zip(np.ravel(axes), records, strict=False):
        sample = dataset[record["index"]]["image"]
        rgb = np.moveaxis(sample[:3].numpy(), 0, -1).copy()
        mask = sample[3].numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 255, 255), 3)
        ax.imshow(rgb)
        flags = "".join(f" {f}" for f, on in [("nodata", record["nodata_frac"] > 0.05), ("border", record["mask_at_border"]),
                                              ("nomask", record["mask_empty"])] if on)
        ax.set_title(f"true {SHORT_NAMES[record['target']]} -> pred {SHORT_NAMES[record['pred']]}  p={record['confidence']:.2f}\n"
                     f"{record['event'].replace('Hurricane ', 'H. ')}  seeds {record['seed_agreement']}/{record['n_seeds']}"
                     f"  mask {record['mask_frac']:.2f}{flags}", fontsize=8)
        ax.axis("off")
    for ax in np.ravel(axes)[len(records):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"sheet written: {output_path}")


def main() -> None:
    """CLI entry point: rank errors, attach diagnostics, write inventory and exemplar sheets."""

    parser = argparse.ArgumentParser(description="Mine highest-confidence ensemble errors from a cached probability tensor.")
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--probs-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--statistics-csv", type=Path, default=Path("data/statistics.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16, help="Panels per contact sheet.")
    parser.add_argument("--inventory-n", type=int, default=100, help="Errors listed in the JSON inventory per class.")
    parser.add_argument("--stratify-mask-frac", action="store_true",
                        help="Also render every chip once to stratify error rate by footprint fraction and border truncation.")
    args = parser.parse_args()

    payload = torch.load(args.probs_cache, weights_only=False)
    targets: torch.Tensor = payload["targets"]
    variant = next(v for v in payload["variants"] if Path(v["variant_root"]).name == args.variant_root.name)
    ensemble_probs: torch.Tensor = variant["ensemble_probs"]
    individual = variant["individual_probs"]
    seed_probs = torch.stack(list(individual) if not isinstance(individual, dict) else list(individual.values()))  # [S, T, K]

    seed_dir = next(p for p in sorted((args.variant_root / args.split).glob("seed_*")) if p.is_dir())
    cfg = load_resolved_config(seed_dir)
    val_loader, _holdout = build_val_loader_from_config(cfg=cfg, data_dir_override=args.data_dir)
    dataset = val_loader.dataset
    rebuilt = torch.tensor([int(inst["label_tensor"].item()) for inst in dataset.instances])
    if not torch.equal(rebuilt, targets):
        raise RuntimeError("Rebuilt instance sequence does not reproduce the cached targets; loader order changed.")
    mosaic_to_event = dict(zip(*(pd.read_csv(args.statistics_csv)[c] for c in ["Orthomosaic", "Event"]), strict=True))
    events = [mosaic_to_event[Path(inst["image_path"]).name] for inst in dataset.instances]
    print(f"instance order verified: {len(events)} buildings")

    # Rank errors: severity first, then ensemble confidence in the wrong class.
    pred = ensemble_probs.argmax(dim=1)
    confidence = ensemble_probs.max(dim=1).values
    seed_pred = seed_probs.argmax(dim=2)  # [S, T]
    agreement = (seed_pred == pred.unsqueeze(0)).sum(dim=0)
    error_idx = torch.nonzero(pred != targets).flatten().tolist()
    severity = (pred - targets).abs()

    def record_for(i: int) -> dict:
        diag = chip_diagnostics(dataset[i]["image"])
        return {
            "index": i,
            "mosaic": Path(dataset.instances[i]["image_path"]).name,
            "event": events[i],
            "target": int(targets[i]),
            "pred": int(pred[i]),
            "severity": int(severity[i]),
            "confidence": float(confidence[i]),
            "seed_agreement": int(agreement[i]),
            "n_seeds": int(seed_probs.shape[0]),
            "probs": [round(float(p), 4) for p in ensemble_probs[i]],
            **diag,
        }

    ranked = sorted(error_idx, key=lambda i: (-int(severity[i]), -float(confidence[i])))
    severe = [i for i in ranked if int(severity[i]) >= 2]
    adjacent = [i for i in ranked if int(severity[i]) == 1]
    severe_records = [record_for(i) for i in severe[:args.inventory_n]]
    adjacent_records = [record_for(i) for i in adjacent[:args.inventory_n]]

    # Aggregate view: where errors sit, how confident they are, how unanimous the seeds are.
    n_err = len(error_idx)
    err_conf = confidence[error_idx]
    err_agree = agreement[error_idx]
    correct_conf = confidence[pred == targets]
    by_event: dict[str, dict] = {}
    for event in sorted(set(events)):
        sel = torch.tensor([i for i, e in enumerate(events) if e == event])
        e_err = sel[(pred[sel] != targets[sel])]
        by_event[event] = {
            "n": len(sel), "errors": len(e_err), "severe": int((severity[e_err] >= 2).sum()),
            "under_call": int((pred[e_err] < targets[e_err]).sum()), "over_call": int((pred[e_err] > targets[e_err]).sum()),
        }
    transitions = torch.zeros(4, 4, dtype=torch.int64)
    for i in error_idx:
        transitions[int(targets[i]), int(pred[i])] += 1

    summary = {
        "n_instances": len(targets), "n_errors": n_err, "n_severe": len(severe), "n_adjacent": len(adjacent),
        "under_call": int((pred[error_idx] < targets[error_idx]).sum()), "over_call": int((pred[error_idx] > targets[error_idx]).sum()),
        "error_confidence": {"median": float(err_conf.median()), "frac_gt_0.9": float((err_conf > 0.9).float().mean())},
        "correct_confidence": {"median": float(correct_conf.median()), "frac_gt_0.9": float((correct_conf > 0.9).float().mean())},
        "errors_unanimous_across_seeds": int((err_agree == seed_probs.shape[0]).sum()),
        "severe_unanimous_across_seeds": int(sum(1 for i in severe if int(agreement[i]) == seed_probs.shape[0])),
        "transitions_true_x_pred": transitions.tolist(),
        "by_event": by_event,
        "mechanical_flags_in_top_severe": {
            "nodata_gt_5pct": sum(1 for r in severe_records[:args.top_k] if r["nodata_frac"] > 0.05),
            "mask_at_border": sum(1 for r in severe_records[:args.top_k] if r["mask_at_border"]),
            "mask_empty": sum(1 for r in severe_records[:args.top_k] if r["mask_empty"]),
        },
    }

    # Optional column-wide stratification: does the fixed chip window's truncation of large
    # footprints (mask fraction near 1, footprint at the border) carry excess error?
    if args.stratify_mask_frac:
        is_error = (pred != targets).numpy()
        is_severe = (severity >= 2).numpy()
        mask_frac = np.empty(len(targets))
        at_border = np.empty(len(targets), dtype=bool)
        nodata_frac = np.empty(len(targets))
        for i in range(len(targets)):
            diag = chip_diagnostics(dataset[i]["image"])
            mask_frac[i], at_border[i], nodata_frac[i] = diag["mask_frac"], diag["mask_at_border"], diag["nodata_frac"]
        bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
        strata = []
        for lo, hi in bins:
            sel = (mask_frac >= lo) & (mask_frac < hi)
            strata.append({"mask_frac_bin": [lo, min(hi, 1.0)], "n": int(sel.sum()),
                           "error_rate": float(is_error[sel].mean()) if sel.any() else None,
                           "severe_rate": float(is_severe[sel].mean()) if sel.any() else None})
        for name, sel in [("mask_at_border", at_border), ("mask_interior", ~at_border), ("nodata_gt_5pct", nodata_frac > 0.05)]:
            strata.append({"group": name, "n": int(sel.sum()), "error_rate": float(is_error[sel].mean()) if sel.any() else None,
                           "severe_rate": float(is_severe[sel].mean()) if sel.any() else None})
        summary["stratification"] = strata
        print("\nerror rate by footprint fraction / truncation:")
        for s in strata:
            label = f"mask_frac {s['mask_frac_bin'][0]:.2f}-{s['mask_frac_bin'][1]:.2f}" if "mask_frac_bin" in s else s["group"]
            if s["n"] == 0:
                print(f"  {label:<22} n=    0  (empty)")
                continue
            print(f"  {label:<22} n={s['n']:>5}  error={s['error_rate']:.3f}  severe={s['severe_rate']:.3f}")

    print(f"\nerrors {n_err}/{len(targets)} ({n_err / len(targets):.1%}); severe {len(severe)}, adjacent {len(adjacent)}; "
          f"under-call {summary['under_call']} vs over-call {summary['over_call']}")
    print(f"median confidence: errors {summary['error_confidence']['median']:.3f} vs correct {summary['correct_confidence']['median']:.3f}; "
          f"errors with p>0.9: {summary['error_confidence']['frac_gt_0.9']:.1%}; "
          f"seed-unanimous errors: {summary['errors_unanimous_across_seeds']} (severe: {summary['severe_unanimous_across_seeds']})")
    print("transitions (rows true, cols pred):")
    for name, row in zip(ORDINAL_CLASS_DISPLAY_NAMES, transitions.tolist(), strict=True):
        print(f"  {name:<10} {row}")
    for event, e in by_event.items():
        print(f"  {event:<22} n={e['n']:>5} errors={e['errors']:>4} severe={e['severe']:>3} under={e['under_call']:>4} over={e['over_call']:>4}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inventory.json").write_text(json.dumps({
        "variant_root": str(args.variant_root), "split": args.split, "probs_cache": str(args.probs_cache),
        "summary": summary, "severe_errors": severe_records, "adjacent_errors": adjacent_records,
    }, indent=2), encoding="utf-8")
    print(f"inventory written: {args.output_dir / 'inventory.json'}")

    render_sheet(records=severe_records[:args.top_k], dataset=dataset, output_path=args.output_dir / "severe_top.png",
                 title=f"Most confident severe errors (|pred-true| >= 2) — {args.variant_root.name}, {args.split[:40]}")
    render_sheet(records=adjacent_records[:args.top_k], dataset=dataset, output_path=args.output_dir / "adjacent_top.png",
                 title=f"Most confident adjacent errors — {args.variant_root.name}, {args.split[:40]}")


if __name__ == "__main__":
    main()
