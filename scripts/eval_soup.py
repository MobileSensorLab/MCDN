"""Evaluate weight-space model soups against the per-seed ensemble (R1-6 deployability).

A model soup [Wortsman et al., 2022] averages the weights of independently fine-tuned
seeds into one deployable network, trading the N-fold inference cost of a probability
ensemble for a single forward pass. For ``MaskCenteredDamageNet`` the pretrained
backbone, the zero-initialized mask stem channel, and the zero-initialized FiLM affines
are shared at initialization, but the typology context MLP, the mask-pooling
projections, the cross-scale mixer/gate, and the classification head are randomly
initialized per seed and FiLM enters mid-backbone, so seeds are not expected to be
linearly mode-connected. This script measures that directly, on the same validation
loader and 8-view TTA path the ensemble evaluation uses, for three single-network
candidates:

- ``reference``: the reference seed alone (first seed in ``--seeds``), the floor;
- ``backbone_soup``: selective soup averaging only ``backbone.*`` tensors, with every
  other tensor taken from the reference seed (Tier 1.5 in ``src/postproc/soup.py``);
- ``full_soup``: uniform soup averaging every floating-point tensor.

Metrics are reported under all three decision rules so they slot directly beside the
per-seed and ensemble rows of the matching ``src.postproc.ensemble`` artifact.

Usage::

    python scripts/eval_soup.py \
        --variant-root outputs/ablation/all_features \
        --split "Hurricane_Idalia+Hurricane_Michael+Mayfield_Tornado+Mussett_Bayou_Fire" \
        --data-dir data \
        --output-json outputs/ablation/all_features/<split>/soup_metrics.json
"""

import argparse
import json
import time

import torch

from pathlib import Path

from src.postproc.ensemble import (
    DEFAULT_SEEDS, build_model_from_config, build_val_loader_from_config, collect_averaged_probabilities,
    format_metric_line, load_resolved_config, metrics_under_all_rules, serialize_for_json
)
from src.postproc.soup import build_uniform_soup

BACKBONE_PREFIXES = ("backbone.",)


def resolve_device(requested: str) -> str:
    """Map the ``auto`` sentinel onto CUDA when available, else CPU."""

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_state_dict(state_dict: dict[str, torch.Tensor], cfg: dict, val_loader: torch.utils.data.DataLoader,
                        device: str) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Load one state_dict into a fresh model and run the standard TTA evaluation.

    Returns:
        Tuple of (metrics under all rules, probabilities [T, K], targets [T]).
    """

    model = build_model_from_config(cfg=cfg, device=device)
    model.load_state_dict(state_dict, strict=True)
    probs, targets = collect_averaged_probabilities(model=model, val_loader=val_loader, device=device)
    return metrics_under_all_rules(probs=probs, targets=targets), probs, targets


def main() -> None:
    """CLI entry point: build the soups, evaluate all candidates, write the artifact."""

    parser = argparse.ArgumentParser(description="Evaluate backbone-only and full model soups against the reference seed.")
    parser.add_argument("--variant-root", type=Path, required=True, help="outputs/ablation/<variant> directory.")
    parser.add_argument("--split", type=str, required=True, help="Split directory name under the variant root.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to soup; the first is the reference seed for non-averaged tensors.")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional data-root override.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Artifact path (default: <variant-root>/<split>/soup_metrics.json).")
    parser.add_argument("--probs-cache", type=Path, default=None,
                        help="Optional .pt cache of per-candidate probabilities and targets.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    split_dir = args.variant_root / args.split
    checkpoint_paths = [split_dir / f"seed_{seed:02d}" / "best_model.pt" for seed in args.seeds]
    missing = [p for p in checkpoint_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    # The reference seed's resolved config defines both the architecture and the val pool.
    reference_dir = checkpoint_paths[0].parent
    cfg = load_resolved_config(reference_dir)
    t0 = time.time()
    val_loader, holdout_event = build_val_loader_from_config(cfg=cfg, data_dir_override=args.data_dir)
    print(f"val loader ready ({time.time() - t0:.1f}s)  holdout={holdout_event}  batches={len(val_loader)}  device={device}")

    # Soups are formed on CPU (one-shot tensor arithmetic) and moved by load_state_dict.
    t0 = time.time()
    candidates: dict[str, dict[str, torch.Tensor]] = {
        "reference": build_uniform_soup(checkpoint_paths=checkpoint_paths[:1], device="cpu"),
        "backbone_soup": build_uniform_soup(checkpoint_paths=checkpoint_paths, device="cpu",
                                            average_key_prefixes=BACKBONE_PREFIXES),
        "full_soup": build_uniform_soup(checkpoint_paths=checkpoint_paths, device="cpu"),
    }
    print(f"soups formed over {len(checkpoint_paths)} seeds ({time.time() - t0:.1f}s)")

    results: dict[str, dict] = {}
    cache: dict[str, torch.Tensor] = {}
    for name, state_dict in candidates.items():
        t0 = time.time()
        metrics, probs, targets = evaluate_state_dict(state_dict=state_dict, cfg=cfg, val_loader=val_loader, device=device)
        results[name] = metrics
        cache[f"{name}_probs"] = probs
        cache.setdefault("targets", targets)
        print(f"\n--- {name} ({time.time() - t0:.0f}s) ---")
        for rule, m in metrics.items():
            print(format_metric_line(rule, m))

    output_json = args.output_json or (split_dir / "soup_metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "variant_root": str(args.variant_root),
        "split": args.split,
        "holdout_event": holdout_event,
        "seeds": list(args.seeds),
        "reference_seed": args.seeds[0],
        "backbone_prefixes": list(BACKBONE_PREFIXES),
        "candidates": {name: serialize_for_json(metrics) for name, metrics in results.items()},
    }
    output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nartifact written: {output_json}")

    if args.probs_cache is not None:
        args.probs_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, args.probs_cache)
        print(f"probability cache written: {args.probs_cache}")


if __name__ == "__main__":
    main()
