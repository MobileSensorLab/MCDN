"""K-fold CV per-class logit-bias calibration on cached ensemble probabilities.

Consumes the ``.pt`` probability cache produced by ``src.postproc.ensemble`` and
fits an additive per-class bias vector ``b in R^K`` that maximizes macro F1 under
``argmax(log_probs + b)``. Uses 5-fold stratified cross-validation on the
validation set so the reported OOF macro F1 is an honest estimate of what the
calibration would yield on a fresh evaluation set -- not an overfit to the
seen labels.

Reports:
    - Uncalibrated baseline (no bias) as a sanity reference.
    - Per-fold fitted bias vectors + per-fold holdout F1 (stability check).
    - Aggregated OOF macro F1 / per-class F1 / confusion matrix (primary
      evaluation number).
    - Fit-all oracle (bias fit on the entire val set) as an overfit ceiling.

Usage:
    uv run python -m src.postproc.calibrate \
        --prob-cache outputs/ablation/baseline/Spatial_Block_East/ensemble_probs.pt \
        --output-json outputs/ablation/baseline/Spatial_Block_East/calibration.json
"""

import argparse
import json
import time

import numpy as np
import torch

from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score


ORDINAL_CLASS_DISPLAY_NAMES = ("No Damage", "Minor", "Major", "Destroyed")
QWK_WEIGHTS_CACHE: dict[int, np.ndarray] = {}


def build_quadratic_weights(num_classes: int) -> np.ndarray:
    """Cache the QWK weight matrix W[i, j] = (i - j)^2 / (K - 1)^2."""

    if num_classes in QWK_WEIGHTS_CACHE:
        return QWK_WEIGHTS_CACHE[num_classes]
    idx = np.arange(num_classes, dtype=np.float64)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (num_classes - 1) ** 2
    QWK_WEIGHTS_CACHE[num_classes] = weights
    return weights


def quadratic_weighted_kappa(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Compute QWK from integer predictions and targets."""

    weights = build_quadratic_weights(num_classes=num_classes)
    observed = confusion_matrix(targets, preds, labels=list(range(num_classes))).astype(np.float64)
    row_marg = observed.sum(axis=1, keepdims=True)
    col_marg = observed.sum(axis=0, keepdims=True)
    expected = row_marg @ col_marg / max(observed.sum(), 1.0)
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator <= 0:
        return 1.0
    return 1.0 - numerator / denominator


def classification_breakdown(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    """Macro F1, per-class F1, confusion matrix, QWK."""

    per_class_f1 = f1_score(targets, preds, labels=list(range(num_classes)), average=None, zero_division=0.0)
    macro_f1 = float(per_class_f1.mean())
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    qwk = quadratic_weighted_kappa(preds=preds, targets=targets, num_classes=num_classes)
    return {
        "macro_f1": macro_f1,
        "per_class_f1": {name: float(per_class_f1[i]) for i, name in enumerate(ORDINAL_CLASS_DISPLAY_NAMES)},
        "confusion_matrix": cm.tolist(),
        "qwk": float(qwk),
        "accuracy": float((preds == targets).mean())
    }


def ensemble_log_probs(individual_probs: torch.Tensor) -> np.ndarray:
    """Collapse [num_seeds, N, K] probabilities into mean log-probs [N, K].

    Log of the averaged softmax (the standard ensemble posterior); floor with
    1e-12 to avoid log(0) on fp16-tight zeros.
    """

    mean_probs = individual_probs.mean(dim=0).clamp(min=1e-12)
    return torch.log(mean_probs).numpy()


def fit_bias_for_f1(log_probs: np.ndarray, targets: np.ndarray, num_classes: int,
                     bounds_half_width: float, seed: int, max_iter: int) -> tuple[np.ndarray, float]:
    """Fit per-class bias with differential_evolution to maximize macro F1.

    Piecewise-constant objective (argmax is non-differentiable), so a
    population-based global optimizer is more robust than Nelder-Mead. Bounds
    enforce a reasonable shift magnitude relative to calibrated log-probs.
    """

    def negative_f1(bias_vec: np.ndarray) -> float:
        preds = np.argmax(log_probs + bias_vec[None, :], axis=1)
        return -f1_score(targets, preds,
                          labels=list(range(num_classes)),
                          average="macro", zero_division=0.0)

    bounds = [(-bounds_half_width, bounds_half_width)] * num_classes
    result = differential_evolution(
        func=negative_f1,
        bounds=bounds,
        seed=seed,
        maxiter=max_iter,
        tol=1e-5,
        polish=False,
        init="sobol"
    )
    return result.x, float(-result.fun)


def apply_bias_argmax(log_probs: np.ndarray, bias_vec: np.ndarray) -> np.ndarray:
    """Apply additive bias and take argmax."""

    return np.argmax(log_probs + bias_vec[None, :], axis=1)


def format_bias(bias_vec: np.ndarray) -> str:
    """Compact one-liner for a bias vector."""

    return " ".join(f"{name[:3]}={bias_vec[i]:+.3f}" for i, name in enumerate(ORDINAL_CLASS_DISPLAY_NAMES))


def format_per_class(per_class: dict) -> str:
    """Compact per-class F1 one-liner."""

    return "  ".join(f"{name[:3]}={per_class[name]:.4f}" for name in ORDINAL_CLASS_DISPLAY_NAMES)


def main() -> None:
    """CLI entry point for CV logit-bias calibration on cached ensemble probabilities."""

    parser = argparse.ArgumentParser(description="CV logit-bias calibration on cached ensemble probabilities.")
    parser.add_argument("--prob-cache", type=Path, required=True,
                        help="Path to ensemble_probs.pt from src.postproc.ensemble.")
    parser.add_argument("--variant-index", type=int, default=0,
                        help="Index into payload['variants'] to calibrate (default: 0, first variant).")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--bias-bound", type=float, default=2.0,
                        help="Half-width of per-class bias search bounds (default: 2.0 logits).")
    parser.add_argument("--max-iter", type=int, default=150,
                        help="differential_evolution maxiter (default: 150).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading probability cache: {args.prob_cache}")
    payload = torch.load(args.prob_cache, weights_only=False, map_location="cpu")

    variant = payload["variants"][args.variant_index]
    individual_probs: torch.Tensor = variant["individual_probs"]
    targets_t: torch.Tensor = payload["targets"]
    split_name: str = payload["split"]
    holdout_event: str = payload["holdout_event"]

    targets = targets_t.numpy().astype(np.int64)
    log_probs = ensemble_log_probs(individual_probs=individual_probs)
    num_samples, num_classes = log_probs.shape

    print(f"Variant: {variant['variant_root']}  seeds={variant['seeds']}")
    print(f"Split: {split_name}  holdout={holdout_event}")
    print(f"N={num_samples} samples, K={num_classes} classes")
    print("Class support: " + ", ".join(
        f"{n}={int((targets == i).sum())}" for i, n in enumerate(ORDINAL_CLASS_DISPLAY_NAMES))
    )

    # Baseline: uncalibrated argmax (bias = 0) ---------------------------
    uncal_preds = apply_bias_argmax(log_probs=log_probs, bias_vec=np.zeros(num_classes))
    uncal_metrics = classification_breakdown(preds=uncal_preds, targets=targets, num_classes=num_classes)
    print("\n=== Uncalibrated ensemble argmax (reference) ===")
    print(f"  F1={uncal_metrics['macro_f1']:.4f}  QWK={uncal_metrics['qwk']:.4f}  "
          f"acc={uncal_metrics['accuracy']:.4f}  [{format_per_class(uncal_metrics['per_class_f1'])}]")

    # K-fold CV calibration ---------------------------------------------
    kfold = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    oof_preds = np.full_like(targets, fill_value=-1)
    per_fold_records: list[dict] = []

    print(f"\n=== {args.num_folds}-fold stratified CV calibration ===")
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(log_probs, targets)):
        t0 = time.time()
        bias_vec, train_f1 = fit_bias_for_f1(
            log_probs=log_probs[train_idx],
            targets=targets[train_idx],
            num_classes=num_classes,
            bounds_half_width=args.bias_bound,
            seed=args.seed + fold_idx,
            max_iter=args.max_iter
        )
        fold_preds = apply_bias_argmax(log_probs=log_probs[test_idx], bias_vec=bias_vec)
        fold_metrics = classification_breakdown(preds=fold_preds, targets=targets[test_idx], num_classes=num_classes)

        oof_preds[test_idx] = fold_preds
        fit_seconds = time.time() - t0
        print(f"  fold {fold_idx+1}/{args.num_folds}  ({fit_seconds:.1f}s)  "
              f"bias=[{format_bias(bias_vec)}]  "
              f"train_F1={train_f1:.4f}  test_F1={fold_metrics['macro_f1']:.4f}  "
              f"test_QWK={fold_metrics['qwk']:.4f}")

        per_fold_records.append({
            "fold_idx": fold_idx,
            "bias": bias_vec.tolist(),
            "train_macro_f1": train_f1,
            "test_metrics": fold_metrics,
            "n_test": len(test_idx)
        })

    assert (oof_preds >= 0).all(), "stratified kfold left some samples unassigned -- should be impossible"

    oof_metrics = classification_breakdown(preds=oof_preds, targets=targets, num_classes=num_classes)
    print("\n=== Aggregated OOF (all 5 folds' test-partition predictions combined) ===")
    print(f"  F1={oof_metrics['macro_f1']:.4f}  QWK={oof_metrics['qwk']:.4f}  "
          f"acc={oof_metrics['accuracy']:.4f}  [{format_per_class(oof_metrics['per_class_f1'])}]")

    # Per-fold bias stability --------------------------------------------
    bias_matrix = np.array([r["bias"] for r in per_fold_records])
    bias_mean = bias_matrix.mean(axis=0)
    bias_std = bias_matrix.std(axis=0)
    print("\nPer-fold bias stability (mean +/- std):")
    for i, name in enumerate(ORDINAL_CLASS_DISPLAY_NAMES):
        print(f"  {name:<10s}  {bias_mean[i]:+.3f} +/- {bias_std[i]:.3f}")

    # Fit-all oracle (upper bound, overfits to val set) ------------------
    print("\n=== Fit-all oracle (overfit ceiling, not a valid test metric) ===")
    t0 = time.time()
    oracle_bias, _oracle_train_f1 = fit_bias_for_f1(
        log_probs=log_probs,
        targets=targets,
        num_classes=num_classes,
        bounds_half_width=args.bias_bound,
        seed=args.seed + 99,
        max_iter=args.max_iter
    )
    oracle_preds = apply_bias_argmax(log_probs=log_probs, bias_vec=oracle_bias)
    oracle_metrics = classification_breakdown(preds=oracle_preds, targets=targets, num_classes=num_classes)
    print(f"  ({time.time() - t0:.1f}s)  bias=[{format_bias(oracle_bias)}]  "
          f"F1={oracle_metrics['macro_f1']:.4f}  QWK={oracle_metrics['qwk']:.4f}  "
          f"acc={oracle_metrics['accuracy']:.4f}  [{format_per_class(oracle_metrics['per_class_f1'])}]")

    # Deltas -------------------------------------------------------------
    delta_oof = oof_metrics["macro_f1"] - uncal_metrics["macro_f1"]
    delta_oracle = oracle_metrics["macro_f1"] - uncal_metrics["macro_f1"]
    print("\n=== Deltas vs uncalibrated ensemble argmax ===")
    print(f"  OOF calibrated  :  F1 delta={delta_oof:+.4f}  (honest)")
    print(f"  Fit-all oracle  :  F1 delta={delta_oracle:+.4f}  (ceiling)")

    # Artifact -----------------------------------------------------------
    if args.output_json is not None:
        artifact = {
            "prob_cache": str(args.prob_cache),
            "variant_root": variant["variant_root"],
            "split": split_name,
            "holdout_event": holdout_event,
            "seeds": variant["seeds"],
            "num_samples": num_samples,
            "num_classes": num_classes,
            "num_folds": args.num_folds,
            "bias_bound": args.bias_bound,
            "uncalibrated": uncal_metrics,
            "oof_calibrated": oof_metrics,
            "oracle_calibrated": oracle_metrics,
            "per_fold": per_fold_records,
            "bias_mean": bias_mean.tolist(),
            "bias_std": bias_std.tolist(),
            "oracle_bias": oracle_bias.tolist(),
            "delta_f1_oof_vs_uncal": delta_oof,
            "delta_f1_oracle_vs_uncal": delta_oracle
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nArtifact written: {args.output_json}")


if __name__ == "__main__":
    main()
