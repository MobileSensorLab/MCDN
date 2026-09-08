"""Per-seed temperature-scaled ensemble calibration.

Loads the probability cache produced by ``src.postproc.ensemble`` and fits a
single scalar temperature ``T`` per seed by minimizing NLL on held-out val
(5-fold stratified CV, matching the protocol in ``src.postproc.calibrate``).
Each seed's softmax is re-scaled by its fitted temperature and the resulting
distributions are averaged across seeds. Reports OOF QWK / macro F1 /
accuracy / macro recall under EV / argmax / hybrid prediction rules.

Mathematical equivalence note:
    Softmax is translation-invariant, so ``softmax(log(p) / T) == softmax(z / T)``
    where ``p == softmax(z)``. Cached probabilities are sufficient; raw logits
    are not required.

Usage:
    uv run python -m src.postproc.temperature \
        --prob-cache outputs/ablation/all_features/Spatial_Block_East/ensemble_probs.pt \
        --output-json outputs/ablation/all_features/Spatial_Block_East/temperature_calibration.json
"""

import argparse
import json

import numpy as np
import torch

from pathlib import Path
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


ORDINAL_CLASS_DISPLAY_NAMES: tuple[str, ...] = ("No Damage", "Minor", "Major", "Destroyed")


def quadratic_weighted_kappa(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Standard QWK from integer predictions and targets."""

    idx = np.arange(num_classes, dtype=np.float64)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (num_classes - 1) ** 2
    observed = confusion_matrix(targets, preds, labels=list(range(num_classes))).astype(np.float64)
    row_marg = observed.sum(axis=1, keepdims=True)
    col_marg = observed.sum(axis=0, keepdims=True)
    expected = row_marg @ col_marg / max(observed.sum(), 1.0)
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator <= 0:
        return 1.0
    return 1.0 - numerator / denominator


def macro_recall(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Macro-averaged recall across all K classes."""

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    per_class = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0 for i in range(num_classes)]
    return float(np.mean(per_class))


def apply_temperature(log_probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-softmax ``log(p) / T`` with numerically stable max-subtraction."""

    scaled = log_probs / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_vals = np.exp(scaled)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def nll_of_temperature(temperature: float, log_probs: np.ndarray, targets: np.ndarray) -> float:
    """NLL under softmax(log(p) / T). Returns +inf if T is non-positive."""

    if temperature <= 0:
        return float("inf")
    probs = apply_temperature(log_probs=log_probs, temperature=temperature)
    sample_probs = np.clip(probs[np.arange(len(targets)), targets], a_min=1e-12, a_max=1.0)
    return float(-np.log(sample_probs).mean())


def fit_temperature(log_probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Bounded scalar minimization of NLL on [0.1, 10]. Returns (T, train_NLL)."""

    result = minimize_scalar(
        fun=nll_of_temperature,
        args=(log_probs, targets),
        bounds=(0.1, 10.0),
        method="bounded",
        options={"xatol": 1e-4})
    return float(result.x), float(result.fun)


def ev_preds(probs: np.ndarray) -> np.ndarray:
    """Expected-value rounding: ``floor(sum_i i * p(i) + 0.5)``."""

    classes = np.arange(probs.shape[1], dtype=np.float64)
    return np.floor((probs * classes).sum(axis=1) + 0.5).astype(np.int64)


def argmax_preds(probs: np.ndarray) -> np.ndarray:
    """Categorical argmax."""

    return probs.argmax(axis=1).astype(np.int64)


def hybrid_preds(probs: np.ndarray) -> np.ndarray:
    """Argmax on extreme classes (0 and K-1), EV rounding on interior classes."""

    am = argmax_preds(probs=probs)
    ev = ev_preds(probs=probs)
    num_classes = probs.shape[1]
    extreme_mask = (am == 0) | (am == (num_classes - 1))
    return np.where(extreme_mask, am, ev)


RULE_FNS = {"EV": ev_preds, "argmax": argmax_preds, "hybrid": hybrid_preds}


def classification_breakdown(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    """Return QWK / macro F1 / accuracy / macro recall / per-class F1 / confusion matrix."""

    per_class_f1 = f1_score(targets, preds, labels=list(range(num_classes)), average=None, zero_division=0.0)
    return {
        "macro_f1": float(per_class_f1.mean()),
        "per_class_f1": {n: float(per_class_f1[i]) for i, n in enumerate(ORDINAL_CLASS_DISPLAY_NAMES)},
        "qwk": float(quadratic_weighted_kappa(preds=preds, targets=targets, num_classes=num_classes)),
        "accuracy": float((preds == targets).mean()),
        "macro_recall": macro_recall(preds=preds, targets=targets, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(targets, preds, labels=list(range(num_classes))).tolist()
    }


def print_rule_row(label: str, m: dict) -> None:
    """One-line QWK / F1 / acc / recall summary for a given prediction rule."""

    print(f"  {label:<10s}  QWK={m['qwk']:.4f}  F1={m['macro_f1']:.4f}  "
          f"acc={m['accuracy']:.4f}  recall={m['macro_recall']:.4f}")


def main() -> None:
    """CLI entry point for per-seed temperature-scaled ensemble calibration."""

    parser = argparse.ArgumentParser(description="Per-seed temperature-scaled ensemble calibration.")
    parser.add_argument("--prob-cache", type=Path, required=True)
    parser.add_argument("--variant-index", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print(f"Loading probability cache: {args.prob_cache}")
    payload = torch.load(args.prob_cache, weights_only=False, map_location="cpu")
    variant = payload["variants"][args.variant_index]
    individual_probs = variant["individual_probs"].numpy()
    targets = payload["targets"].numpy().astype(np.int64)

    num_seeds, num_samples, num_classes = individual_probs.shape
    seeds = variant["seeds"]

    log_probs_per_seed = np.log(np.clip(individual_probs, a_min=1e-12, a_max=1.0))

    print(f"Variant: {variant['variant_root']}  seeds={seeds}")
    print(f"N={num_samples} samples, K={num_classes} classes, S={num_seeds} seeds")

    uncal_probs = individual_probs.mean(axis=0)
    uncal_metrics = {
        rule: classification_breakdown(preds=fn(uncal_probs), targets=targets, num_classes=num_classes)
        for rule, fn in RULE_FNS.items()
    }
    print("\n=== Uncalibrated ensemble (reference) ===")
    for rule in ("EV", "argmax", "hybrid"):
        print_rule_row(label=rule, m=uncal_metrics[rule])

    kfold = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros_like(uncal_probs, dtype=np.float64)
    per_fold_temperatures = np.zeros((args.num_folds, num_seeds), dtype=np.float64)
    per_fold_train_nll = np.zeros((args.num_folds, num_seeds), dtype=np.float64)

    print(f"\n=== {args.num_folds}-fold per-seed temperature scaling ===")
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(uncal_probs, targets)):
        scaled_test = np.zeros((num_seeds, len(test_idx), num_classes), dtype=np.float64)
        for k_idx in range(num_seeds):
            t_kf, nll_kf = fit_temperature(
                log_probs=log_probs_per_seed[k_idx, train_idx],
                targets=targets[train_idx])
            per_fold_temperatures[fold_idx, k_idx] = t_kf
            per_fold_train_nll[fold_idx, k_idx] = nll_kf
            scaled_test[k_idx] = apply_temperature(
                log_probs=log_probs_per_seed[k_idx, test_idx], temperature=t_kf)
        oof_probs[test_idx] = scaled_test.mean(axis=0)
        t_summary = " ".join(f"s{seeds[k]}:{per_fold_temperatures[fold_idx, k]:.2f}"
                              for k in range(num_seeds))
        print(f"  fold {fold_idx + 1}/{args.num_folds}  T=[{t_summary}]")

    oof_metrics = {
        rule: classification_breakdown(preds=fn(oof_probs), targets=targets, num_classes=num_classes)
        for rule, fn in RULE_FNS.items()
    }
    print("\n=== OOF temperature-calibrated ensemble ===")
    for rule in ("EV", "argmax", "hybrid"):
        print_rule_row(label=rule, m=oof_metrics[rule])

    print("\nPer-seed temperature stability across folds:")
    for k_idx in range(num_seeds):
        t_vals = per_fold_temperatures[:, k_idx]
        print(f"  seed {seeds[k_idx]:>3d}  T={t_vals.mean():.3f} +/- {t_vals.std():.3f}  "
              f"(min={t_vals.min():.3f}, max={t_vals.max():.3f})")

    oracle_temperatures = np.zeros(num_seeds, dtype=np.float64)
    scaled_all = np.zeros_like(individual_probs, dtype=np.float64)
    for k_idx in range(num_seeds):
        t_k, _ = fit_temperature(log_probs=log_probs_per_seed[k_idx], targets=targets)
        oracle_temperatures[k_idx] = t_k
        scaled_all[k_idx] = apply_temperature(log_probs=log_probs_per_seed[k_idx], temperature=t_k)
    oracle_probs = scaled_all.mean(axis=0)
    oracle_metrics = {
        rule: classification_breakdown(preds=fn(oracle_probs), targets=targets, num_classes=num_classes)
        for rule, fn in RULE_FNS.items()
    }
    print("\n=== Oracle (per-seed T fit on full val, overfit ceiling) ===")
    for rule in ("EV", "argmax", "hybrid"):
        print_rule_row(label=rule, m=oracle_metrics[rule])
    oracle_summary = "  ".join(f"s{seeds[k]}={oracle_temperatures[k]:.3f}" for k in range(num_seeds))
    print(f"  oracle T:  {oracle_summary}")

    print("\n=== Deltas vs uncalibrated ensemble ===")
    for rule in ("EV", "argmax", "hybrid"):
        delta_oof = oof_metrics[rule]["macro_f1"] - uncal_metrics[rule]["macro_f1"]
        delta_oracle = oracle_metrics[rule]["macro_f1"] - uncal_metrics[rule]["macro_f1"]
        print(f"  {rule:<8s}  OOF F1 delta={delta_oof:+.4f}  oracle F1 delta={delta_oracle:+.4f}")

    if args.output_json is not None:
        artifact = {
            "prob_cache": str(args.prob_cache),
            "variant_root": variant["variant_root"],
            "seeds": seeds,
            "num_samples": int(num_samples),
            "num_classes": int(num_classes),
            "num_seeds": int(num_seeds),
            "num_folds": args.num_folds,
            "uncalibrated": uncal_metrics,
            "oof_calibrated": oof_metrics,
            "oracle_calibrated": oracle_metrics,
            "per_fold_temperatures": per_fold_temperatures.tolist(),
            "per_fold_train_nll": per_fold_train_nll.tolist(),
            "oracle_temperatures": oracle_temperatures.tolist(),
            "temperature_mean_per_seed": per_fold_temperatures.mean(axis=0).tolist(),
            "temperature_std_per_seed": per_fold_temperatures.std(axis=0).tolist()
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nArtifact written: {args.output_json}")


if __name__ == "__main__":
    main()
