"""Post-hoc spatial-context probe: does neighbor information add lift over the ensemble?

Consumes a probability cache written by ``src.postproc.ensemble --prob-cache``, maps every
validation sample back to its physical building through the deterministic dataset instance
index, builds neighbor features from within-mosaic building geometry, and compares four
prediction variants:

    raw        : argmax on ensemble probabilities (published baseline, no fitting)
    recal      : logistic combiner on own log-probabilities only (recalibration control)
    ctx_pred   : combiner on own log-probs + neighbor predicted-probability features
                 (deployable: neighbors' predictions exist at inference over a mosaic)
    ctx_oracle : combiner on own log-probs + neighbor ground-truth features (diagnostic
                 ceiling on post-hoc context; never deployable)

The incremental value of context is ``ctx_pred - recal``; ``recal - raw`` isolates mere
recalibration. The combiner is multinomial logistic regression fit with torch LBFGS from
a zero init (deterministic). Evaluation is leave-one-group-out over per-mosaic spatial
quadrants, so the combiner never sees labels from the quadrant it is scored on; pooled
out-of-fold predictions yield QWK / macro-F1 / per-class F1. Buildings within
``--buffer-m`` of their quadrant boundary share neighbors across folds, so a rescore
excluding that strip bounds the resulting optimism.

Usage::

    uv run python scripts/spatial_context_probe.py \
        --prob-cache outputs/spatial_context/probcache_Hurricane_Ida.pt --data-dir data
"""

import argparse
import json

import numpy as np
import rasterio
import torch

from geopandas import GeoSeries
from pathlib import Path
from scipy.spatial import cKDTree

from src.data.dataset import CRASARUnitemporalDataset
from src.model.trainer import ORDINAL_CLASS_DISPLAY_NAMES, _validation_qwk_and_classification
from src.postproc.ensemble import get_fold_dataframes_from_config, load_resolved_config

NUM_CLASSES = 4
PROB_FLOOR = 1e-6


def rebuild_instance_geometry(cache: dict, data_dir: str) -> tuple[CRASARUnitemporalDataset, np.ndarray, np.ndarray]:
    """Rebuild the fold's validation instances and return building coordinates in meters.

    Reconstructs the exact validation manifest the ensemble inference used (same fold
    semantics, same deterministic ordering), then extracts each building's centroid in a
    per-mosaic UTM projection. No image chips are read; only polygon parsing and one
    raster-header open per mosaic.

    Args:
        cache: Loaded probability-cache payload.
        data_dir: Local data root overriding the checkpoint-recorded (cluster) path.

    Returns:
        Tuple of (dataset, coords [T, 2] in meters, mosaic index [T]).
    """

    variant = cache["variants"][0]
    seed = variant["seeds"][0]
    fold_dir = Path(variant["variant_root"]) / cache["split"] / f"seed_{seed:02d}"
    cfg = load_resolved_config(fold_dir)
    _, val_df, holdout = get_fold_dataframes_from_config(cfg, data_dir_override=data_dir)
    print(f"Fold rebuilt: holdout={holdout}  mosaics={len(val_df)}")

    dataset = CRASARUnitemporalDataset(val_df, chip_size=cfg["data"]["chip_size"], transform=None, is_train=False)

    coords_chunks: list[np.ndarray] = []
    mosaic_idx_chunks: list[np.ndarray] = []
    for mosaic_id, (_, row) in enumerate(val_df.iterrows()):
        gdf = dataset.gdf_cache[str(Path(row["image_path"]))]
        if gdf.empty:
            continue
        centroids: GeoSeries = gdf.geometry.centroid
        if centroids.crs is None:
            with rasterio.open(row["image_path"]) as src:
                centroids = centroids.set_crs(src.crs)
        utm = centroids.to_crs(centroids.estimate_utm_crs())
        coords_chunks.append(np.column_stack([utm.x.to_numpy(), utm.y.to_numpy()]))
        mosaic_idx_chunks.append(np.full(len(utm), mosaic_id, dtype=np.int64))

    coords = np.concatenate(coords_chunks, axis=0)
    mosaic_idx = np.concatenate(mosaic_idx_chunks, axis=0)
    if len(coords) != len(dataset):
        raise RuntimeError(f"Geometry/instance mismatch: {len(coords)} centroids vs {len(dataset)} instances.")
    return dataset, coords, mosaic_idx


def verify_alignment(dataset: CRASARUnitemporalDataset, targets: torch.Tensor) -> None:
    """Assert the cached target sequence matches the rebuilt instance index exactly."""

    rebuilt = torch.tensor([int(inst["label_tensor"]) for inst in dataset.instances], dtype=torch.long)
    if len(rebuilt) != len(targets) or not torch.equal(rebuilt, targets):
        raise RuntimeError(
            f"Alignment failure: rebuilt labels ({len(rebuilt)}) do not match cached targets ({len(targets)}). "
            "The prob cache and the local dataset view disagree; check --data-dir and fold config."
        )
    print(f"Alignment verified: {len(rebuilt)} buildings, cached targets match rebuilt instance labels exactly.")


def neighbor_features(coords: np.ndarray, mosaic_idx: np.ndarray, class_matrix: np.ndarray,
                      radii: list[float]) -> np.ndarray:
    """Inverse-distance-weighted neighbor class summaries at multiple radii.

    Args:
        coords: Building centroids in meters, shape [T, 2].
        mosaic_idx: Mosaic id per building; pairs never cross mosaics.
        class_matrix: Per-building class vectors [T, K] (probabilities or one-hot truth).
        radii: Neighborhood radii in meters.

    Returns:
        Feature array [T, len(radii) * (K + 1)]: weighted mean class vector plus
        log1p(neighbor count) per radius. Buildings without neighbors get zeros.
    """

    features = np.zeros((len(coords), len(radii) * (NUM_CLASSES + 1)), dtype=np.float64)
    for mosaic_id in np.unique(mosaic_idx):
        members = np.flatnonzero(mosaic_idx == mosaic_id)
        tree = cKDTree(coords[members])
        for radius_slot, radius in enumerate(radii):
            neighbor_lists = tree.query_ball_point(coords[members], r=radius)
            base = radius_slot * (NUM_CLASSES + 1)
            for local_i, local_neighbors in enumerate(neighbor_lists):
                neighbor_local = [j for j in local_neighbors if j != local_i]
                if not neighbor_local:
                    continue
                global_i = members[local_i]
                global_neighbors = members[neighbor_local]
                dists = np.linalg.norm(coords[global_neighbors] - coords[global_i], axis=1)
                weights = 1.0 / np.maximum(dists, 1.0)
                weighted_mean = weights @ class_matrix[global_neighbors] / weights.sum()
                features[global_i, base:base + NUM_CLASSES] = weighted_mean
                features[global_i, base + NUM_CLASSES] = np.log1p(len(neighbor_local))
    return features


def quadrant_groups(coords: np.ndarray, mosaic_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign per-mosaic spatial quadrant groups and distance to the quadrant boundary.

    Args:
        coords: Building centroids in meters, shape [T, 2].
        mosaic_idx: Mosaic id per building.

    Returns:
        Tuple of (group id [T], distance in meters to the nearest quadrant split line [T]).
    """

    groups = np.zeros(len(coords), dtype=np.int64)
    boundary_dist = np.zeros(len(coords), dtype=np.float64)
    for mosaic_id in np.unique(mosaic_idx):
        members = np.flatnonzero(mosaic_idx == mosaic_id)
        median_x, median_y = np.median(coords[members, 0]), np.median(coords[members, 1])
        east = (coords[members, 0] >= median_x).astype(np.int64)
        north = (coords[members, 1] >= median_y).astype(np.int64)
        groups[members] = mosaic_id * 4 + east * 2 + north
        boundary_dist[members] = np.minimum(np.abs(coords[members, 0] - median_x),
                                            np.abs(coords[members, 1] - median_y))
    return groups, boundary_dist


def fit_multinomial_lr(features: torch.Tensor, targets: torch.Tensor, l2: float) -> torch.nn.Linear:
    """Fit a multinomial logistic regression with LBFGS from a zero init (deterministic).

    Args:
        features: Standardized training features [N, F].
        targets: Class indices [N].
        l2: L2 penalty on the weight matrix (bias excluded).

    Returns:
        The fitted linear layer.
    """

    linear = torch.nn.Linear(features.shape[1], NUM_CLASSES)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.LBFGS(linear.parameters(), lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        loss = torch.nn.functional.cross_entropy(linear(features), targets) + l2 * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return linear


def grouped_cv_predictions(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, l2: float) -> np.ndarray:
    """Leave-one-group-out predictions from the logistic combiner.

    Standardization statistics are fit on each fold's training rows only, so no
    information from the scored group leaks into the combiner.

    Args:
        features: Raw (unstandardized) features [T, F].
        targets: Class indices [T].
        groups: Spatial group id per building.
        l2: L2 penalty for the combiner.

    Returns:
        Out-of-fold argmax predictions [T].
    """

    predictions = np.zeros(len(targets), dtype=np.int64)
    for group in np.unique(groups):
        test_mask = groups == group
        train_x, train_y = features[~test_mask], targets[~test_mask]
        mean, std = train_x.mean(axis=0), train_x.std(axis=0)
        std[std == 0.0] = 1.0
        model = fit_multinomial_lr(
            features=torch.tensor((train_x - mean) / std, dtype=torch.float32),
            targets=torch.tensor(train_y, dtype=torch.long),
            l2=l2
        )
        with torch.no_grad():
            logits = model(torch.tensor((features[test_mask] - mean) / std, dtype=torch.float32))
        predictions[test_mask] = logits.argmax(dim=1).numpy()
    return predictions


def score(preds: np.ndarray, targets: np.ndarray) -> dict:
    """QWK, macro-F1, accuracy, and per-class F1 for one prediction vector."""

    qwk, class_metrics = _validation_qwk_and_classification(
        preds=torch.tensor(preds, dtype=torch.long), targets=torch.tensor(targets, dtype=torch.long))
    return {
        "qwk": float(qwk),
        "macro_f1": float(class_metrics["macro_f1"]),
        "accuracy": float(class_metrics["accuracy"]),
        "per_class_f1": {name: float(class_metrics["per_class"][name]["f1"]) for name in ORDINAL_CLASS_DISPLAY_NAMES}
    }


def format_row(label: str, metrics: dict) -> str:
    """Single summary line for one variant."""

    per_class = "  ".join(f"{name[:3]}={metrics['per_class_f1'][name]:.3f}" for name in ORDINAL_CLASS_DISPLAY_NAMES)
    return (f"  {label:<12s} QWK={metrics['qwk']:.4f}  F1={metrics['macro_f1']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  [{per_class}]")


def main() -> None:
    """Run the four-variant spatial-context comparison for one probability cache."""

    parser = argparse.ArgumentParser(description="Post-hoc spatial-context probe over a probability cache.")
    parser.add_argument("--prob-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--radii", type=float, nargs="+", default=[25.0, 50.0, 100.0])
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--buffer-m", type=float, default=50.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cache = torch.load(args.prob_cache, map_location="cpu", weights_only=False)
    targets_t: torch.Tensor = cache["targets"]
    probs = cache["variants"][0]["ensemble_probs"].numpy().astype(np.float64)
    targets = targets_t.numpy().astype(np.int64)
    print(f"Cache loaded: split={cache['split']}  holdout={cache['holdout_event']}  buildings={len(targets)}")

    dataset, coords, mosaic_idx = rebuild_instance_geometry(cache, data_dir=args.data_dir)
    verify_alignment(dataset, targets_t)

    own_logprobs = np.log(np.clip(probs, PROB_FLOOR, 1.0))
    onehot = np.eye(NUM_CLASSES)[targets]
    context_pred = neighbor_features(coords, mosaic_idx, class_matrix=probs, radii=args.radii)
    context_oracle = neighbor_features(coords, mosaic_idx, class_matrix=onehot, radii=args.radii)
    groups, boundary_dist = quadrant_groups(coords, mosaic_idx)
    print(f"Features ready: radii={args.radii}  groups={len(np.unique(groups))}  l2={args.l2}")

    variant_features = {
        "recal": own_logprobs,
        "ctx_pred": np.concatenate([own_logprobs, context_pred], axis=1),
        "ctx_oracle": np.concatenate([own_logprobs, context_oracle], axis=1)
    }
    variant_preds = {"raw": probs.argmax(axis=1)}
    for name, feats in variant_features.items():
        variant_preds[name] = grouped_cv_predictions(feats, targets, groups, l2=args.l2)

    results = {name: score(preds, targets) for name, preds in variant_preds.items()}
    interior = boundary_dist >= args.buffer_m
    results_interior = {name: score(preds[interior], targets[interior]) for name, preds in variant_preds.items()}

    print(f"\n=== Full holdout pool (n={len(targets)}) ===")
    for name in ("raw", "recal", "ctx_pred", "ctx_oracle"):
        print(format_row(name, results[name]))
    print(f"\n  deltas: recal-raw QWK {results['recal']['qwk'] - results['raw']['qwk']:+.4f} "
          f"F1 {results['recal']['macro_f1'] - results['raw']['macro_f1']:+.4f}  |  "
          f"ctx_pred-recal QWK {results['ctx_pred']['qwk'] - results['recal']['qwk']:+.4f} "
          f"F1 {results['ctx_pred']['macro_f1'] - results['recal']['macro_f1']:+.4f}  |  "
          f"ctx_oracle-recal QWK {results['ctx_oracle']['qwk'] - results['recal']['qwk']:+.4f} "
          f"F1 {results['ctx_oracle']['macro_f1'] - results['recal']['macro_f1']:+.4f}")

    print(f"\n=== Interior only, >= {args.buffer_m:.0f} m from quadrant boundary (n={int(interior.sum())}) ===")
    for name in ("raw", "recal", "ctx_pred", "ctx_oracle"):
        print(format_row(name, results_interior[name]))

    output_path = args.output_json or args.prob_cache.parent / f"probe_{cache['split'].replace('+', '_')}.json"
    payload = {
        "prob_cache": str(args.prob_cache), "split": cache["split"], "holdout_event": cache["holdout_event"],
        "n_buildings": len(targets), "radii_m": args.radii, "l2": args.l2, "buffer_m": args.buffer_m,
        "full_pool": results, "interior_pool": {"n": int(interior.sum()), "results": results_interior}
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nProbe artifact written: {output_path}")


if __name__ == "__main__":
    main()
