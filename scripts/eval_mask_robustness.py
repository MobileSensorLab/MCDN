"""Footprint misalignment / staleness robustness evaluation (T-6, reviewer R1-2).

Evaluates trained checkpoints under perturbed structure-footprint caches, reusing the
exact validation pipeline (fold reconstruction, chip windowing, 8-view D4 TTA, prediction
rules) from ``src.postproc.ensemble``. Three experiment arms map onto the T-6 design:

    (a) Real raw-cache evaluation (headline). The CRASAR-U-DROIDs annotation JSONs ship
        polygons at their *unadjusted* public-cache positions (Microsoft Building
        Footprints where ``source == "Microsoft"``); the training pipeline applies the
        dataset's tie-point alignment field at load time (``apply_alignment_adjustments``).
        The ``raw_cache`` condition therefore simply skips that adjustment step, which
        reproduces the genuine uncorrected footprint cache exactly — no inversion and no
        synthetic proxy. Manually digitized ``custom``-source polygons are likewise left
        at their shipped positions.
    (b) Calibrated synthetic sweep (dose-response). ``offset:<px>`` applies a per-mosaic
        coherent translation field to the aligned reference: one shared direction per
        orthomosaic with per-polygon wrapped-normal angular dispersion matched to the
        published within-mosaic circular variance (~0.28), at a fixed pixel magnitude.
        ``buffer:<+/-px>`` grows or shrinks footprints (dilation / erosion analogue).
    (c) Footprint deletion. ``mask_zero:<rate|auto>`` zeroes the mask channel for a
        deterministic random subset of chips, exercising the documented mask-pooling
        fallback to global statistics. ``auto`` calibrates the rate to the observed
        ``custom``-source polygon fraction in the holdout (a measure of real public-cache
        incompleteness).

Conditions are evaluated per variant and seed pool under the deployed inference
configuration (seed-ensemble x 8-view TTA), with per-seed dispersion and ensemble point
metrics under all prediction rules, plus deltas against the ``aligned`` reference
condition when it is included in the run.

Outputs:
    - printed per-condition table (all prediction rules) with deltas vs. ``aligned``
    - JSON artifact at ``outputs/ablation/mask_robustness.json`` (overridable)
    - optional per-condition probability caches ([seeds, samples, classes]) via
      ``--probs-cache-dir`` for later confusion / paired-delta analysis

Usage:
    uv run python -m scripts.eval_mask_robustness \
        --variant-roots outputs/ablation/baseline \
        --split Spatial_Block_East \
        --conditions aligned raw_cache offset:15 offset:60 offset:75 offset:150 offset:240 mask_zero:auto \
        --data-dir data \
        --device auto
"""

import argparse
import json
import math
import time
import zlib

import albumentations
import numpy as np
import pandas as pd
import rasterio
import torch

from dataclasses import asdict, dataclass, replace
from geopandas import GeoDataFrame
from pathlib import Path
from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader

from src.data.dataset import CRASARUnitemporalDataset
from src.data.transform import get_val_transforms
from src.model.trainer import ORDINAL_CLASS_DISPLAY_NAMES, _resolve_device, _seed_worker
from src.postproc.decompose import _summarize_across_seeds
from src.postproc.ensemble import (
    DEFAULT_SEEDS,
    RULE_FNS,
    build_model_from_config,
    get_fold_dataframes_from_config,
    load_resolved_config,
    metrics_under_all_rules,
    tta_mean_softmax_probs,
)


# Chip tensors are [RGB, mask] = [4, H, W]; the footprint mask occupies the last channel.
MASK_CHANNEL_INDEX = 3

# Within-mosaic circular variance of raw-vs-adjusted offset directions reported by
# Manzini et al. 2025 (mean across orthomosaics); default dispersion for the offset sweep.
DEFAULT_CIRCULAR_VARIANCE = 0.28

# Perturbation RNG seed, independent of the model training seeds so the same synthetic
# misalignment field is replayed identically for every seed and variant.
DEFAULT_PERTURBATION_SEED = 20260827

VALID_MODES = frozenset({"aligned", "raw_cache", "offset", "buffer", "mask_zero"})

REPORT_METRIC_KEYS = ("qwk", "macro_f1", "accuracy")


class SeedTargetMismatchError(Exception):
    """Raised when two seeds of one variant yield differently ordered validation targets."""


@dataclass(frozen=True)
class PerturbationSpec:
    """One footprint-perturbation condition.

    Args:
        mode: One of ``aligned``, ``raw_cache``, ``offset``, ``buffer``, ``mask_zero``.
        magnitude_px: Translation magnitude in orthomosaic pixels (``offset`` mode).
        circular_variance: Angular dispersion of the per-mosaic offset field (``offset`` mode).
        buffer_px: Signed footprint buffer in pixels; negative erodes (``buffer`` mode).
        rate: Fraction of chips whose mask channel is zeroed (``mask_zero`` mode);
            ``None`` requests calibration to the observed custom-source polygon rate.
        seed: Perturbation RNG seed, mixed with a per-mosaic hash for the offset field.
    """

    mode: str
    magnitude_px: float = 0.0
    circular_variance: float = DEFAULT_CIRCULAR_VARIANCE
    buffer_px: float = 0.0
    rate: float | None = None
    seed: int = DEFAULT_PERTURBATION_SEED

    @property
    def label(self) -> str:
        """Compact condition name used in tables, artifacts, and cache filenames."""

        if self.mode == "offset":
            return f"offset_{self.magnitude_px:g}px"
        if self.mode == "buffer":
            return f"buffer_{self.buffer_px:+g}px"
        if self.mode == "mask_zero":
            return "mask_zero_auto" if self.rate is None else f"mask_zero_{self.rate:g}"
        return self.mode


def parse_condition(spec_str: str, circular_variance: float = DEFAULT_CIRCULAR_VARIANCE,
                    perturbation_seed: int = DEFAULT_PERTURBATION_SEED) -> PerturbationSpec:
    """Parse a CLI condition token such as ``raw_cache``, ``offset:75``, or ``mask_zero:auto``.

    Args:
        spec_str: Condition token; parameterized modes take a single ``:``-separated argument.
        circular_variance: Angular dispersion applied to every ``offset`` condition.
        perturbation_seed: Base RNG seed recorded on every parsed spec.

    Returns:
        The parsed, validated PerturbationSpec.
    """

    head, _, arg = spec_str.strip().partition(":")
    mode = head.strip().lower()
    arg = arg.strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown condition mode '{mode}' in '{spec_str}'; expected one of {sorted(VALID_MODES)}.")

    if mode in ("aligned", "raw_cache"):
        if arg:
            raise ValueError(f"Condition '{mode}' takes no argument (got '{spec_str}').")
        return PerturbationSpec(mode=mode, seed=perturbation_seed)

    if mode == "offset":
        magnitude = float(arg)
        if magnitude <= 0:
            raise ValueError(f"offset magnitude must be positive (got '{spec_str}').")
        return PerturbationSpec(mode=mode, magnitude_px=magnitude, circular_variance=circular_variance, seed=perturbation_seed)

    if mode == "buffer":
        buffer_px = float(arg)
        if buffer_px == 0:
            raise ValueError(f"buffer offset must be non-zero (got '{spec_str}').")
        return PerturbationSpec(mode=mode, buffer_px=buffer_px, seed=perturbation_seed)

    # mask_zero: explicit rate in [0, 1], or "auto" for custom-source-rate calibration.
    if arg == "auto":
        return PerturbationSpec(mode=mode, rate=None, seed=perturbation_seed)
    rate = float(arg)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"mask_zero rate must lie in [0, 1] (got '{spec_str}').")
    return PerturbationSpec(mode=mode, rate=rate, seed=perturbation_seed)


# Synthetic offset field ------------------------------------------------------

def wrapped_normal_sigma(circular_variance: float) -> float:
    """Angular std of a wrapped normal with the requested circular variance.

    Uses the wrapped-normal identity ``CV = 1 - exp(-sigma^2 / 2)``, so
    ``sigma = sqrt(-2 ln(1 - CV))``; CV = 0 collapses to a shared direction.
    """

    if not 0.0 <= circular_variance < 1.0:
        raise ValueError(f"circular_variance must lie in [0, 1) (got {circular_variance}).")

    # CV = 0 yields -0.0 under the log identity; normalize so numpy accepts the scale.
    variance_term = -2.0 * math.log(1.0 - circular_variance)
    return math.sqrt(variance_term) if variance_term > 0.0 else 0.0


def mosaic_rng(base_seed: int, image_path: Path | str) -> np.random.Generator:
    """Deterministic per-orthomosaic generator, stable across runs, seeds, and variants."""

    name_hash = zlib.crc32(Path(image_path).name.encode("utf-8"))
    return np.random.default_rng([base_seed, name_hash])


def sample_offset_field(num_polygons: int, magnitude_px: float, circular_variance: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample one mosaic's coherent translation field in pixel units.

    One shared direction is drawn per mosaic; per-polygon angles disperse around it with
    a wrapped-normal std matched to ``circular_variance``, replicating the spatially
    correlated (not i.i.d.) structure of real footprint registration lag.

    Returns:
        Array [N, 2] of (dx, dy) pixel offsets with ``|offset| == magnitude_px`` per row.
    """

    base_angle = rng.uniform(0.0, 2.0 * math.pi)
    sigma = wrapped_normal_sigma(circular_variance)
    angles = base_angle + rng.normal(loc=0.0, scale=sigma, size=num_polygons)
    return np.stack([magnitude_px * np.cos(angles), magnitude_px * np.sin(angles)], axis=1)


def _linear_pixel_shift(transform: rasterio.Affine, dx_px: float, dy_px: float) -> tuple[float, float]:
    """Map a pixel-space offset through the linear part of a raster affine transform."""

    return transform.a * dx_px + transform.b * dy_px, transform.d * dx_px + transform.e * dy_px


def apply_coherent_offsets(gdf: GeoDataFrame, raster_path: Path | str, magnitude_px: float,
                           circular_variance: float, rng: np.random.Generator) -> GeoDataFrame:
    """Translate every polygon by one draw from the mosaic's coherent offset field."""

    with rasterio.open(raster_path) as source:
        transform = source.transform

    field = sample_offset_field(num_polygons=len(gdf), magnitude_px=magnitude_px,
                                circular_variance=circular_variance, rng=rng)
    output = gdf.copy()
    output.geometry = [
        translate(geom, *_linear_pixel_shift(transform, dx_px, dy_px))
        for geom, (dx_px, dy_px) in zip(gdf.geometry, field, strict=True)
    ]
    return output


def apply_footprint_buffer(gdf: GeoDataFrame, raster_path: Path | str, buffer_px: float) -> GeoDataFrame:
    """Buffer every polygon by a signed pixel distance (negative erodes).

    Footprints eroded to nothing are replaced by a sub-pixel marker at the original
    centroid: chip centering is preserved while the rasterized mask stays empty, so a
    fully eroded footprint degrades into the mask-deletion condition rather than
    dropping the instance.
    """

    with rasterio.open(raster_path) as source:
        transform = source.transform
    pixel_size = (abs(transform.a) + abs(transform.e)) / 2.0

    def buffer_geometry(geometry: BaseGeometry) -> BaseGeometry:
        buffered = geometry.buffer(buffer_px * pixel_size)
        if buffered.is_empty:
            return geometry.centroid.buffer(pixel_size * 1e-6)
        return buffered

    output = gdf.copy()
    output.geometry = output.geometry.apply(buffer_geometry)
    return output


# Perturbed dataset -----------------------------------------------------------

class RobustnessDataset(CRASARUnitemporalDataset):
    """Evaluation-only CRASAR dataset with a footprint-cache perturbation applied.

    Geometry perturbations run inside ``_load_clean_polygons`` so downstream behavior
    (chip centering on the perturbed centroid, window rasterization of every intersecting
    footprint, augmentation-free val transforms) exactly mirrors what a deployed pipeline
    fed the perturbed cache would produce. ``mask_zero`` instead intercepts the final
    chip tensor and zeroes the mask channel for a deterministic subset of instances.

    Args:
        manifest: Validation manifest slice for one fold.
        perturbation: Condition specification; ``mask_zero`` rate must be resolved
            (not ``None``) before construction.
        chip_size: Square chip edge in pixels.
        transform: Albumentations pipeline (validation transforms; may be ``None``).
        mask_dilation_px: Post-augmentation mask dilation radius, mirroring training.
        cache_validation_tensors: Cache deterministic chips across epochs/seeds.
    """

    def __init__(self, manifest: pd.DataFrame, perturbation: PerturbationSpec, chip_size: int = 512,
                 transform: albumentations.Compose | None = None, mask_dilation_px: int = 0,
                 cache_validation_tensors: bool = False) -> None:
        """Initialize the perturbed dataset over one fold's validation manifest."""

        if perturbation.mode == "mask_zero" and perturbation.rate is None:
            raise ValueError("mask_zero rate must be resolved before dataset construction (got rate=None).")

        # The base initializer builds the instance index, which routes through our
        # _load_clean_polygons override, so the spec must be attached first.
        self.perturbation = perturbation
        super().__init__(manifest, chip_size=chip_size, transform=transform, is_train=False,
                         cache_validation_tensors=cache_validation_tensors, mask_dilation_px=mask_dilation_px)

        if perturbation.mode == "mask_zero":
            rng = np.random.default_rng(perturbation.seed)
            num_zeroed = round(perturbation.rate * len(self.instances))
            zeroed = rng.choice(len(self.instances), size=num_zeroed, replace=False)
            self._zeroed_indices = frozenset(int(index) for index in zeroed)
        else:
            self._zeroed_indices = frozenset()

    def _load_clean_polygons(self, image_path: Path, label_path: Path, alignment_path: Path | None) -> GeoDataFrame:
        """Load one mosaic's polygons with the configured cache perturbation applied."""

        spec = self.perturbation
        if spec.mode == "raw_cache":
            # Shipped polygon coordinates are the unadjusted public-cache positions;
            # skipping the tie-point adjustment reproduces the raw cache exactly.
            return CRASARUnitemporalDataset._load_clean_polygons(
                image_path=image_path, label_path=label_path, alignment_path=None)

        gdf = CRASARUnitemporalDataset._load_clean_polygons(
            image_path=image_path, label_path=label_path, alignment_path=alignment_path)
        if gdf.empty or spec.mode in ("aligned", "mask_zero"):
            return gdf
        if spec.mode == "offset":
            return apply_coherent_offsets(gdf=gdf, raster_path=image_path, magnitude_px=spec.magnitude_px,
                                          circular_variance=spec.circular_variance,
                                          rng=mosaic_rng(spec.seed, image_path))
        return apply_footprint_buffer(gdf=gdf, raster_path=image_path, buffer_px=spec.buffer_px)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a chip, zeroing the footprint-mask channel for deletion-flagged instances."""

        item = super().__getitem__(idx)
        if idx in self._zeroed_indices:
            item["image"][MASK_CHANNEL_INDEX].zero_()
        return item


# Custom-source rate (mask_zero:auto calibration) -----------------------------

def count_polygon_sources(label_paths: list[str]) -> dict[str, int]:
    """Tally the per-polygon ``source`` field across raw CRASAR annotation JSONs.

    Counts every entry (including labels later filtered from the ordinal set), which is
    adequate for calibrating the deletion rate; entries without a ``source`` field are
    tallied under ``unknown``.
    """

    counts: dict[str, int] = {}
    for label_path in label_paths:
        try:
            data = json.loads(Path(label_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"WARNING: skipping unreadable annotation file {label_path}: {error}")
            continue
        for item in data:
            if isinstance(item, dict):
                source = str(item.get("source", "unknown"))
                counts[source] = counts.get(source, 0) + 1
    return counts


def resolve_auto_rate(val_df: pd.DataFrame) -> float:
    """Observed custom-source polygon fraction over one holdout's annotation files."""

    counts = count_polygon_sources(list(val_df["label_path"]))
    total = sum(counts.values())
    custom = counts.get("custom", 0)
    rate = custom / total if total else 0.0
    print(f"  custom-source polygons: {custom}/{total} = {rate:.4f}  (per-source: {counts})")
    return rate


# Inference orchestration -----------------------------------------------------

def build_perturbed_val_loader(cfg: dict, val_df: pd.DataFrame, perturbation: PerturbationSpec) -> DataLoader:
    """Deterministic validation loader over a perturbed footprint cache.

    Mirrors the ensemble evaluation loader (val transforms, mask dilation, batch sizing)
    with the RobustnessDataset substituted for the stock dataset.
    """

    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]
    training_cfg = cfg["training"]
    ablation = cfg["ablation"]

    dataset = RobustnessDataset(
        val_df,
        perturbation=perturbation,
        chip_size=data_cfg["chip_size"],
        transform=get_val_transforms(synthetic_gsd_factor=data_cfg.get("synthetic_gsd_factor", 1.0)),
        mask_dilation_px=ablation.get("mask_dilation_px", 0),
        cache_validation_tensors=False,
    )

    if runtime_cfg["persistent_workers"] and runtime_cfg["num_workers"] == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    generator = torch.Generator()
    seed = runtime_cfg.get("seed")
    if seed is not None:
        generator.manual_seed(int(seed))

    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": max(1, int(training_cfg["batch_size"] * runtime_cfg["val_batch_size_factor"])),
        "shuffle": False,
        "num_workers": runtime_cfg["num_workers"],
        "pin_memory": runtime_cfg["pin_memory"],
        "persistent_workers": runtime_cfg["persistent_workers"],
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if runtime_cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = runtime_cfg["prefetch_factor"]
    return DataLoader(**loader_kwargs)


@torch.no_grad()
def collect_tta_probs(model: torch.nn.Module, val_loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run 8-view TTA over a loader and return mean softmax probs [T, K] with targets [T]."""

    prob_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for batch in val_loader:
        probs = tta_mean_softmax_probs(model=model, images_u8=batch["image"], context=batch["context"], device=device)
        prob_chunks.append(probs.cpu())
        target_chunks.append(batch["label"].cpu())
    return torch.cat(prob_chunks, dim=0), torch.cat(target_chunks, dim=0)


def evaluate_condition(fold_dirs: dict[int, Path], cfg: dict, val_df: pd.DataFrame,
                       spec: PerturbationSpec, device: str) -> dict:
    """Evaluate every seed of one variant under one perturbation condition.

    Returns:
        Record with the condition spec, per-seed metrics, across-seed summary, ensemble
        (mean-softmax) metrics, and the stacked probability tensor [S, T, K] + targets.
    """

    loader = build_perturbed_val_loader(cfg=cfg, val_df=val_df, perturbation=spec)

    seed_probs: list[torch.Tensor] = []
    reference_targets: torch.Tensor | None = None
    ordered_seeds = sorted(fold_dirs.keys())

    for seed in ordered_seeds:
        fold_dir = fold_dirs[seed]
        fold_cfg = load_resolved_config(fold_dir)
        model = build_model_from_config(cfg=fold_cfg, device=device)
        state_dict = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        t0 = time.time()
        probs, targets = collect_tta_probs(model=model, val_loader=loader, device=device)
        print(f"    seed {seed}: probs {tuple(probs.shape)}  ({time.time() - t0:.1f}s)")

        if reference_targets is None:
            reference_targets = targets
        elif not torch.equal(reference_targets, targets):
            raise SeedTargetMismatchError(
                f"Validation target order diverged between seeds {ordered_seeds[0]} and {seed} "
                f"under condition '{spec.label}'.")

        seed_probs.append(probs)
        del model, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stacked = torch.stack(seed_probs, dim=0)  # [S, T, K]
    per_seed = [
        {"seed": seed, "metrics": metrics_under_all_rules(probs=stacked[index], targets=reference_targets)}
        for index, seed in enumerate(ordered_seeds)
    ]
    return {
        "condition": spec.label,
        "spec": asdict(spec),
        "seeds": ordered_seeds,
        "per_seed": per_seed,
        "summary": _summarize_across_seeds([record["metrics"] for record in per_seed]),
        "ensemble": metrics_under_all_rules(probs=stacked.mean(dim=0), targets=reference_targets),
        "probs": stacked,
        "targets": reference_targets,
    }


def run_variant_robustness(variant_root: Path, seeds: list[int], split_name: str,
                           condition_specs: list[PerturbationSpec], data_dir: str | None,
                           device: str, probs_cache_dir: Path | None = None) -> dict:
    """Evaluate one variant's seed pool under every requested perturbation condition."""

    fold_dirs: dict[int, Path] = {}
    for seed in seeds:
        candidate = variant_root / split_name / f"seed_{seed:02d}"
        if candidate.is_dir():
            fold_dirs[seed] = candidate
    if not fold_dirs:
        raise FileNotFoundError(f"No seed folds found under {variant_root / split_name}")

    print(f"\n=== {variant_root.name} ===  seeds present: {sorted(fold_dirs.keys())}")
    first_cfg = load_resolved_config(fold_dirs[sorted(fold_dirs.keys())[0]])
    _train_df, val_df, holdout_event = get_fold_dataframes_from_config(first_cfg, data_dir)
    del _train_df
    print(f"  holdout={holdout_event}  val mosaics={len(val_df)}")

    # Resolve auto-calibrated deletion rates once per variant/holdout.
    if any(spec.mode == "mask_zero" and spec.rate is None for spec in condition_specs):
        auto_rate = resolve_auto_rate(val_df)
        condition_specs = [
            replace(spec, rate=auto_rate) if spec.mode == "mask_zero" and spec.rate is None else spec
            for spec in condition_specs
        ]

    condition_records: list[dict] = []
    for spec in condition_specs:
        print(f"  --- condition: {spec.label} ---")
        record = evaluate_condition(fold_dirs=fold_dirs, cfg=first_cfg, val_df=val_df, spec=spec, device=device)
        condition_records.append(record)
        if probs_cache_dir is not None:
            probs_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = probs_cache_dir / f"{variant_root.name}_{split_name}_{record['condition']}.pt"
            torch.save({"condition": record["condition"], "spec": record["spec"], "seeds": record["seeds"],
                        "probs": record["probs"], "targets": record["targets"],
                        "class_names": list(ORDINAL_CLASS_DISPLAY_NAMES)}, cache_path)
            print(f"    probability cache written: {cache_path}")

    return {
        "variant_root": str(variant_root),
        "holdout_event": holdout_event,
        "conditions": condition_records,
    }


# Reporting -------------------------------------------------------------------

def compute_deltas_vs_aligned(condition_records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Ensemble-metric deltas of every condition against the ``aligned`` reference.

    Returns an empty mapping when no ``aligned`` condition was evaluated.
    """

    aligned = next((record for record in condition_records if record["condition"] == "aligned"), None)
    if aligned is None:
        return {}

    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for record in condition_records:
        if record is aligned:
            continue
        deltas[record["condition"]] = {
            rule_name: {
                metric: record["ensemble"][rule_name][metric] - aligned["ensemble"][rule_name][metric]
                for metric in REPORT_METRIC_KEYS
            }
            for rule_name in RULE_FNS
        }
    return deltas


def print_robustness_table(variant_record: dict) -> None:
    """Print per-condition ensemble metrics (all rules) with deltas vs. ``aligned``."""

    conditions = variant_record["conditions"]
    deltas = compute_deltas_vs_aligned(conditions)

    print(f"\n{'=' * 110}")
    print(f"FOOTPRINT ROBUSTNESS  variant={Path(variant_record['variant_root']).name}  "
          f"holdout={variant_record['holdout_event']}  seeds={len(conditions[0]['seeds'])}")
    print("=" * 110)

    for rule_name in RULE_FNS:
        print(f"\n  rule={rule_name}")
        print(f"    {'condition':<20s}  {'ens QWK':>9s}  {'dQWK':>8s}  {'ens mF1':>9s}  {'ens acc':>9s}  "
              f"{'seed QWK mean +/- std':>22s}")
        for record in conditions:
            ensemble = record["ensemble"][rule_name]
            summary = record["summary"][rule_name]["qwk"]
            delta = deltas.get(record["condition"], {}).get(rule_name, {}).get("qwk")
            delta_text = f"{delta:+8.4f}" if delta is not None else f"{'--':>8s}"
            print(f"    {record['condition']:<20s}  {ensemble['qwk']:>9.4f}  {delta_text}  "
                  f"{ensemble['macro_f1']:>9.4f}  {ensemble['accuracy']:>9.4f}  "
                  f"{summary['mean']:>13.4f} +/- {summary['std']:.4f}")


def write_robustness_artifact(path: Path, variant_records: list[dict], split_name: str) -> None:
    """Write the JSON artifact covering every variant and condition (tensors excluded)."""

    payload = {
        "split": split_name,
        "holdout_event": variant_records[0]["holdout_event"],
        "class_names": list(ORDINAL_CLASS_DISPLAY_NAMES),
        "variants": [
            {
                "variant_root": record["variant_root"],
                "conditions": [
                    {key: condition[key] for key in ("condition", "spec", "seeds", "per_seed", "summary", "ensemble")}
                    for condition in record["conditions"]
                ],
                "deltas_vs_aligned": compute_deltas_vs_aligned(record["conditions"]),
            }
            for record in variant_records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nArtifact written: {path}")


def main() -> None:
    """CLI entry point for the footprint-robustness evaluation."""

    parser = argparse.ArgumentParser(description="Footprint misalignment / staleness robustness evaluation (T-6).")
    parser.add_argument("--variant-roots", type=Path, nargs="+", required=True,
                        help="One or more outputs/ablation/<variant> directories.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to attempt per variant (missing ones are silently skipped).")
    parser.add_argument("--split", type=str, default="Spatial_Block_East",
                        help="Split/fold directory name under each variant root.")
    parser.add_argument("--conditions", type=str, nargs="+", default=["aligned", "raw_cache"],
                        help="Condition tokens: aligned, raw_cache, offset:<px>, buffer:<+/-px>, mask_zero:<rate|auto>.")
    parser.add_argument("--circular-variance", type=float, default=DEFAULT_CIRCULAR_VARIANCE,
                        help="Angular dispersion for every offset condition (published within-mosaic mean: 0.28).")
    parser.add_argument("--perturbation-seed", type=int, default=DEFAULT_PERTURBATION_SEED,
                        help="Base RNG seed for offset fields and mask-deletion sampling.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override the data root recorded in each seed's config_resolved.yaml.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/ablation/mask_robustness.json"))
    parser.add_argument("--probs-cache-dir", type=Path, default=None,
                        help="Optional directory for per-condition probability caches (.pt).")
    parser.add_argument("--report-custom-rate", action="store_true",
                        help="Only report the custom-source polygon rate per variant holdout, then exit.")
    args = parser.parse_args()

    condition_specs = [
        parse_condition(token, circular_variance=args.circular_variance, perturbation_seed=args.perturbation_seed)
        for token in args.conditions
    ]

    if args.report_custom_rate:
        for variant_root in args.variant_roots:
            seed_dirs = sorted(path for path in (variant_root / args.split).glob("seed_*") if path.is_dir())
            if not seed_dirs:
                raise FileNotFoundError(f"No seed folds found under {variant_root / args.split}")
            cfg = load_resolved_config(seed_dirs[0])
            _train_df, val_df, holdout_event = get_fold_dataframes_from_config(cfg, args.data_dir)
            print(f"\n=== {variant_root.name} ===  holdout={holdout_event}")
            resolve_auto_rate(val_df)
        return

    resolved_device = _resolve_device(device=args.device)
    print(f"Device: {resolved_device}")
    print(f"Variants: {[path.name for path in args.variant_roots]}")
    print(f"Split: {args.split}")
    print(f"Conditions: {[spec.label for spec in condition_specs]}")

    variant_records = [
        run_variant_robustness(
            variant_root=variant_root,
            seeds=args.seeds,
            split_name=args.split,
            condition_specs=condition_specs,
            data_dir=args.data_dir,
            device=resolved_device,
            probs_cache_dir=args.probs_cache_dir,
        )
        for variant_root in args.variant_roots
    ]

    for record in variant_records:
        print_robustness_table(record)

    write_robustness_artifact(path=args.output_json, variant_records=variant_records, split_name=args.split)


if __name__ == "__main__":
    main()
