"""Root execution script for the hybrid unitemporal damage classifier."""

import argparse
import json
import os
import statistics
import sys
import yaml

from pathlib import Path

ABLATION_PRESET_FILES = {
    "baseline": "ablation_baseline.yaml",
    "mask": "ablation_mask.yaml",
    "typology": "ablation_typology.yaml",
    "resolution": "ablation_resolution.yaml",
    "rgb_only": "ablation_rgb_only.yaml",
    "no_smoothing": "ablation_no_smoothing.yaml",
    "mask_channel_only": "ablation_mask_channel_only.yaml",
    "pooling_only": "ablation_pooling_only.yaml",
    "ce_loss": "ablation_ce_loss.yaml",
    "downsample_15cm": "ablation_downsample_15cm.yaml",
    "deployed_split": "ablation_deployed_split.yaml"
}

def _resolve_early_config_path_for_cuda_launch_blocking() -> Path:
    """Match train.py CLI resolution for which YAML is loaded (before argparse runs)."""

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if argv[i].startswith("--config="):
            return Path(argv[i].split("=", 1)[1])
        i += 1
    if "--ablation-preset" in argv:
        j = argv.index("--ablation-preset")
        if j + 1 < len(argv):
            key = argv[j + 1]
            if key in ABLATION_PRESET_FILES:
                return Path("config") / "presets" / ABLATION_PRESET_FILES[key]
    return Path("config") / "config.yaml"


def _apply_cuda_launch_blocking_early() -> None:
    """Set CUDA_LAUNCH_BLOCKING before any PyTorch import when requested.

    Honors ``GEOG594_CUDA_LAUNCH_BLOCKING`` (1/true/yes), ``--cuda-launch-blocking``,
    or ``runtime.cuda_launch_blocking: true`` in the resolved config YAML. Set
    ``GEOG594_CUDA_LAUNCH_BLOCKING`` to 0/false/no to force-disable. Runs before
    ``torch`` initializes CUDA.
    """

    env = os.environ.get("GEOG594_CUDA_LAUNCH_BLOCKING", "").strip().lower()
    if env in {"0", "false", "no"}:
        return
    if env in {"1", "true", "yes"}:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        return
    if "--cuda-launch-blocking" in sys.argv:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        return
    path = _resolve_early_config_path_for_cuda_launch_blocking()
    if not path.is_file():
        return
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(raw, dict):
        return
    runtime = raw.get("runtime")
    if isinstance(runtime, dict) and runtime.get("cuda_launch_blocking") is True:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

_apply_cuda_launch_blocking_early()

from src.config.settings import AppConfig, load_config
from src.model.trainer import run_training_pipeline

FIXED_ABLATION_SEEDS = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

REQUIRED_RUN_ARTIFACTS = ("metrics.json", "config_resolved.yaml", "best_model.pt")


def _seed_run_is_complete(checkpoint_root: Path) -> bool:
    """True when a run directory already holds the full per-seed artifact triplet.

    Keyed on the same three files the aggregation step requires, so a skipped run is by
    definition one downstream tooling can consume. Partial artifacts (e.g. a preempted
    run that only wrote a log) never trigger a skip.
    """

    return all((checkpoint_root / name).is_file() for name in REQUIRED_RUN_ARTIFACTS)


def _resolve_split_dir_name(holdout_event: str | list[str] | None) -> str:
    """Directory name for the configured holdout under the ablation output layout.

    Multi-event holdouts join their sorted event names with ``+`` to mirror the composite
    fold name emitted by ``build_multi_event_fold``.
    """

    if isinstance(holdout_event, list):
        return "+".join(sorted(holdout_event)).replace(" ", "_")
    return (holdout_event or "Spatial_Block_East").replace(" ", "_")


def _ablation_variant_root(preset: str) -> Path:
    """Output directory for a named ablation preset under outputs/ablation/."""

    return Path("outputs") / "ablation" / preset


def _parse_bool(v: str) -> bool:
    """Parse boolean from string."""
    return str(v).lower() in {"yes", "true", "t", "1"}


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for config path and optional overrides."""

    parser = argparse.ArgumentParser(description="Train the hybrid unitemporal classifier.")
    parser.add_argument("--config", type=Path, default=Path("config") / "config.yaml", help="Path to YAML or JSON config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--accum-steps", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--sampler-mode", type=str, choices=["weighted", "uniform"], default=None, help="Override sampler mode.")
    parser.add_argument("--class-weighting-enabled", type=_parse_bool, default=None, help="Override class weighting enabled.")
    parser.add_argument("--class-weighting-strategy", type=str, default=None, help="Override class weighting strategy.")
    parser.add_argument("--class-weighting-power", type=float, default=None, help="Override class weighting power.")
    parser.add_argument("--class-weighting-eps", type=float, default=None, help="Override class weighting eps.")
    parser.add_argument("--class-weighting-max-ratio", type=float, default=None, help="Override class weighting max ratio.")
    parser.add_argument("--chip-size", type=int, default=None, help="Override data chip size.")
    parser.add_argument("--holdout-event", type=str, default=None, help="Override holdout event for LOEO.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory path.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Execute one run per provided seed value.")
    parser.add_argument("--fixed-seeds", action="store_true", help="Use the fixed ablation seed protocol: 0, 11, 22, 33, 44, 55, 66, 77, 88, 99.")
    parser.add_argument(
        "--ablation-preset",
        type=str,
        choices=sorted(ABLATION_PRESET_FILES.keys()),
        default=None,
        help="Run using a named preset from config/presets."
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip a seed whose checkpoint root already holds metrics.json, config_resolved.yaml, and best_model.pt (idempotent backfill for requeued or resubmitted job arrays)."
    )
    parser.add_argument(
        "--cuda-launch-blocking",
        action="store_true",
        help="Set CUDA_LAUNCH_BLOCKING=1 before importing torch (sync CUDA; slower). Same as runtime.cuda_launch_blocking: true or GEOG594_CUDA_LAUNCH_BLOCKING=1."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit verbose per-batch tabular printouts during training (default: labeled-prose summaries with heartbeats)."
    )
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI overrides to loaded config and re-validate."""

    payload = config.model_dump(mode="python")
    if args.epochs is not None:
        payload["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        payload["training"]["batch_size"] = args.batch_size
    if args.accum_steps is not None:
        payload["training"]["accum_steps"] = args.accum_steps
    if args.lr is not None:
        payload["training"]["lr"] = args.lr
    if getattr(args, "sampler_mode", None) is not None:
        payload["training"]["sampler_mode"] = args.sampler_mode
    if getattr(args, "class_weighting_enabled", None) is not None:
        payload["training"]["class_weighting_enabled"] = args.class_weighting_enabled
    if getattr(args, "class_weighting_strategy", None) is not None:
        payload["training"]["class_weighting_strategy"] = args.class_weighting_strategy
    if getattr(args, "class_weighting_power", None) is not None:
        payload["training"]["class_weighting_power"] = args.class_weighting_power
    if getattr(args, "class_weighting_eps", None) is not None:
        payload["training"]["class_weighting_eps"] = args.class_weighting_eps
    if getattr(args, "class_weighting_max_ratio", None) is not None:
        payload["training"]["class_weighting_max_ratio"] = args.class_weighting_max_ratio
    if args.chip_size is not None:
        payload["data"]["chip_size"] = args.chip_size
    if args.holdout_event is not None:
        payload["data"]["holdout_event"] = args.holdout_event
    if args.data_dir is not None:
        payload["data"]["dir"] = args.data_dir
    if getattr(args, "verbose", False):
        payload.setdefault("runtime", {})["verbose"] = True
    # Auto-enable file logging for ablation runs so multi-seed sweeps always produce a tee'd
    # train_log.txt alongside their metrics without requiring an additional flag.
    if getattr(args, "ablation_preset", None) is not None:
        payload.setdefault("runtime", {})["log_to_file"] = True
    return AppConfig.model_validate(payload)


def _compute_seed_macro_qwk(variant_root: Path, seed: int) -> tuple[float, int]:
    """Compute macro-QWK across every ``<split>/seed_<seed>`` under ``variant_root``.

    Reflects the canonical ablation layout where each (split, seed) pair writes its
    metrics directly into ``<variant>/<split>/seed_<n>/metrics.json``. Returns the mean
    ``best_val_qwk`` across available splits and the number of splits contributing.
    """

    split_dirs = sorted(directory for directory in variant_root.iterdir() if directory.is_dir())
    fold_scores: list[float] = []
    for split_dir in split_dirs:
        seed_dir = split_dir / f"seed_{seed:02d}"
        metrics_file = seed_dir / "metrics.json"
        config_file = seed_dir / "config_resolved.yaml"
        model_file = seed_dir / "best_model.pt"
        if not metrics_file.is_file():
            continue
        missing_artifacts = [str(path.name) for path in (metrics_file, config_file, model_file) if not path.exists()]
        if missing_artifacts:
            raise ValueError(f"Missing required fold artifacts in {seed_dir}: {missing_artifacts}")

        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        if "best_val_qwk" not in payload:
            raise ValueError(f"Missing 'best_val_qwk' in metrics file: {metrics_file}")
        fold_scores.append(float(payload["best_val_qwk"]))

    if not fold_scores:
        raise ValueError(f"No split metrics found for seed {seed} under variant root: {variant_root}")

    return statistics.mean(fold_scores), len(fold_scores)


def _aggregate_ablation_metrics(variant_root: Path, expected_seeds: list[int]) -> dict[str, object]:
    """Aggregate seed-level macro-QWK and persist summary metrics for ablation runs."""

    per_seed: list[dict[str, float | int]] = []
    missing_seeds: list[int] = []
    seed_macros: list[float] = []
    for seed in expected_seeds:
        has_any_split = any(
            (split_dir / f"seed_{seed:02d}" / "metrics.json").is_file()
            for split_dir in variant_root.iterdir()
            if split_dir.is_dir()
        )
        if not has_any_split:
            missing_seeds.append(seed)
            continue

        macro_qwk, fold_count = _compute_seed_macro_qwk(variant_root=variant_root, seed=seed)
        seed_macros.append(macro_qwk)
        per_seed.append({
            "seed": seed,
            "macro_qwk": macro_qwk,
            "num_folds": fold_count
        })

    if missing_seeds:
        raise ValueError(f"Missing expected seed outputs for aggregation: {missing_seeds}")
    if not seed_macros:
        raise ValueError(f"No seed metrics found under ablation variant root: {variant_root}")

    payload = {
        "variant": variant_root.name,
        "metric": "best_val_qwk",
        "seed_protocol": expected_seeds,
        "seed_macro_qwk": per_seed,
        "macro_qwk_mean": statistics.mean(seed_macros),
        "macro_qwk_std": 0.0 if len(seed_macros) == 1 else statistics.stdev(seed_macros)
    }
    summary_path = variant_root / "summary_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Ablation summary saved: {summary_path}")
    return payload


def main() -> None:
    """Load runtime config, apply optional overrides, and launch training."""

    parser = _build_parser()
    args = parser.parse_args()
    if args.fixed_seeds and args.seeds is not None:
        parser.error("Use either --seeds or --fixed-seeds, not both.")
    if args.fixed_seeds and args.ablation_preset is None:
        parser.error("--fixed-seeds requires --ablation-preset.")

    if args.ablation_preset is not None:
        preset_path = Path("config") / "presets" / ABLATION_PRESET_FILES[args.ablation_preset]
        config = load_config(preset_path)
    else:
        config = load_config(args.config)
    config = _apply_overrides(config=config, args=args)
    if args.fixed_seeds:
        seed_values: list[int | None] = FIXED_ABLATION_SEEDS
    elif args.seeds is not None:
        seed_values: list[int | None] = list(dict.fromkeys(args.seeds))
    else:
        seed_values = [config.runtime.seed]

    for seed in seed_values:
        if args.ablation_preset is not None:
            ablation_root = _ablation_variant_root(args.ablation_preset)
            split_name = _resolve_split_dir_name(config.data.holdout_event)
            checkpoint_root = ablation_root / split_name / f"seed_{seed:02d}"
        else:
            checkpoint_root = config.runtime.checkpoint_root if len(seed_values) == 1 else config.runtime.checkpoint_root / f"seed_{seed:02d}"
        if getattr(args, "skip_if_complete", False) and _seed_run_is_complete(Path(checkpoint_root)):
            print(f"Skipping seed {seed}: complete artifacts already present under {checkpoint_root}")
            continue
        run_training_pipeline(
            config=config,
            seed=seed,
            checkpoint_root=str(checkpoint_root),
        )

    if args.ablation_preset is not None and args.fixed_seeds:
        _aggregate_ablation_metrics(
            variant_root=_ablation_variant_root(args.ablation_preset),
            expected_seeds=FIXED_ABLATION_SEEDS,
        )


if __name__ == "__main__":
    main()
