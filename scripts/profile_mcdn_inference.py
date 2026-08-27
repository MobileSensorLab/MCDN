"""Profile MCDN inference: latency, fvcore FLOPs, and torch.profiler summaries for edge-deployment reporting.

Two modes:
    - default (forward): bare-forward latency, fvcore FLOPs, and a torch.profiler table
      for one checkpoint — kernel-level diagnostics.
    - ``--deployment``: measured latency for the three deployment configurations (T-8) —
      single model / no TTA, single model / 8-view D4 TTA, and the seed-ensemble x 8-view
      TTA headline — timed through the actual inference entry points used at evaluation
      time (``tta_mean_softmax_probs`` / ``ensemble_mean_tta_probs``), so uint8-to-float
      normalization, view transforms, autocast, and softmax are all inside the timed
      region. Reports chips/sec, per-chip latency, peak CUDA memory, and derived
      time-to-full-holdout figures (val chip counts read from ``split_summary.json``)
      plus the 415-building operational yardstick.

Deployment-mode usage:
    uv run python -m scripts.profile_mcdn_inference --deployment \
        --checkpoint-dir outputs/ablation/baseline/Spatial_Block_East/seed_00 \
        --deployment-batch-sizes 1 16 64 --output-json outputs/ablation/latency_profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from fvcore.nn import FlopCountAnalysis

from src.data.dataset import to_normalized_float
from src.model.mcdn import MaskCenteredDamageNet
from src.postproc.ensemble import ensemble_mean_tta_probs, tta_mean_softmax_probs

DEFAULT_CHECKPOINT_DIR = Path("outputs/ablation/baseline/Spatial_Block_East/seed_00")

# Operational yardstick: the deployed CRASAR baseline assessed 415 buildings in ~18 min
# at Hurricanes Debby/Helene; used as a citable time-to-assessment anchor in T-8.
YARDSTICK_CHIPS = 415


class ProfileConfigError(ValueError):
    """Raised when ``config_resolved.yaml`` is missing required keys for profiling."""


@dataclass(frozen=True)
class ResolvedFoldConfig:
    """Subset of training snapshot fields required to rebuild ``MaskCenteredDamageNet``."""

    chip_size: int
    model_name: str
    model_pretrained: bool
    mask_enabled: bool
    typology_enabled: bool
    mask_weighted_pooling_enabled: bool


def load_resolved_fold_config(config_path: Path) -> ResolvedFoldConfig:
    """Parse ``config_resolved.yaml`` written by the training pipeline.

    Args:
        config_path: Path to ``config_resolved.yaml`` beside ``best_model.pt``.

    Returns:
        Fields needed to construct the network and synthetic tensors.

    Raises:
        ProfileConfigError: If required nested keys are absent or invalid.
    """

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Expected mapping at root of YAML: {config_path}"
        raise ProfileConfigError(msg)

    data = raw.get("data")
    model = raw.get("model")
    ablation = raw.get("ablation")
    if not isinstance(data, dict) or not isinstance(model, dict) or not isinstance(ablation, dict):
        msg = f"Missing or invalid 'data', 'model', or 'ablation' in {config_path}"
        raise ProfileConfigError(msg)

    chip = data.get("chip_size")
    name = model.get("name")
    pretrained = model.get("pretrained")
    mask_enabled = ablation.get("mask_enabled")
    typology_enabled = ablation.get("typology_enabled")
    if not isinstance(chip, int) or chip <= 0:
        msg = f"data.chip_size must be a positive int in {config_path}"
        raise ProfileConfigError(msg)
    if not isinstance(name, str) or not name:
        msg = f"model.name must be a non-empty string in {config_path}"
        raise ProfileConfigError(msg)
    if not isinstance(pretrained, bool):
        msg = f"model.pretrained must be a bool in {config_path}"
        raise ProfileConfigError(msg)
    if not isinstance(mask_enabled, bool) or not isinstance(typology_enabled, bool):
        msg = f"ablation.mask_enabled and ablation.typology_enabled must be bools in {config_path}"
        raise ProfileConfigError(msg)

    mask_weighted = bool(ablation.get("mask_weighted_pooling_enabled", False))

    return ResolvedFoldConfig(
        chip_size=chip,
        model_name=name,
        model_pretrained=pretrained,
        mask_enabled=mask_enabled,
        typology_enabled=typology_enabled,
        mask_weighted_pooling_enabled=mask_weighted,
    )


def build_model(cfg: ResolvedFoldConfig) -> MaskCenteredDamageNet:
    """Instantiate MCDN from resolved fold config."""

    return MaskCenteredDamageNet(
        backbone_name=cfg.model_name,
        pretrained=cfg.model_pretrained,
        mask_enabled=cfg.mask_enabled,
        typology_enabled=cfg.typology_enabled,
        mask_weighted_pooling_enabled=cfg.mask_weighted_pooling_enabled,
    )


def load_state_dict_into_model(model: nn.Module, weights_path: Path, device: torch.device) -> None:
    """Load ``best_model.pt`` state dict onto ``model``."""

    try:
        payload = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(weights_path, map_location=device)
    if not isinstance(payload, dict):
        msg = f"Checkpoint must contain a state_dict mapping: {weights_path}"
        raise TypeError(msg)
    model.load_state_dict(payload)


def synthetic_batch(
    *,
    batch_size: int,
    chip_size: int,
    mask_enabled: bool,
    typology_enabled: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic random-like tensors for forward profiling."""

    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    c = 4 if mask_enabled else 3
    x = torch.randn(batch_size, c, chip_size, chip_size, device=device, dtype=dtype, generator=rng)
    ctx = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    if typology_enabled:
        ctx[:, 0] = 1.0
    return x, ctx


def percentile_sorted(sorted_vals: list[float], p: float) -> float:
    """Return the ``p``-th percentile (0-100) of a pre-sorted sequence."""

    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    w = idx - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def _forward_step(
    model: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
) -> None:
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type=amp_device, dtype=torch.float16):
            model(x, context)
    else:
        model(x, context)


def benchmark_latency_ms(
    model: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    use_amp: bool,
) -> dict[str, float]:
    """Measure per-forward latency in milliseconds."""

    model.eval()
    times_ms: list[float] = []

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _forward_step(model, x, context, device=device, use_amp=use_amp)
            sync()

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iterations):
                sync()
                start.record()
                _forward_step(model, x, context, device=device, use_amp=use_amp)
                end.record()
                sync()
                times_ms.append(float(start.elapsed_time(end)))
        else:
            for _ in range(iterations):
                t_a = time.perf_counter()
                _forward_step(model, x, context, device=device, use_amp=use_amp)
                t_b = time.perf_counter()
                times_ms.append((t_b - t_a) * 1000.0)

    sorted_t = sorted(times_ms)
    batch = x.shape[0]
    per_chip_ms = [t / batch for t in sorted_t]
    return {
        "latency_batch_ms_mean": float(statistics.mean(times_ms)),
        "latency_batch_ms_std": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "latency_batch_ms_p50": percentile_sorted(sorted_t, 50.0),
        "latency_batch_ms_p95": percentile_sorted(sorted_t, 95.0),
        "latency_batch_ms_p99": percentile_sorted(sorted_t, 99.0),
        "latency_per_chip_ms_mean": float(statistics.mean(per_chip_ms)),
        "throughput_chips_per_s_mean": float(batch * 1000.0 / statistics.mean(times_ms)) if times_ms else float("nan"),
    }


def total_flops_fvcore(model: nn.Module, x: torch.Tensor, context: torch.Tensor) -> float:
    """Compute total FLOPs for one forward using fvcore (runs on CPU for analyzer stability)."""

    device_backup = next(model.parameters()).device
    try:
        model_cpu = model.cpu()
        x_cpu = x.detach().cpu()
        ctx_cpu = context.detach().cpu()
        model_cpu.eval()
        analysis = FlopCountAnalysis(model_cpu, (x_cpu, ctx_cpu))
        return float(analysis.total())
    finally:
        model.to(device_backup)


def profiler_table_string(
    model: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    *,
    device: torch.device,
    row_limit: int,
) -> str:
    """Run one profiled forward and return the ``key_averages`` table as text."""

    activities: list[Any] = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model.eval()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=False,
        acc_events=True,
    ) as prof:
        with torch.inference_mode():
            model(x, context)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    return prof.key_averages().table(sort_by=sort_key, row_limit=row_limit)


def run_profile_suite(
    *,
    checkpoint_dir: Path,
    batch_size: int,
    warmup: int,
    iterations: int,
    device_str: str,
    use_amp: bool,
    profiler_rows: int,
    output_json: Path | None,
) -> dict[str, Any]:
    """Load checkpoint, run latency / FLOPs / profiler, optionally write JSON."""

    config_path = checkpoint_dir / "config_resolved.yaml"
    weights_path = checkpoint_dir / "best_model.pt"
    if not config_path.is_file():
        msg = f"Missing config: {config_path}"
        raise FileNotFoundError(msg)
    if not weights_path.is_file():
        msg = f"Missing weights: {weights_path}"
        raise FileNotFoundError(msg)

    cfg = load_resolved_fold_config(config_path)
    device = torch.device(device_str)
    model = build_model(cfg)
    load_state_dict_into_model(model, weights_path, device=torch.device("cpu"))
    model.to(device)
    x, ctx = synthetic_batch(
        batch_size=batch_size,
        chip_size=cfg.chip_size,
        mask_enabled=cfg.mask_enabled,
        typology_enabled=cfg.typology_enabled,
        device=device,
    )

    param_count = sum(p.numel() for p in model.parameters())
    checkpoint_bytes = weights_path.stat().st_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    lat = benchmark_latency_ms(
        model,
        x,
        ctx,
        device=device,
        warmup=warmup,
        iterations=iterations,
        use_amp=use_amp,
    )

    peak_cuda_bytes: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))

    flops_total = total_flops_fvcore(model, x, ctx)

    print(
        f"checkpoint_bytes={checkpoint_bytes} parameter_count={param_count} "
        f"fvcore_total_flops={flops_total:.6g}"
    )
    print(
        f"latency_batch_ms mean={lat['latency_batch_ms_mean']:.4f} std={lat['latency_batch_ms_std']:.4f} "
        f"p50={lat['latency_batch_ms_p50']:.4f} p95={lat['latency_batch_ms_p95']:.4f} p99={lat['latency_batch_ms_p99']:.4f} "
        f"throughput_chips_per_s_mean={lat['throughput_chips_per_s_mean']:.4f}"
    )
    if peak_cuda_bytes is not None:
        print(f"peak_cuda_memory_allocated_bytes={peak_cuda_bytes}")

    table_text = profiler_table_string(
        model,
        x,
        ctx,
        device=device,
        row_limit=profiler_rows,
    )
    print(table_text)

    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "config_resolved": str(config_path.resolve()),
        "best_model_pt": str(weights_path.resolve()),
        "checkpoint_bytes": checkpoint_bytes,
        "parameter_count": param_count,
        "batch_size": batch_size,
        "chip_size": cfg.chip_size,
        "warmup": warmup,
        "iterations": iterations,
        "amp": use_amp,
        "latency": lat,
        "fvcore_total_flops": flops_total,
        "peak_cuda_memory_allocated_bytes": peak_cuda_bytes,
        "profiler_key_averages_table": table_text,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {output_json.resolve()}")

    return payload


# Deployment-configuration benchmarking (T-8) --------------------------------

@torch.no_grad()
def no_tta_softmax_probs(model: MaskCenteredDamageNet, images_u8: torch.Tensor, context: torch.Tensor,
                         device: str) -> torch.Tensor:
    """Single-view softmax probabilities [B, K] under the deployed inference conventions.

    Mirrors ``tta_view_softmax_probs`` (GPU-side uint8 normalization, autocast, softmax)
    restricted to the identity view — the deployed no-TTA configuration.
    """

    images_u8 = images_u8.to(device, non_blocking=True)
    context = context.to(device, non_blocking=True)
    images = to_normalized_float(images_u8)
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.amp.autocast(autocast_device):
        return F.softmax(model(images, context).float(), dim=1)


def discover_seed_checkpoints(split_root: Path, limit: int | None = None) -> list[Path]:
    """Sorted ``seed_*`` fold directories under one split root that contain weights.

    Args:
        split_root: ``outputs/ablation/<variant>/<split>`` directory.
        limit: Optional cap on the number of returned fold directories.

    Returns:
        Fold directories sorted by name, truncated to ``limit`` when given.
    """

    folds = sorted(
        path for path in split_root.glob("seed_*")
        if path.is_dir() and (path / "best_model.pt").is_file()
    )
    return folds[:limit] if limit is not None else folds


def read_event_chip_counts(variant_root: Path) -> dict[str, int]:
    """Validation chip counts per split, read from any seed's ``split_summary.json``.

    Splits without a readable summary are skipped; an empty mapping disables the
    derived time-to-event reporting.
    """

    counts: dict[str, int] = {}
    for split_dir in sorted(path for path in variant_root.iterdir() if path.is_dir()):
        for fold_dir in discover_seed_checkpoints(split_dir):
            summary_path = fold_dir / "split_summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            val_instances = summary.get("val_instances")
            if isinstance(val_instances, int) and val_instances > 0:
                counts[split_dir.name] = val_instances
                break
    return counts


def derive_operational_times(per_chip_ms: float, event_chip_counts: dict[str, int],
                             yardstick_chips: int = YARDSTICK_CHIPS) -> dict[str, float]:
    """Seconds to process each holdout's validation set plus the operational yardstick."""

    times = {f"{event}_s": count * per_chip_ms / 1000.0 for event, count in event_chip_counts.items()}
    times[f"yardstick_{yardstick_chips}_chips_s"] = yardstick_chips * per_chip_ms / 1000.0
    return times


def synthetic_uint8_batch(batch_size: int, chip_size: int, device: torch.device,
                          mask_fill_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Loader-shaped uint8 chips ([B, 4, H, W], mask channel in {0, 1}) plus context.

    The mask channel is partially filled so mask-weighted pooling exercises its standard
    (non-fallback) path, matching typical deployed inputs.
    """

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    rgb = torch.randint(0, 256, (batch_size, 3, chip_size, chip_size), dtype=torch.uint8, generator=rng)
    mask = (torch.rand((batch_size, 1, chip_size, chip_size), generator=rng) < mask_fill_fraction).to(torch.uint8)
    images_u8 = torch.cat([rgb, mask], dim=1).to(device)
    context = torch.zeros(batch_size, 4, dtype=torch.float32, device=device)
    context[:, 0] = 1.0
    return images_u8, context


def benchmark_callable_ms(step_fn: Callable[[], None], *, device: torch.device,
                          warmup: int, iterations: int) -> dict[str, float]:
    """Time a zero-arg inference step with CUDA events (or perf_counter on CPU)."""

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        step_fn()
        sync()

    times_ms: list[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iterations):
            sync()
            start.record()
            step_fn()
            end.record()
            sync()
            times_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iterations):
            t_a = time.perf_counter()
            step_fn()
            t_b = time.perf_counter()
            times_ms.append((t_b - t_a) * 1000.0)

    sorted_t = sorted(times_ms)
    return {
        "latency_batch_ms_mean": float(statistics.mean(times_ms)),
        "latency_batch_ms_std": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "latency_batch_ms_p50": percentile_sorted(sorted_t, 50.0),
        "latency_batch_ms_p95": percentile_sorted(sorted_t, 95.0),
    }


def _load_deployment_model(fold_dir: Path, device: torch.device) -> MaskCenteredDamageNet:
    """Build one fold's MCDN with weights loaded and moved to ``device``."""

    cfg = load_resolved_fold_config(fold_dir / "config_resolved.yaml")
    model = build_model(cfg)
    load_state_dict_into_model(model, fold_dir / "best_model.pt", device=torch.device("cpu"))
    model.to(device)
    model.eval()
    return model


def run_deployment_suite(*, checkpoint_dir: Path, batch_sizes: list[int], warmup: int, iterations: int,
                         device_str: str, ensemble_size: int, output_json: Path | None) -> dict[str, Any]:
    """Measure the three deployment configurations and derive operational timings.

    Configurations: ``single_no_tta`` (1 forward/chip), ``single_tta8`` (8 forwards/chip),
    and ``ensemble<N>_tta8`` (8N forwards/chip), each timed through the deployed inference
    entry points. Autocast mixed precision is applied inside those entry points, matching
    evaluation behavior exactly.
    """

    device = torch.device(device_str)
    split_root = checkpoint_dir.parent
    variant_root = split_root.parent

    fold_dirs = discover_seed_checkpoints(split_root, limit=ensemble_size)
    if not fold_dirs:
        msg = f"No seed_* folds with best_model.pt under {split_root}"
        raise FileNotFoundError(msg)

    cfg = load_resolved_fold_config(checkpoint_dir / "config_resolved.yaml")
    single_model = _load_deployment_model(checkpoint_dir, device)
    ensemble_models = [_load_deployment_model(fold_dir, device) for fold_dir in fold_dirs]
    event_chip_counts = read_event_chip_counts(variant_root)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    ensemble_label = f"ensemble{len(ensemble_models)}_tta8"
    print(f"Deployment profile: device={gpu_name or device}  chip={cfg.chip_size}  "
          f"backbone={cfg.model_name}  ensemble seeds={len(ensemble_models)}")
    print(f"Event chip counts: {event_chip_counts or 'unavailable (time-to-event skipped)'}")

    batch_sweeps: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        images_u8, context = synthetic_uint8_batch(batch_size=batch_size, chip_size=cfg.chip_size, device=device)
        config_steps: dict[str, Callable[[], None]] = {
            "single_no_tta": lambda x=images_u8, c=context: no_tta_softmax_probs(
                model=single_model, images_u8=x, context=c, device=device_str),
            "single_tta8": lambda x=images_u8, c=context: tta_mean_softmax_probs(
                model=single_model, images_u8=x, context=c, device=device_str),
            ensemble_label: lambda x=images_u8, c=context: ensemble_mean_tta_probs(
                models=ensemble_models, images_u8=x, context=c, device=device_str),
        }

        config_results: dict[str, dict[str, Any]] = {}
        for config_name, step_fn in config_steps.items():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            stats = benchmark_callable_ms(step_fn, device=device, warmup=warmup, iterations=iterations)
            per_chip_ms = stats["latency_batch_ms_mean"] / batch_size
            result: dict[str, Any] = {
                **stats,
                "per_chip_ms": per_chip_ms,
                "chips_per_s": 1000.0 / per_chip_ms,
                "peak_cuda_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
            }
            if event_chip_counts:
                result["operational"] = derive_operational_times(per_chip_ms, event_chip_counts)
            config_results[config_name] = result

        batch_sweeps.append({"batch_size": batch_size, "configs": config_results})
        _print_deployment_rows(batch_size=batch_size, config_results=config_results, event_chip_counts=event_chip_counts)

    payload: dict[str, Any] = {
        "mode": "deployment",
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "variant_root": str(variant_root.resolve()),
        "split": split_root.name,
        "backbone": cfg.model_name,
        "chip_size": cfg.chip_size,
        "ensemble_seed_dirs": [str(path) for path in fold_dirs],
        "warmup": warmup,
        "iterations": iterations,
        "event_chip_counts": event_chip_counts,
        "yardstick_chips": YARDSTICK_CHIPS,
        "batch_sweeps": batch_sweeps,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {output_json.resolve()}")
    return payload


def _print_deployment_rows(*, batch_size: int, config_results: dict[str, dict[str, Any]],
                           event_chip_counts: dict[str, int]) -> None:
    """Print one batch size's deployment table rows."""

    print(f"\nbatch_size={batch_size}")
    event_headers = "  ".join(f"{event}({count})s" for event, count in event_chip_counts.items())
    print(f"  {'config':<18s}  {'batch ms p50':>13s}  {'per-chip ms':>12s}  {'chips/s':>9s}  {'peak MB':>8s}  "
          f"{event_headers}  yardstick_{YARDSTICK_CHIPS}s")
    for config_name, result in config_results.items():
        peak_mb = result["peak_cuda_memory_allocated_bytes"]
        peak_text = f"{peak_mb / 2**20:8.1f}" if peak_mb is not None else f"{'n/a':>8s}"
        operational = result.get("operational", {})
        event_text = "  ".join(
            f"{operational.get(f'{event}_s', float('nan')):>{len(event) + len(str(count)) + 4}.1f}"
            for event, count in event_chip_counts.items()
        )
        yardstick = operational.get(f"yardstick_{YARDSTICK_CHIPS}_chips_s", float("nan"))
        print(f"  {config_name:<18s}  {result['latency_batch_ms_p50']:>13.2f}  {result['per_chip_ms']:>12.3f}  "
              f"{result['chips_per_s']:>9.1f}  {peak_text}  {event_text}  {yardstick:>12.1f}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MCDN inference (latency, FLOPs, torch profiler).")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing config_resolved.yaml and best_model.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use autocast FP16 on CUDA during latency timing.")
    parser.add_argument("--profiler-rows", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--deployment", action="store_true",
                        help="Measure the three deployment configurations (no-TTA / TTA / ensemble) instead of the forward suite.")
    parser.add_argument("--deployment-batch-sizes", type=int, nargs="+", default=[1, 16, 64],
                        help="Batch sizes swept in deployment mode.")
    parser.add_argument("--ensemble-size", type=int, default=10,
                        help="Number of sibling seed checkpoints loaded for the ensemble configuration.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = _parse_args(argv)
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA is not available; use --device cpu or run on a GPU machine.", file=sys.stderr)
        return 1
    if args.deployment:
        run_deployment_suite(
            checkpoint_dir=args.checkpoint_dir,
            batch_sizes=args.deployment_batch_sizes,
            warmup=args.warmup,
            iterations=args.iterations,
            device_str=args.device,
            ensemble_size=args.ensemble_size,
            output_json=args.output_json,
        )
        return 0
    run_profile_suite(
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device_str=args.device,
        use_amp=args.amp,
        profiler_rows=args.profiler_rows,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
