"""Profile MCDN inference: latency, fvcore FLOPs, and torch.profiler summaries for edge-deployment reporting."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from fvcore.nn import FlopCountAnalysis

from src.model.mcdn import MaskConditionedDamageNet

DEFAULT_CHECKPOINT_DIR = Path("outputs/ablation/baseline/Spatial_Block_East/seed_00")


class ProfileConfigError(ValueError):
    """Raised when ``config_resolved.yaml`` is missing required keys for profiling."""


@dataclass(frozen=True)
class ResolvedFoldConfig:
    """Subset of training snapshot fields required to rebuild ``MaskConditionedDamageNet``."""

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


def build_model(cfg: ResolvedFoldConfig) -> MaskConditionedDamageNet:
    """Instantiate MCDN from resolved fold config."""

    return MaskConditionedDamageNet(
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = _parse_args(argv)
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA is not available; use --device cpu or run on a GPU machine.", file=sys.stderr)
        return 1
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
