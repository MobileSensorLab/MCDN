"""Uniform model soup over per-seed EMA-shadow checkpoints.

A model soup [Wortsman et al., 2022] arithmetically averages the weights of N
independently-fine-tuned models from the same pretrained initialization,
producing a single deployable artifact whose accuracy is often competitive
with the corresponding probability-space ensemble at 1/N the inference cost.

This module provides uniform averaging and a **selective** variant: only
state_dict keys whose names start with a configured prefix tuple are
averaged; all other float tensors are taken from the **first** checkpoint
(typically seed 00). Tier 1.5 uses ``("backbone.",)`` on
``MaskConditionedDamageNet`` so timm backbone weights are soup-averaged while
randomly initialized scaffolding (FiLM, mask-pooling readout, classifier)
stays fixed to one seed's basin.

The soup primitive assumes:

* All checkpoints share an identical state_dict schema (key set, per-key
  tensor shapes, per-key dtypes). The function hard-fails on mismatch.
* Floating-point parameters (weights, biases, LayerNorm gain/bias) are the
  averaging targets. They are accumulated in fp32 for numerical stability,
  then cast back to the source dtype to preserve the checkpoint format.
* Integer and boolean buffers (e.g., ``num_batches_tracked``, mask metadata)
  are passed through from the first checkpoint, with an optional cross-seed
  agreement check that warns rather than raises.

Soup formation is fully deterministic given an ordered checkpoint list.

Usage::

    from src.postproc.soup import build_uniform_soup

    paths = [Path(f"outputs/ablation/baseline/Spatial_Block_East/seed_{s:02d}/best_model.pt")
             for s in (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)]
    soup_state = build_uniform_soup(checkpoint_paths=paths, average_key_prefixes=("backbone.",))
    model.load_state_dict(soup_state, strict=True)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch


_FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64
)


def _load_state_dict(checkpoint_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load a single weights-only state_dict from disk."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(str(checkpoint_path), map_location=device, weights_only=True)


def _validate_schema_consistency(state_dicts: list[dict[str, torch.Tensor]],
                                 checkpoint_paths: list[Path]) -> None:
    """Assert every state_dict shares the same key set, shapes, and dtypes.

    Hard-fails with a per-key diagnostic on the first mismatch encountered so
    silent architecture drift across seeds never propagates into a soup.
    """

    reference_keys = set(state_dicts[0].keys())
    for idx, sd in enumerate(state_dicts[1:], start=1):
        candidate_keys = set(sd.keys())
        only_in_reference = reference_keys - candidate_keys
        only_in_candidate = candidate_keys - reference_keys
        if only_in_reference or only_in_candidate:
            raise KeyError(
                f"Key-set mismatch between {checkpoint_paths[0]} and {checkpoint_paths[idx]}: "
                f"only_in_reference={sorted(only_in_reference)[:5]}{'...' if len(only_in_reference) > 5 else ''}  "
                f"only_in_candidate={sorted(only_in_candidate)[:5]}{'...' if len(only_in_candidate) > 5 else ''}"
            )

    for key in reference_keys:
        ref_tensor = state_dicts[0][key]
        for idx, sd in enumerate(state_dicts[1:], start=1):
            candidate = sd[key]
            if candidate.shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at key {key!r}: "
                    f"{checkpoint_paths[0]} -> {tuple(ref_tensor.shape)}, "
                    f"{checkpoint_paths[idx]} -> {tuple(candidate.shape)}"
                )
            if candidate.dtype != ref_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch at key {key!r}: "
                    f"{checkpoint_paths[0]} -> {ref_tensor.dtype}, "
                    f"{checkpoint_paths[idx]} -> {candidate.dtype}"
                )


def _is_float_dtype(dtype: torch.dtype) -> bool:
    """Return True iff the dtype is one of fp16 / bf16 / fp32 / fp64."""

    return dtype in _FLOAT_DTYPES


def _normalize_average_prefixes(average_key_prefixes: Iterable[str] | None,
                               n_checkpoints: int) -> tuple[str, ...] | None:
    """Validate and freeze ``average_key_prefixes`` for selective soup.

    ``None`` means uniform soup (every float tensor averaged). A non-empty
    tuple means only keys starting with one of those prefixes are averaged;
    all other float tensors are copied from the first checkpoint.

    Args:
        average_key_prefixes: Optional iterable of state_dict key prefixes.
        n_checkpoints: Number of checkpoints; used to reject an empty-prefix
            selective request when N > 1.

    Returns:
        ``None`` for full uniform soup, or a non-empty ``tuple`` of prefixes.

    Raises:
        ValueError: If selective soup is requested with an empty prefix list
            while ``n_checkpoints > 1``.
    """

    if average_key_prefixes is None:
        return None
    frozen = tuple(str(p) for p in average_key_prefixes)
    if len(frozen) == 0 and n_checkpoints > 1:
        raise ValueError(
            "average_key_prefixes cannot be empty when combining multiple checkpoints; "
            "pass None for full uniform soup"
        )
    return frozen


def _key_matches_any_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if ``key`` starts with any element of ``prefixes``."""

    return any(key.startswith(prefix) for prefix in prefixes)


def build_uniform_soup(checkpoint_paths: list[Path], device: str = "cpu",
                       strict_buffer_agreement: bool = False,
                       average_key_prefixes: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    r"""Average EMA-shadow checkpoints into one state_dict (uniform or selective).

    Floating-point tensors are either averaged across all checkpoints or taken
    from the first checkpoint only, depending on ``average_key_prefixes``.
    Integer/bool buffers are always taken from the first checkpoint (with an
    optional agreement check). The first path in ``checkpoint_paths`` should be
    the reference seed (e.g. seed ``00``) when using selective averaging so
    head weights match that run.

    Args:
        checkpoint_paths: Ordered list of N checkpoint paths to soup. Must be
            non-empty; N=1 returns the single checkpoint's state_dict (clone).
        device: ``map_location`` argument for the underlying ``torch.load``
            calls. Defaults to CPU since soup formation is a one-shot tensor
            arithmetic step that does not benefit from GPU residency.
        strict_buffer_agreement: When True, integer/bool buffers must match
            across all checkpoints exactly or a ``ValueError`` is raised.
            When False (default), the first-checkpoint buffer is taken with
            no cross-seed check; this is the right setting when buffers carry
            seed-incidental counters such as ``num_batches_tracked``.
        average_key_prefixes: If ``None``, every float tensor is uniformly
            averaged (full soup). If a non-empty iterable, only keys starting
            with one of these prefixes are averaged; other float tensors are
            copied from the first checkpoint. Tier 1.5 for ``MaskConditionedDamageNet``
            uses ``(\"backbone.\",)`` so timm backbone weights are averaged while
            FiLM, mixer, and classifier stay on the reference seed.

    Returns:
        A state_dict suitable for ``model.load_state_dict(soup, strict=True)``
        on the same architecture every input checkpoint was produced by.

    Raises:
        ValueError: If ``checkpoint_paths`` is empty, selective prefixes are
            empty with N > 1, or any tensor shape / dtype mismatches across
            checkpoints.
        KeyError: If state_dict keys differ across checkpoints.
        FileNotFoundError: If any checkpoint path does not exist.
    """

    if len(checkpoint_paths) == 0:
        raise ValueError("checkpoint_paths must contain at least one path")

    state_dicts = [_load_state_dict(checkpoint_path=p, device=device) for p in checkpoint_paths]
    _validate_schema_consistency(state_dicts=state_dicts, checkpoint_paths=checkpoint_paths)

    n = len(state_dicts)
    prefixes = _normalize_average_prefixes(
        average_key_prefixes=average_key_prefixes, n_checkpoints=n
    )

    if n == 1:
        return {key: tensor.detach().clone() for key, tensor in state_dicts[0].items()}

    soup: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        ref_tensor = state_dicts[0][key]
        do_average = prefixes is None or _key_matches_any_prefix(key=key, prefixes=prefixes)

        if _is_float_dtype(ref_tensor.dtype):
            if do_average:
                accumulator = ref_tensor.detach().to(torch.float32).clone()
                for sd in state_dicts[1:]:
                    accumulator.add_(sd[key].detach().to(torch.float32))
                accumulator.div_(n)
                soup[key] = accumulator.to(ref_tensor.dtype)
            else:
                soup[key] = ref_tensor.detach().clone()
        else:
            if strict_buffer_agreement:
                for idx, sd in enumerate(state_dicts[1:], start=1):
                    if not torch.equal(sd[key], ref_tensor):
                        raise ValueError(
                            f"Buffer disagreement at key {key!r} between "
                            f"{checkpoint_paths[0]} and {checkpoint_paths[idx]} "
                            f"(strict_buffer_agreement=True)"
                        )
            soup[key] = ref_tensor.detach().clone()

    return soup


# Prefixes for Tier 1.5: average only timm ``backbone.*``; FiLM, pool readout, head from seed 0.
MCDN_BACKBONE_SOUP_PREFIXES: tuple[str, ...] = ("backbone.",)
