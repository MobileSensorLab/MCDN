"""Unit tests for ``src.postproc.soup`` uniform-soup primitive."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.postproc.soup import MCDN_BACKBONE_SOUP_PREFIXES, build_uniform_soup


def _write_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Write a state_dict to disk in the format ``torch.load(weights_only=True)`` reads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(path))


def test_arithmetic_mean_three_float_checkpoints(tmp_path: Path) -> None:
    """A soup of three known-valued state_dicts produces the per-element arithmetic mean."""

    paths = []
    values = (1.0, 2.0, 6.0)
    for idx, v in enumerate(values):
        sd = {
            "weight": torch.full((2, 3), v, dtype=torch.float32),
            "bias": torch.full((3,), v * 0.5, dtype=torch.float32)
        }
        path = tmp_path / f"ckpt_{idx}.pt"
        _write_state_dict(path=path, state_dict=sd)
        paths.append(path)

    soup = build_uniform_soup(checkpoint_paths=paths)

    expected_mean = sum(values) / len(values)
    assert torch.allclose(soup["weight"], torch.full((2, 3), expected_mean, dtype=torch.float32))
    assert torch.allclose(soup["bias"], torch.full((3,), expected_mean * 0.5, dtype=torch.float32))


def test_integer_buffers_pass_through_first_checkpoint(tmp_path: Path) -> None:
    """Integer-dtype tensors (e.g., counters) are copied from the first checkpoint, not averaged."""

    sd_a = {
        "weight": torch.ones((2,), dtype=torch.float32),
        "step_counter": torch.tensor([10], dtype=torch.int64)
    }
    sd_b = {
        "weight": torch.full((2,), 3.0, dtype=torch.float32),
        "step_counter": torch.tensor([99], dtype=torch.int64)
    }
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    soup = build_uniform_soup(checkpoint_paths=[path_a, path_b])

    assert torch.allclose(soup["weight"], torch.full((2,), 2.0, dtype=torch.float32))
    assert soup["step_counter"].dtype == torch.int64
    assert soup["step_counter"].item() == 10


def test_strict_buffer_agreement_raises_on_disagreement(tmp_path: Path) -> None:
    """``strict_buffer_agreement=True`` raises ValueError when integer buffers differ."""

    sd_a = {"step_counter": torch.tensor([1], dtype=torch.int64)}
    sd_b = {"step_counter": torch.tensor([2], dtype=torch.int64)}
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    with pytest.raises(ValueError, match="Buffer disagreement"):
        build_uniform_soup(checkpoint_paths=[path_a, path_b], strict_buffer_agreement=True)


def test_mismatched_keys_raise_key_error(tmp_path: Path) -> None:
    """Diverging key sets across checkpoints produce a KeyError with a per-side diagnostic."""

    sd_a = {"alpha": torch.zeros((1,)), "beta": torch.zeros((1,))}
    sd_b = {"alpha": torch.zeros((1,)), "gamma": torch.zeros((1,))}
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    with pytest.raises(KeyError, match="Key-set mismatch"):
        build_uniform_soup(checkpoint_paths=[path_a, path_b])


def test_mismatched_shapes_raise_value_error(tmp_path: Path) -> None:
    """Diverging tensor shapes for the same key produce a ValueError."""

    sd_a = {"weight": torch.zeros((2, 3), dtype=torch.float32)}
    sd_b = {"weight": torch.zeros((2, 4), dtype=torch.float32)}
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    with pytest.raises(ValueError, match="Shape mismatch"):
        build_uniform_soup(checkpoint_paths=[path_a, path_b])


def test_mismatched_dtypes_raise_value_error(tmp_path: Path) -> None:
    """Diverging dtypes for the same key produce a ValueError (no silent upcast)."""

    sd_a = {"weight": torch.zeros((2,), dtype=torch.float32)}
    sd_b = {"weight": torch.zeros((2,), dtype=torch.float16)}
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        build_uniform_soup(checkpoint_paths=[path_a, path_b])


def test_empty_path_list_raises_value_error() -> None:
    """An empty checkpoint_paths argument is rejected up front."""

    with pytest.raises(ValueError, match="at least one path"):
        build_uniform_soup(checkpoint_paths=[])


def test_single_checkpoint_returns_clone(tmp_path: Path) -> None:
    """A soup of N=1 returns a value-equal clone, not a reference to the input tensor."""

    sd = {"weight": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    path = tmp_path / "solo.pt"
    _write_state_dict(path=path, state_dict=sd)

    soup = build_uniform_soup(checkpoint_paths=[path])

    assert torch.equal(soup["weight"], sd["weight"])
    soup["weight"].add_(100.0)
    assert sd["weight"][0].item() == 1.0


def test_dtype_preserved_through_round_trip(tmp_path: Path) -> None:
    """The output dtype matches the input dtype regardless of the intermediate fp32 accumulation."""

    sd_a = {"weight": torch.full((4,), 1.0, dtype=torch.float16)}
    sd_b = {"weight": torch.full((4,), 3.0, dtype=torch.float16)}
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    _write_state_dict(path=path_a, state_dict=sd_a)
    _write_state_dict(path=path_b, state_dict=sd_b)

    soup = build_uniform_soup(checkpoint_paths=[path_a, path_b])

    assert soup["weight"].dtype == torch.float16
    assert torch.allclose(soup["weight"], torch.full((4,), 2.0, dtype=torch.float16))


def test_missing_checkpoint_raises_file_not_found(tmp_path: Path) -> None:
    """A nonexistent checkpoint path produces FileNotFoundError, not a silent failure."""

    sd = {"weight": torch.zeros((1,))}
    real_path = tmp_path / "real.pt"
    _write_state_dict(path=real_path, state_dict=sd)
    fake_path = tmp_path / "ghost.pt"

    with pytest.raises(FileNotFoundError, match=r"ghost\.pt"):
        build_uniform_soup(checkpoint_paths=[real_path, fake_path])


def test_selective_soup_averages_only_prefix_matches(tmp_path: Path) -> None:
    """Keys matching a prefix are averaged; other floats copy the first checkpoint."""

    sd_0 = {
        "backbone.w": torch.ones((2,), dtype=torch.float32),
        "head.w": torch.full((2,), 10.0, dtype=torch.float32),
    }
    sd_1 = {
        "backbone.w": torch.full((2,), 3.0, dtype=torch.float32),
        "head.w": torch.full((2,), 50.0, dtype=torch.float32),
    }
    path_0 = tmp_path / "s0.pt"
    path_1 = tmp_path / "s1.pt"
    _write_state_dict(path=path_0, state_dict=sd_0)
    _write_state_dict(path=path_1, state_dict=sd_1)

    soup = build_uniform_soup(
        checkpoint_paths=[path_0, path_1],
        average_key_prefixes=MCDN_BACKBONE_SOUP_PREFIXES,
    )

    assert torch.allclose(soup["backbone.w"], torch.full((2,), 2.0, dtype=torch.float32))
    assert torch.allclose(soup["head.w"], torch.full((2,), 10.0, dtype=torch.float32))


def test_empty_average_prefixes_with_multiple_checkpoints_raises(tmp_path: Path) -> None:
    """Selective soup with an empty prefix list and N > 1 is rejected."""

    sd = {"w": torch.zeros((1,), dtype=torch.float32)}
    p0 = tmp_path / "a.pt"
    p1 = tmp_path / "b.pt"
    _write_state_dict(path=p0, state_dict=sd)
    _write_state_dict(path=p1, state_dict=sd)

    with pytest.raises(ValueError, match="average_key_prefixes cannot be empty"):
        build_uniform_soup(checkpoint_paths=[p0, p1], average_key_prefixes=())
