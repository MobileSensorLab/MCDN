"""Tests for the stdout tee and config-hash helpers."""

import sys

from pathlib import Path

import pytest

from src.common.logging import _Tee, config_hash, tee_stdout_to_file
from src.config.settings import AppConfig, DataConfig


def _make_config(*, holdout_event: str | None = None, seed: int | None = 11) -> AppConfig:
    """Build a minimal AppConfig for hash-stability smoke tests."""

    config_dict = AppConfig(
        data=DataConfig(dir=Path("dataset-root"), holdout_event=holdout_event),
    ).model_dump(mode="python")
    config_dict["runtime"]["seed"] = seed
    return AppConfig.model_validate(config_dict)


def test_tee_fans_write_to_every_stream() -> None:
    """_Tee duplicates every write into each underlying stream."""

    class _Buffer:
        def __init__(self) -> None:
            self.data: list[str] = []

        def write(self, chunk: str) -> int:
            self.data.append(chunk)
            return len(chunk)

        def flush(self) -> None:
            self.data.append("<flush>")

    a = _Buffer()
    b = _Buffer()
    tee = _Tee(a, b)
    written = tee.write("hello")
    tee.flush()

    assert written == 5
    assert a.data == ["hello", "<flush>"]
    assert b.data == ["hello", "<flush>"]
    assert tee.isatty() is False


def test_tee_stdout_to_file_captures_prints_and_writes_header(tmp_path: Path) -> None:
    """tee_stdout_to_file echoes stdout to disk and prepends header lines only to the file."""

    log_path = tmp_path / "logs" / "train_log.txt"
    with tee_stdout_to_file(log_path, header_lines=["# run meta", "# config abc123"]):
        print("first line")
        print("second line")

    assert sys.stdout is not None
    contents = log_path.read_text(encoding="utf-8").splitlines()
    assert contents[:2] == ["# run meta", "# config abc123"]
    assert "first line" in contents
    assert "second line" in contents


def test_tee_stdout_to_file_restores_stdout_on_error(tmp_path: Path) -> None:
    """Context manager restores stdout even when the wrapped body raises."""

    log_path = tmp_path / "train_log.txt"
    original = sys.stdout
    with pytest.raises(RuntimeError), tee_stdout_to_file(log_path):  # noqa: PT012
        print("captured-before-raise")
        raise RuntimeError("boom")
    assert sys.stdout is original
    assert "captured-before-raise" in log_path.read_text(encoding="utf-8")


def test_config_hash_is_stable_for_identical_configs() -> None:
    """Same validated config reliably produces the same short digest."""

    first = _make_config(seed=11)
    second = _make_config(seed=11)
    assert config_hash(first) == config_hash(second)


def test_config_hash_changes_when_any_field_changes() -> None:
    """A non-trivial field change flips the short digest."""

    base = _make_config(seed=11, holdout_event=None)
    altered = _make_config(seed=22, holdout_event=None)
    assert config_hash(base) != config_hash(altered)


def test_config_hash_respects_requested_length() -> None:
    """Explicit length truncates the digest to the requested prefix."""

    config = _make_config()
    assert len(config_hash(config, length=12)) == 12
    assert len(config_hash(config, length=4)) == 4
