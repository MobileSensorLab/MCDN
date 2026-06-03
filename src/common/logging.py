"""Stdout tee utility and config-hash helper for the train-time printout stack.

The trainer funnels identity box, timeline markers, and per-epoch blocks through ``print``.
This module provides the thin plumbing that complements those calls for preset-gated
ablation runs: a ``Tee``-like object that fans writes to a second file stream, a context
manager that wires that object into ``sys.stdout`` for the duration of a ``run_training_pipeline``
call, and a ``config_hash`` helper whose short digest appears in identity-box headers and
the tee'd log's metadata prelude.
"""

import contextlib
import hashlib
import sys

from collections.abc import Iterator
from pathlib import Path
from typing import IO

from src.config.settings import AppConfig


class _Tee:
    """File-like that fans writes to multiple streams.

    Minimal surface intentionally: ``write``, ``flush``, ``isatty``. The trainer only
    emits ``print`` calls so the ``TextIO`` subset exercised is small.

    Args:
        streams: Output streams (stdout + log file is the canonical pair).
    """

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        """Forward ``data`` to every underlying stream and return its length."""

        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush every underlying stream."""

        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        """Report as a non-tty so colorization libraries stay silent when teeing."""

        return False


@contextlib.contextmanager
def tee_stdout_to_file(log_path: Path, *, header_lines: list[str] | None = None) -> Iterator[Path]:
    """Tee ``sys.stdout`` to ``log_path`` for the duration of the context.

    Creates the parent directory on demand, opens the log file in append mode with
    line-buffering, and swaps ``sys.stdout`` for a ``_Tee`` that fans writes to both
    the original stdout and the log file.

    Args:
        log_path: Destination file. Opened in append mode so re-runs of the same
            fold accumulate history rather than truncating.
        header_lines: Optional lines written only to the log file (no stdout echo)
            before the stdout tee engages. Use for identifying metadata that should
            live in the log but not noisify the terminal.

    Yields:
        The resolved ``log_path`` so callers can record it in upstream artifacts.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        if header_lines:
            for line in header_lines:
                log_file.write(line + "\n")
            log_file.flush()

        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)
        try:
            yield log_path
        finally:
            sys.stdout = original_stdout


def config_hash(config: AppConfig, *, length: int = 8) -> str:
    """Short sha256 prefix of the validated config's JSON representation.

    Pydantic's ``model_dump_json`` is field-order stable for a fixed schema, so the
    digest is reproducible across runs that share the same validated configuration.
    The intent is log identification only; this is not an integrity guarantee.

    Args:
        config: Validated root config.
        length: Hex-character prefix length. Default 8 matches the identity-box
            display width.
    """

    payload = config.model_dump_json(exclude_none=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
