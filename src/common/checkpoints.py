"""Shared helpers for the published MCDN checkpoint pool on Hugging Face.

Trained checkpoints (``best_model.pt``, about 134 MB each) are not tracked in git. They
are published as a Hugging Face model repository whose tree mirrors this project's
``outputs/ablation/<arm>/<fold>/seed_XX/`` layout, so a fetched file lands exactly where
the evaluation and figure scripts expect it. This module owns the manifest format that
ties every published file to its SHA-256, plus the selection and token helpers shared by
``scripts/fetch_checkpoints.py`` and the (untracked) staging and upload scripts.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

DEFAULT_REPO_ID = "mobilesensorlab/mcdn"
REPO_TYPE = "model"

CHECKPOINT_NAME = "best_model.pt"
CONFIG_NAME = "config_resolved.yaml"
MANIFEST_JSON = "MANIFEST.json"
MANIFEST_SHA256 = "MANIFEST.sha256"

DEFAULT_TOKEN_ENV = "HF_TOKEN"
CHUNK_BYTES = 8 << 20


class CheckpointError(Exception):
    """Base class for checkpoint publishing and fetching failures."""


class ChecksumMismatchError(CheckpointError):
    """Raised when a file's digest disagrees with the manifest."""


class ManifestError(CheckpointError):
    """Raised when a manifest is missing, malformed, or does not describe the requested cell."""


@dataclass(frozen=True)
class FileEntry:
    """Manifest row describing one published file.

    Args:
        path: Path relative to the repository root (``arm/fold/seed_XX/name``), POSIX separators.
        arm: Ablation arm directory name.
        fold: Holdout fold directory name.
        seed: Seed directory name (``seed_00``).
        size: Size in bytes.
        sha256: Hex SHA-256 of the file.
    """

    path: str
    arm: str
    fold: str
    seed: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Rebuild an entry from :meth:`to_dict` output."""

        return cls(path=payload["path"], arm=payload["arm"], fold=payload["fold"], seed=payload["seed"],
                   size=int(payload["size"]), sha256=payload["sha256"])

    @property
    def is_checkpoint(self) -> bool:
        """True for the weights file, False for its resolved-config sidecar."""

        return self.path.endswith(f"/{CHECKPOINT_NAME}")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path`` in one streaming pass."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(entries: Iterable[FileEntry], out_dir: Path, *, repo_id: str) -> tuple[Path, Path]:
    """Write ``MANIFEST.json`` and a ``sha256sum -c``-compatible ``MANIFEST.sha256`` to ``out_dir``."""

    ordered = sorted(entries, key=lambda e: e.path)
    json_path = out_dir / MANIFEST_JSON
    payload = {"repo_id": repo_id, "files": [e.to_dict() for e in ordered]}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sha_path = out_dir / MANIFEST_SHA256
    sha_path.write_text("".join(f"{e.sha256}  {e.path}\n" for e in ordered), encoding="utf-8")
    return json_path, sha_path


def read_manifest(path: Path) -> list[FileEntry]:
    """Load ``MANIFEST.json``; raise :class:`ManifestError` if it is unreadable or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [FileEntry.from_dict(item) for item in payload["files"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc


def select_entries(entries: Iterable[FileEntry], *, arms: set[str] | None, folds: set[str] | None) -> list[FileEntry]:
    """Filter manifest entries by arm and fold; ``None`` for either means no restriction."""

    return [e for e in entries if (arms is None or e.arm in arms) and (folds is None or e.fold in folds)]


def group_by_cell(entries: Iterable[FileEntry]) -> dict[tuple[str, str], list[FileEntry]]:
    """Group manifest entries by ``(arm, fold)``, preserving manifest order within each cell."""

    cells: dict[tuple[str, str], list[FileEntry]] = {}
    for entry in entries:
        cells.setdefault((entry.arm, entry.fold), []).append(entry)
    return cells


def resolve_token(token_file: Path | None = None, env_var: str = DEFAULT_TOKEN_ENV) -> str | None:
    """Return an access token from ``token_file`` if given, else from ``env_var``, else ``None``.

    A file keeps the token out of shell history; anonymous access (``None``) is enough
    once the repository is public.
    """

    if token_file is not None:
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise CheckpointError(f"Token file {token_file} is empty")
        return token
    return os.environ.get(env_var) or None
