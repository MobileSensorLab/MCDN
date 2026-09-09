"""Fetch the CRASAR-U-DROIDs dataset from Hugging Face into this clone's ``data/`` directory.

The dataset repository's tree (``train/``, ``test/``, ``format/``, ``statistics.csv``) is
exactly the layout ``src.data`` expects under ``data/``, so files are mirrored in place
with ``huggingface_hub`` (parallel, resumable). The selection is by sensor and split:
the sUAS imagery and annotations that every MCDN arm trains on are the default, the
crewed-aircraft subset backs the ``resolution`` arm, and the satellite subset is offered
for completeness. ``statistics.csv`` and ``format/`` are always included because the
sampler reads event metadata from them. Files already present at the published size are
skipped unless ``--force`` is given; ``data/readme/`` is never touched.

The sUAS subset is about 196 GB, so run ``--list`` first to see what a selection costs.

Usage::

    python -m scripts.fetch_dataset --list
    python -m scripts.fetch_dataset                              # sUAS train + test (~196 GB)
    python -m scripts.fetch_dataset --sensor uas --split test   # evaluate published checkpoints only (~66 GB)
    python -m scripts.fetch_dataset --sensor uas --sensor crewed
"""

import argparse
import time

from pathlib import Path
from typing import NamedTuple

from huggingface_hub import HfApi, snapshot_download

from src.common.checkpoints import DEFAULT_TOKEN_ENV, resolve_token

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data"
DEFAULT_REPO_ID = "CRASAR/CRASAR-U-DROIDs"
REPO_TYPE = "dataset"

# CLI sensor names -> directory names under <split>/imagery and <split>/annotations.
SENSOR_DIRS: dict[str, str] = {"uas": "UAS", "crewed": "CREWED", "satellite": "SATELLITE", "dsm": "UAS_DSM"}
SPLITS = ("train", "test")
# Small top-level files the data pipeline reads regardless of sensor selection.
REQUIRED_PREFIXES = ("statistics.csv", "format/")


class DatasetFetchError(Exception):
    """Raised when the request cannot be satisfied or a fetched file disagrees with the Hub."""


class RemoteFile(NamedTuple):
    """One file in the dataset repository."""

    path: str
    size: int

    @property
    def split(self) -> str | None:
        """``train`` or ``test`` for split-scoped files, else None."""

        head = self.path.split("/", 1)[0]
        return head if head in SPLITS else None

    @property
    def sensor_dir(self) -> str | None:
        """Sensor directory name (``UAS``, ``CREWED``, ...) for split-scoped files, else None."""

        parts = self.path.split("/")
        return parts[2] if len(parts) >= 4 and parts[0] in SPLITS and parts[1] in ("imagery", "annotations") else None

    @property
    def required(self) -> bool:
        """True for the top-level files every selection includes."""

        return self.path.startswith(REQUIRED_PREFIXES)


def list_remote_files(repo_id: str, *, revision: str | None = None, token: str | None = None) -> list[RemoteFile]:
    """Enumerate every file in the dataset repository with its size."""

    tree = HfApi(token=token).list_repo_tree(repo_id, repo_type=REPO_TYPE, revision=revision, recursive=True)
    return [RemoteFile(path=item.path, size=item.size) for item in tree if getattr(item, "size", None) is not None]


def select_files(files: list[RemoteFile], *, sensors: set[str], splits: set[str]) -> list[RemoteFile]:
    """Keep the required top-level files plus imagery and annotations for the requested sensors and splits."""

    unknown = sensors - set(SENSOR_DIRS)
    if unknown:
        raise DatasetFetchError(f"Unknown sensor(s) {sorted(unknown)}; choose from {sorted(SENSOR_DIRS)}")
    sensor_dirs = {SENSOR_DIRS[s] for s in sensors}
    return [f for f in files if f.required or (f.split in splits and f.sensor_dir in sensor_dirs)]


def is_present(entry: RemoteFile, dest: Path) -> bool:
    """Return True when the file exists under ``dest`` at the size the Hub reports."""

    target = dest / entry.path
    return target.is_file() and target.stat().st_size == entry.size


def verify_sizes(entries: list[RemoteFile], dest: Path) -> None:
    """Check every fetched file against its published size; delete and report any that disagree."""

    mismatched: list[str] = []
    for entry in entries:
        target = dest / entry.path
        if not target.is_file():
            mismatched.append(f"{entry.path}: not downloaded")
            continue
        actual = target.stat().st_size
        if actual != entry.size:
            target.unlink()
            mismatched.append(f"{entry.path}: size {actual} != published {entry.size}")
    if mismatched:
        raise DatasetFetchError(f"{len(mismatched)} file(s) failed verification (removed):\n  " + "\n  ".join(mismatched))


def print_listing(files: list[RemoteFile], dest: Path) -> None:
    """Print one line per (split, sensor) group with file count, size, and on-disk status."""

    groups: dict[str, list[RemoteFile]] = {}
    for f in files:
        key = f"{f.split}/{f.sensor_dir}" if f.sensor_dir else ("required" if f.required else "other")
        groups.setdefault(key, []).append(f)
    for key in sorted(groups):
        group = groups[key]
        present = sum(is_present(f, dest) for f in group)
        status = "present" if present == len(group) else ("partial" if present else "missing")
        print(f"{key:<18} {len(group):>4} files {sum(f.size for f in group) / 1e9:7.1f} GB  {status}")
    print(f"{len(files)} files, {sum(f.size for f in files) / 1e9:.1f} GB total")


def fetch(repo_id: str = DEFAULT_REPO_ID, *, dest: Path = DEFAULT_DEST, sensors: set[str] | None = None, splits: set[str] | None = None,
          list_only: bool = False, force: bool = False, revision: str | None = None, token: str | None = None,
          max_workers: int = 8) -> int:
    """Fetch the requested subset of the dataset into ``dest`` and return how many files were downloaded."""

    files = list_remote_files(repo_id, revision=revision, token=token)
    if list_only:
        print_listing(files, dest)
        return 0

    selected = select_files(files, sensors=sensors or {"uas"}, splits=splits or set(SPLITS))
    if not selected:
        raise DatasetFetchError("No files match the requested --sensor/--split")

    pending = [f for f in selected if force or not is_present(f, dest)]
    skipped = len(selected) - len(pending)
    if not pending:
        print(f"All {len(selected)} files already present under {dest}")
        return 0

    total_bytes = sum(f.size for f in pending)
    print(f"Fetching {len(pending)} files ({total_bytes / 1e9:.1f} GB) from {repo_id} into {dest}; {skipped} already present")
    started = time.perf_counter()
    snapshot_download(repo_id, repo_type=REPO_TYPE, revision=revision, token=token, local_dir=dest,
                      allow_patterns=[f.path for f in pending], force_download=force, max_workers=max_workers)
    elapsed = time.perf_counter() - started

    verify_sizes(pending, dest)
    print(f"Done: {len(pending)} files verified ({total_bytes / 1e9:.1f} GB in {elapsed:,.0f} s, "
          f"{total_bytes / 1e6 / max(elapsed, 1e-6):.1f} MB/s), {skipped} skipped")
    return len(pending)


def main() -> None:
    """Parse CLI arguments and fetch the dataset."""

    parser = argparse.ArgumentParser(description="Fetch CRASAR-U-DROIDs from Hugging Face into data/.")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help="Dataset repository id.")
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit to fetch (default: main).")
    parser.add_argument("--sensor", action="append", choices=sorted(SENSOR_DIRS),
                        help="Sensor subset to fetch (repeatable; default uas). crewed backs the resolution arm.")
    parser.add_argument("--split", action="append", choices=SPLITS, help="Split to fetch (repeatable; default both).")
    parser.add_argument("--list", action="store_true", help="List the repository by split and sensor with on-disk status.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Data root to download into.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if present.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    parser.add_argument("--token-file", type=Path, default=None, help="File holding a Hub token (optional; raises rate limits).")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help=f"Environment variable holding a token (default {DEFAULT_TOKEN_ENV}).")
    args = parser.parse_args()

    token = resolve_token(args.token_file, args.token_env)
    try:
        fetch(args.repo, dest=args.dest, sensors=set(args.sensor) if args.sensor else None, splits=set(args.split) if args.split else None,
              list_only=args.list, force=args.force, revision=args.revision, token=token, max_workers=args.workers)
    except DatasetFetchError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
