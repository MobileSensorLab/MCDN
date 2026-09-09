"""Fetch published MCDN checkpoints from Hugging Face into this clone's ``outputs/ablation/``.

Checkpoints are not tracked in git; they live in the ``mobilesensorlab/mcdn`` model
repository, whose tree mirrors ``outputs/ablation/<arm>/<fold>/seed_XX/``. This script
reads the repository's ``MANIFEST.json``, downloads the requested cells with
``huggingface_hub`` (parallel, resumable), verifies every fetched file's SHA-256 against
the manifest, and leaves ``best_model.pt`` and ``config_resolved.yaml`` exactly where the
evaluation and figure scripts expect them. Files already present with the manifest size
are skipped unless ``--force`` is given.

Usage::

    python -m scripts.fetch_checkpoints --list
    python -m scripts.fetch_checkpoints --arm all_features
    python -m scripts.fetch_checkpoints --arm all_features --fold Hurricane_Ida
    python -m scripts.fetch_checkpoints --all
    python -m scripts.fetch_checkpoints --repo mobilesensorlab/mcdn --revision <commit> --arm rgb_only
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from src.common.checkpoints import (
    DEFAULT_REPO_ID, DEFAULT_TOKEN_ENV, MANIFEST_JSON, REPO_TYPE, CheckpointError, ChecksumMismatchError, FileEntry, group_by_cell,
    read_manifest, resolve_token, select_entries, sha256_file
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "outputs" / "ablation"


def load_remote_manifest(repo_id: str, dest: Path, *, revision: str | None = None, token: str | None = None) -> list[FileEntry]:
    """Download the repository's ``MANIFEST.json`` into ``dest`` and parse it."""

    dest.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id, MANIFEST_JSON, repo_type=REPO_TYPE, revision=revision, token=token, local_dir=dest,
                           force_download=True)
    return read_manifest(Path(path))


def is_present(entry: FileEntry, dest: Path) -> bool:
    """Return True when the entry is already on disk.

    Checkpoints must match the manifest size. Config sidecars only need to exist: a clone
    already tracks ``config_resolved.yaml`` for every seed, and the published copies have
    their run-bookkeeping paths normalized, so an existing file is never overwritten.
    """

    target = dest / entry.path
    if not target.is_file():
        return False
    return target.stat().st_size == entry.size if entry.is_checkpoint else True


def verify_files(entries: list[FileEntry], dest: Path) -> None:
    """Hash every entry under ``dest`` against the manifest; delete and report any that disagree."""

    mismatched: list[str] = []
    for entry in entries:
        target = dest / entry.path
        if not target.is_file():
            mismatched.append(f"{entry.path}: not downloaded")
            continue
        digest = sha256_file(target)
        if digest != entry.sha256:
            target.unlink()
            mismatched.append(f"{entry.path}: sha256 {digest} != manifest {entry.sha256}")
    if mismatched:
        raise ChecksumMismatchError(f"{len(mismatched)} file(s) failed verification (removed):\n  " + "\n  ".join(mismatched))


def print_listing(entries: list[FileEntry], dest: Path) -> None:
    """Print one line per cell with checkpoint count, size, and on-disk status."""

    for (arm, fold), group in group_by_cell(entries).items():
        present = sum(is_present(e, dest) for e in group)
        status = "present" if present == len(group) else ("partial" if present else "missing")
        checkpoints = sum(1 for e in group if e.is_checkpoint)
        print(f"{arm:<20} {fold:<72} {checkpoints:>2} ckpt {sum(e.size for e in group) / 1e9:5.2f} GB  {status}")
    print(f"{len(group_by_cell(entries))} cells, {sum(e.size for e in entries) / 1e9:.2f} GB total")


def fetch(repo_id: str = DEFAULT_REPO_ID, *, dest: Path = DEFAULT_DEST, arms: set[str] | None = None, folds: set[str] | None = None,
          everything: bool = False, list_only: bool = False, force: bool = False, revision: str | None = None,
          token: str | None = None, max_workers: int = 8) -> int:
    """Fetch the requested cells and return how many files were downloaded."""

    entries = load_remote_manifest(repo_id, dest, revision=revision, token=token)
    if list_only:
        print_listing(entries, dest)
        return 0

    if not everything and not arms and not folds:
        raise CheckpointError("Specify --arm and/or --fold, or --all")
    selected = entries if everything else select_entries(entries, arms=arms, folds=folds)
    if not selected:
        raise CheckpointError("No cells match the requested --arm/--fold")

    pending = [e for e in selected if force or not is_present(e, dest)]
    skipped = len(selected) - len(pending)
    if not pending:
        print(f"All {len(selected)} files already present under {dest}")
        return 0

    total_bytes = sum(e.size for e in pending)
    print(f"Fetching {len(pending)} files ({total_bytes / 1e9:.2f} GB) from {repo_id} into {dest}; {skipped} already present")
    started = time.perf_counter()
    snapshot_download(repo_id, repo_type=REPO_TYPE, revision=revision, token=token, local_dir=dest,
                      allow_patterns=[e.path for e in pending], force_download=force, max_workers=max_workers)
    elapsed = time.perf_counter() - started

    verify_files(pending, dest)
    print(f"Done: {len(pending)} files verified ({total_bytes / 1e9:.2f} GB in {elapsed:,.0f} s, "
          f"{total_bytes / 1e6 / max(elapsed, 1e-6):.1f} MB/s), {skipped} skipped")
    return len(pending)


def main() -> None:
    """Parse CLI arguments and fetch checkpoints."""

    parser = argparse.ArgumentParser(description="Fetch MCDN checkpoints from Hugging Face into outputs/ablation/.")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help="Model repository id.")
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit to fetch (default: main).")
    parser.add_argument("--arm", action="append", help="Arm to fetch (repeatable).")
    parser.add_argument("--fold", action="append", help="Fold to fetch (repeatable).")
    parser.add_argument("--all", action="store_true", help="Fetch every cell (~43 GB).")
    parser.add_argument("--list", action="store_true", help="List cells and on-disk status without downloading.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Ablation root to download into.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if present.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    parser.add_argument("--token-file", type=Path, default=None, help="File holding a read token (private repository only).")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help=f"Environment variable holding a token (default {DEFAULT_TOKEN_ENV}).")
    args = parser.parse_args()

    token = resolve_token(args.token_file, args.token_env)
    try:
        fetch(args.repo, dest=args.dest, arms=set(args.arm) if args.arm else None, folds=set(args.fold) if args.fold else None,
              everything=args.all, list_only=args.list, force=args.force, revision=args.revision, token=token,
              max_workers=args.workers)
    except CheckpointError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
