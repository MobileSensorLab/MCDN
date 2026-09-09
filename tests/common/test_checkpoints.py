"""Unit tests for the shared checkpoint-publishing helpers in ``src/common/checkpoints.py``."""

import hashlib
import json
from pathlib import Path

import pytest

from src.common.checkpoints import (
    CheckpointError, FileEntry, ManifestError, group_by_cell, read_manifest, resolve_token, select_entries, sha256_file, write_manifest
)


def _entry(arm: str, fold: str, seed: str, name: str = "best_model.pt", size: int = 10) -> FileEntry:
    return FileEntry(path=f"{arm}/{fold}/{seed}/{name}", arm=arm, fold=fold, seed=seed, size=size, sha256="ab" * 32)


class TestFileEntry:
    """``FileEntry`` serialization and classification."""

    def test_roundtrip_and_checkpoint_flag(self) -> None:
        """``to_dict``/``from_dict`` round-trip; only ``best_model.pt`` counts as a checkpoint."""

        entry = _entry("all_features", "Hurricane_Ida", "seed_00")
        config = _entry("all_features", "Hurricane_Ida", "seed_00", name="config_resolved.yaml")

        assert FileEntry.from_dict(entry.to_dict()) == entry
        assert entry.is_checkpoint
        assert not config.is_checkpoint


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """Streaming digest equals a one-shot ``hashlib`` digest."""

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 1000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


class TestManifest:
    """Manifest writing and reading."""

    def test_write_then_read_roundtrip_sorted_by_path(self, tmp_path: Path) -> None:
        """Entries are written sorted by path with the repo id, and the sha256 file is ``sha256sum`` compatible."""

        entries = [_entry("rgb_only", "Hurricane_Ida", "seed_00"), _entry("all_features", "Hurricane_Ida", "seed_11")]

        json_path, sha_path = write_manifest(entries, tmp_path, repo_id="org/repo")

        loaded = read_manifest(json_path)
        assert [e.path for e in loaded] == sorted(e.path for e in entries)
        assert json.loads(json_path.read_text())["repo_id"] == "org/repo"
        assert sha_path.read_text().splitlines()[0] == f"{'ab' * 32}  all_features/Hurricane_Ida/seed_11/best_model.pt"

    def test_read_rejects_missing_and_malformed(self, tmp_path: Path) -> None:
        """A missing file or a row lacking fields raises ``ManifestError``."""

        with pytest.raises(ManifestError):
            read_manifest(tmp_path / "absent.json")
        bad = tmp_path / "bad.json"
        bad.write_text('{"files": [{"path": "x"}]}')
        with pytest.raises(ManifestError):
            read_manifest(bad)


def test_select_entries_and_group_by_cell() -> None:
    """Arm/fold filters combine with AND; ``None`` means unrestricted; grouping preserves first-seen cell order."""

    entries = [_entry("a", "F1", "seed_00"), _entry("a", "F2", "seed_00"), _entry("b", "F1", "seed_00"),
               _entry("a", "F1", "seed_11")]

    assert [e.path for e in select_entries(entries, arms={"a"}, folds={"F1"})] == ["a/F1/seed_00/best_model.pt", "a/F1/seed_11/best_model.pt"]
    assert len(select_entries(entries, arms=None, folds={"F1"})) == 3
    assert len(select_entries(entries, arms=None, folds=None)) == 4
    assert list(group_by_cell(entries)) == [("a", "F1"), ("a", "F2"), ("b", "F1")]


class TestResolveToken:
    """Token resolution from file or environment."""

    def test_prefers_file_over_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token file wins over ``HF_TOKEN``; without a file the environment is used."""

        token_file = tmp_path / "tok.txt"
        token_file.write_text("hf_fromfile\n")
        monkeypatch.setenv("HF_TOKEN", "hf_fromenv")

        assert resolve_token(token_file) == "hf_fromfile"
        assert resolve_token(None) == "hf_fromenv"

    def test_none_when_unset_and_error_when_file_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No file and no variable yields ``None``; an empty file is an error."""

        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert resolve_token(None) is None

        empty = tmp_path / "empty.txt"
        empty.write_text("  \n")
        with pytest.raises(CheckpointError):
            resolve_token(empty)
