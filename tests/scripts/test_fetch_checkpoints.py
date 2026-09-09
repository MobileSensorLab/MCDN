"""Unit tests for ``scripts/fetch_checkpoints.py`` against an in-memory stand-in for the Hub."""

import fnmatch
import shutil
import sys
from pathlib import Path

import pytest

import scripts.fetch_checkpoints as fetch_mod
from scripts.fetch_checkpoints import fetch, is_present, main, verify_files
from src.common.checkpoints import MANIFEST_JSON, CheckpointError, ChecksumMismatchError, FileEntry, read_manifest, sha256_file, write_manifest


class FakeHub:
    """Serves a payload directory as if it were the remote repository."""

    def __init__(self, remote: Path) -> None:
        """Remember the directory that plays the repository."""

        self.remote = remote
        self.manifest_calls: list[dict] = []
        self.snapshot_calls: list[dict] = []

    def hf_hub_download(self, repo_id: str, filename: str, *, local_dir: Path, **kwargs: object) -> str:
        """Copy one file into ``local_dir`` and return its path, as the real function does."""

        self.manifest_calls.append({"repo_id": repo_id, "filename": filename, "local_dir": local_dir, **kwargs})
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.remote / filename, target)
        return str(target)

    def snapshot_download(self, repo_id: str, *, local_dir: Path, allow_patterns: list[str], **kwargs: object) -> str:
        """Copy every remote file matching ``allow_patterns`` into ``local_dir``."""

        self.snapshot_calls.append({"repo_id": repo_id, "local_dir": local_dir, "allow_patterns": allow_patterns, **kwargs})
        for source in self.remote.rglob("*"):
            relative = source.relative_to(self.remote).as_posix()
            if source.is_file() and any(fnmatch.fnmatch(relative, p) for p in allow_patterns):
                target = Path(local_dir) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return str(local_dir)


@pytest.fixture
def remote(ablation_tree: Path, tmp_path: Path) -> Path:
    """A published-style payload (checkpoint + config per seed, manifest) built from the fake ablation tree."""

    payload = tmp_path / "remote"
    entries: list[FileEntry] = []
    for seed_dir in sorted(ablation_tree.glob("*/*/seed_*")):
        arm, fold = seed_dir.parts[-3], seed_dir.parts[-2]
        if fold == "Spatial_Block_East" or not (seed_dir / "best_model.pt").is_file():
            continue
        for name in ("best_model.pt", "config_resolved.yaml"):
            relative = f"{arm}/{fold}/{seed_dir.name}/{name}"
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed_dir / name, target)
            entries.append(FileEntry(path=relative, arm=arm, fold=fold, seed=seed_dir.name, size=target.stat().st_size,
                                     sha256=sha256_file(target)))
    write_manifest(entries, payload, repo_id="org/repo")
    return payload


@pytest.fixture
def hub(remote: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """Route the script's Hub calls to the fake remote."""

    fake = FakeHub(remote)
    monkeypatch.setattr(fetch_mod, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(fetch_mod, "snapshot_download", fake.snapshot_download)
    return fake


class TestFetch:
    """``fetch`` end to end against the fake Hub."""

    def test_downloads_selected_cell_and_verifies(self, hub: FakeHub, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Only the requested arm is downloaded, the manifest is refreshed, and every file is hash-verified."""

        dest = tmp_path / "dest"

        count = fetch("org/repo", dest=dest, arms={"rgb_only"}, token="tok")

        assert count == 4  # two seeds x (checkpoint + config)
        assert (dest / "rgb_only" / "Hurricane_Ida" / "seed_00" / "best_model.pt").is_file()
        assert not (dest / "all_features").exists()
        call = hub.snapshot_calls[0]
        assert call["repo_type"] == "model"
        assert call["token"] == "tok"
        assert sorted(call["allow_patterns"]) == sorted(e.path for e in read_manifest(dest / MANIFEST_JSON) if e.arm == "rgb_only")
        assert hub.manifest_calls[0]["force_download"] is True
        assert "4 files verified" in capsys.readouterr().out

    def test_skips_present_files_and_force_redownloads(self, hub: FakeHub, tmp_path: Path) -> None:
        """A second fetch is a no-op; ``force`` re-downloads and passes ``force_download`` through."""

        dest = tmp_path / "dest"
        fetch("org/repo", dest=dest, arms={"rgb_only"})

        assert fetch("org/repo", dest=dest, arms={"rgb_only"}) == 0
        assert len(hub.snapshot_calls) == 1

        assert fetch("org/repo", dest=dest, arms={"rgb_only"}, force=True) == 4
        assert hub.snapshot_calls[-1]["force_download"] is True

    @pytest.mark.usefixtures("hub")
    def test_all_fetches_every_cell_and_fold_filter_works(self, tmp_path: Path) -> None:
        """``everything`` takes the whole manifest; arm and fold filters combine."""

        assert fetch("org/repo", dest=tmp_path / "a", everything=True) == 8
        assert fetch("org/repo", dest=tmp_path / "b", folds={"Hurricane_Ida"}, arms={"all_features"}) == 4

    def test_requires_a_selection_and_rejects_empty_match(self, hub: FakeHub, tmp_path: Path) -> None:
        """No selection and an empty selection are both errors raised before any download."""

        with pytest.raises(CheckpointError, match="--all"):
            fetch("org/repo", dest=tmp_path / "d")
        with pytest.raises(CheckpointError, match="No cells match"):
            fetch("org/repo", dest=tmp_path / "d", arms={"nope"})
        assert hub.snapshot_calls == []

    def test_list_prints_status_without_downloading(self, hub: FakeHub, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """``list_only`` reports present/partial/missing per cell and does not download."""

        dest = tmp_path / "dest"
        fetch("org/repo", dest=dest, arms={"rgb_only"})
        (dest / "rgb_only" / "Hurricane_Ida" / "seed_11" / "best_model.pt").unlink()

        assert fetch("org/repo", dest=dest, list_only=True) == 0

        out = capsys.readouterr().out
        all_features_line = next(line for line in out.splitlines() if line.startswith("all_features"))
        rgb_line = next(line for line in out.splitlines() if line.startswith("rgb_only"))
        assert all_features_line.endswith("missing")
        assert rgb_line.endswith("partial")
        assert "2 cells" in out
        assert len(hub.snapshot_calls) == 1

    @pytest.mark.usefixtures("hub")
    def test_corrupt_download_is_removed_and_reported(self, remote: Path, tmp_path: Path) -> None:
        """A file whose content disagrees with the manifest is deleted and named in the error."""

        corrupt = remote / "rgb_only" / "Hurricane_Ida" / "seed_00" / "best_model.pt"
        good = corrupt.read_bytes()
        corrupt.write_bytes(good[:-1] + bytes([good[-1] ^ 0xFF]))  # same size, different content
        dest = tmp_path / "dest"

        with pytest.raises(ChecksumMismatchError, match=r"seed_00/best_model\.pt: sha256"):
            fetch("org/repo", dest=dest, arms={"rgb_only"})
        assert not (dest / "rgb_only" / "Hurricane_Ida" / "seed_00" / "best_model.pt").exists()
        assert (dest / "rgb_only" / "Hurricane_Ida" / "seed_11" / "best_model.pt").exists()


def test_is_present_requires_matching_size_for_checkpoints_only(remote: Path, tmp_path: Path) -> None:
    """A checkpoint counts as present only at the manifest size; a config counts as present if it exists at all."""

    entries = read_manifest(remote / MANIFEST_JSON)
    checkpoint = next(e for e in entries if e.is_checkpoint)
    config = next(e for e in entries if not e.is_checkpoint)
    for entry in (checkpoint, config):
        (tmp_path / entry.path).parent.mkdir(parents=True, exist_ok=True)

    assert not is_present(checkpoint, tmp_path)
    (tmp_path / checkpoint.path).write_bytes(b"x")
    assert not is_present(checkpoint, tmp_path)
    shutil.copyfile(remote / checkpoint.path, tmp_path / checkpoint.path)
    assert is_present(checkpoint, tmp_path)

    assert not is_present(config, tmp_path)
    (tmp_path / config.path).write_text("locally tracked copy\n")
    assert is_present(config, tmp_path)


def test_verify_files_reports_missing(remote: Path, tmp_path: Path) -> None:
    """A manifest entry with no file on disk is reported as not downloaded."""

    entries = read_manifest(remote / MANIFEST_JSON)[:1]

    with pytest.raises(ChecksumMismatchError, match="not downloaded"):
        verify_files(entries, tmp_path)


class TestMain:
    """CLI wiring."""

    def test_cli_options_reach_fetch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every option maps onto the matching ``fetch`` keyword, including the token read from a file."""

        captured: dict[str, object] = {}

        def fake_fetch(repo_id: str, **kwargs: object) -> int:
            captured.update(repo_id=repo_id, **kwargs)
            return 0

        token_file = tmp_path / "tok.txt"
        token_file.write_text("hf_read")
        monkeypatch.setattr(fetch_mod, "fetch", fake_fetch)
        monkeypatch.setattr(sys, "argv", ["fetch", "--repo", "org/x", "--arm", "a", "--arm", "b", "--fold", "F", "--dest", str(tmp_path),
                                          "--force", "--revision", "deadbeef", "--workers", "3", "--token-file", str(token_file)])

        main()

        assert captured == {"repo_id": "org/x", "dest": tmp_path, "arms": {"a", "b"}, "folds": {"F"}, "everything": False,
                            "list_only": False, "force": True, "revision": "deadbeef", "token": "hf_read", "max_workers": 3}

    @pytest.mark.usefixtures("hub")
    def test_errors_become_exit_codes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``CheckpointError`` surfaces as a ``SystemExit`` with the message."""

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(sys, "argv", ["fetch", "--dest", str(tmp_path / "d")])

        with pytest.raises(SystemExit, match="--all"):
            main()
