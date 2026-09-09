"""Unit tests for ``scripts/fetch_dataset.py`` against an in-memory stand-in for the Hub."""

import fnmatch
import shutil
import sys

from pathlib import Path

import pytest

import scripts.fetch_dataset as fetch_mod

from scripts.fetch_dataset import DatasetFetchError, RemoteFile, fetch, is_present, main, select_files, verify_sizes

# A miniature CRASAR-U-DROIDs tree: two splits, three sensors, plus the required top-level files.
REMOTE_LAYOUT: dict[str, int] = {
    "README.md": 40,
    "statistics.csv": 300,
    "format/annotations/JDS/1.1.json": 50,
    "train/imagery/UAS/a.tif": 4000,
    "train/imagery/UAS/b.tif": 5000,
    "train/annotations/UAS/building_damage_assessment/a.tif.json": 120,
    "train/annotations/UAS/building_alignment_adjustments/a.tif.json": 60,
    "train/imagery/CREWED/c.tif": 2000,
    "train/annotations/CREWED/building_damage_assessment/c.tif.json": 90,
    "train/imagery/SATELLITE/s.tif": 700,
    "test/imagery/UAS/d.tif": 3000,
    "test/annotations/UAS/building_damage_assessment/d.tif.json": 100,
    "test/imagery/UAS_DSM/d_dsm.tif": 10
}


class FakeHub:
    """Serves a directory as if it were the dataset repository."""

    def __init__(self, remote: Path) -> None:
        """Remember the directory that plays the repository."""

        self.remote = remote
        self.list_calls: list[dict] = []
        self.snapshot_calls: list[dict] = []

    def list_remote_files(self, repo_id: str, **kwargs: object) -> list[RemoteFile]:
        """Report every file under the fake repository with its size."""

        self.list_calls.append({"repo_id": repo_id, **kwargs})
        return [RemoteFile(path=p.relative_to(self.remote).as_posix(), size=p.stat().st_size)
                for p in sorted(self.remote.rglob("*")) if p.is_file()]

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
def remote(tmp_path: Path) -> Path:
    """Materialize ``REMOTE_LAYOUT`` on disk."""

    root = tmp_path / "remote"
    for relative, size in REMOTE_LAYOUT.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes([len(relative) % 251]) * size)
    return root


@pytest.fixture
def hub(remote: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """Route the module's Hub calls to the fake repository."""

    fake = FakeHub(remote)
    monkeypatch.setattr(fetch_mod, "list_remote_files", fake.list_remote_files)
    monkeypatch.setattr(fetch_mod, "snapshot_download", fake.snapshot_download)
    return fake


@pytest.fixture
def files(hub: FakeHub) -> list[RemoteFile]:
    """The fake repository's inventory."""

    return hub.list_remote_files("CRASAR/CRASAR-U-DROIDs")


def test_remote_file_classifies_paths() -> None:
    """Split, sensor, and required-ness derive from the path alone."""

    uas = RemoteFile("train/imagery/UAS/a.tif", 1)
    assert (uas.split, uas.sensor_dir, uas.required) == ("train", "UAS", False)
    ann = RemoteFile("test/annotations/CREWED/building_damage_assessment/c.tif.json", 1)
    assert (ann.split, ann.sensor_dir) == ("test", "CREWED")
    stats = RemoteFile("statistics.csv", 1)
    assert (stats.split, stats.sensor_dir, stats.required) == (None, None, True)
    assert RemoteFile("format/annotations/JDS/1.1.json", 1).required
    assert not RemoteFile("README.md", 1).required


def test_select_files_defaults_to_uas_with_required_extras(files: list[RemoteFile]) -> None:
    """The sUAS selection includes both splits' UAS imagery and annotations plus statistics.csv and format/."""

    chosen = {f.path for f in select_files(files, sensors={"uas"}, splits={"train", "test"})}
    assert chosen == {
        "statistics.csv", "format/annotations/JDS/1.1.json",
        "train/imagery/UAS/a.tif", "train/imagery/UAS/b.tif",
        "train/annotations/UAS/building_damage_assessment/a.tif.json",
        "train/annotations/UAS/building_alignment_adjustments/a.tif.json",
        "test/imagery/UAS/d.tif", "test/annotations/UAS/building_damage_assessment/d.tif.json"
    }


def test_select_files_honors_sensor_and_split(files: list[RemoteFile]) -> None:
    """Crewed + test narrows to the crewed test files (none here) plus the required extras; dsm maps to UAS_DSM."""

    crewed_test = {f.path for f in select_files(files, sensors={"crewed"}, splits={"test"})}
    assert crewed_test == {"statistics.csv", "format/annotations/JDS/1.1.json"}
    crewed_train = {f.path for f in select_files(files, sensors={"crewed"}, splits={"train"})}
    assert "train/imagery/CREWED/c.tif" in crewed_train
    assert "train/imagery/SATELLITE/s.tif" not in crewed_train
    dsm = {f.path for f in select_files(files, sensors={"dsm"}, splits={"test"})}
    assert "test/imagery/UAS_DSM/d_dsm.tif" in dsm


def test_select_files_rejects_unknown_sensor(files: list[RemoteFile]) -> None:
    """An unrecognized sensor name is an error, not an empty selection."""

    with pytest.raises(DatasetFetchError, match="Unknown sensor"):
        select_files(files, sensors={"lidar"}, splits={"train"})


def test_is_present_requires_matching_size(tmp_path: Path) -> None:
    """Presence means the file exists at the published size."""

    entry = RemoteFile("train/imagery/UAS/a.tif", 4000)
    target = tmp_path / entry.path
    target.parent.mkdir(parents=True)
    assert not is_present(entry, tmp_path)
    target.write_bytes(b"x" * 10)
    assert not is_present(entry, tmp_path)
    target.write_bytes(b"x" * 4000)
    assert is_present(entry, tmp_path)


def test_fetch_downloads_default_selection_and_skips_present(hub: FakeHub, tmp_path: Path) -> None:
    """The default fetch mirrors the sUAS subset in place; a second call downloads nothing."""

    dest = tmp_path / "data"
    (dest / "readme").mkdir(parents=True)
    (dest / "readme" / "badge.svg").write_text("keep me")

    downloaded = fetch(dest=dest)
    assert downloaded == 8
    assert (dest / "train" / "imagery" / "UAS" / "b.tif").stat().st_size == 5000
    assert (dest / "statistics.csv").is_file()
    assert not (dest / "train" / "imagery" / "CREWED").exists()
    assert not (dest / "README.md").exists()
    assert (dest / "readme" / "badge.svg").read_text() == "keep me"
    call = hub.snapshot_calls[0]
    assert call["repo_type"] == "dataset"
    assert sorted(call["allow_patterns"]) == sorted(f.path for f in select_files(hub.list_remote_files("x"), sensors={"uas"}, splits={"train", "test"}))

    assert fetch(dest=dest) == 0
    assert len(hub.snapshot_calls) == 1


def test_fetch_adds_only_missing_files(hub: FakeHub, tmp_path: Path) -> None:
    """Files already on disk at the right size are excluded from the download request."""

    dest = tmp_path / "data"
    fetch(dest=dest, sensors={"uas"}, splits={"test"})
    assert len(hub.snapshot_calls[0]["allow_patterns"]) == 4  # d.tif, its annotation, statistics.csv, format json

    fetch(dest=dest, sensors={"uas", "crewed"}, splits={"train", "test"})
    patterns = hub.snapshot_calls[1]["allow_patterns"]
    assert "test/imagery/UAS/d.tif" not in patterns
    assert "statistics.csv" not in patterns
    assert "train/imagery/CREWED/c.tif" in patterns


def test_fetch_force_redownloads(hub: FakeHub, tmp_path: Path) -> None:
    """``force`` re-requests every selected file and passes ``force_download`` through."""

    dest = tmp_path / "data"
    fetch(dest=dest, sensors={"uas"}, splits={"test"})
    assert fetch(dest=dest, sensors={"uas"}, splits={"test"}, force=True) == 4
    assert hub.snapshot_calls[1]["force_download"] is True


def test_fetch_list_only_prints_groups_without_downloading(hub: FakeHub, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--list`` reports per split/sensor groups and their on-disk status, and downloads nothing."""

    assert fetch(dest=tmp_path / "data", list_only=True) == 0
    out = capsys.readouterr().out
    assert "train/UAS" in out
    assert "test/UAS_DSM" in out
    assert "required" in out
    assert "missing" in out
    assert "files," in out
    assert not hub.snapshot_calls


@pytest.mark.usefixtures("hub")
def test_fetch_raises_on_empty_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A selection that matches nothing at all is reported rather than silently completing."""

    monkeypatch.setattr(fetch_mod, "list_remote_files", lambda *_a, **_k: [])
    with pytest.raises(DatasetFetchError, match="No files match"):
        fetch(dest=tmp_path / "data")


def test_verify_sizes_removes_short_files(remote: Path, tmp_path: Path) -> None:
    """A truncated download is deleted and reported; an absent one is reported."""

    dest = tmp_path / "data"
    good = RemoteFile("statistics.csv", 300)
    bad = RemoteFile("train/imagery/UAS/a.tif", 4000)
    absent = RemoteFile("train/imagery/UAS/b.tif", 5000)
    for entry in (good, bad):
        (dest / entry.path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / entry.path, dest / entry.path)
    (dest / bad.path).write_bytes(b"short")

    with pytest.raises(DatasetFetchError, match="2 file\\(s\\) failed") as excinfo:
        verify_sizes([good, bad, absent], dest)
    assert "a.tif: size 5 != published 4000" in str(excinfo.value)
    assert "b.tif: not downloaded" in str(excinfo.value)
    assert not (dest / bad.path).exists()
    assert (dest / good.path).exists()


def test_fetch_verification_failure_surfaces(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that lands at the wrong size fails the fetch."""

    def truncating_snapshot(repo_id: str, *, local_dir: Path, allow_patterns: list[str], **kwargs: object) -> str:
        hub.snapshot_download(repo_id, local_dir=local_dir, allow_patterns=allow_patterns, **kwargs)
        (Path(local_dir) / "test" / "imagery" / "UAS" / "d.tif").write_bytes(b"truncated")
        return str(local_dir)

    monkeypatch.setattr(fetch_mod, "snapshot_download", truncating_snapshot)
    with pytest.raises(DatasetFetchError, match="failed verification"):
        fetch(dest=tmp_path / "data", sensors={"uas"}, splits={"test"})


def test_main_parses_repeatable_flags(hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI maps repeated --sensor/--split onto sets and passes the destination through."""

    dest = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["fetch_dataset", "--sensor", "uas", "--sensor", "crewed", "--split", "train", "--dest", str(dest)])
    main()
    patterns = set(hub.snapshot_calls[0]["allow_patterns"])
    assert "train/imagery/CREWED/c.tif" in patterns
    assert "train/imagery/UAS/a.tif" in patterns
    assert "test/imagery/UAS/d.tif" not in patterns
    assert hub.snapshot_calls[0]["local_dir"] == dest


@pytest.mark.usefixtures("hub")
def test_main_reports_errors_as_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Domain errors become a clean SystemExit with the message, not a traceback."""

    monkeypatch.setattr(fetch_mod, "list_remote_files", lambda *_a, **_k: [])
    monkeypatch.setattr(sys, "argv", ["fetch_dataset", "--dest", str(tmp_path / "data")])
    with pytest.raises(SystemExit, match="error: No files match"):
        main()
