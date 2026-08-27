"""Unit tests for the Tier A SLURM job-manifest generator."""

from pathlib import Path

from cluster import make_job_manifest as mjm
from train import ABLATION_PRESET_FILES, FIXED_ABLATION_SEEDS


def test_build_job_matrix_default_composition() -> None:
    """Default matrix is 190 runs: six full-protocol arms x 3 holdouts x 10 seeds + deployed_split x 10."""

    jobs = mjm.build_job_matrix()
    assert len(jobs) == 190

    per_arm: dict[str, int] = {}
    for preset, _holdout, _seed in jobs:
        per_arm[preset] = per_arm.get(preset, 0) + 1
    assert per_arm == {
        "rgb_only": 30,
        "no_smoothing": 30,
        "mask_channel_only": 30,
        "pooling_only": 30,
        "ce_loss": 30,
        "downsample_15cm": 30,
        "deployed_split": 10
    }


def test_build_job_matrix_include_baseline_adds_sanity_runs() -> None:
    """The T-23 flag appends 30 baseline runs across all three holdouts, totalling 220."""

    jobs = mjm.build_job_matrix(include_baseline=True)
    assert len(jobs) == 220

    baseline_jobs = [job for job in jobs if job[0] == "baseline"]
    assert len(baseline_jobs) == 30
    assert {holdout for _preset, holdout, _seed in baseline_jobs} == set(mjm.FULL_PROTOCOL_HOLDOUTS)


def test_build_job_matrix_stays_consistent_with_train_cli() -> None:
    """Every referenced preset is registered in train.py and the seed protocol matches."""

    jobs = mjm.build_job_matrix(include_baseline=True)
    assert {preset for preset, _holdout, _seed in jobs}.issubset(ABLATION_PRESET_FILES.keys())
    assert mjm.FIXED_ABLATION_SEEDS == FIXED_ABLATION_SEEDS

    for arm in mjm.FULL_PROTOCOL_ARMS:
        seeds = [seed for preset, holdout, seed in jobs if preset == arm and holdout == mjm.PRESET_DEFAULT_HOLDOUT]
        assert seeds == FIXED_ABLATION_SEEDS


def test_deployed_split_uses_preset_default_holdout() -> None:
    """The deployed-baseline arm never overrides the holdout: its split lives in the preset."""

    jobs = mjm.build_job_matrix()
    deployed_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "deployed_split"}
    assert deployed_holdouts == {mjm.PRESET_DEFAULT_HOLDOUT}


def test_write_manifest_emits_tab_separated_lf_lines(tmp_path: Path) -> None:
    """Manifest lines are three tab-separated fields with LF endings (sbatch-side sed/read contract)."""

    output_path = tmp_path / "jobs.tsv"
    mjm.write_manifest(jobs=mjm.build_job_matrix(), output_path=output_path)

    raw = output_path.read_bytes().decode("utf-8")
    assert "\r" not in raw
    assert raw.endswith("\n")

    lines = raw.splitlines()
    assert len(lines) == 190
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 3
        assert fields[0] in ABLATION_PRESET_FILES
        assert int(fields[2]) in FIXED_ABLATION_SEEDS
    assert lines[0] == "rgb_only\t-\t0"
