"""Unit tests for the Tier A SLURM job-manifest generator."""

from pathlib import Path

from cluster import make_job_manifest as mjm
from train import ABLATION_PRESET_FILES, FIXED_ABLATION_SEEDS


def test_build_job_matrix_composition() -> None:
    """Matrix is 310 runs: 6 arms x 4 holdouts x 10 seeds + typology completion + baseline block + deployed_split."""

    jobs = mjm.build_job_matrix()
    assert len(jobs) == 310

    per_arm: dict[str, int] = {}
    for preset, _holdout, _seed in jobs:
        per_arm[preset] = per_arm.get(preset, 0) + 1
    assert per_arm == {
        "rgb_only": 40,
        "no_smoothing": 40,
        "mask_channel_only": 40,
        "pooling_only": 40,
        "ce_loss": 40,
        "downsample_15cm": 40,
        "typology": 20,
        "baseline": 40,
        "deployed_split": 10
    }


def test_full_protocol_arms_cover_default_split_and_all_feasible_loeo() -> None:
    """Each full-protocol arm runs the dataset-default composite plus LOEO {Ida, Michael, Mayfield}."""

    jobs = mjm.build_job_matrix()
    for arm in mjm.FULL_PROTOCOL_ARMS:
        holdouts = {holdout for preset, holdout, _seed in jobs if preset == arm}
        assert holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"}


def test_default_split_composite_matches_deployed_split_preset() -> None:
    """The '+'-joined composite names exactly the four DROIDs test events."""

    events = set(mjm.DEFAULT_SPLIT_HOLDOUT.split("+"))
    assert events == {"Hurricane Michael", "Hurricane Idalia", "Mussett Bayou Fire", "Mayfield Tornado"}


def test_typology_factorial_completion_runs_only_new_columns() -> None:
    """The published typology arm adds only the default-split and LOEO-Ida columns."""

    jobs = mjm.build_job_matrix()
    typology_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "typology"}
    assert typology_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_baseline_block_covers_ida_column_and_t23_sanity() -> None:
    """Baseline runs LOEO-Ida (new column) plus the spatial/Michael/Mayfield T-23 sanity reruns."""

    jobs = mjm.build_job_matrix()
    baseline_jobs = [job for job in jobs if job[0] == "baseline"]
    assert len(baseline_jobs) == 40
    holdouts = {holdout for _preset, holdout, _seed in baseline_jobs}
    assert holdouts == {mjm.PRESET_DEFAULT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"}


def test_build_job_matrix_stays_consistent_with_train_cli() -> None:
    """Every referenced preset is registered in train.py and the seed protocol matches."""

    jobs = mjm.build_job_matrix()
    assert {preset for preset, _holdout, _seed in jobs}.issubset(ABLATION_PRESET_FILES.keys())
    assert mjm.FIXED_ABLATION_SEEDS == FIXED_ABLATION_SEEDS

    for arm in mjm.FULL_PROTOCOL_ARMS:
        seeds = [seed for preset, holdout, seed in jobs if preset == arm and holdout == mjm.DEFAULT_SPLIT_HOLDOUT]
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
    assert len(lines) == 310
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 3
        assert fields[0] in ABLATION_PRESET_FILES
        assert int(fields[2]) in FIXED_ABLATION_SEEDS
    assert lines[0] == f"rgb_only\t{mjm.DEFAULT_SPLIT_HOLDOUT}\t0"
