"""Unit tests for the Tier A SLURM job-manifest generator."""

from pathlib import Path

from cluster import make_job_manifest as mjm
from train import ABLATION_PRESET_FILES, FIXED_ABLATION_SEEDS


def test_build_job_matrix_composition() -> None:
    """Matrix is 430 runs: 6 arms x 4 holdouts x 10 seeds + typology completion + all_features block + crewed-GSD + deliverable + mask completion + pooling+typology + crewed-geometry."""

    jobs = mjm.build_job_matrix()
    assert len(jobs) == 430

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
        "all_features": 50,
        "downsample_crewed": 20,
        "downsample_deliverable": 20,
        "mask": 20,
        "pooling_typology": 40,
        "downsample_crewed_fov": 20
    }


def test_crewed_gsd_arm_appended_after_original_rows() -> None:
    """Crewed-GSD rows occupy positions 311-330 only: rows 1-310 keep their launch-time meaning.

    The shepherd maps done-markers to manifest rows by line number, so the appended arm
    must never displace an original row.
    """

    jobs = mjm.build_job_matrix()
    assert all(preset != "downsample_crewed" for preset, _holdout, _seed in jobs[:310])
    assert all(preset == "downsample_crewed" for preset, _holdout, _seed in jobs[310:330])
    crewed_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "downsample_crewed"}
    assert crewed_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_deliverable_arm_appended_after_crewed_rows() -> None:
    """Deliverable-matched rows occupy positions 331-350 only: rows 1-330 keep their meaning."""

    jobs = mjm.build_job_matrix()
    assert all(preset != "downsample_deliverable" for preset, _holdout, _seed in jobs[:330])
    assert all(preset == "downsample_deliverable" for preset, _holdout, _seed in jobs[330:350])
    deliverable_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "downsample_deliverable"}
    assert deliverable_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_mask_completion_arm_appended_after_deliverable_rows() -> None:
    """Mask-ablated completion rows occupy positions 351-370 only: rows 1-350 keep their meaning."""

    jobs = mjm.build_job_matrix()
    assert all(preset != "mask" for preset, _holdout, _seed in jobs[:350])
    assert all(preset == "mask" for preset, _holdout, _seed in jobs[350:370])
    mask_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "mask"}
    assert mask_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_pooling_typology_arm_runs_full_protocol_after_mask_rows() -> None:
    """Pooling+typology rows occupy positions 371-410 and cover all four reported columns."""

    jobs = mjm.build_job_matrix()
    assert all(preset != "pooling_typology" for preset, _holdout, _seed in jobs[:370])
    assert all(preset == "pooling_typology" for preset, _holdout, _seed in jobs[370:410])
    pt_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "pooling_typology"}
    assert pt_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"}


def test_fov_arm_occupies_final_rows() -> None:
    """Crewed-geometry rows occupy positions 411-430 only, so a truncated deploy can hold them back."""

    jobs = mjm.build_job_matrix()
    assert all(preset != "downsample_crewed_fov" for preset, _holdout, _seed in jobs[:410])
    assert all(preset == "downsample_crewed_fov" for preset, _holdout, _seed in jobs[410:])
    fov_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "downsample_crewed_fov"}
    assert fov_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_full_protocol_arms_cover_default_split_and_all_feasible_loeo() -> None:
    """Each full-protocol arm runs the dataset-default composite plus LOEO {Ida, Michael, Mayfield}."""

    jobs = mjm.build_job_matrix()
    for arm in mjm.FULL_PROTOCOL_ARMS:
        holdouts = {holdout for preset, holdout, _seed in jobs if preset == arm}
        assert holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"}


def test_default_split_composite_names_the_droids_test_events() -> None:
    """The '+'-joined composite names exactly the four DROIDs test events."""

    events = set(mjm.DEFAULT_SPLIT_HOLDOUT.split("+"))
    assert events == {"Hurricane Michael", "Hurricane Idalia", "Mussett Bayou Fire", "Mayfield Tornado"}


def test_typology_factorial_completion_runs_only_new_columns() -> None:
    """The published typology arm adds only the default-split and LOEO-Ida columns."""

    jobs = mjm.build_job_matrix()
    typology_holdouts = {holdout for preset, holdout, _seed in jobs if preset == "typology"}
    assert typology_holdouts == {mjm.DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"}


def test_all_features_block_covers_sanity_loeo_and_default_split() -> None:
    """all_features runs the T-23 spatial sanity block, all three LOEO columns, and the dataset-default split."""

    jobs = mjm.build_job_matrix()
    all_features_jobs = [job for job in jobs if job[0] == "all_features"]
    assert len(all_features_jobs) == 50
    holdouts = {holdout for _preset, holdout, _seed in all_features_jobs}
    assert holdouts == {mjm.PRESET_DEFAULT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado", mjm.DEFAULT_SPLIT_HOLDOUT}


def test_all_features_rows_keep_their_pre_rename_positions() -> None:
    """Rows 261-310 are all_features, with the former deployed_split rows (301-310) on the composite holdout."""

    jobs = mjm.build_job_matrix()
    assert all(preset == "all_features" for preset, _holdout, _seed in jobs[260:310])
    assert all(holdout == mjm.DEFAULT_SPLIT_HOLDOUT for _preset, holdout, _seed in jobs[300:310])
    assert all(holdout == mjm.PRESET_DEFAULT_HOLDOUT for _preset, holdout, _seed in jobs[260:270])


def test_build_job_matrix_stays_consistent_with_train_cli() -> None:
    """Every referenced preset is registered in train.py and the seed protocol matches."""

    jobs = mjm.build_job_matrix()
    assert {preset for preset, _holdout, _seed in jobs}.issubset(ABLATION_PRESET_FILES.keys())
    assert mjm.FIXED_ABLATION_SEEDS == FIXED_ABLATION_SEEDS

    for arm in mjm.FULL_PROTOCOL_ARMS:
        seeds = [seed for preset, holdout, seed in jobs if preset == arm and holdout == mjm.DEFAULT_SPLIT_HOLDOUT]
        assert seeds == FIXED_ABLATION_SEEDS


def test_write_manifest_emits_tab_separated_lf_lines(tmp_path: Path) -> None:
    """Manifest lines are three tab-separated fields with LF endings (sbatch-side sed/read contract)."""

    output_path = tmp_path / "jobs.tsv"
    mjm.write_manifest(jobs=mjm.build_job_matrix(), output_path=output_path)

    raw = output_path.read_bytes().decode("utf-8")
    assert "\r" not in raw
    assert raw.endswith("\n")

    lines = raw.splitlines()
    assert len(lines) == 430
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == 3
        assert fields[0] in ABLATION_PRESET_FILES
        assert int(fields[2]) in FIXED_ABLATION_SEEDS
    assert lines[0] == f"rgb_only\t{mjm.DEFAULT_SPLIT_HOLDOUT}\t0"
