"""Generate the Tier A SLURM job manifest: one tab-separated line per training run.

Each line holds the tab-separated fields ``preset``, ``holdout``, and ``seed``, consumed
by ``train_array.sbatch`` where the SLURM array task ID selects its line. The holdout
column is an explicit event name (passed to ``train.py --holdout-event``), a
``'+'``-joined composite of event names (parsed by the CLI into a multi-event fold), or
the ``-`` sentinel, meaning the flag is omitted and the preset's own split protocol
applies (the default spatial-block split, used only for the T-23 sanity runs).

The matrix implements the revised evaluation funnel (27 Aug decision, R1-1): the
dataset-default DROIDs split (train events {Ian, Harvey, Ida, Laura, Kilauea, Champlain}
vs test events {Michael, Idalia, Mussett Bayou, Mayfield}, verified event-disjoint and
identical to the split used by Manzini et al.) is the headline column, with LOEO over the
three feasible events {Ida, Michael, Mayfield} beside it. The former spatial-block fold
is retained only as a T-23 sanity run of the full configuration.

Naming note (4 Sep, D-12): the full configuration was renamed ``baseline`` ->
``all_features`` and the former ``deployed_split`` arm (the same configuration on the
dataset-default split) was merged into it. Rows 261-310 kept their positions and their
run semantics; only the preset label (and, for rows 301-310, the explicit composite
holdout replacing the retired preset's baked-in split) changed. The manifest deployed on
the cluster predates the rename and still carries the old labels; the two agree row for row.

Deliberately stdlib-only so it runs anywhere (head node, container, laptop) without the
project venv; the unit tests cross-check the hardcoded matrix against ``train.py``.
"""

import argparse

from pathlib import Path

# Mirrors train.FIXED_ABLATION_SEEDS (cross-checked by tests/scripts/test_make_job_manifest.py).
FIXED_ABLATION_SEEDS = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

# Omitting --holdout-event selects each preset's own split protocol.
PRESET_DEFAULT_HOLDOUT = "-"

# The dataset-default DROIDs test pool as a '+'-joined composite for --holdout-event.
# Every arm passes this same string, so all arms share one split directory name.
DEFAULT_SPLIT_HOLDOUT = "Hurricane Michael+Hurricane Idalia+Mussett Bayou Fire+Mayfield Tornado"

# All feasible single-event holdouts: four-class validation support >= ~1000 instances
# without starving training (rules out Ian at 66% of instances, the class-degenerate
# Harvey/Idalia/Laura/Kilauea pools, and the tiny Champlain/Mussett events).
LOEO_HOLDOUTS = ["Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"]

# Full-protocol arms run the dataset-default split plus all three LOEO holdouts.
FULL_PROTOCOL_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, *LOEO_HOLDOUTS]

# The six new revision arms (Tier A) run the full 4-holdout x 10-seed protocol.
FULL_PROTOCOL_ARMS = ["rgb_only", "no_smoothing", "mask_channel_only", "pooling_only", "ce_loss", "downsample_15cm"]

# The published typology arm (the both-on cell of the mask-channel x pooling factorial)
# needs only the two new columns; its Michael/Mayfield cells are published.
FACTORIAL_COMPLETION_ARM = "typology"
FACTORIAL_COMPLETION_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"]

# Full-configuration runs (rows 261-310). LOEO-Ida is a new column; the '-' sentinel
# (default spatial split) and the Michael/Mayfield reruns are the T-23 A100 sanity block
# that bounds the hardware effect against the published RTX 5090 numbers; the final ten
# rows are the dataset-default split column (formerly the separate `deployed_split` arm).
ALL_FEATURES_ARM = "all_features"
ALL_FEATURES_HOLDOUTS = [PRESET_DEFAULT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado", DEFAULT_SPLIT_HOLDOUT]

# Crewed-GSD degradation arm (29 Aug, T-3 calibration finding): factor 7.0 matches the
# real crewed pool's building-weighted GSD (median 25 cm vs ~3.4 cm native sUAS), which
# the 3x arm under-shoots (~10 cm effective). Replaced the MTF arm's rows 311-330 the
# same day, before any had started; occupies positions AFTER all original rows so
# manifest row numbers (and the shepherd's done-markers) for rows 1-310 are preserved.
CREWED_GSD_ARM = "downsample_crewed"
CREWED_GSD_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"]

# Deliverable-matched degradation arm (30 Aug): factor 7.0 plus a product-referenced
# unsharp mask (amount 0.2) regressed against the trustworthy overlapping Ian crewed
# sorties, landing the end-to-end transfer at ~0.44 at the crewed Nyquist (real
# deliverables measure 0.39-0.55). Appended AFTER rows 1-330 so existing manifest row
# numbers and the shepherd's done-markers are preserved.
DELIVERABLE_ARM = "downsample_deliverable"
DELIVERABLE_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"]

# Mask-ablated factorial-completion arm (30 Aug, author-directed): the published 'mask'
# arm (mask channel and pooling off, typology KEPT) was never trained on the DROIDs
# default split or LOEO-Ida in either lineage, leaving the typology x mask factorial
# open on the two headline columns. Appended AFTER rows 1-350 (done-markers preserved)
# and AHEAD of the FOV rows so it dispatches first.
MASK_COMPLETION_ARM = "mask"
MASK_COMPLETION_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"]

# Pooling + typology arm (30 Aug, author-directed): the two components the ensemble
# matrices identified as carrying the architecture's value, without the 4th-channel mask
# input — the one cell the existing factorial cannot adjudicate (is a leaner channel-free
# configuration equivalent to the full configuration?). Full protocol so it lands on every
# reported column. Appended AFTER rows 1-370 (mask-arm done-markers preserved).
POOLING_TYPOLOGY_ARM = "pooling_typology"
POOLING_TYPOLOGY_HOLDOUTS = FULL_PROTOCOL_HOLDOUTS

# Scale-preserving crewed-geometry arm (30 Aug): 7x-wider ground windows area-decimated
# into 512 px chips, reproducing the real crewed arm's chip presentation (wide context,
# small buildings, coarse-grid mask) with sUAS pixels and labels. Isolates the
# presentation/scale term of the sUAS-vs-crewed gap from the label and capture terms.
# Occupies the FINAL rows (411-430) so it can be held back by truncating the deployed
# manifest without perturbing any earlier row.
FOV_ARM = "downsample_crewed_fov"
FOV_HOLDOUTS = [DEFAULT_SPLIT_HOLDOUT, "Hurricane Ida"]


def build_job_matrix() -> list[tuple[str, str, int]]:
    """Compose the (preset, holdout, seed) run matrix for the Tier A array.

    Returns:
        Ordered list of ``(preset, holdout, seed)`` triples, grouped by arm then
        holdout then seed so related runs are adjacent in the array. 430 entries:
        6 arms x 4 holdouts x 10 seeds, plus 20 typology factorial-completion runs,
        plus 50 all_features runs (rows 261-310: T-23 sanity block, LOEO-Ida column,
        and the dataset-default split column), plus 20
        crewed-GSD factor-7 downsample runs (rows 311-330, appended post-launch),
        plus 20 deliverable-matched runs (rows 331-350, appended post-launch),
        plus 20 mask-ablated factorial-completion runs (rows 351-370, appended
        post-launch), plus 40 pooling+typology full-protocol runs (rows 371-410),
        plus 20 crewed-geometry wide-window runs (rows 411-430).
    """

    jobs = [
        (arm, holdout, seed)
        for arm in FULL_PROTOCOL_ARMS
        for holdout in FULL_PROTOCOL_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    ]
    jobs.extend(
        (FACTORIAL_COMPLETION_ARM, holdout, seed)
        for holdout in FACTORIAL_COMPLETION_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    jobs.extend((ALL_FEATURES_ARM, holdout, seed) for holdout in ALL_FEATURES_HOLDOUTS for seed in FIXED_ABLATION_SEEDS)
    jobs.extend(
        (CREWED_GSD_ARM, holdout, seed)
        for holdout in CREWED_GSD_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    jobs.extend(
        (DELIVERABLE_ARM, holdout, seed)
        for holdout in DELIVERABLE_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    jobs.extend(
        (MASK_COMPLETION_ARM, holdout, seed)
        for holdout in MASK_COMPLETION_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    jobs.extend(
        (POOLING_TYPOLOGY_ARM, holdout, seed)
        for holdout in POOLING_TYPOLOGY_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    jobs.extend(
        (FOV_ARM, holdout, seed)
        for holdout in FOV_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    )
    return jobs


def write_manifest(jobs: list[tuple[str, str, int]], output_path: Path) -> None:
    """Write the job matrix as a tab-separated manifest with LF line endings."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(f"{preset}\t{holdout}\t{seed}\n" for preset, holdout, seed in jobs)
    output_path.write_text(lines, encoding="utf-8", newline="\n")


def main() -> None:
    """CLI entry point: build the matrix and write the manifest file."""

    parser = argparse.ArgumentParser(description="Generate the Tier A SLURM job manifest.")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "jobs.tsv",
        help="Manifest destination (default: jobs.tsv next to this script)."
    )
    args = parser.parse_args()

    jobs = build_job_matrix()
    write_manifest(jobs=jobs, output_path=args.output)
    print(f"Wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
