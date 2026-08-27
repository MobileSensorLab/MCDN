"""Generate the Tier A SLURM job manifest: one tab-separated line per training run.

Each line holds the tab-separated fields ``preset``, ``holdout``, and ``seed``, consumed
by ``train_array.sbatch`` where the SLURM array task ID selects its line. The holdout
column is an explicit event name (passed to ``train.py --holdout-event``), a
``'+'``-joined composite of event names (parsed by the CLI into a multi-event fold), or
the ``-`` sentinel, meaning the flag is omitted and the preset's own split protocol
applies: the default spatial-block split for the baseline sanity runs, or the composite
holdout baked into ``ablation_deployed_split.yaml``.

The matrix implements the revised evaluation funnel (27 Aug decision, R1-1): the
dataset-default DROIDs split (train events {Ian, Harvey, Ida, Laura, Kilauea, Champlain}
vs test events {Michael, Idalia, Mussett Bayou, Mayfield}, verified event-disjoint and
identical to the deployed-baseline split) is the headline column, with LOEO over the
three feasible events {Ida, Michael, Mayfield} beside it. The former spatial-block fold
is retained only as a T-23 baseline sanity run.

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
# Matches ablation_deployed_split.yaml, so all arms share one split directory name.
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

# Baseline runs: LOEO-Ida is a new column; the '-' sentinel (default spatial split) and
# the Michael/Mayfield reruns are the T-23 A100 sanity block that bounds the hardware
# effect against the published RTX 5090 numbers. Baseline-on-default-split is covered by
# the deployed_split arm below.
BASELINE_ARM = "baseline"
BASELINE_HOLDOUTS = [PRESET_DEFAULT_HOLDOUT, "Hurricane Ida", "Hurricane Michael", "Mayfield Tornado"]

# The deployed-baseline replication arm defines its composite split in its preset,
# so it runs one 10-seed pool with no holdout override.
PRESET_SPLIT_ARMS = ["deployed_split"]


def build_job_matrix() -> list[tuple[str, str, int]]:
    """Compose the (preset, holdout, seed) run matrix for the Tier A array.

    Returns:
        Ordered list of ``(preset, holdout, seed)`` triples, grouped by arm then
        holdout then seed so related runs are adjacent in the array. 310 entries:
        6 arms x 4 holdouts x 10 seeds, plus 20 typology factorial-completion runs,
        plus 40 baseline runs (LOEO-Ida column + T-23 sanity block), plus 10
        deployed_split runs (baseline on the dataset-default split).
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
    jobs.extend((BASELINE_ARM, holdout, seed) for holdout in BASELINE_HOLDOUTS for seed in FIXED_ABLATION_SEEDS)
    jobs.extend((arm, PRESET_DEFAULT_HOLDOUT, seed) for arm in PRESET_SPLIT_ARMS for seed in FIXED_ABLATION_SEEDS)
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
