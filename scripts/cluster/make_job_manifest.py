"""Generate the Tier A SLURM job manifest: one tab-separated line per training run.

Each line holds the tab-separated fields ``preset``, ``holdout``, and ``seed``, consumed
by ``train_array.sbatch`` where the SLURM array task ID selects its line. The holdout
column is either an explicit
event name (passed to ``train.py --holdout-event``) or the ``-`` sentinel, meaning the
flag is omitted and the preset's own split protocol applies: the default spatial-block
split for ordinary arms, or the composite multi-event holdout baked into
``ablation_deployed_split.yaml``.

Deliberately stdlib-only so it runs anywhere (head node, container, laptop) without the
project venv; the unit tests cross-check the hardcoded matrix against ``train.py``.
"""

import argparse

from pathlib import Path

# Mirrors train.FIXED_ABLATION_SEEDS (cross-checked by tests/scripts/test_make_job_manifest.py).
FIXED_ABLATION_SEEDS = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

# Omitting --holdout-event selects each preset's own split protocol.
PRESET_DEFAULT_HOLDOUT = "-"

# Full-protocol arms run the default spatial split plus both LOEO holdouts.
FULL_PROTOCOL_HOLDOUTS = [PRESET_DEFAULT_HOLDOUT, "Hurricane Michael", "Mayfield Tornado"]

# The six new revision arms (Tier A) run the full 3-holdout x 10-seed protocol.
FULL_PROTOCOL_ARMS = ["rgb_only", "no_smoothing", "mask_channel_only", "pooling_only", "ce_loss", "downsample_15cm"]

# The deployed-baseline replication arm defines its composite split in its preset,
# so it runs one 10-seed pool with no holdout override.
PRESET_SPLIT_ARMS = ["deployed_split"]

# Optional T-23 sanity rerun of the published baseline on cluster hardware.
BASELINE_ARM = "baseline"


def build_job_matrix(*, include_baseline: bool = False) -> list[tuple[str, str, int]]:
    """Compose the (preset, holdout, seed) run matrix for the Tier A array.

    Args:
        include_baseline: Also emit the optional baseline sanity-check runs (T-23),
            growing the matrix from 190 to 220 entries.

    Returns:
        Ordered list of ``(preset, holdout, seed)`` triples, grouped by arm then
        holdout then seed so related runs are adjacent in the array.
    """

    arms = [*FULL_PROTOCOL_ARMS, BASELINE_ARM] if include_baseline else list(FULL_PROTOCOL_ARMS)
    jobs = [
        (arm, holdout, seed)
        for arm in arms
        for holdout in FULL_PROTOCOL_HOLDOUTS
        for seed in FIXED_ABLATION_SEEDS
    ]
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
    parser.add_argument(
        "--include-baseline", action="store_true",
        help="Append the optional baseline sanity-check runs (T-23), 190 -> 220 entries."
    )
    args = parser.parse_args()

    jobs = build_job_matrix(include_baseline=args.include_baseline)
    write_manifest(jobs=jobs, output_path=args.output)
    print(f"Wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
