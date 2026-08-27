#!/usr/bin/env bash
# Tier A training array for the UAlbany DGX On-Prem cluster (SLURM + Pyxis/Enroot).
#
# Each array task runs one (preset, holdout, seed) line from the job manifest through
# train.py inside an NGC PyTorch container, writing into the standard
# outputs/ablation/<arm>/<split>/seed_<NN>/ layout on the lab share so all existing
# aggregation and postproc tooling works unchanged. --skip-if-complete makes requeued
# or resubmitted tasks no-ops when their artifact triplet already exists, so
# preemption (free tier, REQUEUE mode) and backfill resubmission are both safe.
#
# Prerequisites (one-time):
#   1. Repo checked out at $MCDN_REPO_ROOT and bootstrap_env.sh run (builds .venv).
#   2. Dataset staged at $MCDN_DATA_DIR (see tasks.md T-22).
#   3. Job manifest generated: python scripts/cluster/make_job_manifest.py
#
# Submission (use submit_tier_a.sh, which launches the pump shepherd): the freetier
# QOS caps a user at 8 submitted jobs, so the full manifest cannot be queued as one
# array. pump_tier_a.sbatch feeds single-task jobs through this script instead,
# passing the manifest row via MCDN_TASK_ID. Array invocation still works for
# smoke tests within the cap:
#   sbatch --array=1-1 scripts/cluster/train_array.sbatch
#
#SBATCH --job-name=mcdn_tier_a
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/network/rit/lab/mobilesensorlab/mcdn/outputs/cluster_logs/%x_%A_%a.out
# Pre-imported squashfs (seconds to start) instead of a per-task registry import (~4 min each).
# Regenerate after an image bump with:
#   srun --partition=dgx --gpus=0 --cpus-per-task=16 --mem=64G --time=01:00:00 \
#     enroot import -o /network/rit/dgx/dgx_mobilesensorlab/containers/nvidia+pytorch+25.01-py3.sqsh \
#     'docker://nvcr.io#nvidia/pytorch:25.01-py3'
#SBATCH --container-image=/network/rit/dgx/dgx_mobilesensorlab/containers/nvidia+pytorch+25.01-py3.sqsh
#SBATCH --container-mounts=/network/rit/lab/mobilesensorlab:/network/rit/lab/mobilesensorlab,/network/rit/dgx/dgx_mobilesensorlab:/network/rit/dgx/dgx_mobilesensorlab

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
DATA_DIR=${MCDN_DATA_DIR:-/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids}
MANIFEST=${MCDN_JOB_MANIFEST:-$REPO_ROOT/scripts/cluster/jobs.tsv}

# Shared HF cache so the pretrained backbone downloads once, not once per task.
export HF_HOME=${MCDN_HF_HOME:-$REPO_ROOT/.hf_cache}
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

# Row selection: array tasks use SLURM_ARRAY_TASK_ID; pump-fed single jobs pass
# the manifest row explicitly via MCDN_TASK_ID.
TASK_ID=${SLURM_ARRAY_TASK_ID:-${MCDN_TASK_ID:?need SLURM_ARRAY_TASK_ID or MCDN_TASK_ID}}

line=$(sed -n "${TASK_ID}p" "$MANIFEST")
if [[ -z "$line" ]]; then
    echo "ERROR: no manifest line ${TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=$'\t' read -r preset holdout seed <<< "$line"
echo "[task ${TASK_ID}] preset=${preset} holdout=${holdout} seed=${seed} node=$(hostname)"

args=(--ablation-preset "$preset" --seeds "$seed" --data-dir "$DATA_DIR" --skip-if-complete)
if [[ "$holdout" != "-" ]]; then
    args+=(--holdout-event "$holdout")
fi

"$REPO_ROOT/.venv/bin/python" train.py "${args[@]}"

# Success marker for the pump shepherd: reached only when train.py exits 0 (set -e
# aborts us otherwise), including --skip-if-complete no-ops on already-finished rows.
mkdir -p "$REPO_ROOT/outputs/cluster_logs/done"
touch "$REPO_ROOT/outputs/cluster_logs/done/t${TASK_ID}.done"
