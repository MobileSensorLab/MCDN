#!/usr/bin/env bash
# T-22 dataset staging as a batch job (see stage_dataset.sh for the transfer itself).
#
# Runs on a compute node because head01 caps virtual memory at 500 MB (ulimit -v 512000),
# which aborts uv on startup. Requests no GPU: the pull is network- and I/O-bound, so it
# neither needs nor should consume the free tier's 8-GPU-per-user allowance.
#
# Free-tier jobs are preemptible by paidtier-1. --requeue plus the script's resumable,
# verifying download makes preemption harmless: a requeued task re-verifies what landed
# and continues from there.
#
# Submit:
#   sbatch scripts/cluster/stage_dataset.sbatch
# Watch:
#   tail -f /network/rit/lab/mobilesensorlab/mcdn/outputs/cluster_logs/mcdn_stage_<jobid>.out
#
#SBATCH --job-name=mcdn_stage
#SBATCH --partition=dgx
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/network/rit/lab/mobilesensorlab/mcdn/outputs/cluster_logs/%x_%j.out

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
export PYTHONUNBUFFERED=1

echo "[sbatch] job=${SLURM_JOB_ID:-none} node=$(hostname) start=$(date -Is)"
cd "$REPO_ROOT"
exec scripts/cluster/stage_dataset.sh
