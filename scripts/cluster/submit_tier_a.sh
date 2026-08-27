#!/usr/bin/env bash
# Submit the Tier A training array, sized from the job manifest.
#
# Usage (defaults in brackets):
#   MCDN_REPO_ROOT=...        repo checkout [/network/rit/lab/mobilesensorlab/mcdn]
#   MCDN_JOB_MANIFEST=...     manifest path [$MCDN_REPO_ROOT/scripts/cluster/jobs.tsv]
#   MCDN_ARRAY_THROTTLE=N     max concurrent tasks [8 — the free-tier GPU cap]
#   MCDN_QOS=training         optional QOS override [unset — free tier]
#   scripts/cluster/submit_tier_a.sh
#
# Resubmitting the same array after preemptions or failures is safe: completed
# (preset, holdout, seed) runs exit immediately via --skip-if-complete.

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
MANIFEST=${MCDN_JOB_MANIFEST:-$REPO_ROOT/scripts/cluster/jobs.tsv}
THROTTLE=${MCDN_ARRAY_THROTTLE:-8}
QOS=${MCDN_QOS:-}

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: job manifest not found at ${MANIFEST}." >&2
    echo "Generate it first: python scripts/cluster/make_job_manifest.py" >&2
    exit 1
fi

count=$(wc -l < "$MANIFEST")
mkdir -p "$REPO_ROOT/outputs/cluster_logs"

args=(--array="1-${count}%${THROTTLE}")
if [[ -n "$QOS" ]]; then
    args+=(--qos="$QOS")
fi

echo "[submit] ${count} tasks, throttle ${THROTTLE}${QOS:+, qos ${QOS}}"
sbatch "${args[@]}" "$REPO_ROOT/scripts/cluster/train_array.sbatch"
