#!/usr/bin/env bash
# Launch the Tier A campaign via the spawner shepherd (spawn_tier_a.sbatch).
#
# The freetier QOS caps a user at 8 submitted jobs (pending + running), so the
# full manifest cannot be queued as one array. Instead a CPU-only pump job keeps
# single-task freetier training jobs in flight until every manifest row has
# completed. The dgx partition QOS (QoS=freetier) counts every job in the
# partition against the 8-job cap — the pump included, whatever its own QOS — so
# capacity is 7 training jobs either way. The pump prefers the training QOS
# (ualbany account) and falls back to freetier if refused.
#
# Usage (defaults in brackets):
#   MCDN_REPO_ROOT=...      repo checkout [/network/rit/lab/mobilesensorlab/mcdn]
#   MCDN_JOB_MANIFEST=...   manifest path [$MCDN_REPO_ROOT/scripts/cluster/jobs.tsv]
#   MCDN_TIME=HH:MM:SS      per-task wall limit [06:00:00]
#   scripts/cluster/submit_tier_a.sh
#
# The 06:00:00 task default overrides the sbatch header's 04:00:00: the published
# early-stop tail reaches 65 epochs (resolution arm) and a worst-case
# no-early-stop run (100 epochs at ~2.5-2.8 min/epoch on the A100) needs
# ~4.5-4.7 h. Under the 4 h header such a run would be wall-killed, requeued, and
# restarted from scratch in a loop; 6 h covers the hard maximum with margin.
#
# Resubmitting after any interruption is safe: the pump is stateless (it rescans
# done-markers and the queue each cycle), refuses to run alongside another pump,
# and rows already complete on disk no-op via --skip-if-complete.

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
MANIFEST=${MCDN_JOB_MANIFEST:-$REPO_ROOT/scripts/cluster/jobs.tsv}
PUMP=$REPO_ROOT/scripts/cluster/spawn_tier_a.sbatch

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: job manifest not found at ${MANIFEST}." >&2
    echo "Generate it first: python scripts/cluster/make_job_manifest.py" >&2
    exit 1
fi

count=$(wc -l < "$MANIFEST")
mkdir -p "$REPO_ROOT/outputs/cluster_logs/done"

# Knobs are passed as plain env assignments (default sbatch behavior propagates
# the full submission environment): an explicit --export=ALL,VAR=... triggers
# user-env retrieval on this cluster, which fails and leaves jobs requeued-held.
echo "[submit] pump shepherd for ${count} manifest rows"
if out=$(sbatch "$PUMP" 2>&1); then
    echo "[submit] ${out} (training QOS pump, 7 freetier GPU slots)"
else
    echo "[submit] training QOS refused (${out}); falling back to freetier pump"
    sbatch --account=mobilesensorlab --qos=freetier --time=3-00:00:00 "$PUMP"
fi
