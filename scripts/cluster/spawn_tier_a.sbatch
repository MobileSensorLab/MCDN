#!/usr/bin/env bash
# Pump shepherd for the Tier A campaign.
#
# Why this exists: the freetier QOS caps a user at 8 submitted jobs
# (pending + running combined), so the full job manifest cannot sit in the queue
# as one array (sbatch rejects it with QOSMaxSubmitJobPerUserLimit). This CPU-only
# job keeps the queue topped up with single-task training jobs, one manifest row
# each, until every row has completed successfully.
#
# Capacity is 7, not 8: the dgx partition carries a partition QOS (QoS=freetier,
# see scontrol show partition dgx) whose per-user submit cap counts every job in
# the partition — including this pump — regardless of the job's own QOS. Running
# the pump under the training QOS therefore does not buy a slot back; 7 training
# jobs + 1 pump = the hard ceiling of 8.
#
# Self-healing properties:
#   - Stateless: every cycle rescans done-markers and squeue, so a requeued or
#     resubmitted pump resumes exactly where the previous one stopped.
#   - A failed/preempted-and-lost task lacks its marker and is resubmitted, up to
#     MCDN_PUMP_MAX_TRIES fresh submissions per row (SLURM-level requeues of a
#     still-queued job do not consume tries).
#   - Rows already complete on disk no-op via --skip-if-complete, then write
#     their marker.
#   - Duplicate-pump guard: exits immediately if another mcdn_pump job is queued.
#
# Knobs (as plain env assignments at submission, e.g. MCDN_PUMP_SLEEP=60 sbatch ...):
#   MCDN_PUMP_CAPACITY  concurrent training jobs [7 — the partition cap of 8
#                       minus this pump]
#   MCDN_PUMP_MAX_TRIES fresh submissions per row before giving up [3]
#   MCDN_PUMP_SLEEP     seconds between cycles [180]
#   MCDN_TIME           per-task wall limit [06:00:00 — covers the 65-epoch tail]
#
#SBATCH --job-name=mcdn_pump
#SBATCH --partition=dgx
#SBATCH --account=ualbany
#SBATCH --qos=training
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/network/rit/lab/mobilesensorlab/mcdn/outputs/cluster_logs/%x_%j.out

set -uo pipefail  # deliberately no -e: the loop must ride out transient squeue/sbatch failures

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
MANIFEST=${MCDN_JOB_MANIFEST:-$REPO_ROOT/scripts/cluster/jobs.tsv}
SBATCH_SCRIPT=$REPO_ROOT/scripts/cluster/train_array.sbatch
DONE_DIR=$REPO_ROOT/outputs/cluster_logs/done
CAPACITY=${MCDN_PUMP_CAPACITY:-7}
MAX_TRIES=${MCDN_PUMP_MAX_TRIES:-3}
SLEEP_SECS=${MCDN_PUMP_SLEEP:-180}
TASK_TIME=${MCDN_TIME:-06:00:00}

PUMP_JOB_ID=${SLURM_JOB_ID:-manual}

# sbatch treats inherited SLURM_*/SBATCH_* variables as input environment, which
# outranks the child script's #SBATCH headers. Scrub them so training jobs get
# their own 16-CPU/64G/1-GPU headers instead of this job's 2-CPU/4G shape.
# SLURM_CONF is exempt: client commands (squeue/sbatch) need it on compute nodes
# to locate the controller.
while IFS= read -r var; do
    [[ "$var" == "SLURM_CONF" ]] && continue
    unset "$var"
done < <(compgen -e | grep -E '^(SLURM_|SBATCH_)')

mkdir -p "$DONE_DIR"
total=$(wc -l < "$MANIFEST")

# Refuse to run alongside another pump: double-submission races would follow.
other_pumps=$(squeue -h -u "$USER" -n mcdn_pump -o %i 2>/dev/null | grep -cv "^${PUMP_JOB_ID}\$")
if (( other_pumps > 0 )); then
    echo "[pump ${PUMP_JOB_ID}] another mcdn_pump is already queued/running; exiting."
    exit 0
fi

echo "[pump ${PUMP_JOB_ID}] shepherding ${total} rows: capacity ${CAPACITY}, task time ${TASK_TIME}, max tries ${MAX_TRIES}"

cycle=0
while :; do
    cycle=$((cycle + 1))

    if ! qnames=$(squeue -h -u "$USER" -o %j 2>"$DONE_DIR/.squeue_err"); then
        echo "[pump] cycle ${cycle}: squeue failed ($(head -1 "$DONE_DIR/.squeue_err" 2>/dev/null)); retrying in ${SLEEP_SECS}s"
        sleep "$SLEEP_SECS"
        continue
    fi
    in_flight=$(printf '%s\n' "$qnames" | grep -cE '^mcdn_t[0-9]+$')
    slots=$(( CAPACITY - in_flight ))

    # Classify rows: done, in flight, exhausted, or eligible for (re)submission.
    done_n=0
    exhausted=0
    eligible=()
    for (( row=1; row<=total; row++ )); do
        if [[ -f "$DONE_DIR/t${row}.done" ]]; then
            done_n=$((done_n + 1))
        elif printf '%s\n' "$qnames" | grep -qx "mcdn_t${row}"; then
            :
        else
            tries=0
            [[ -f "$DONE_DIR/t${row}.tries" ]] && tries=$(<"$DONE_DIR/t${row}.tries")
            if (( tries >= MAX_TRIES )); then
                exhausted=$((exhausted + 1))
            else
                eligible+=("$row")
            fi
        fi
    done

    submitted=0
    if (( ${#eligible[@]} > 0 )); then
        for row in "${eligible[@]}"; do
            (( slots <= 0 )) && break
            tries=0
            [[ -f "$DONE_DIR/t${row}.tries" ]] && tries=$(<"$DONE_DIR/t${row}.tries")
            # MCDN_TASK_ID rides the default full-env propagation: an explicit
            # --export=ALL,VAR=... triggers user-env retrieval on this cluster,
            # which fails and leaves jobs requeued-held.
            if out=$(MCDN_TASK_ID="$row" sbatch --job-name="mcdn_t${row}" --time="$TASK_TIME" \
                        --output="$REPO_ROOT/outputs/cluster_logs/mcdn_t${row}_%j.out" \
                        "$SBATCH_SCRIPT" 2>&1); then
                echo $(( tries + 1 )) > "$DONE_DIR/t${row}.tries"
                slots=$((slots - 1))
                submitted=$((submitted + 1))
                echo "[pump] cycle ${cycle}: row ${row} submitted (try $((tries + 1))): ${out}"
            else
                echo "[pump] cycle ${cycle}: sbatch refused row ${row} (${out}); deferring to next cycle"
                break
            fi
        done
    fi

    if (( done_n >= total )); then
        echo "[pump] all ${total} rows complete after ${cycle} cycles."
        exit 0
    fi
    if (( in_flight == 0 && submitted == 0 && ${#eligible[@]} == 0 )); then
        echo "[pump] stopping: ${done_n}/${total} done; ${exhausted} rows exhausted ${MAX_TRIES} tries:"
        for (( row=1; row<=total; row++ )); do
            [[ -f "$DONE_DIR/t${row}.done" ]] || echo "[pump]   incomplete row ${row}: $(sed -n "${row}p" "$MANIFEST")"
        done
        exit 1
    fi

    if (( cycle % 10 == 1 || submitted > 0 )); then
        echo "[pump] cycle ${cycle}: ${done_n}/${total} done, ${in_flight} in flight, ${submitted} submitted, ${exhausted} exhausted"
    fi
    sleep "$SLEEP_SECS"
done
