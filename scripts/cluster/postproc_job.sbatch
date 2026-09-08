#!/usr/bin/env bash
# Phase 1 inference jobs for the UAlbany DGX On-Prem cluster (SLURM + Pyxis/Enroot).
#
# One job evaluates one (kind, variant, split) cell against finished checkpoints on the
# lab share, using the same container, mounts, venv, and data root as the Tier A trainers.
# Kinds:
#   robustness  scripts.eval_mask_robustness  (T-6: footprint alignment / offset / buffer / deletion)
#   decompose   src.postproc.decompose        (T-7: single-seed vs ensemble x TTA vs no-TTA)
#   soup        scripts.eval_soup             (D-16: backbone-only and full weight soups)
#
# Knobs are plain env assignments at submission (an explicit --export=ALL,VAR=... triggers
# user-environment retrieval on this cluster and is refused):
#   MCDN_JOB_KIND=robustness MCDN_SPLIT="Hurricane Ida" sbatch --job-name=mcdn_rob_ida \
#       scripts/cluster/postproc_job.sbatch
#   MCDN_VARIANT   variant directory under outputs/ablation [all_features]
#   MCDN_SPLIT     one or more split directory names (event name with '_' or the '+'-joined
#                  composite), ';'-separated; processed sequentially inside the one job
#   MCDN_CONDITIONS  (robustness only) space-separated condition tokens overriding the
#                  default sweep; MCDN_TAG_SUFFIX distinguishes the output artifact
#
#SBATCH --job-name=mcdn_postproc
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/network/rit/lab/mobilesensorlab/mcdn/outputs/cluster_logs/%x_%j.out
#SBATCH --container-image=/network/rit/dgx/dgx_mobilesensorlab/containers/nvidia+pytorch+25.01-py3.sqsh
#SBATCH --container-mounts=/network/rit/lab/mobilesensorlab:/network/rit/lab/mobilesensorlab,/network/rit/dgx/dgx_mobilesensorlab:/network/rit/dgx/dgx_mobilesensorlab

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
DATA_DIR=${MCDN_DATA_DIR:-/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids}
KIND=${MCDN_JOB_KIND:?need MCDN_JOB_KIND (robustness|decompose|soup)}
SPLIT_LIST=${MCDN_SPLIT:?need MCDN_SPLIT}
VARIANT=${MCDN_VARIANT:-all_features}
CONDITIONS=${MCDN_CONDITIONS:-"aligned raw_cache offset:15 offset:60 offset:75 offset:150 offset:240 buffer:-30 buffer:+30 mask_zero:auto"}
TAG_SUFFIX=${MCDN_TAG_SUFFIX:-}

export HF_HOME=${MCDN_HF_HOME:-$REPO_ROOT/.hf_cache}
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
VARIANT_ROOT="outputs/ablation/$VARIANT"
echo "[postproc] kind=${KIND} variant=${VARIANT} splits=${SPLIT_LIST} node=$(hostname) start=$(date -Is)"

run_one() {
    local SPLIT=$1
    local TAG="${VARIANT}__${SPLIT}${TAG_SUFFIX}"
    echo "[postproc] --- split=${SPLIT} start=$(date -Is)"
    case "$KIND" in
    robustness)
        OUT_DIR="outputs/mask_robustness"
        mkdir -p "$OUT_DIR"
        "$PY" -m scripts.eval_mask_robustness \
            --variant-roots "$VARIANT_ROOT" \
            --split "$SPLIT" \
            --data-dir "$DATA_DIR" \
            --conditions $CONDITIONS \
            --output-json "$OUT_DIR/${TAG}.json" \
            --probs-cache-dir "$OUT_DIR/probs/${TAG}"
        ;;
    decompose)
        OUT_DIR="outputs/decomposition"
        mkdir -p "$OUT_DIR"
        "$PY" -m src.postproc.decompose \
            --variant-roots "$VARIANT_ROOT" \
            --split "$SPLIT" \
            --data-dir "$DATA_DIR" \
            --output-json "$OUT_DIR/${TAG}.json" \
            --view-cache "$OUT_DIR/${TAG}_views.pt"
        ;;
    soup)
        OUT_DIR="outputs/soup"
        mkdir -p "$OUT_DIR"
        "$PY" -m scripts.eval_soup \
            --variant-root "$VARIANT_ROOT" \
            --split "$SPLIT" \
            --data-dir "$DATA_DIR" \
            --output-json "$OUT_DIR/${TAG}.json" \
            --probs-cache "$OUT_DIR/${TAG}_probs.pt"
        ;;
    *)
        echo "ERROR: unknown MCDN_JOB_KIND '${KIND}'" >&2
        exit 2
        ;;
    esac
    echo "[postproc] --- split=${SPLIT} done=$(date -Is)"
}

IFS=';' read -ra SPLITS <<< "$SPLIT_LIST"
for split in "${SPLITS[@]}"; do
    run_one "$split"
done

echo "[postproc] done kind=${KIND} splits=${SPLIT_LIST} end=$(date -Is)"
