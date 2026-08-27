#!/usr/bin/env bash
# One-time environment bootstrap for the UAlbany DGX cluster.
#
# Installs uv (if absent), provisions the Python 3.13 interpreter, and syncs the
# project venv with the linux CUDA torch backend (--extra cu128; A100 is sm_80).
# Run from the head node with the repo checked out on the lab share:
#   MCDN_REPO_ROOT=/network/rit/lab/mobilesensorlab/mcdn scripts/cluster/bootstrap_env.sh
#
# The venv lands at $MCDN_REPO_ROOT/.venv, which train_array.sbatch invokes directly.
# All wheels are manylinux (torch bundles its own CUDA runtime), so building here and
# executing inside the NGC container is safe.
#
# Must run on a COMPUTE node: head01 enforces ulimit -v 512000 (500 MB virtual memory),
# which aborts uv on startup. Wrap it in srun, e.g.
#   srun --partition=dgx --gpus=0 --cpus-per-task=8 --mem=32G --time=01:00:00 \
#       scripts/cluster/bootstrap_env.sh

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
cd "$REPO_ROOT"

# Caches on the lab share: the CUDA torch wheels alone would overrun the 13 GB home quota.
export UV_CACHE_DIR=${MCDN_UV_CACHE_DIR:-/network/rit/lab/mobilesensorlab/.uv_cache}
export HF_HOME=${MCDN_HF_HOME:-$REPO_ROOT/.hf_cache}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

if ! command -v uv >/dev/null 2>&1; then
    echo "[bootstrap] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[bootstrap] provisioning Python 3.13"
uv python install 3.13

echo "[bootstrap] syncing project venv (cu128 backend, no dev group)"
uv sync --frozen --extra cu128 --no-dev

echo "[bootstrap] sanity check"
"$REPO_ROOT/.venv/bin/python" -c "import torch; print(f'torch {torch.__version__}  cuda {torch.version.cuda}')"
echo "[bootstrap] done"
