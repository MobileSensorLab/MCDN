#!/usr/bin/env bash
# One-time environment bootstrap for the UAlbany DGX cluster.
#
# Installs uv (if absent), provisions the Python 3.13 interpreter, and syncs the
# project venv with the linux CUDA torch backend (--extra cu128; A100 is sm_80).
# Run from the head node with the repo checked out on the lab share:
#   MCDN_REPO_ROOT=/network/rit/lab/mobilesensorlab/mcdn scripts/cluster/bootstrap_env.sh
#
# The venv lands at $MCDN_REPO_ROOT/.venv, which train_array.sbatch invokes directly.
# All wheels are manylinux (torch bundles its own CUDA runtime), so building on the
# head node and executing inside the NGC container is safe.

set -euo pipefail

REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
cd "$REPO_ROOT"

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
