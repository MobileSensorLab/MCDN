#!/usr/bin/env bash
# T-22: stage the UAS-only CRASAR-U-DROIDs payload onto the DGX flash share.
#
# Pulls imagery + annotations for the UAS sensor branch only (~182 GiB; the CREWED /
# SATELLITE / UAS_DSM branches are never read by scan_dataset) from Hugging Face at a
# pinned revision, into the layout the manifest tooling expects:
#   $MCDN_DATA_DIR/statistics.csv
#   $MCDN_DATA_DIR/{train,test}/imagery/UAS/*.tif
#   $MCDN_DATA_DIR/{train,test}/annotations/UAS/{building_damage_assessment,building_alignment_adjustments}/*.json
#
# statistics.csv is mandatory, not incidental: _attach_source_metadata reads the per-
# orthomosaic Event and Source columns from it. Without it every mosaic collapses to a
# single synthetic event label, which silently disables LOEO splitting and the
# sensor_profile filter (observed 27 Aug on a first staging pass that omitted it).
#
# Idempotent: the HF downloader verifies and resumes existing files, so rerun after any
# interruption (including free-tier preemption, hence --requeue in the sbatch wrapper).
#
# Must run on a COMPUTE node, not the head node: head01 enforces ulimit -v 512000 (500 MB
# virtual memory), which aborts uv immediately ("memory allocation of N bytes failed").
# Submit through the wrapper, which requests no GPU:
#   sbatch scripts/cluster/stage_dataset.sbatch
#
# Auth: reads an optional read-scoped token from $MCDN_HF_TOKEN_FILE (default
# ~/.hf_token, chmod 600) for per-account instead of per-IP rate limits; anonymous
# works too. Verification compares local file counts and byte sizes against the HF
# tree at the pinned revision and writes STAGING_MANIFEST.txt beside the data; the
# script exits nonzero on any mismatch.

set -euo pipefail
umask 0002

DATASET_REPO=${MCDN_DATASET_REPO:-CRASAR/CRASAR-U-DROIDs}
DATA_DIR=${MCDN_DATA_DIR:-/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids}
REPO_ROOT=${MCDN_REPO_ROOT:-/network/rit/lab/mobilesensorlab/mcdn}
TOKEN_FILE=${MCDN_HF_TOKEN_FILE:-$HOME/.hf_token}

# Shared HF and uv caches on the lab share; nothing may land in the 13 GB home quota.
export HF_HOME=${MCDN_HF_HOME:-$REPO_ROOT/.hf_cache}
export UV_CACHE_DIR=${MCDN_UV_CACHE_DIR:-/network/rit/lab/mobilesensorlab/.uv_cache}
export HF_HUB_ENABLE_HF_TRANSFER=1

# Pin below 1.0 so the huggingface-cli entry point and hf_transfer extra are guaranteed.
HUB_SPEC='huggingface_hub[cli,hf_transfer]<1.0'

if [[ -f "$TOKEN_FILE" ]]; then
    HF_TOKEN=$(<"$TOKEN_FILE")
    export HF_TOKEN
    echo "[stage] HF token loaded from $TOKEN_FILE"
else
    echo "[stage] no token file at $TOKEN_FILE - proceeding anonymously"
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uvx >/dev/null 2>&1; then
    echo "[stage] installing uv (provides uvx)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

mkdir -p "$DATA_DIR" "$HF_HOME" "$UV_CACHE_DIR"

echo "[stage] resolving current dataset revision"
REVISION=$(uvx --from "$HUB_SPEC" python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('$DATASET_REPO').sha)")
echo "[stage] pinned revision: $REVISION"

echo "[stage] downloading UAS payload (resumable; ~182 GiB)"
nice -n 10 uvx --from "$HUB_SPEC" huggingface-cli download "$DATASET_REPO" \
    --repo-type dataset --revision "$REVISION" --local-dir "$DATA_DIR" --max-workers 16 \
    --include "statistics.csv" "train/imagery/UAS/*" "test/imagery/UAS/*" "train/annotations/UAS/*" "test/annotations/UAS/*"

echo "[stage] verifying local tree against the pinned HF revision"
uvx --from "$HUB_SPEC" python - "$DATASET_REPO" "$REVISION" "$DATA_DIR" <<'PY'
import fnmatch
import sys

from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

repo, revision, data_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
patterns = ["statistics.csv", "train/imagery/UAS/*", "test/imagery/UAS/*", "train/annotations/UAS/*", "test/annotations/UAS/*"]

info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
expected = {}
for sibling in info.siblings:
    if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in patterns):
        expected[sibling.rfilename] = sibling.lfs.size if sibling.lfs is not None else sibling.size

failures = []
total_bytes = 0
for rel_path, size in sorted(expected.items()):
    local = data_dir / rel_path
    if not local.is_file():
        failures.append(f"MISSING  {rel_path}")
    elif size is not None and local.stat().st_size != size:
        failures.append(f"SIZE     {rel_path}  local={local.stat().st_size}  expected={size}")
    else:
        total_bytes += local.stat().st_size

print(f"expected files: {len(expected)}   verified bytes: {total_bytes / 2**30:.1f} GiB")

# statistics.csv drives event labeling and the sensor filter, so require the columns the
# manifest builder reads rather than merely the file's presence.
stats_path = data_dir / "statistics.csv"
if not stats_path.is_file():
    failures.append("MISSING  statistics.csv (event labels and sensor filter depend on it)")
else:
    header = stats_path.read_text(encoding="utf-8").splitlines()[0] if stats_path.stat().st_size else ""
    missing_columns = [column for column in ("Orthomosaic", "Event", "Source") if column not in header]
    if missing_columns:
        failures.append(f"COLUMNS  statistics.csv lacks {missing_columns}; header was: {header!r}")

for line in failures[:20]:
    print(line)
if failures:
    print(f"VERIFICATION FAILED: {len(failures)} problems")
    sys.exit(1)

manifest = data_dir / "STAGING_MANIFEST.txt"
manifest.write_text(
    f"dataset={repo}\nrevision={revision}\nstaged_utc={datetime.now(timezone.utc).isoformat()}\n"
    f"patterns={patterns}\nfiles={len(expected)}\nbytes={total_bytes}\n",
    encoding="utf-8",
)
print(f"manifest written: {manifest}")
PY

echo "[stage] share usage:"
du -sh "$DATA_DIR"
df -h "$DATA_DIR" | tail -n 1
echo "[stage] done"
