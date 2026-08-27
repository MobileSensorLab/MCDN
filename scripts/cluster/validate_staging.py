"""Post-staging validation: inventory, event labeling, chip read, and GPU AMP forward.

Complements the byte-level checks in ``stage_dataset.sh`` by exercising the repo's own
data path against a staged dataset, which catches gaps that a file-count check cannot.
The motivating example (found 27 Aug): a first staging pass omitted ``statistics.csv``,
so every orthomosaic collapsed to one synthetic event label — byte verification passed
while LOEO splitting was silently impossible.

Checks, in order:
    1. ``scan_dataset`` discovers imagery with matching labels and alignment files.
    2. ``build_valid_manifest`` attaches real event labels (more than one event, and no
       fallback ``Test_Event`` label) and applies the sensor-profile filter.
    3. ``generate_loeo_splits`` produces the expected per-event folds.
    4. A real chip read through ``CRASARUnitemporalDataset`` (windowed rasterio read off
       the staged GeoTIFFs, footprint rasterization, mask channel populated).
    5. On CUDA hosts: an AMP forward pass and fused-AdamW construction, which is the
       T-19 numerics/optimizer smoke test.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/cluster/validate_staging.py --data-dir <path>
"""

import argparse

import pandas as pd
import torch

from pathlib import Path

from src.data.dataset import CRASARUnitemporalDataset
from src.data.io import scan_dataset
from src.data.sampling import build_valid_manifest, generate_loeo_splits
from src.model.mcdn import MaskCenteredDamageNet


# Assigned by _ensure_event_column when no real event metadata is available; its presence
# means statistics.csv was missing or unreadable.
FALLBACK_EVENT_LABEL = "Test_Event"


class StagingValidationError(Exception):
    """Raised when a staged dataset fails a structural or semantic check."""


def validate_inventory(data_dir: str) -> None:
    """Check that imagery, damage labels, and alignment adjustments are all present."""

    inventory = scan_dataset(data_dir)
    if inventory.empty:
        raise StagingValidationError(f"scan_dataset found no records under {data_dir}")

    valid = int(inventory["valid"].sum())
    with_alignment = int(inventory["alignment_path"].notna().sum())
    print(f"[1] scan_dataset: {len(inventory)} records  valid={valid}  with_alignment={with_alignment}")
    if valid != len(inventory):
        raise StagingValidationError(f"{len(inventory) - valid} records lack damage-assessment labels")
    if with_alignment != len(inventory):
        raise StagingValidationError(
            f"{len(inventory) - with_alignment} records lack alignment adjustments (T-6a needs all of them)")


def validate_manifest(data_dir: str, sensor_profile: str) -> pd.DataFrame:
    """Check that real event metadata was attached, and return the manifest."""

    manifest = build_valid_manifest(data_dir=data_dir, sensor_profile=sensor_profile)
    events = manifest["event"].value_counts().to_dict()
    print(f"[2] manifest: {len(manifest)} rows  events={events}")

    if FALLBACK_EVENT_LABEL in events:
        raise StagingValidationError(
            f"event labeling fell back to '{FALLBACK_EVENT_LABEL}' - statistics.csv is missing "
            f"or unreadable under {data_dir}; LOEO splitting would be impossible")
    if len(events) < 2:
        raise StagingValidationError(f"expected multiple events for LOEO splitting, found {list(events)}")
    if "source" not in manifest.columns:
        raise StagingValidationError("no 'source' column; the sensor_profile filter was skipped")
    return manifest


def validate_splits(manifest: pd.DataFrame) -> None:
    """Check that LOEO fold generation yields per-event validation sets."""

    splits = list(generate_loeo_splits(manifest))
    if not splits:
        raise StagingValidationError("generate_loeo_splits produced no folds")
    print(f"[3] LOEO folds: {[(split[0], len(split[2])) for split in splits]}")


def validate_chip_read(manifest: pd.DataFrame) -> dict:
    """Read one real chip end to end and return it."""

    first_event = manifest["event"].iloc[0]
    subset = manifest[manifest["event"] == first_event].head(2)
    dataset = CRASARUnitemporalDataset(subset, chip_size=512, transform=None, is_train=False)
    if len(dataset) == 0:
        raise StagingValidationError(f"no instances built from event '{first_event}'")

    sample = dataset[0]
    mask_pixels = int(sample["image"][3].sum())
    print(f"[4] chip: shape={tuple(sample['image'].shape)} dtype={sample['image'].dtype} "
          f"mask_px={mask_pixels} label={int(sample['label'])} instances={len(dataset)}")
    if mask_pixels == 0:
        raise StagingValidationError("footprint mask channel is empty; polygon rasterization failed")
    return sample


def validate_gpu_forward(sample: dict, backbone: str) -> None:
    """Run an AMP forward pass and construct a fused optimizer (T-19 smoke)."""

    if not torch.cuda.is_available():
        print("[5] CUDA unavailable; skipping AMP forward and fused-optimizer checks")
        return

    print(f"[5] cuda: {torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}")
    model = MaskCenteredDamageNet(backbone_name=backbone, pretrained=False).cuda().eval()
    images = sample["image"].unsqueeze(0).cuda().float().div(255.0)
    context = sample["context"].unsqueeze(0).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        logits = model(images, context)
    print(f"    AMP forward OK: logits={tuple(logits.shape)} dtype={logits.dtype}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    optimizer.zero_grad(set_to_none=True)
    print("    fused AdamW OK")


def main() -> None:
    """CLI entry point for staged-dataset validation."""

    parser = argparse.ArgumentParser(description="Validate a staged CRASAR-U-DROIDs tree against the repo data path.")
    parser.add_argument("--data-dir", type=str, default="/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids")
    parser.add_argument("--sensor-profile", type=str, default="uas_5cm")
    parser.add_argument("--backbone", type=str, default="convnextv2_nano.fcmae_ft_in22k_in1k")
    args = parser.parse_args()

    print(f"Validating staged dataset at {args.data_dir}")
    manifest_path = Path(args.data_dir) / "STAGING_MANIFEST.txt"
    if manifest_path.is_file():
        print(manifest_path.read_text(encoding="utf-8").strip())

    validate_inventory(args.data_dir)
    manifest = validate_manifest(args.data_dir, args.sensor_profile)
    validate_splits(manifest)
    sample = validate_chip_read(manifest)
    validate_gpu_forward(sample, args.backbone)
    print("\nAll staging checks passed.")


if __name__ == "__main__":
    main()
