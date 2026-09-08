"""Report the event composition of every evaluation split (T-11 / R1-1).

Reviewer 1's primary comment asserts the Spatial_Block_East validation pool is
"dominated by Hurricane Michael" and asks for a table of events and sample counts
per split. This script computes that table from the dataset itself, using the repo's
own split code so the reported composition is exactly what the trainer sees:

    1. Per-mosaic chip inventory via ``CRASARUnitemporalDataset`` extraction.
    2. The default spatial fold reconstructed through ``build_spatial_split_manifest``
       + ``generate_loeo_splits`` (raw-CRS centroid-x sort, 80/20 mosaic cut), with each
       mosaic mapped back to its real event.
    3. True WGS84 centroid longitudes alongside the raw-CRS sort key, to flag any
       divergence between the split's ordering and actual geography.
    4. LOEO fold sizes per event and the deployed-split composite pool.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/cluster/report_split_composition.py \
        --data-dir /network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids
"""

import argparse

import pandas as pd
import rasterio

from collections import Counter
from pathlib import Path
from rasterio.warp import transform_bounds

from src.data.dataset import CRASARUnitemporalDataset
from src.data.sampling import build_spatial_split_manifest, build_valid_manifest, generate_loeo_splits

DEPLOYED_SPLIT_HOLDOUT = ["Hurricane Michael", "Hurricane Idalia", "Mussett Bayou Fire", "Mayfield Tornado"]

CLASS_ORDER = ["no damage", "minor damage", "major damage", "destroyed"]


def per_mosaic_inventory(manifest: pd.DataFrame) -> pd.DataFrame:
    """Count extracted instances per mosaic using the dataset's own extraction logic.

    Returns one row per mosaic with the real event, raw-CRS centroid x (the spatial
    split's sort key), true WGS84 centroid longitude, and per-class instance counts.
    """

    rows = []
    for _, record in manifest.iterrows():
        with rasterio.open(record["image_path"]) as src:
            raw_cx = (src.bounds.left + src.bounds.right) / 2.0
            west, _south, east, _north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            lon = (west + east) / 2.0

        subset = manifest[manifest["image_name"] == record["image_name"]]
        dataset = CRASARUnitemporalDataset(subset, chip_size=512, transform=None, is_train=False)
        counts = Counter(instance["damage_label"] for instance in dataset.instances)
        rows.append({
            "image_name": record["image_name"], "event": record["event"],
            "raw_cx": raw_cx, "lon": lon, "instances": len(dataset.instances),
            **{cls: counts.get(cls, 0) for cls in CLASS_ORDER}
        })
    return pd.DataFrame(rows)


def print_event_table(title: str, inventory: pd.DataFrame, names: set[str]) -> None:
    """Print per-event mosaic and instance counts for the mosaics in ``names``."""

    pool = inventory[inventory["image_name"].isin(names)]
    print(f"\n  {title}: {pool['instances'].sum()} instances / {len(pool)} mosaics")
    grouped = pool.groupby("event").agg(
        mosaics=("image_name", "count"), instances=("instances", "sum"),
        **{cls.replace(" ", "_"): (cls, "sum") for cls in CLASS_ORDER}
    ).sort_values("instances", ascending=False)
    for event, row in grouped.iterrows():
        share = 100.0 * row["instances"] / max(pool["instances"].sum(), 1)
        print(f"    {event:<28} mosaics {row['mosaics']:>2}   instances {row['instances']:>6} ({share:4.1f}%)   "
              f"noD {row['no_damage']:>5}  min {row['minor_damage']:>5}  maj {row['major_damage']:>5}  "
              f"des {row['destroyed']:>5}")


def report_spatial_fold(manifest: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Reconstruct the default spatial fold and report its real-event composition."""

    spatial_manifest = build_spatial_split_manifest(manifest)
    folds = list(generate_loeo_splits(spatial_manifest))
    fold_name, train_df, val_df = folds[0]
    print(f"\n=== Default spatial fold ({fold_name}) — mosaics sorted by raw-CRS centroid x, easternmost 20% held out ===")

    ordered = inventory.sort_values("raw_cx").reset_index(drop=True)
    val_names = set(val_df["image_name"])
    print("\n  West-to-east mosaic order (V = validation block):")
    for _, row in ordered.iterrows():
        marker = "V" if row["image_name"] in val_names else " "
        print(f"    {marker}  raw_cx {row['raw_cx']:>14.1f}   lon {row['lon']:>9.4f}   {row['event']:<28} "
              f"{row['image_name']:<44} instances {row['instances']:>6}")

    # A raw-CRS sort across mixed UTM zones is not guaranteed to match true longitude order.
    lon_order = ordered.sort_values("lon")["image_name"].tolist()
    raw_order = ordered["image_name"].tolist()
    if lon_order != raw_order:
        mismatches = sum(1 for a, b in zip(raw_order, lon_order, strict=True) if a != b)
        print(f"\n  NOTE: raw-CRS ordering diverges from true WGS84 longitude ordering at {mismatches} positions.")

    print_event_table("TRAIN pool", inventory, set(train_df["image_name"]))
    print_event_table("VAL pool", inventory, val_names)


def report_loeo_and_deployed(manifest: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Report per-event LOEO fold sizes and the deployed-split composite pool."""

    print("\n=== LOEO folds (val = entire event) ===")
    total = inventory["instances"].sum()
    for event, group in inventory.groupby("event"):
        event_instances = group["instances"].sum()
        print(f"    {event:<28} val {event_instances:>6} instances ({100.0 * event_instances / total:4.1f}%)   "
              f"train {total - event_instances:>6}")

    print("\n=== Deployed-split composite (T-24) ===")
    val_names = set(manifest[manifest["event"].isin(DEPLOYED_SPLIT_HOLDOUT)]["image_name"])
    train_names = set(manifest["image_name"]) - val_names
    print_event_table("TRAIN pool", inventory, train_names)
    print_event_table("VAL pool", inventory, val_names)


def main() -> None:
    """CLI entry point: build the inventory and print all split compositions."""

    parser = argparse.ArgumentParser(description="Report event composition of every evaluation split.")
    parser.add_argument("--data-dir", type=str, default="/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids")
    parser.add_argument("--sensor-profile", type=str, default="uas_5cm")
    args = parser.parse_args()

    manifest = build_valid_manifest(data_dir=args.data_dir, sensor_profile=args.sensor_profile)
    inventory = per_mosaic_inventory(manifest)
    report_spatial_fold(manifest, inventory)
    report_loeo_and_deployed(manifest, inventory)

    csv_path = Path("outputs") / "split_composition.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.sort_values("raw_cx").to_csv(csv_path, index=False)
    print(f"\nPer-mosaic inventory written to {csv_path}")


if __name__ == "__main__":
    main()
