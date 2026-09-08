"""Check the default spatial fold for physically overlapping train/val mosaics (T-11 / R1-1).

The spatial fold holds out whole mosaics, so chip-level leakage requires two mosaics whose
ground footprints intersect to land on opposite sides of the cut (e.g. repeat passes over
the same neighborhood on different days). This script reconstructs the fold and reports
every train x val pair whose WGS84 bounds intersect, with the intersection area as a
fraction of the smaller mosaic.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/cluster/check_split_overlap.py \
        --data-dir /network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids
"""

import argparse

import rasterio

from rasterio.warp import transform_bounds
from shapely.geometry import box

from src.data.sampling import build_spatial_split_manifest, build_valid_manifest, generate_loeo_splits


def main() -> None:
    """CLI entry point: reconstruct the spatial fold and report train/val bounds overlaps."""

    parser = argparse.ArgumentParser(description="Check spatial-fold train/val mosaics for ground-footprint overlap.")
    parser.add_argument("--data-dir", type=str, default="/network/rit/dgx/dgx_mobilesensorlab/crasar-u-droids")
    parser.add_argument("--sensor-profile", type=str, default="uas_5cm")
    args = parser.parse_args()

    manifest = build_valid_manifest(data_dir=args.data_dir, sensor_profile=args.sensor_profile)
    fold_name, train_df, val_df = next(iter(generate_loeo_splits(build_spatial_split_manifest(manifest))))

    def wgs84_box(image_path: str) -> box:
        with rasterio.open(image_path) as src:
            return box(*transform_bounds(src.crs, "EPSG:4326", *src.bounds))

    train_boxes = {row["image_name"]: wgs84_box(row["image_path"]) for _, row in train_df.iterrows()}
    val_boxes = {row["image_name"]: wgs84_box(row["image_path"]) for _, row in val_df.iterrows()}

    print(f"Fold {fold_name}: {len(train_boxes)} train x {len(val_boxes)} val mosaic pairs")
    overlaps = 0
    for train_name, train_geom in train_boxes.items():
        for val_name, val_geom in val_boxes.items():
            if train_geom.intersects(val_geom):
                overlaps += 1
                smaller = min(train_geom.area, val_geom.area)
                fraction = train_geom.intersection(val_geom).area / smaller
                print(f"  OVERLAP {fraction:6.1%} of smaller mosaic:  train={train_name}  val={val_name}")
    if overlaps == 0:
        print("  No train/val ground-footprint overlaps found.")
    else:
        print(f"  {overlaps} overlapping pair(s) found.")


if __name__ == "__main__":
    main()
