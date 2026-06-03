"""I/O utilities for dataset inventory and CRASAR label parsing."""

import json

import geopandas as gpd
import pandas as pd

from pathlib import Path
from shapely.geometry import Polygon


def parse_crasar_json(json_path: str | Path) -> gpd.GeoDataFrame:
    """Parse a CRASAR-U-DROIDs JSON label file into a GeoDataFrame.

    Args:
        json_path: Path to the `.json` annotation file.

    Returns:
        GeoDataFrame with `geometry` and `label` columns in `EPSG:4326`.
        Returns an empty GeoDataFrame when file data is missing or malformed.
    """

    path = Path(json_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: Failed to load JSON {path}: {error}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    polygons: list[Polygon] = []
    labels: list[str | None] = []
    for item in data:
        if not isinstance(item, dict) or "EPSG:4326" not in item:
            continue
        coords = item["EPSG:4326"]
        if not isinstance(coords, list) or len(coords) < 3:
            continue
        label = item.get("label")
        points = [(pt["lon"], pt["lat"]) for pt in coords]
        try:
            polygon = Polygon(points)
            if polygon.is_valid:
                polygons.append(polygon)
                labels.append(label)
            else:
                repaired = polygon.buffer(0)
                if repaired.is_valid and not repaired.is_empty:
                    polygons.append(repaired)
                    labels.append(label)
        except Exception as error:
            print(f"WARNING: Invalid polygon in {path.name}: {error}")
            continue

    if not polygons:
        print(f"WARNING: No valid polygons found in {path.name}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(geometry=polygons, data={"label": labels}, crs="EPSG:4326")


def scan_dataset(data_root: Path | str, splits: list[str] | None = None, sensor_types: list[str] | None = None, include_orphans: bool = False) -> pd.DataFrame:
    """Scan the CRASAR-U-DROIDs folder structure and build an inventory table.

    Args:
        data_root: Root path for local dataset directory.
        splits: Dataset split names. Defaults to `["train", "test"]`.
        sensor_types: Sensor type names. Defaults to `["UAS"]`.
        include_orphans: Include imagery without matching labels.

    Returns:
        DataFrame with paths and metadata for each discovered orthomosaic.
    """

    split_values = ["train", "test"] if splits is None else splits
    root = Path(data_root)
    records: list[dict[str, str | bool | None]] = []

    for split in split_values:
        split_imagery_root = root / split / "imagery"
        if sensor_types is None:
            if not split_imagery_root.exists():
                print(f"WARNING: Imagery directory not found: {split_imagery_root}")
                continue
            sensor_values = sorted(directory.name for directory in split_imagery_root.iterdir() if directory.is_dir())
            if not sensor_values:
                print(f"WARNING: No sensor directories found in {split_imagery_root}")
                continue
        else:
            sensor_values = sensor_types

        for sensor in sensor_values:
            image_dir = root / split / "imagery" / sensor
            label_dir = root / split / "annotations" / sensor / "building_damage_assessment"
            align_dir = root / split / "annotations" / sensor / "building_alignment_adjustments"

            if not image_dir.exists():
                print(f"WARNING: Imagery directory not found: {image_dir}")
                continue

            images = list(image_dir.glob("*.tif"))
            if not images:
                print(f"WARNING: No images found in {image_dir}")

            for image_path in images:
                label_name = f"{image_path.name}.json"
                label_path = label_dir / label_name if label_dir.exists() else None
                align_path = align_dir / label_name if align_dir.exists() else None

                label_exists = label_path.exists() if label_path is not None else False
                align_exists = align_path.exists() if align_path is not None else False
                if not label_exists and not include_orphans:
                    continue

                records.append({
                    "image_path": str(image_path.absolute()),
                    "label_path": str(label_path.absolute()) if label_exists else None,
                    "alignment_path": str(align_path.absolute()) if align_exists else None,
                    "split": split,
                    "sensor": sensor,
                    "image_name": image_path.name,
                    "valid": label_exists
                })

    if not records:
        print("WARNING: No records found. Check data root and structure.")
        return pd.DataFrame(columns=["image_path", "label_path", "alignment_path", "split", "sensor", "image_name", "valid"])
    return pd.DataFrame(records)
