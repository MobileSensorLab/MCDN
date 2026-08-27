"""Manifest preparation, fold selection, and sampling utilities.

Hosts the public helpers that build valid manifests, assemble spatial-block-friendly
event labels, and select folds from generated splits. These live here (rather than
under ``src/model/trainer``) because multiple post-training scripts reuse the same
pipeline wiring and should not need to reach through trainer internals.
"""

import pandas as pd
import rasterio
import torch

from collections.abc import Generator
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

from src.data.dataset import CRASARUnitemporalDataset, ORDINAL_MAP
from src.data.geometry import validate_integrity
from src.data.io import scan_dataset


DEFAULT_SPATIAL_HOLDOUT_CANONICAL = "spatial_block_east"


def generate_loeo_splits(manifest: pd.DataFrame) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """Generates geographic train/val splits.

    If multiple events exist, uses strict Leave-One-Event-Out (LOEO).
    If only one event exists, falls back to a Spatial Block Split by
    sorting tiles geographically (West to East) to prevent data leakage.
    """
    if "event" not in manifest.columns:
        raise ValueError("Manifest must contain an 'event' column for LOEO splitting.")

    events = manifest["event"].dropna().unique()

    if len(events) <= 1:
        # --- SPATIAL BLOCK SPLIT FALLBACK ---
        # The synthetic event label on the source manifest (e.g. ``"Spatial_Block"``) is
        # intentionally ignored for the emitted fold name so callers always see a
        # canonical ``"Spatial_Block_East"`` identifier rather than the legacy doubled
        # ``"Spatial_Block_Spatial_Block_East"``.

        centroids_x = []
        for _, row in manifest.iterrows():
            try:
                with rasterio.open(row["image_path"]) as src:
                    cx = (src.bounds.left + src.bounds.right) / 2.0
                    centroids_x.append(cx)
            except Exception:
                centroids_x.append(0.0)

        manifest_spatial = manifest.copy()
        manifest_spatial["cx"] = centroids_x
        manifest_spatial = manifest_spatial.sort_values(by="cx").reset_index(drop=True)

        split_idx = int(len(manifest_spatial) * 0.8)
        train_df = manifest_spatial.iloc[:split_idx].drop(columns=["cx"])
        val_df = manifest_spatial.iloc[split_idx:].drop(columns=["cx"])

        yield "Spatial_Block_East", train_df, val_df

    else:
        for holdout_event in events:
            val_df = manifest[manifest["event"] == holdout_event].copy()
            train_df = manifest[manifest["event"] != holdout_event].copy()

            if not train_df.empty and not val_df.empty:
                yield holdout_event, train_df, val_df


def create_weighted_sampler(dataset: CRASARUnitemporalDataset) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler to mitigate class imbalance."""
    if not dataset.instances:
        raise ValueError("Dataset contains no instances to sample.")

    labels = [ORDINAL_MAP[inst["damage_label"]] for inst in dataset.instances]
    label_tensor = torch.tensor(labels, dtype=torch.long)

    class_counts = torch.bincount(label_tensor)
    class_weights = 1.0 / torch.where(class_counts > 0, class_counts, torch.tensor(1.0))
    instance_weights = [class_weights[label].item() for label in labels]

    sampler = WeightedRandomSampler(
        weights=instance_weights,
        num_samples=len(instance_weights),
        replacement=True
    )

    return sampler


def _ensure_event_column(valid_manifest: pd.DataFrame) -> pd.DataFrame:
    """Ensure manifest has an `event` column, deriving one from image name when missing."""

    if "event" in valid_manifest.columns:
        output = valid_manifest.copy()
        missing_events = output["event"].fillna("").astype(str).str.strip() == ""
        if not missing_events.any():
            return output
        output.loc[missing_events, "event"] = output.loc[missing_events, "image_name"].apply(
            lambda name: name.split("_")[0] if "_" in name else "Test_Event"
        )
        return output
    output = valid_manifest.copy()
    output["event"] = output["image_name"].apply(lambda name: name.split("_")[0] if "_" in name else "Test_Event")
    return output


def _assert_one_event_per_image(valid_manifest: pd.DataFrame) -> None:
    """Validate that each image maps to exactly one canonical event label."""

    grouped = valid_manifest.groupby("image_name")["event"].nunique(dropna=False)
    inconsistent_images = grouped[grouped > 1]
    if inconsistent_images.empty:
        return

    sample_names = inconsistent_images.index.tolist()[:5]
    sample_pairs: list[str] = []
    for image_name in sample_names:
        labels = sorted(valid_manifest.loc[valid_manifest["image_name"] == image_name, "event"].astype(str).unique().tolist())
        sample_pairs.append(f"{image_name}: {labels}")
    sample_text = "; ".join(sample_pairs)

    raise ValueError(
        "Inconsistent event mapping detected: each image_name must map to one event. "
        f"Found {len(inconsistent_images)} inconsistent image(s). Examples -> {sample_text}"
    )


def _canonicalize_event_name(event_name: str) -> str:
    """Normalize event naming for robust fold matching."""

    canonical = "_".join(event_name.lower().replace("-", "_").split())
    while "__" in canonical:
        canonical = canonical.replace("__", "_")
    return canonical.strip("_")


def _attach_source_metadata(manifest: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Attach optional source and event metadata from statistics.csv when available."""

    if "image_name" not in manifest.columns:
        return manifest

    stats_path = Path(data_dir) / "statistics.csv"
    if not stats_path.exists():
        return manifest

    try:
        stats_df = pd.read_csv(stats_path)
    except Exception as error:
        print(f"WARNING: Failed to parse source metadata at {stats_path}: {error}")
        return manifest

    if "Orthomosaic" not in stats_df.columns:
        print(f"WARNING: statistics.csv missing 'Orthomosaic' column at {stats_path}; skipping metadata attachment.")
        return manifest

    metadata_df = stats_df.dropna(subset=["Orthomosaic"]).drop_duplicates(subset=["Orthomosaic"]).set_index("Orthomosaic")
    output = manifest.copy()
    if "Source" in metadata_df.columns:
        output["source"] = output["image_name"].map(metadata_df["Source"])
    if "Event" in metadata_df.columns:
        mapped_event = output["image_name"].map(metadata_df["Event"])
        if "event" in output.columns:
            output["event"] = mapped_event.fillna(output["event"])
        else:
            output["event"] = mapped_event
    return output


def _filter_manifest_for_sensor_profile(manifest: pd.DataFrame, sensor_profile: str, data_dir: str) -> pd.DataFrame:
    """Filter manifest rows by configured sensor profile when source metadata is available."""

    if sensor_profile not in {"uas_5cm", "manned_15cm"}:
        raise ValueError(f"Unsupported data.sensor_profile value: {sensor_profile}")

    if "source" not in manifest.columns:
        print(f"WARNING: No source metadata available under {data_dir}; sensor_profile filter is skipped.")
        return manifest

    source_values = manifest["source"].fillna("").astype(str).str.lower()
    if (source_values == "").all():
        print(f"WARNING: Source metadata is empty under {data_dir}; sensor_profile filter is skipped.")
        return manifest

    if sensor_profile == "uas_5cm":
        mask = source_values.str.contains("uas")
    else:
        mask = source_values.str.contains("crewed") | source_values.str.contains("manned")

    filtered_manifest = manifest[mask].copy()
    if filtered_manifest.empty:
        raise ValueError(
            f"No records matched data.sensor_profile='{sensor_profile}' under data root: {data_dir}. "
            "Verify source metadata in statistics.csv and dataset availability."
        )
    return filtered_manifest


def build_spatial_split_manifest(valid_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a manifest suited for spatial-block default splitting.

    Prefers explicit ``spatial_block_*`` labels when encoded in image names. If
    unavailable, forces a single synthetic event label so ``generate_loeo_splits``
    takes its built-in geographic east/west fallback.
    """

    output = valid_manifest.copy()
    normalized_names = output["image_name"].astype(str).str.lower().str.replace("-", "_", regex=False)
    spatial_blocks = normalized_names.str.extract(r"(spatial_block_[a-z0-9]+)")[0]
    missing_mask = spatial_blocks.isna()
    if missing_mask.any():
        output["event"] = "Spatial_Block"
    else:
        output["event"] = spatial_blocks
    return output


def build_valid_manifest(data_dir: str, sensor_profile: str) -> pd.DataFrame:
    """Scan, validate, and prepare a manifest for splitting.

    Emits a single diagnostic line with the resolved count after integrity checks and
    sensor-profile filtering. The trainer and downstream post-processing scripts share
    this helper so every consumer sees the same valid-instance view of the dataset.
    """

    manifest = scan_dataset(Path(data_dir))
    manifest = _attach_source_metadata(manifest=manifest, data_dir=data_dir)
    manifest = _filter_manifest_for_sensor_profile(manifest=manifest, sensor_profile=sensor_profile, data_dir=data_dir)
    required_columns = {"valid", "image_name"}
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(
            "Dataset scan did not return the required manifest columns "
            f"{missing_columns}. Check data root layout and source path: {data_dir}"
        )

    validation_results = validate_integrity(manifest)
    if validation_results.get("corrupt", 0) > 0:
        print(f"WARNING: Found {validation_results['corrupt']} corrupt images during scan.")

    valid_manifest = manifest[manifest["valid"]].copy()
    if valid_manifest.empty:
        raise ValueError(f"No valid records found under data root: {data_dir}")
    valid_manifest = _ensure_event_column(valid_manifest=valid_manifest)
    _assert_one_event_per_image(valid_manifest=valid_manifest)
    return valid_manifest


def select_fold(splits: list[tuple[str, pd.DataFrame, pd.DataFrame]], holdout_event: str | None) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Choose a fold from generated splits.

    Args:
        splits: List of ``(event_name, train_df, val_df)`` tuples.
        holdout_event: Optional explicit holdout event.

    Returns:
        Selected ``(holdout, train_df, val_df, holdout_selection)`` tuple where
        ``holdout_selection`` is ``"explicit"`` or ``"default_spatial"``.

    Raises:
        ValueError: If ``holdout_event`` is set but not present, or no default
            spatial fold is available when ``holdout_event`` is ``None``.
    """

    canonical_lookup: dict[str, list[tuple[str, pd.DataFrame, pd.DataFrame]]] = {}
    for split in splits:
        canonical_name = _canonicalize_event_name(split[0])
        canonical_lookup.setdefault(canonical_name, []).append(split)

    if holdout_event is not None:
        direct_fold = next((split for split in splits if split[0] == holdout_event), None)
        if direct_fold is not None:
            return (*direct_fold, "explicit")

        requested_canonical = _canonicalize_event_name(holdout_event)
        canonical_matches = canonical_lookup.get(requested_canonical, [])
        if len(canonical_matches) == 1:
            return (*canonical_matches[0], "explicit")
        if len(canonical_matches) > 1:
            matched_names = ", ".join(match[0] for match in canonical_matches)
            raise ValueError(
                f"Event '{holdout_event}' matched multiple folds after canonicalization: {matched_names}. "
                "Use an exact holdout event string."
            )
        raise ValueError(f"Event '{holdout_event}' not found in manifest.")

    spatial_matches = [split for split in splits if DEFAULT_SPATIAL_HOLDOUT_CANONICAL in _canonicalize_event_name(split[0])]
    if len(spatial_matches) == 1:
        return (*spatial_matches[0], "default_spatial")
    if len(spatial_matches) > 1:
        matched_names = ", ".join(match[0] for match in spatial_matches)
        raise ValueError(
            "Default spatial holdout selection is ambiguous. "
            f"Found multiple candidates containing '{DEFAULT_SPATIAL_HOLDOUT_CANONICAL}': {matched_names}. "
            "Set data.holdout_event explicitly."
        )

    available_events = ", ".join(split[0] for split in splits)
    raise ValueError(
        "No default spatial holdout event found. Expected an event containing "
        f"'{DEFAULT_SPATIAL_HOLDOUT_CANONICAL}'. Available events: {available_events}. "
        "Set data.holdout_event explicitly."
    )


def build_multi_event_fold(manifest: pd.DataFrame, holdout_events: list[str]) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Build one composite fold holding out several events at once.

    Supports split-replication experiments whose published protocol reserves a fixed
    multi-disaster test pool (e.g. the deployed-baseline split: four test disasters,
    six training disasters) rather than a single LOEO event. Event matching mirrors
    ``select_fold``: exact manifest labels first, then unique canonicalized matches
    (case / hyphen / whitespace insensitive).

    Args:
        manifest: Valid manifest with an ``event`` column (see ``build_valid_manifest``).
        holdout_events: Event names forming the validation pool. Training keeps every
            other event.

    Returns:
        ``(fold_name, train_df, val_df, "explicit_multi")`` where ``fold_name`` joins the
        matched manifest event labels, sorted, with ``+``.

    Raises:
        ValueError: If the manifest lacks an ``event`` column, a requested event is missing
            or ambiguous after canonicalization, or no training events would remain.
    """

    if "event" not in manifest.columns:
        raise ValueError("Manifest must contain an 'event' column for multi-event holdout.")

    available_events = [str(event) for event in manifest["event"].dropna().unique()]
    canonical_lookup: dict[str, list[str]] = {}
    for event_name in available_events:
        canonical_lookup.setdefault(_canonicalize_event_name(event_name), []).append(event_name)

    matched: list[str] = []
    for requested in holdout_events:
        if requested in available_events:
            matched.append(requested)
            continue
        candidates = canonical_lookup.get(_canonicalize_event_name(requested), [])
        if len(candidates) == 1:
            matched.append(candidates[0])
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"Event '{requested}' matched multiple manifest events after canonicalization: "
                f"{', '.join(candidates)}. Use exact holdout event strings."
            )
        raise ValueError(f"Event '{requested}' not found in manifest. Available events: {', '.join(sorted(available_events))}")

    matched_unique = list(dict.fromkeys(matched))
    val_mask = manifest["event"].isin(matched_unique)
    val_df = manifest[val_mask].copy()
    train_df = manifest[~val_mask].copy()
    if train_df.empty:
        raise ValueError("Multi-event holdout would leave no training events. Reduce data.holdout_event.")

    fold_name = "+".join(sorted(matched_unique))
    return fold_name, train_df, val_df, "explicit_multi"
