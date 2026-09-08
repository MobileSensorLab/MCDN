"""Cross-source damage-label agreement audit: sUAS vs crewed annotations of the same buildings.

The CRASAR-U-DROIDs corpus annotates overlapping sUAS and crewed-aircraft mosaics
independently, so the same physical building can carry different damage labels in the
two sources. This audit matches buildings across every overlapping (sUAS, crewed)
mosaic pair — the crewed filename embeds its parent sUAS mosaic per the dataset naming
contract ``<uas_name>.geo.tif_<sortie>_RGB.geo.tif`` — and quantifies the label
agreement: 4x4 confusion matrix, exact/adjacent agreement rates, and Cohen's quadratic
kappa between the label sources. The result prices the annotation-disagreement term of
the sUAS-vs-crewed performance gap.

Matching runs in EPSG:4326 on polygon centroids with a local meter conversion:
a crewed building matches the nearest sUAS building within ``--match-radius-m``
(default 4 m, about one small rooftop) when the match is mutually nearest.

Usage::

    uv run python scripts/audit_label_agreement.py --data-dir data --output-json outputs/label_agreement/summary.json
"""

import argparse
import json
import math

import numpy as np
import pandas as pd

from pathlib import Path

from src.data.geometry import VALID_ORDINAL_LABELS
from src.data.io import parse_crasar_json

ORDINAL_ORDER = ["no damage", "minor damage", "major damage", "destroyed"]
ORDINAL_INDEX = {label: idx for idx, label in enumerate(ORDINAL_ORDER)}


def find_overlapping_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Discover (event, sUAS BDA json, crewed BDA json) triples via the filename contract.

    Both splits' annotation trees are searched on both sides: a crewed sortie in the
    train pool can overlap a sUAS mosaic in the test pool and vice versa.

    Returns:
        List of ``(event, uas_json_path, crewed_json_path)`` tuples, one per crewed
        sortie whose parent sUAS mosaic annotation exists locally.
    """

    stats = pd.read_csv(data_dir / "statistics.csv").dropna(subset=["Orthomosaic"]).set_index("Orthomosaic")

    uas_jsons: dict[str, Path] = {}
    for split in ["train", "test"]:
        bda_dir = data_dir / split / "annotations" / "UAS" / "building_damage_assessment"
        for json_path in bda_dir.glob("*.json"):
            uas_jsons[json_path.name.removesuffix(".json")] = json_path

    pairs: list[tuple[str, Path, Path]] = []
    seen_crewed: set[str] = set()
    for split in ["train", "test"]:
        bda_dir = data_dir / split / "annotations" / "CREWED" / "building_damage_assessment"
        for json_path in sorted(bda_dir.glob("*.json")):
            crewed_name = json_path.name.removesuffix(".json")
            # A handful of crewed annotation files ship in both split trees under the
            # same name (imagery withheld upstream); count each sortie once.
            if crewed_name in seen_crewed:
                continue
            parent_name = crewed_name.split(".geo.tif_")[0] + ".geo.tif"
            if parent_name not in uas_jsons:
                continue
            seen_crewed.add(crewed_name)
            event = str(stats.loc[crewed_name, "Event"]) if crewed_name in stats.index else "Unknown"
            pairs.append((event, uas_jsons[parent_name], json_path))
    return pairs


def load_labeled_centroids(json_path: Path) -> pd.DataFrame:
    """Parse one BDA json to a DataFrame of valid-ordinal-label centroids (EPSG:4326)."""

    gdf = parse_crasar_json(json_path)
    if gdf.empty:
        return pd.DataFrame(columns=["lon", "lat", "label"])
    gdf = gdf[gdf["label"].isin(VALID_ORDINAL_LABELS)]
    centroids = gdf.geometry.centroid
    return pd.DataFrame({"lon": centroids.x, "lat": centroids.y, "label": gdf["label"].to_numpy()})


def match_mutual_nearest(uas: pd.DataFrame, crewed: pd.DataFrame, radius_m: float) -> list[tuple[int, int]]:
    """Mutually-nearest centroid matching within a metric radius.

    Distances are computed in a local equirectangular meter frame anchored at the
    crewed pool's mean latitude — adequate at building scale.

    Returns:
        List of ``(uas_row, crewed_row)`` index pairs.
    """

    if uas.empty or crewed.empty:
        return []

    lat0 = math.radians(float(crewed["lat"].mean()))
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2.0 * lat0) + 1.175 * math.cos(4.0 * lat0)
    m_per_deg_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3.0 * lat0)

    uas_xy = np.column_stack([uas["lon"].to_numpy() * m_per_deg_lon, uas["lat"].to_numpy() * m_per_deg_lat])
    crewed_xy = np.column_stack([crewed["lon"].to_numpy() * m_per_deg_lon, crewed["lat"].to_numpy() * m_per_deg_lat])

    # Pairwise distances [num_crewed, num_uas]; corpora are a few thousand buildings
    # per mosaic at most, so the dense matrix is fine.
    diffs = crewed_xy[:, None, :] - uas_xy[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))

    nearest_uas = np.argmin(dists, axis=1)          # per crewed row
    nearest_crewed = np.argmin(dists, axis=0)       # per uas row

    matches: list[tuple[int, int]] = []
    for crewed_idx, uas_idx in enumerate(nearest_uas):
        if dists[crewed_idx, uas_idx] <= radius_m and nearest_crewed[uas_idx] == crewed_idx:
            matches.append((int(uas_idx), int(crewed_idx)))
    return matches


def quadratic_kappa(confusion: np.ndarray) -> float:
    """Cohen's quadratically-weighted kappa from a KxK confusion matrix."""

    num_classes = confusion.shape[0]
    total = confusion.sum()
    if total == 0:
        return float("nan")
    weights = np.array([[(i - j) ** 2 for j in range(num_classes)] for i in range(num_classes)], dtype=np.float64)
    weights /= (num_classes - 1) ** 2
    observed = confusion / total
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / (total * total)
    denom = float(np.sum(weights * expected))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(weights * observed)) / denom


def main() -> None:
    """CLI entry point: run the audit and print/serialize per-event and pooled results."""

    parser = argparse.ArgumentParser(description="sUAS vs crewed damage-label agreement audit.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--match-radius-m", type=float, default=4.0)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/label_agreement/summary.json"))
    args = parser.parse_args()

    pairs = find_overlapping_pairs(args.data_dir)
    print(f"Overlapping (sUAS, crewed) annotation pairs found: {len(pairs)}")

    per_event_conf: dict[str, np.ndarray] = {}
    per_event_matched: dict[str, int] = {}
    per_event_crewed_total: dict[str, int] = {}
    pair_records: list[dict] = []

    for event, uas_path, crewed_path in pairs:
        uas_df = load_labeled_centroids(uas_path)
        crewed_df = load_labeled_centroids(crewed_path)
        matches = match_mutual_nearest(uas=uas_df, crewed=crewed_df, radius_m=args.match_radius_m)

        conf = np.zeros((4, 4), dtype=np.int64)
        for uas_idx, crewed_idx in matches:
            row = ORDINAL_INDEX[uas_df.iloc[uas_idx]["label"]]
            col = ORDINAL_INDEX[crewed_df.iloc[crewed_idx]["label"]]
            conf[row, col] += 1

        per_event_conf[event] = per_event_conf.get(event, np.zeros((4, 4), dtype=np.int64)) + conf
        per_event_matched[event] = per_event_matched.get(event, 0) + len(matches)
        per_event_crewed_total[event] = per_event_crewed_total.get(event, 0) + len(crewed_df)

        pair_records.append({
            "event": event,
            "uas_json": uas_path.name,
            "crewed_json": crewed_path.name,
            "uas_buildings": len(uas_df),
            "crewed_buildings": len(crewed_df),
            "matched": len(matches),
        })
        print(f"  [{event}] {crewed_path.name.removesuffix('.json')}: "
              f"uas={len(uas_df)} crewed={len(crewed_df)} matched={len(matches)}")

    print("\n=== Per-event agreement (rows = sUAS label, cols = crewed label) ===")
    pooled = np.zeros((4, 4), dtype=np.int64)
    event_summaries: dict[str, dict] = {}
    for event in sorted(per_event_conf):
        conf = per_event_conf[event]
        total = int(conf.sum())
        pooled += conf
        if total == 0:
            print(f"\n--- {event}: no matches ---")
            continue
        exact = float(np.trace(conf) / total)
        adjacent = float(sum(conf[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / total)
        kappa = quadratic_kappa(conf)
        print(f"\n--- {event}: matched={total} (crewed pool coverage "
              f"{per_event_matched[event]}/{per_event_crewed_total[event]}) ---")
        header = "            " + "  ".join(f"{label[:9]:>9}" for label in ORDINAL_ORDER)
        print(header)
        for i, label in enumerate(ORDINAL_ORDER):
            print(f"  {label[:9]:>9}  " + "  ".join(f"{conf[i, j]:>9d}" for j in range(4)))
        print(f"  exact agreement={exact:.4f}  within-one={adjacent:.4f}  quadratic kappa={kappa:.4f}")
        event_summaries[event] = {
            "matched": total,
            "crewed_pool": per_event_crewed_total[event],
            "confusion_uas_rows_crewed_cols": conf.tolist(),
            "exact_agreement": exact,
            "within_one_agreement": adjacent,
            "quadratic_kappa": kappa,
        }

    pooled_total = int(pooled.sum())
    pooled_exact = float(np.trace(pooled) / pooled_total) if pooled_total else float("nan")
    pooled_adjacent = float(sum(pooled[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / pooled_total) if pooled_total else float("nan")
    pooled_kappa = quadratic_kappa(pooled)
    print(f"\n=== Pooled over all events: matched={pooled_total} ===")
    print(f"  exact agreement={pooled_exact:.4f}  within-one={pooled_adjacent:.4f}  quadratic kappa={pooled_kappa:.4f}")
    print("  Interpretation: the pooled quadratic kappa is the ceiling any model evaluated across")
    print("  label sources can reach; disagreement here is annotation noise, not model error.")

    payload = {
        "match_radius_m": args.match_radius_m,
        "pairs": pair_records,
        "per_event": event_summaries,
        "pooled": {
            "matched": pooled_total,
            "confusion_uas_rows_crewed_cols": pooled.tolist(),
            "exact_agreement": pooled_exact,
            "within_one_agreement": pooled_adjacent,
            "quadratic_kappa": pooled_kappa,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSummary written: {args.output_json}")


if __name__ == "__main__":
    main()
