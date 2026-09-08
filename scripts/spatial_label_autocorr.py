"""Spatial autocorrelation of sUAS damage labels from annotation geometry alone.

Quantifies whether damage labels of nearby buildings meaningfully relate to each other,
as a de-risking study for wide-context (neighborhood-scale) model inputs. Two tiers:

Tier 1 (is there structure?): global Moran's I of the ordinal damage label per
orthomosaic, using row-standardized k-nearest-neighbor weights on building centroids
with a within-mosaic permutation test, aggregated per event.

Tier 2 (at what distance?): per-event spatial correlograms of within-mosaic building
pairs -- Moran-style pair correlation and same-label agreement versus centroid
distance -- yielding the decorrelation range in meters that informs chip-window sizing.

Runs on annotation JSONs only; imagery is never opened. Building pairs are always formed
within a single mosaic, never across mosaics, so repeated sorties of one area cannot pair
with themselves. Centroid geometry tolerates the known meter-scale footprint
misregistration because inter-building spacing is tens of meters.

Outputs under ``outputs/spatial_autocorr/``: ``moran_per_mosaic.csv``,
``moran_per_event.csv``, ``correlogram.csv``, ``correlogram.png``, plus a printed summary.

Usage::

    uv run python scripts/spatial_label_autocorr.py --data-dir data
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from src.data.geometry import filter_polygons_valid_labels
from src.data.io import parse_crasar_json, scan_dataset

ORDINAL_ORDER = ["no damage", "minor damage", "major damage", "destroyed"]
ORDINAL_INDEX = {label: idx for idx, label in enumerate(ORDINAL_ORDER)}


def load_mosaic_points(label_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load building centroids (meters) and ordinal labels for one mosaic.

    Args:
        label_path: Path to a CRASAR building-damage-assessment JSON.

    Returns:
        Tuple of (coords [n, 2] in a local UTM projection, ordinal labels [n]),
        or None when the mosaic has no usable ordinal-labeled polygons.
    """

    gdf = parse_crasar_json(label_path)
    if gdf.empty:
        return None
    gdf = filter_polygons_valid_labels(gdf)
    if gdf.empty:
        return None

    utm = gdf.to_crs(gdf.estimate_utm_crs())
    centroids = utm.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    values = utm["label"].map(ORDINAL_INDEX).to_numpy(dtype=np.float64)
    return coords, values


def morans_i(coords: np.ndarray, values: np.ndarray, k: int, permutations: int, rng: np.random.Generator) -> tuple[float, float]:
    """Global Moran's I with row-standardized kNN weights and a permutation pseudo p-value.

    Args:
        coords: Building centroids in meters, shape [n, 2].
        values: Ordinal damage labels, shape [n].
        k: Number of nearest neighbors defining the spatial weights.
        permutations: Number of label permutations for the null distribution.
        rng: Random generator for the permutation test.

    Returns:
        Tuple of (observed I, one-sided pseudo p-value for positive autocorrelation).
    """

    n = len(values)
    neighbors_k = min(k, n - 1)
    _, neighbor_idx = cKDTree(coords).query(coords, k=neighbors_k + 1)
    rows = np.repeat(np.arange(n), neighbors_k)
    cols = neighbor_idx[:, 1:].ravel()
    weights = csr_matrix((np.full(rows.shape, 1.0 / neighbors_k), (rows, cols)), shape=(n, n))

    z = values - values.mean()
    denom = float(z @ z)
    observed = float(z @ (weights @ z)) / denom

    exceed = 0
    for _ in range(permutations):
        z_perm = rng.permutation(z)
        if float(z_perm @ (weights @ z_perm)) / denom >= observed:
            exceed += 1
    pseudo_p = (1 + exceed) / (permutations + 1)
    return observed, pseudo_p


def mosaic_pairs(coords: np.ndarray, values: np.ndarray, max_range_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate within-mosaic building pairs for the correlogram.

    Args:
        coords: Building centroids in meters, shape [n, 2].
        values: Ordinal damage labels, shape [n].
        max_range_m: Maximum pair separation to enumerate.

    Returns:
        Tuple of per-pair arrays (distance, product of standardized labels, same-label
        flag, expected same-label probability under within-mosaic permutation).
    """

    pairs = cKDTree(coords).query_pairs(r=max_range_m, output_type="ndarray")
    if len(pairs) == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty

    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    z = (values - values.mean()) / values.std()
    products = z[pairs[:, 0]] * z[pairs[:, 1]]
    same = (values[pairs[:, 0]] == values[pairs[:, 1]]).astype(np.float64)

    # Expected agreement under random label placement within this mosaic: sum of squared
    # class frequencies. Repeated per pair so binning can weight by pair counts exactly.
    _, counts = np.unique(values, return_counts=True)
    frequencies = counts / counts.sum()
    expected = np.full(distances.shape, float(np.sum(frequencies**2)))
    return distances, products, same, expected


def bin_correlogram(event: str, distances: np.ndarray, products: np.ndarray, same: np.ndarray,
                    expected: np.ndarray, bin_edges: np.ndarray) -> pd.DataFrame:
    """Bin pooled pair statistics into a per-event correlogram table.

    Args:
        event: Event name for the output rows.
        distances: Pair separations in meters.
        products: Products of within-mosaic standardized labels.
        same: Same-label indicator per pair.
        expected: Expected same-label probability per pair.
        bin_edges: Distance bin edges in meters.

    Returns:
        DataFrame with one row per non-empty bin: pair correlation, observed and
        expected agreement, and an approximate 95% null envelope for the correlation.
    """

    bin_index = np.digitize(distances, bin_edges) - 1
    records: list[dict[str, float | str]] = []
    for bin_id in range(len(bin_edges) - 1):
        mask = bin_index == bin_id
        n_pairs = int(mask.sum())
        if n_pairs == 0:
            continue
        records.append({
            "event": event,
            "bin_low_m": float(bin_edges[bin_id]),
            "bin_high_m": float(bin_edges[bin_id + 1]),
            "bin_center_m": float((bin_edges[bin_id] + bin_edges[bin_id + 1]) / 2),
            "n_pairs": n_pairs,
            "pair_correlation": float(products[mask].mean()),
            "null_envelope_95": float(1.96 / np.sqrt(n_pairs)),
            "agreement": float(same[mask].mean()),
            "expected_agreement": float(expected[mask].mean())
        })
    return pd.DataFrame(records)


def first_crossing(correlogram: pd.DataFrame, threshold: float) -> str:
    """Report the first bin center where pair correlation falls below a threshold.

    Args:
        correlogram: Single-event correlogram table sorted by distance.
        threshold: Correlation threshold defining decorrelation.

    Returns:
        Human-readable range string in meters, or a note when never/immediately crossed.
    """

    below = correlogram["pair_correlation"] < threshold
    if bool(below.iloc[0]):
        return f"<{correlogram['bin_high_m'].iloc[0]:.0f}"
    if not bool(below.any()):
        return f">{correlogram['bin_high_m'].iloc[-1]:.0f}"
    return f"~{correlogram.loc[below.idxmax(), 'bin_center_m']:.0f}"


def main() -> None:
    """Run the two-tier spatial-autocorrelation analysis and write outputs."""

    parser = argparse.ArgumentParser(description="Spatial autocorrelation of sUAS damage labels.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial_autocorr"))
    parser.add_argument("--k", type=int, default=8, help="Neighbors for Moran's I weights.")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--min-buildings", type=int, default=30, help="Skip mosaics below this building count.")
    parser.add_argument("--max-range-m", type=float, default=250.0)
    parser.add_argument("--bin-width-m", type=float, default=10.0)
    args = parser.parse_args()

    plt.switch_backend("Agg")
    rng = np.random.default_rng(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Inventory the sUAS annotations and attach event names from statistics.csv.
    manifest = scan_dataset(args.data_dir, splits=["train", "test"], sensor_types=["UAS"])
    stats = pd.read_csv(args.data_dir / "statistics.csv").drop_duplicates(subset=["Orthomosaic"]).set_index("Orthomosaic")
    manifest["event"] = manifest["image_name"].map(stats["Event"])
    if manifest["event"].isna().any():
        missing = manifest.loc[manifest["event"].isna(), "image_name"].tolist()
        print(f"WARNING: {len(missing)} mosaics missing event metadata, excluded: {missing}")
        manifest = manifest.dropna(subset=["event"])

    moran_rows: list[dict[str, float | int | str]] = []
    pair_pool: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    skipped_small, skipped_degenerate = 0, 0

    for record in manifest.itertuples():
        points = load_mosaic_points(Path(record.label_path))
        if points is None:
            continue
        coords, values = points
        if len(values) < args.min_buildings:
            skipped_small += 1
            continue
        if len(np.unique(values)) < 2:
            skipped_degenerate += 1
            continue

        observed_i, pseudo_p = morans_i(coords, values, k=args.k, permutations=args.permutations, rng=rng)
        moran_rows.append({
            "event": record.event, "split": record.split, "image_name": record.image_name,
            "n_buildings": len(values), "morans_i": observed_i, "pseudo_p": pseudo_p
        })
        pair_pool.setdefault(str(record.event), []).append(mosaic_pairs(coords, values, max_range_m=args.max_range_m))

    moran_df = pd.DataFrame(moran_rows).sort_values(["event", "image_name"])
    moran_df.to_csv(args.output_dir / "moran_per_mosaic.csv", index=False)

    # Tier 1 aggregation: building-weighted Moran's I and significance share per event.
    event_summary = moran_df.groupby("event").apply(
        lambda group: pd.Series({
            "mosaics": len(group),
            "buildings": int(group["n_buildings"].sum()),
            "weighted_morans_i": float(np.average(group["morans_i"], weights=group["n_buildings"])),
            "min_i": float(group["morans_i"].min()),
            "max_i": float(group["morans_i"].max()),
            "share_p_le_001": float((group["pseudo_p"] <= 0.001).mean())
        }), include_groups=False
    ).sort_values("buildings", ascending=False)
    event_summary.to_csv(args.output_dir / "moran_per_event.csv")

    # Tier 2: pool pairs per event and bin into correlograms.
    bin_edges = np.arange(0.0, args.max_range_m + args.bin_width_m, args.bin_width_m)
    correlograms: list[pd.DataFrame] = []
    for event, chunks in pair_pool.items():
        distances = np.concatenate([chunk[0] for chunk in chunks])
        products = np.concatenate([chunk[1] for chunk in chunks])
        same = np.concatenate([chunk[2] for chunk in chunks])
        expected = np.concatenate([chunk[3] for chunk in chunks])
        if len(distances) == 0:
            continue
        correlograms.append(bin_correlogram(event, distances, products, same, expected, bin_edges=bin_edges))
    correlogram_df = pd.concat(correlograms, ignore_index=True)
    correlogram_df.to_csv(args.output_dir / "correlogram.csv", index=False)

    # Figure: pair correlation and excess agreement versus distance, one line per event.
    events_ordered = [event for event in event_summary.index if event in set(correlogram_df["event"])]
    fig, (ax_corr, ax_agree) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for event in events_ordered:
        subset = correlogram_df[correlogram_df["event"] == event].sort_values("bin_center_m")
        ax_corr.plot(subset["bin_center_m"], subset["pair_correlation"], marker="o", markersize=3, label=event)
        ax_agree.plot(subset["bin_center_m"], subset["agreement"] - subset["expected_agreement"],
                      marker="o", markersize=3, label=event)
    ax_corr.axhline(0.0, color="gray", linewidth=0.8)
    ax_corr.set_ylabel("Pair correlation (standardized labels)")
    ax_corr.set_title("Spatial correlogram of sUAS damage labels (within-mosaic pairs)")
    ax_agree.axhline(0.0, color="gray", linewidth=0.8)
    ax_agree.set_ylabel("Agreement minus permutation expectation")
    ax_agree.set_xlabel("Centroid separation (m)")
    ax_agree.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "correlogram.png", dpi=150)

    # Printed summary: Tier 1 per event plus Tier 2 decorrelation ranges.
    print(f"\nMosaics analyzed: {len(moran_df)} (skipped {skipped_small} small, {skipped_degenerate} single-label)")
    print(f"\n{'event':<28}{'mosaics':>8}{'bldgs':>8}{'Moran I':>9}{'p<=.001':>9}{'r(0.1)':>8}{'r(0.05)':>9}")
    for event, row in event_summary.iterrows():
        event_correlogram = correlogram_df[correlogram_df["event"] == event].sort_values("bin_center_m")
        range_10 = first_crossing(event_correlogram, threshold=0.10) if not event_correlogram.empty else "n/a"
        range_05 = first_crossing(event_correlogram, threshold=0.05) if not event_correlogram.empty else "n/a"
        print(f"{event!s:<28}{int(row['mosaics']):>8}{int(row['buildings']):>8}"
              f"{row['weighted_morans_i']:>9.3f}{row['share_p_le_001']:>9.0%}{range_10:>8}{range_05:>9}")
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
