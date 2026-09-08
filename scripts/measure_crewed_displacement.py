"""Per-building roof displacement between crewed and sUAS mosaics via phase correlation.

The crewed products are conventional orthos (bare-earth DEM rectification), so
buildings exhibit relief displacement that grows with off-nadir angle: roofs shift
away from their ground-plane footprints. The sUAS mosaics (low-altitude, high-overlap)
are near-true-ortho and carry footprint alignment adjustments, so sUAS pixels define
the reference position for each footprint. This script measures, for every
co-annotated building in the overlapping (sUAS, crewed) mosaic pairs, the local
image-to-image displacement between the two sources:

- read the same ground window from both mosaics on a common metric grid (the sUAS
  side is box-decimated exactly like the synthetic-GSD chain);
- Hann-window the luminance and phase-correlate (subpixel);
- record the displacement vector in meters, plus correlation response and coverage.

Per mosaic, the mean displacement vector estimates the global georegistration bias,
while the residual dispersion after removing the mean estimates the spatially varying
relief-displacement (lean) field — pure georeferencing error is spatially uniform,
off-nadir lean is not. The per-building table (JSON) keys on centroid lon/lat so it
can be joined against transfer-evaluation predictions and the T-6 mask-translation
sensitivity curves.

Usage::

    uv run python scripts/measure_crewed_displacement.py --data-dir data --output-dir outputs/crewed_displacement
"""

import argparse
import json
import math
import time

import cv2
import rasterio

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from scripts.audit_label_agreement import find_overlapping_pairs, load_labeled_centroids, match_mutual_nearest

# Common correlation grid: 25.6 m ground window at 20 cm/px. Wide enough that the
# dominant structure plus immediate context drives the correlation peak; fine enough
# that subpixel peaks resolve fractions of a meter.
GROUND_WINDOW_M = 25.6
GRID_PX = 128
GRID_GSD_M = GROUND_WINDOW_M / GRID_PX

# Quality gates: minimum non-empty pixel fraction per window and minimum phase
# correlation response for a confident measurement.
MIN_COVERAGE = 0.70
MIN_RESPONSE = 0.03


def ground_gsds(src: rasterio.DatasetReader) -> tuple[float, float]:
    """Per-axis ground sample distances in meters, handling geographic-CRS rasters."""

    step_x, step_y = abs(src.transform.a), abs(src.transform.e)
    if src.crs is None or not src.crs.is_geographic:
        return step_x, step_y
    lat = math.radians((src.bounds.bottom + src.bounds.top) / 2.0)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2.0 * lat) + 1.175 * math.cos(4.0 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3.0 * lat)
    return step_x * m_per_deg_lon, step_y * m_per_deg_lat


def read_ground_window(src: rasterio.DatasetReader, lon: float, lat: float) -> tuple[np.ndarray, float]:
    """Read a ``GROUND_WINDOW_M`` square window centered on (lon, lat) onto the common grid.

    Native pixels are read boundless and resampled with cv2 (area-average when
    shrinking, bilinear when enlarging), matching the project's degradation semantics.

    Returns:
        Tuple of (luminance array [GRID_PX, GRID_PX] in [0, 1], coverage fraction).
    """

    xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
    col_c, row_c = ~src.transform * (xs[0], ys[0])
    gsd_x, gsd_y = ground_gsds(src)
    half_w = GROUND_WINDOW_M / 2.0 / gsd_x
    half_h = GROUND_WINDOW_M / 2.0 / gsd_y
    window = Window(col_off=round(col_c - half_w), row_off=round(row_c - half_h),
                    width=max(4, round(2 * half_w)), height=max(4, round(2 * half_h)))

    rgb = src.read(window=window, indexes=(1, 2, 3), boundless=True, fill_value=0)
    coverage = float(np.mean(np.any(rgb > 0, axis=0)))

    hwc = np.moveaxis(rgb, 0, -1)
    interp = cv2.INTER_AREA if hwc.shape[0] > GRID_PX else cv2.INTER_LINEAR
    resized = cv2.resize(hwc, (GRID_PX, GRID_PX), interpolation=interp).astype(np.float32) / 255.0

    # Rec.601 luminance for a single correlation channel.
    luminance = 0.299 * resized[:, :, 0] + 0.587 * resized[:, :, 1] + 0.114 * resized[:, :, 2]
    return luminance, coverage


def measure_pair_displacements(uas_image: Path, crewed_image: Path, matched: pd.DataFrame) -> list[dict]:
    """Phase-correlate every matched building's windows between one mosaic pair.

    Args:
        uas_image: Parent sUAS mosaic path.
        crewed_image: Overlapping crewed mosaic path.
        matched: DataFrame with ``lon``, ``lat``, ``uas_label``, ``crewed_label`` rows.

    Returns:
        One record per building: displacement vector (meters; the shift of the crewed
        window relative to the sUAS window in the ground frame, +x east, +y south),
        correlation response, and coverage diagnostics.
    """

    hann = cv2.createHanningWindow((GRID_PX, GRID_PX), cv2.CV_32F)
    records: list[dict] = []
    with rasterio.open(uas_image) as uas_src, rasterio.open(crewed_image) as crewed_src:
        for _, row in matched.iterrows():
            uas_win, uas_cov = read_ground_window(src=uas_src, lon=row["lon"], lat=row["lat"])
            crewed_win, crewed_cov = read_ground_window(src=crewed_src, lon=row["lon"], lat=row["lat"])
            if uas_cov < MIN_COVERAGE or crewed_cov < MIN_COVERAGE:
                continue

            # phaseCorrelate(src1, src2) returns the shift of src2 relative to src1;
            # with src1 = sUAS reference, the vector points from the building's sUAS
            # position to its apparent crewed position on the common ground grid.
            (dx_px, dy_px), response = cv2.phaseCorrelate(uas_win, crewed_win, hann)
            records.append({
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "uas_label": row["uas_label"],
                "crewed_label": row["crewed_label"],
                "dx_m": float(dx_px * GRID_GSD_M),
                "dy_m": float(dy_px * GRID_GSD_M),
                "offset_m": float(math.hypot(dx_px, dy_px) * GRID_GSD_M),
                "response": float(response),
                "uas_coverage": uas_cov,
                "crewed_coverage": crewed_cov,
            })
    return records


def imagery_path_for_annotation(json_path: Path) -> Path | None:
    """Map a BDA annotation path to its imagery file, searching both split trees."""

    name = json_path.name.removesuffix(".json")
    sensor = json_path.parent.parent.name
    # .../<data_root>/<split>/annotations/<sensor>/building_damage_assessment/<name>.json
    data_root = json_path.parents[4]
    for split in ["train", "test"]:
        candidate = data_root / split / "imagery" / sensor / name
        if candidate.exists():
            return candidate
    return None


def summarize_mosaic(records: list[dict]) -> dict:
    """Mean-vector / residual decomposition of one mosaic's displacement field."""

    confident = [r for r in records if r["response"] >= MIN_RESPONSE]
    if not confident:
        return {"n": 0}
    dx = np.array([r["dx_m"] for r in confident])
    dy = np.array([r["dy_m"] for r in confident])
    mags = np.hypot(dx, dy)
    mean_dx, mean_dy = float(dx.mean()), float(dy.mean())
    residual = np.hypot(dx - mean_dx, dy - mean_dy)
    return {
        "n": len(confident),
        "n_low_response": len(records) - len(confident),
        "median_offset_m": float(np.median(mags)),
        "p90_offset_m": float(np.percentile(mags, 90)),
        "mean_vector_m": [mean_dx, mean_dy],
        "mean_vector_mag_m": float(math.hypot(mean_dx, mean_dy)),
        "median_residual_m": float(np.median(residual)),
        "p90_residual_m": float(np.percentile(residual, 90)),
    }


def plot_fields(per_mosaic: dict[str, list[dict]], output_path: Path) -> None:
    """Quiver plots of the displacement fields plus pooled magnitude histograms."""

    names = sorted(per_mosaic)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    flat = axes.ravel()
    for ax, name in zip(flat[:len(names)], names, strict=False):
        confident = [r for r in per_mosaic[name] if r["response"] >= MIN_RESPONSE]
        if not confident:
            ax.set_title(f"{name}\n(no confident measurements)")
            continue
        lon = np.array([r["lon"] for r in confident])
        lat = np.array([r["lat"] for r in confident])
        dx = np.array([r["dx_m"] for r in confident])
        dy = np.array([r["dy_m"] for r in confident])
        mags = np.hypot(dx, dy)
        # -dy: pixel-frame +y is south; plot in map frame (+y north).
        q = ax.quiver(lon, lat, dx, -dy, mags, cmap="plasma", scale=60, scale_units="width", width=0.003)
        ax.set_title(f"{name.split('.geo.tif_')[0][:28]}...{name.split('_')[-3]}\n"
                     f"median {np.median(mags):.2f} m, p90 {np.percentile(mags, 90):.2f} m", fontsize=9)
        ax.tick_params(labelsize=6)
        fig.colorbar(q, ax=ax, shrink=0.8)

    pooled_ax = flat[len(names)]
    for name in names:
        confident = [r for r in per_mosaic[name] if r["response"] >= MIN_RESPONSE]
        if confident:
            pooled_ax.hist([r["offset_m"] for r in confident], bins=40, range=(0, 8),
                           histtype="step", label=name.split(".geo.tif_")[0][:20])
    pooled_ax.set_xlabel("|displacement| (m)")
    pooled_ax.set_ylabel("buildings")
    pooled_ax.legend(fontsize=6)
    pooled_ax.set_title("displacement magnitude by mosaic")

    for ax in flat[len(names) + 1:]:
        ax.axis("off")

    fig.suptitle("Crewed-vs-sUAS per-building displacement fields (phase correlation, 25.6 m windows, 20 cm grid)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    print(f"figure written: {output_path}")


def main() -> None:
    """CLI entry point: measure all overlapping pairs and write table, summary, figure."""

    parser = argparse.ArgumentParser(description="Crewed-vs-sUAS building displacement measurement.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--match-radius-m", type=float, default=4.0)
    parser.add_argument("--events", type=str, nargs="+", default=["Hurricane Michael", "Hurricane Idalia"],
                        help="Restrict to these events (default: the crewed transfer-eval pool).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crewed_displacement"))
    args = parser.parse_args()

    pairs = [(event, uas_json, crewed_json) for event, uas_json, crewed_json in find_overlapping_pairs(args.data_dir)
             if event in set(args.events)]
    print(f"pairs in scope: {len(pairs)}")

    per_mosaic: dict[str, list[dict]] = {}
    t0 = time.time()
    for event, uas_json, crewed_json in pairs:
        uas_image = imagery_path_for_annotation(uas_json)
        crewed_image = imagery_path_for_annotation(crewed_json)
        if uas_image is None or crewed_image is None:
            print(f"  SKIP {crewed_json.name}: imagery missing (uas={uas_image}, crewed={crewed_image})")
            continue

        uas_df = load_labeled_centroids(uas_json)
        crewed_df = load_labeled_centroids(crewed_json)
        matches = match_mutual_nearest(uas=uas_df, crewed=crewed_df, radius_m=args.match_radius_m)
        matched = pd.DataFrame([{
            "lon": uas_df.iloc[u]["lon"], "lat": uas_df.iloc[u]["lat"],
            "uas_label": uas_df.iloc[u]["label"], "crewed_label": crewed_df.iloc[c]["label"],
        } for u, c in matches])

        records = measure_pair_displacements(uas_image=uas_image, crewed_image=crewed_image, matched=matched)
        crewed_name = crewed_json.name.removesuffix(".json")
        per_mosaic[crewed_name] = records
        summary = summarize_mosaic(records)
        print(f"  [{event}] {crewed_name}: measured {summary.get('n', 0)}/{len(matched)} "
              f"median={summary.get('median_offset_m', float('nan')):.2f} m "
              f"p90={summary.get('p90_offset_m', float('nan')):.2f} m "
              f"global-bias={summary.get('mean_vector_mag_m', float('nan')):.2f} m "
              f"median-residual={summary.get('median_residual_m', float('nan')):.2f} m "
              f"({time.time() - t0:.0f}s elapsed)")

    summaries = {name: summarize_mosaic(records) for name, records in per_mosaic.items()}
    all_confident = [r for records in per_mosaic.values() for r in records if r["response"] >= MIN_RESPONSE]
    pooled_mags = np.array([r["offset_m"] for r in all_confident])
    print(f"\npooled: n={len(all_confident)}  median={np.median(pooled_mags):.2f} m  "
          f"p90={np.percentile(pooled_mags, 90):.2f} m  "
          f"fraction > 1.5 m = {float((pooled_mags > 1.5).mean()):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_building.json").write_text(json.dumps(per_mosaic, indent=1), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "grid_gsd_m": GRID_GSD_M, "ground_window_m": GROUND_WINDOW_M,
        "min_response": MIN_RESPONSE, "per_mosaic": summaries,
        "pooled": {
            "n": len(all_confident),
            "median_offset_m": float(np.median(pooled_mags)),
            "p90_offset_m": float(np.percentile(pooled_mags, 90)),
            "fraction_gt_1p5m": float((pooled_mags > 1.5).mean()),
        },
    }, indent=2), encoding="utf-8")
    print(f"tables written: {args.output_dir}")

    plot_fields(per_mosaic=per_mosaic, output_path=args.output_dir / "displacement_fields.png")


if __name__ == "__main__":
    main()
