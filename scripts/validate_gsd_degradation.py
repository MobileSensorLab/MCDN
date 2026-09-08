"""Validate SyntheticGsdDegradation against real crewed-aircraft overflights.

CRASAR-U-DROIDs ships crewed-aircraft mosaics clipped to the footprints of specific
sUAS orthomosaics (filename contract ``<uas>.geo.tif_<sortie>_RGB.geo.tif``), giving
precisely overlapping same-scene imagery at ~3 cm and ~15 cm GSD. This script samples
co-located ground tiles from such pairs and compares radially averaged power spectral
densities (PSD) across four conditions:

- ``uas_native``: the sUAS chip as the pipeline sees it (upper spectral envelope).
- ``sampling_only``: ``SyntheticGsdDegradation(factor=f)`` — detector-aperture-only.
- ``mtf_matched``:   ``SyntheticGsdDegradation(factor=f, mtf_at_nyquist=0.3)``.
- ``crewed_real``:   the actual crewed sensor over the same ground.

The degradation factor per pair is ``crewed_gsd / uas_gsd`` measured from the
geotransforms, so the test probes the physical model itself rather than one hardcoded
factor. Spectra are computed per image on its own pixel grid in physical frequency
units (cycles/meter; PSD scaled by GSD^2) — no cross-grid resampling, whose
interpolation MTF would bias the comparison. Radial averaging makes the statistics
invariant to the small co-registration offsets expected between independently
orthorectified products. Crewed spectra are gain-aligned to the sUAS in a low band
(0.2-0.8 cycles/m) where both sensors must agree on scene layout, so residual
differences near the crewed Nyquist reflect resolution, not radiometry.

Success criterion: the ``mtf_matched`` PSD tracks ``crewed_real`` up to the crewed
Nyquist (and the empirical transfer function sqrt(PSD/PSD_native) of the synthetic
arm brackets the crewed sensor's), while ``sampling_only`` over-preserves high
frequencies — quantifying how optimistic the launched arm is.

Usage (from the repo root):

    uv run python scripts/validate_gsd_degradation.py --data-dir data
"""

import argparse
import json
import math

import matplotlib
import numpy as np
import rasterio

from dataclasses import dataclass
from pathlib import Path
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from src.data.transform import SyntheticGsdDegradation, gaussian_sigma_for_mtf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # backend must be set before pyplot import

# Default validation pairs: crewed sorties at true ~15 cm GSD with solid building
# content. Michael/Ian/Harvey crewed products are 25-30 cm and excluded from the
# default set. Paths are relative to --data-dir.
DEFAULT_PAIRS = [
    ("train/imagery/UAS/20210902-LA-DIV-01.geo.tif", "train/imagery/CREWED/20210902-LA-DIV-01.geo.tif_20210831a_RGB.geo.tif"),
    ("train/imagery/UAS/20210831-LA-DIV-01.geo.tif", "train/imagery/CREWED/20210831-LA-DIV-01.geo.tif_20210831a_RGB.geo.tif"),
    ("train/imagery/UAS/20210901-Cocodrie-3.geo.tif", "train/imagery/CREWED/20210901-Cocodrie-3.geo.tif_20210831a_RGB.geo.tif"),
    ("train/imagery/UAS/0827-B-02.geo.tif", "train/imagery/CREWED/0827-B-02.geo.tif_20200829b_RGB.geo.tif")
]

# Low-frequency band (cycles/m) used to gain-align crewed spectra to the sUAS.
GAIN_BAND = (0.2, 0.8)

# Resolution-sensitive report band, as fractions of the crewed Nyquist frequency.
REPORT_BAND_FRACTIONS = (0.5, 1.0)


class GsdValidationError(Exception):
    """Raised when a validation pair cannot be processed."""


@dataclass(frozen=True)
class RadialPsd:
    """Radially averaged power spectral density on a physical frequency axis.

    Args:
        freq: Bin-center spatial frequencies in cycles/meter.
        psd: Mean power spectral density per bin (variance * m^2 units).
    """

    freq: np.ndarray
    psd: np.ndarray


def luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """Rec.601 luminance in [0, 1] from an HWC uint8 RGB array."""

    rgb = rgb_u8.astype(np.float64) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ground_gsds(src: rasterio.DatasetReader) -> tuple[float, float]:
    """Per-axis ground sample distances in meters, handling geographic-CRS rasters.

    The crewed NOAA products are gridded in degrees (EPSG:4326-style), where the
    geotransform step is not meters and ground pixels are anisotropic by the
    cos(latitude) factor. Projected rasters return their native step directly.

    Returns:
        Tuple ``(gsd_x_m, gsd_y_m)``.
    """

    step_x, step_y = abs(src.transform.a), abs(src.transform.e)
    if not src.crs.is_geographic:
        return step_x, step_y
    lat = math.radians((src.bounds.bottom + src.bounds.top) / 2.0)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2.0 * lat) + 1.175 * math.cos(4.0 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3.0 * lat)
    return step_x * m_per_deg_lon, step_y * m_per_deg_lat


def radial_psd(image_01: np.ndarray, gsd_x: float, gsd_y: float, bin_width: float = 0.05) -> RadialPsd:
    """Radially averaged, Hann-windowed PSD of a single-channel image.

    The DC component is removed and the window power compensated, so magnitudes are
    comparable across images sampled on different grids over the same scene class.
    Anisotropic ground pixels are handled by building the frequency grid per axis.

    Args:
        image_01: 2D luminance array in [0, 1].
        gsd_x: Ground sample distance along image columns, meters.
        gsd_y: Ground sample distance along image rows, meters.
        bin_width: Radial bin width in cycles/meter.

    Returns:
        RadialPsd on bin centers from 0 up to this grid's fully sampled radius
        (the smaller of the two per-axis Nyquist frequencies).
    """

    field = image_01 - float(image_01.mean())
    h, w = field.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    field = field * window

    spectrum = np.fft.fftshift(np.fft.fft2(field))
    # Physical PSD scaling: pixel area / (N*M), further corrected for the window power loss.
    power = (np.abs(spectrum) ** 2) * (gsd_x * gsd_y) / (h * w * float((window**2).mean()))

    fy = np.fft.fftshift(np.fft.fftfreq(h, d=gsd_y))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=gsd_x))
    radius = np.hypot(fy[:, None], fx[None, :])

    nyquist = min(1.0 / (2.0 * gsd_x), 1.0 / (2.0 * gsd_y))
    edges = np.arange(0.0, nyquist + bin_width, bin_width)
    which = np.digitize(radius.ravel(), edges) - 1
    valid = (which >= 0) & (which < len(edges) - 1)
    sums = np.bincount(which[valid], weights=power.ravel()[valid], minlength=len(edges) - 1)
    counts = np.bincount(which[valid], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        mean_psd = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return RadialPsd(freq=centers, psd=mean_psd)


def mean_log_psd(curves: list[RadialPsd]) -> RadialPsd:
    """Average PSD curves in log space (geometric mean) on their shared bin grid."""

    n_bins = min(len(c.psd) for c in curves)
    stack = np.stack([np.log10(np.maximum(c.psd[:n_bins], 1e-30)) for c in curves])
    return RadialPsd(freq=curves[0].freq[:n_bins], psd=10.0 ** np.nanmean(stack, axis=0))


def band_mean_log(curve: RadialPsd, f_lo: float, f_hi: float) -> float:
    """Mean log10-PSD over a frequency band, NaN-safe."""

    mask = (curve.freq >= f_lo) & (curve.freq <= f_hi) & np.isfinite(curve.psd)
    if not mask.any():
        raise GsdValidationError(f"No finite PSD bins in band [{f_lo:.2f}, {f_hi:.2f}] cyc/m.")
    return float(np.nanmean(np.log10(np.maximum(curve.psd[mask], 1e-30))))


def read_rgb_window(src: rasterio.DatasetReader, window: Window) -> tuple[np.ndarray, float]:
    """Read an RGB uint8 window and its valid-mask fraction from an open dataset."""

    rgb = src.read(indexes=[1, 2, 3], window=window, boundless=False)
    mask = src.dataset_mask(window=window)
    chip = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)  # [C, H, W] -> [H, W, C]
    return chip, float((mask > 0).mean())


def collect_tiles(uas_path: Path, crewed_path: Path, tile_px: int, max_tiles: int,
                  min_valid: float = 0.99, min_std: float = 0.04) -> list[dict]:
    """Sample co-located ground tiles from an overlapping sUAS/crewed mosaic pair.

    Tiles are scanned on a deterministic grid over the geographic intersection and
    accepted when both reads are fully valid (no nodata) and the sUAS luminance shows
    enough structure to carry a meaningful spectrum.

    Returns:
        List of dicts with keys ``uas`` (HWC uint8), ``crewed`` (HWC uint8),
        ``uas_gsd``, ``crewed_gsd``.
    """

    tiles: list[dict] = []
    with rasterio.open(uas_path) as uas, rasterio.open(crewed_path) as crewed:
        uas_gsd_x, uas_gsd_y = ground_gsds(uas)
        crewed_gsd_x, crewed_gsd_y = ground_gsds(crewed)
        uas_gsd = math.sqrt(uas_gsd_x * uas_gsd_y)
        if abs(uas.transform.b) > 1e-9 or abs(crewed.transform.b) > 1e-9:
            raise GsdValidationError("Rotated geotransforms are not supported.")
        if uas.crs.is_geographic:
            raise GsdValidationError("Expected a projected (meter-unit) CRS for the sUAS mosaic.")

        crewed_bounds_in_uas = (transform_bounds(crewed.crs, uas.crs, *crewed.bounds)
                                if crewed.crs != uas.crs else tuple(crewed.bounds))
        minx = max(uas.bounds.left, crewed_bounds_in_uas[0])
        miny = max(uas.bounds.bottom, crewed_bounds_in_uas[1])
        maxx = min(uas.bounds.right, crewed_bounds_in_uas[2])
        maxy = min(uas.bounds.top, crewed_bounds_in_uas[3])
        if minx >= maxx or miny >= maxy:
            raise GsdValidationError(f"No geographic overlap between {uas_path.name} and {crewed_path.name}.")

        ground = tile_px * uas_gsd
        n_x = int((maxx - minx) // ground)
        n_y = int((maxy - miny) // ground)
        for iy in range(n_y):
            for ix in range(n_x):
                if len(tiles) >= max_tiles:
                    return tiles
                x0 = minx + ix * ground
                y1 = maxy - iy * ground
                bounds = (x0, y1 - ground, x0 + ground, y1)

                uas_win = from_bounds(*bounds, transform=uas.transform)
                uas_chip, uas_valid = read_rgb_window(uas, uas_win)
                if uas_valid < min_valid or uas_chip.shape[0] < tile_px or uas_chip.shape[1] < tile_px:
                    continue
                lum = luminance(uas_chip)
                if float(lum.std()) < min_std:
                    continue

                crewed_query = (transform_bounds(uas.crs, crewed.crs, *bounds)
                                if crewed.crs != uas.crs else bounds)
                crewed_win = from_bounds(*crewed_query, transform=crewed.transform)
                crewed_chip, crewed_valid = read_rgb_window(crewed, crewed_win)
                if crewed_valid < min_valid or min(crewed_chip.shape[:2]) < 64:
                    continue

                tiles.append({"uas": uas_chip[:tile_px, :tile_px], "crewed": crewed_chip,
                              "uas_gsds": (uas_gsd_x, uas_gsd_y), "crewed_gsds": (crewed_gsd_x, crewed_gsd_y)})
    return tiles


def analyze_pair(uas_path: Path, crewed_path: Path, tile_px: int, max_tiles: int,
                 mtf_at_nyquist: float, out_dir: Path) -> dict:
    """Run the four-condition spectral comparison for one mosaic pair."""

    tiles = collect_tiles(uas_path, crewed_path, tile_px=tile_px, max_tiles=max_tiles)
    if len(tiles) < 3:
        raise GsdValidationError(f"Only {len(tiles)} usable tiles for {uas_path.name}; need >= 3.")

    uas_gsd_x, uas_gsd_y = tiles[0]["uas_gsds"]
    crewed_gsd_x, crewed_gsd_y = tiles[0]["crewed_gsds"]
    uas_gsd = math.sqrt(uas_gsd_x * uas_gsd_y)
    crewed_gsd = math.sqrt(crewed_gsd_x * crewed_gsd_y)
    factor = crewed_gsd / uas_gsd
    crewed_nyq = min(1.0 / (2.0 * crewed_gsd_x), 1.0 / (2.0 * crewed_gsd_y))
    sigma = gaussian_sigma_for_mtf(factor, mtf_at_nyquist)
    print(f"\n=== {uas_path.name} vs {crewed_path.name}")
    print(f"    uas_gsd {uas_gsd*100:.2f} cm  crewed_gsd {crewed_gsd*100:.2f} cm"
          f" (x {crewed_gsd_x*100:.2f} / y {crewed_gsd_y*100:.2f})  factor {factor:.2f}"
          f"  blur sigma {sigma:.2f} px  tiles {len(tiles)}")

    sampling = SyntheticGsdDegradation(factor=factor)
    matched = SyntheticGsdDegradation(factor=factor, mtf_at_nyquist=mtf_at_nyquist)

    curves: dict[str, list[RadialPsd]] = {"uas_native": [], "sampling_only": [], "mtf_matched": [], "crewed_real": []}
    for tile in tiles:
        uas_chip = tile["uas"]
        curves["uas_native"].append(radial_psd(luminance(uas_chip), uas_gsd_x, uas_gsd_y))
        curves["sampling_only"].append(radial_psd(luminance(sampling.apply(uas_chip)), uas_gsd_x, uas_gsd_y))
        curves["mtf_matched"].append(radial_psd(luminance(matched.apply(uas_chip)), uas_gsd_x, uas_gsd_y))
        curves["crewed_real"].append(radial_psd(luminance(tile["crewed"]), crewed_gsd_x, crewed_gsd_y))

    mean_curves = {name: mean_log_psd(per_tile) for name, per_tile in curves.items()}

    # Gain-align the crewed curve to the sUAS native curve in the low band where both
    # sensors resolve the same scene layout; apply as a log-PSD offset.
    gain_offset = band_mean_log(mean_curves["uas_native"], *GAIN_BAND) - band_mean_log(mean_curves["crewed_real"], *GAIN_BAND)
    aligned_crewed = RadialPsd(freq=mean_curves["crewed_real"].freq, psd=mean_curves["crewed_real"].psd * 10.0**gain_offset)

    # Resolution-sensitive band scores: mean log-PSD delta vs the real crewed sensor.
    f_lo, f_hi = (REPORT_BAND_FRACTIONS[0] * crewed_nyq, REPORT_BAND_FRACTIONS[1] * crewed_nyq)
    crewed_band = band_mean_log(aligned_crewed, f_lo, f_hi)
    deltas = {name: band_mean_log(mean_curves[name], f_lo, f_hi) - crewed_band
              for name in ("uas_native", "sampling_only", "mtf_matched")}
    print(f"    band [{f_lo:.2f}, {f_hi:.2f}] cyc/m mean log10-PSD delta vs crewed (0 = matches real sensor):")
    for name, delta in deltas.items():
        print(f"      {name:<14} {delta:+.3f} dex ({10.0**delta:.2f}x power)")

    # Empirical transfer functions vs the shared native scene, evaluated at the crewed Nyquist.
    def transfer_at(name: str, freq: float) -> float:
        num = mean_curves[name] if name != "crewed_real" else aligned_crewed
        idx = int(np.nanargmin(np.abs(num.freq - freq)))
        ref_idx = int(np.nanargmin(np.abs(mean_curves["uas_native"].freq - freq)))
        return float(np.sqrt(num.psd[idx] / mean_curves["uas_native"].psd[ref_idx]))

    mtf_at_nyq = {name: transfer_at(name, crewed_nyq) for name in ("sampling_only", "mtf_matched", "crewed_real")}
    print(f"    empirical system MTF at the crewed Nyquist ({crewed_nyq:.2f} cyc/m), scene-referenced:")
    for name, value in mtf_at_nyq.items():
        print(f"      {name:<14} {value:.3f}")

    figure_path = out_dir / f"psd_{uas_path.stem.replace('.geo', '')}.png"
    plot_pair(mean_curves=mean_curves, aligned_crewed=aligned_crewed, crewed_nyq=crewed_nyq,
              uas_nyq=1.0 / (2.0 * uas_gsd), factor=factor, title=uas_path.name, out_path=figure_path)
    print(f"    figure: {figure_path}")

    return {"pair": [uas_path.name, crewed_path.name], "uas_gsd_m": uas_gsd, "crewed_gsd_m": crewed_gsd,
            "factor": factor, "blur_sigma_px": sigma, "tiles": len(tiles), "gain_offset_dex": gain_offset,
            "band_cyc_per_m": [f_lo, f_hi], "band_log_psd_delta_vs_crewed": deltas,
            "empirical_mtf_at_crewed_nyquist": mtf_at_nyq, "figure": str(figure_path)}


def plot_pair(mean_curves: dict[str, RadialPsd], aligned_crewed: RadialPsd, crewed_nyq: float,
              uas_nyq: float, factor: float, title: str, out_path: Path) -> None:
    """Render the four-condition PSD comparison for one pair."""

    styles = {"uas_native": ("#888888", "sUAS native"),
              "sampling_only": ("#d62728", f"synthetic {factor:.1f}x sampling-only"),
              "mtf_matched": ("#1f77b4", f"synthetic {factor:.1f}x MTF-matched (0.3)")}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (color, label) in styles.items():
        curve = mean_curves[name]
        ok = np.isfinite(curve.psd) & (curve.freq > 0)
        ax.loglog(curve.freq[ok], curve.psd[ok], color=color, label=label, linewidth=1.6)
    ok = np.isfinite(aligned_crewed.psd) & (aligned_crewed.freq > 0)
    ax.loglog(aligned_crewed.freq[ok], aligned_crewed.psd[ok], color="#2ca02c",
              label="crewed real (gain-aligned)", linewidth=2.2, linestyle="--")
    ax.axvline(crewed_nyq, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(uas_nyq, color="#888888", linestyle=":", linewidth=1.0)
    ax.text(crewed_nyq, ax.get_ylim()[0] * 1.5, " crewed Nyquist", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("spatial frequency (cycles/m)")
    ax.set_ylabel("radially averaged PSD (variance $\\cdot$ m$^2$)")
    ax.set_title(f"{title}: synthetic GSD degradation vs real crewed sensor")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry point: run the spectral validation over the default or given pairs."""

    parser = argparse.ArgumentParser(description="Validate SyntheticGsdDegradation against crewed overflights.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory.")
    parser.add_argument("--tile-px", type=int, default=2048, help="sUAS tile size in pixels.")
    parser.add_argument("--max-tiles", type=int, default=12, help="Maximum tiles per pair.")
    parser.add_argument("--mtf", type=float, default=0.3, help="MTF-at-Nyquist target for the matched arm.")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "gsd_validation", help="Output directory.")
    args = parser.parse_args()

    results = []
    for uas_rel, crewed_rel in DEFAULT_PAIRS:
        uas_path = args.data_dir / uas_rel
        crewed_path = args.data_dir / crewed_rel
        if not uas_path.exists() or not crewed_path.exists():
            print(f"[skip] missing file(s) for pair {uas_path.name}")
            continue
        try:
            results.append(analyze_pair(uas_path=uas_path, crewed_path=crewed_path, tile_px=args.tile_px,
                                         max_tiles=args.max_tiles, mtf_at_nyquist=args.mtf, out_dir=args.out))
        except GsdValidationError as error:
            print(f"[skip] {uas_path.name}: {error}")

    if not results:
        raise SystemExit("No pairs analyzed.")
    summary_path = args.out / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
