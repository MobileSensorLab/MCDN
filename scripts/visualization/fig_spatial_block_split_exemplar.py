r"""Exemplar figure for spatial-block train/validation construction (Harlem Heights).

Left: one orthomosaic with a vertical line at the horizontal midpoint of its
raster bounds---the same scalar ``cx = (left + right) / 2`` used when the
trainer sorts orthos west-to-east before an 80/20 count split (eastern tail
in validation under the ``Spatial_Block_East`` fold).

Right: ranks of every valid ``uas_5cm`` ortho in the training manifest by that
``cx``, with the 80% cut and the highlighted rank of the exemplar ortho.

The script does not modify ``src``; it only imports read-only helpers to match
the trainer's manifest filter.

Usage::

    uv run python -m scripts.visualization.fig_spatial_block_split_exemplar

    uv run python -m scripts.visualization.fig_spatial_block_split_exemplar \\
        --ortho "C:/path/to/1001-Harlem-Heights.geo.tif_20220930d_RGB.geo.tif"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from scripts.visualization._common import (
    WIDTH_2COL,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT: Final[Path] = _REPO_ROOT / "data"
_DEFAULT_UAS_DIR: Final[Path] = _DEFAULT_DATA_ROOT / "train" / "imagery" / "UAS"
_FIGURE_STEM: Final[str] = "spatial_block_split_harlem_exemplar"
_TRAIN_FRAC: Final[float] = 0.8


def _resolve_uas_imagery_dir(data_root: Path) -> Path:
    """Return ``train/imagery/UAS`` under data_root, tolerating case on the sensor folder."""

    base = data_root / "train" / "imagery"
    if not base.is_dir():
        return _DEFAULT_UAS_DIR
    for name in ("UAS", "uas"):
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return _DEFAULT_UAS_DIR


def resolve_exemplar_ortho_path(*, data_root: Path, ortho_arg: str | None) -> Path:
    """Resolve the orthomosaic path from CLI or Harlem-Heights defaults."""

    if ortho_arg:
        raw = Path(ortho_arg)
        bases: list[Path] = [raw]
        if raw.suffix.lower() == ".geo":
            bases.append(raw.with_suffix(".geo.tif"))

        candidates: list[Path] = []
        for base in bases:
            candidates.extend(
                [
                    base,
                    _REPO_ROOT / base,
                    data_root.parent / base,
                    Path.cwd() / base,
                ]
            )
        for path in candidates:
            if path.is_file():
                return path.resolve()
        msg = f"Orthomosaic not found: {ortho_arg!r} (tried .geo, .geo.tif, and repo-relative paths)."
        raise FileNotFoundError(msg)

    uas_dir = _resolve_uas_imagery_dir(data_root)
    matches = sorted(uas_dir.glob("*1001-Harlem-Heights*.tif"))
    if not matches:
        msg = (
            f"No '*1001-Harlem-Heights*.tif' under {uas_dir}. "
            "Pass --ortho with a full path to the GeoTIFF."
        )
        raise FileNotFoundError(msg)
    return matches[0].resolve()


def _ortho_horizontal_midpoint_x(ortho_path: Path) -> float:
    """Return ``(bounds.left + bounds.right) / 2`` in the raster CRS (matches ``sampling.py``)."""

    with rasterio.open(ortho_path) as src:
        bounds = src.bounds
    return (bounds.left + bounds.right) / 2.0


def read_rgb_preview(ortho_path: Path, *, max_dim: int = 1800) -> tuple[np.ndarray, rasterio.coords.BoundingBox]:
    """Read RGB as a float preview array in ``[0, 1]`` plus geographic bounds."""

    with rasterio.open(ortho_path) as src:
        bounds = src.bounds
        height = src.height
        width = src.width
        scale = min(1.0, max_dim / max(height, width))
        out_h = max(1, round(height * scale))
        out_w = max(1, round(width * scale))
        arr = src.read(
            indexes=(1, 2, 3),
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        )
    rgb = np.moveaxis(arr.astype(np.float32), 0, -1)
    rgb = np.clip(rgb, 0.0, None)
    lo, hi = np.percentile(rgb, (2.0, 98.0))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return rgb, bounds


def _manifest_cx_table(data_root: Path) -> tuple[list[str], list[float]]:
    """Paths and midpoint-x for each valid row under ``uas_5cm`` (trainer-aligned)."""

    from src.data.sampling import build_valid_manifest

    manifest = build_valid_manifest(data_dir=str(data_root), sensor_profile="uas_5cm")
    paths: list[str] = []
    xs: list[float] = []
    for _, row in manifest.iterrows():
        image_path = Path(str(row["image_path"]))
        try:
            cx = _ortho_horizontal_midpoint_x(image_path)
        except (OSError, rasterio.errors.RasterioIOError) as error:
            print(f"WARNING: skip {image_path.name}: {error}")
            continue
        paths.append(str(image_path.resolve()))
        xs.append(cx)
    return paths, xs


def _plot_ortho_panel(ax: plt.Axes, ortho_path: Path) -> tuple[float, rasterio.coords.BoundingBox]:
    """Draw the ortho with a vertical line at the sorting key ``cx``."""

    rgb, bounds = read_rgb_preview(ortho_path)
    cx = (bounds.left + bounds.right) / 2.0
    ax.imshow(
        rgb,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        interpolation="bilinear",
    )
    ax.axvline(cx, color="#00e5ff", linestyle="--", linewidth=2.0, alpha=0.95)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return cx, bounds


def _plot_rank_panel(
    ax: plt.Axes,
    *,
    paths: list[str],
    xs: list[float],
    exemplar_resolved: str,
) -> None:
    """Scatter orthos by sorted rank; draw the 80% cut and highlight the exemplar."""

    n = len(xs)
    if n == 0:
        ax.text(0.5, 0.5, "No manifest rows.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    order = np.argsort(np.asarray(xs, dtype=np.float64))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)

    split_idx = int(n * _TRAIN_FRAC)
    exemplar_key = Path(exemplar_resolved).resolve().as_posix().lower()
    keys = [Path(p).resolve().as_posix().lower() for p in paths]
    try:
        orig_idx = keys.index(exemplar_key)
    except ValueError:
        orig_idx = len(xs) // 2
        print(
            "WARNING: exemplar ortho not in uas_5cm manifest; "
            f"highlighting mid-rank marker instead (n={n}).",
            file=sys.stderr,
        )
    target_rank = int(ranks[orig_idx])

    jitter = (np.random.default_rng(0).random(n) - 0.5) * 0.12
    ax.scatter(
        ranks,
        jitter,
        s=10,
        c="#94a3b8",
        alpha=0.65,
        linewidths=0,
        label="Orthos (sorted west $\\rightarrow$ east)",
    )
    ax.scatter(
        [target_rank],
        [0.0],
        s=120,
        c="#f97316",
        marker="*",
        zorder=5,
        edgecolors="#1e293b",
        linewidths=0.6,
        label="Exemplar ortho",
    )
    ax.axvline(
        split_idx - 0.5,
        color="#22c55e",
        linestyle="-",
        linewidth=2.0,
        alpha=0.9,
        label=f"80 / 20 cut (rank {split_idx} of {n})",
    )
    ax.axvspan(split_idx - 0.5, n - 0.5, facecolor="#38bdf8", alpha=0.08, zorder=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("West $\\rightarrow$ East rank (one dot per ortho)")
    ax.set_yticks([])
    ax.legend(loc="upper center", fontsize=7.5, frameon=True, ncol=1)


def main() -> None:
    """Render the two-panel spatial-block split exemplar."""

    parser = argparse.ArgumentParser(description="Render spatial-block split exemplar (Harlem Heights ortho).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Dataset root (expects train/imagery/UAS/... GeoTIFFs).",
    )
    parser.add_argument(
        "--ortho",
        type=str,
        default=None,
        help="Path to the exemplar GeoTIFF. Default: first *1001-Harlem-Heights*.tif under train/imagery/UAS.",
    )
    args = parser.parse_args()

    ortho_path = resolve_exemplar_ortho_path(data_root=args.data_root, ortho_arg=args.ortho)
    print(f"Exemplar ortho: {ortho_path}")

    setup_publication_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_2COL, 3.55),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.22},
    )
    ax_left, ax_right = axes

    cx, bounds = _plot_ortho_panel(ax_left, ortho_path=ortho_path)

    paths, xs = _manifest_cx_table(args.data_root)
    _plot_rank_panel(ax_right, paths=paths, xs=xs, exemplar_resolved=str(ortho_path))

    fig.suptitle(
        f"Spatial block east fold - sorting key on {ortho_path.name}",
        fontsize=9.5,
        y=0.98,
    )
    print(
        f"Bounds: left={bounds.left:.2f} right={bounds.right:.2f}  "
        f"cx={cx:.2f}  manifest orthos: {len(xs)}",
    )

    save_figure(fig=fig, name=_FIGURE_STEM, close=True)

    save_caption(
        name=_FIGURE_STEM,
        title="Spatial-block split: ortho midpoint and west-to-east rank",
        body=(
            "The trainer assigns each orthomosaic one west-to-east scalar "
            "$c_x = (x_{\\min}+x_{\\max})/2$ from its raster bounds (same CRS as the GeoTIFF), "
            "sorts all valid orthos by $c_x$, and places the eastern 20% by count in validation "
            "for the **Spatial_Block_East** fold. **Left:** the exemplar ortho with $c_x$ as a dashed line "
            "(not an interior partition of the mosaic). **Right:** one marker per ortho in the "
            "`uas_5cm` training manifest at its sorted rank, the green cut at 80%, and the orange star "
            "for this ortho when present in that manifest."
        ),
    )


if __name__ == "__main__":
    main()
