"""Damage-class exemplars for the manuscript's opening pages.

Renders one unambiguous sUAS chip per ordinal damage grade (No Damage, Minor,
Major, Destroyed) in a single-column 2 x 2 grid so a reader unfamiliar with
the four-grade building damage scale sees what each grade looks like from
nadir at sUAS resolution before the method is introduced. Each chip is drawn
at the model's native 512 px window with the structure footprint traced as a
white-with-black-halo contour (the polygon that centers the chip and scopes
the label) and a 5 m scale bar derived from the source orthomosaic's GSD.

Chips are pinned by validation-pool index in the DROIDs default split so the
figure is reproducible; the pinned chips were chosen from the eight
highest-confidence correctly classified candidates per class whose footprint
covers 10-55 % of the chip, does not touch the chip border, and comes from a
fine-GSD (<= 4 cm) orthomosaic so all four chips share a similar ground
extent (see `_PINNED`). Swap an index to change an exemplar.

Source: the DROIDs default split val pool via ``load_reference_val_pool`` and
``data/statistics.csv`` for per-orthomosaic GSD.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Rectangle

from src.data.dataset import CRASARUnitemporalDataset
from scripts.visualization._common import (
    DEFAULT_SPLIT_DIR,
    FIG_FONT_PT,
    ORDINAL_CLASS_NAMES,
    WIDTH_1COL,
    load_reference_val_pool,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_STATISTICS_CSV: Final[Path] = _REPO_ROOT / "data" / "statistics.csv"

# Validation-pool indices (DROIDs default split, all_features ensemble ordering) per class.
_PINNED: Final[dict[str, int]] = {
    "No Damage": 1360,
    "Minor": 2613,
    "Major": 155,
    "Destroyed": 2953
}

# Ground-window multiplier per class. Chips are read at ``512 * scale`` source pixels and resampled to
# 512 so a large footprint sits inside the panel with the same clearance from the scale bar as the others;
# the scale bar is drawn from the effective (scaled) GSD so it stays metrically correct.
_WINDOW_SCALE: Final[dict[str, float]] = {
    "No Damage": 1.0,
    "Minor": 1.4,
    "Major": 1.35,
    "Destroyed": 1.0
}

_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 2.4
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 1.1
_SCALE_BAR_M: Final[float] = 5.0


def _gsd_lookup() -> dict[str, float]:
    """Map orthomosaic filename to GSD in metres per pixel from the dataset statistics table."""

    table = pd.read_csv(_STATISTICS_CSV)
    return {str(name): float(gsd) for name, gsd in zip(table["Orthomosaic"], table["GSD (m/px)"], strict=True)}


def _draw_scale_bar(ax: plt.Axes, *, gsd_m: float, chip_px: int) -> None:
    """Draw a 5 m scale bar with a dark halo in the lower-left corner of a chip panel."""

    length_px = _SCALE_BAR_M / gsd_m
    x0, y0 = 0.05 * chip_px, 0.93 * chip_px
    ax.add_patch(Rectangle((x0 - 2, y0 - 2), length_px + 4, 8, facecolor="black", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((x0, y0), length_px, 4, facecolor="white", edgecolor="none", zorder=5))
    ax.text(x0 + length_px / 2, y0 - 6, f"{_SCALE_BAR_M:.0f} m", ha="center", va="bottom", fontsize=FIG_FONT_PT, color="white",
            path_effects=None, zorder=6, bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.2, "alpha": 0.75})


def _chip_at_window_scale(dataset: CRASARUnitemporalDataset, idx: int, *, scale: float) -> dict[str, Any]:
    """Read one chip with its mosaic's read window widened by ``scale``, leaving the dataset unchanged afterwards.

    The dataset resolves the read window per mosaic at index-build time; this widens it for a single read
    (figure use only, no jitter is applied outside training) and restores the original value.
    """

    image_path = dataset.instances[idx]["image_path"]
    original_px = dataset._read_window_px[image_path]
    dataset._read_window_px[image_path] = max(8, round(original_px * scale))
    try:
        return dataset[idx]
    finally:
        dataset._read_window_px[image_path] = original_px


def main() -> None:
    """Render the 2 x 2 damage-class exemplar figure and its caption."""

    setup_publication_style()

    pool = load_reference_val_pool(holdout=DEFAULT_SPLIT_DIR)
    dataset, targets, probs = pool["dataset"], pool["targets"], pool["ensemble_probs"]
    gsd_by_mosaic = _gsd_lookup()

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_1COL, WIDTH_1COL + 0.35),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.16, "left": 0.005, "right": 0.995, "top": 0.93, "bottom": 0.005})

    provenance: list[str] = []
    for ax, class_name in zip(axes.ravel(), ORDINAL_CLASS_NAMES, strict=True):
        idx = _PINNED[class_name]
        class_idx = ORDINAL_CLASS_NAMES.index(class_name)
        if int(targets[idx]) != class_idx:
            raise ValueError(f"Pinned index {idx} is labeled {ORDINAL_CLASS_NAMES[int(targets[idx])]}, not {class_name}.")

        window_scale = _WINDOW_SCALE[class_name]
        sample = _chip_at_window_scale(dataset, idx, scale=window_scale)
        image = sample["image"].numpy()  # [4, H, W] uint8
        rgb = image[:3].transpose(1, 2, 0)
        mask = image[3]
        instance = dataset.instances[idx]
        mosaic = Path(instance["image_path"]).name
        gsd_m = gsd_by_mosaic[mosaic]

        ax.imshow(rgb, interpolation="bilinear")
        ax.contour(mask.astype(float), levels=[0.5], colors=[_MASK_HALO_COLOR], linewidths=_MASK_HALO_WIDTH)
        ax.contour(mask.astype(float), levels=[0.5], colors=[_MASK_OUTLINE_COLOR], linewidths=_MASK_OUTLINE_WIDTH)
        _draw_scale_bar(ax, gsd_m=gsd_m * window_scale, chip_px=mask.shape[0])
        ax.set_title(class_name, fontsize=FIG_FONT_PT, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        provenance.append(f"{class_name}: {instance['event_name']}, {mosaic}, {100 * gsd_m:.1f} cm GSD, "
                          f"{window_scale:.1f}x ground window, val index {idx}, ensemble p = {probs[idx, class_idx]:.3f}")

    save_figure(fig=fig, name="damage_class_exemplars")

    save_caption(
        name="damage_class_exemplars",
        title="The four-grade building damage scale as seen from nadir sUAS imagery.",
        body=(
            "One structure per grade from the CRASAR-U-DROIDs annotations, shown at MCDN's 512 px chip "
            "window (10-16 m across at the 2-3 cm GSD of these orthomosaics; the Minor and Major chips are read at a "
            "1.4x / 1.35x wider ground window so their larger footprints clear the scale bar) with the annotated structure "
            "footprint traced as a white-with-black-halo contour and a metrically correct 5 m scale bar. No Damage: roof and walls intact. Minor: "
            "superficial envelope damage such as stripped shingles or displaced roofing, structure intact. Major: "
            "structural damage such as roof-deck failure or partial collapse with the building still standing. "
            "Destroyed: total collapse or the structure swept from its footprint, leaving slab and debris. All four "
            "are correctly classified by the 10-seed ensemble on the DROIDs default split. Provenance - "
            + "; ".join(provenance) + "."
        ),
    )


if __name__ == "__main__":
    main()
