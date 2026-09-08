"""R3: per-class F1 of the full MCDN configuration across the four reported columns.

Compresses the per-class-decomposition tables into a single polyline plot.
Each line tracks one ordinal damage class across the DROIDs default split and
the three leave-one-event-out holdouts (Michael, Mayfield, Ida), so the
per-class narrative reads as visual trajectories rather than four separately
tabulated rows. The Ida column is where the interior classes diverge most:
Minor F1 collapses while Major F1 rises, the signature of the confident
Minor-to-No Damage under-calling documented in the error inventory.

Visual conventions:
    - Lines colored by the FEMA-aligned ColorBrewer Set1 ordinal palette
      from ``_common.py`` (green / orange / red / purple) for sign-level
      visual consistency with the other figures.
    - Markers are class-specific shapes (circle, square, triangle, diamond)
      for redundant non-color encoding - the figure remains parseable in
      greyscale print and for color-vision-deficient readers.
    - Compact figure footprint with a 2 x 2 in-axes legend.

Source: ``outputs/ablation_dgx/_ensembles/all_features__<split>.json`` ->
``cross_variant.equal_seed_metrics.argmax.per_class_f1``.
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualization._common import (
    ORDINAL_CLASS_NAMES,
    ORDINAL_CLASS_PALETTE,
    REPORTED_COLUMNS,
    WIDTH_2COL,
    ensemble_rule_metrics,
    save_caption,
    save_figure,
    setup_publication_style,
)

# Class-specific marker shapes for redundant non-color encoding. Order
# matches ORDINAL_CLASS_NAMES so the i-th marker pairs with the i-th palette
# color.
_CLASS_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")


def _load_per_class_f1(split_dir: str) -> dict[str, float]:
    """Read per-class ensemble argmax F1 for one split."""

    per_class = ensemble_rule_metrics(split_dir, rule="argmax")["per_class_f1"]
    return {name: float(per_class[name]) for name in ORDINAL_CLASS_NAMES}


def main() -> None:
    """Render the per-class F1 polyline plot."""

    setup_publication_style()

    splits_data = [(label, _load_per_class_f1(d)) for d, label in REPORTED_COLUMNS]
    split_labels = [label for label, _ in splits_data]
    x_positions = np.arange(len(splits_data))

    class_trajectories: dict[str, np.ndarray] = {
        name: np.array([per_class[name] for _, per_class in splits_data])
        for name in ORDINAL_CLASS_NAMES
    }

    fig, ax = plt.subplots(
        figsize=(WIDTH_2COL, 3.0),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "top": 0.94,
            "bottom": 0.14,
        },
    )

    for class_name, color, marker in zip(ORDINAL_CLASS_NAMES, ORDINAL_CLASS_PALETTE, _CLASS_MARKERS, strict=True):
        ax.plot(
            x_positions,
            class_trajectories[class_name],
            color=color,
            marker=marker,
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=1.6,
            label=class_name,
            zorder=3,
        )

    all_values = np.concatenate(list(class_trajectories.values()))
    ax.set_xlim(-0.2, len(splits_data) - 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_labels, fontsize=8)
    ax.set_ylim(np.floor(all_values.min() * 20) / 20, min(1.0, np.ceil(all_values.max() * 20) / 20))
    ax.set_ylabel("Per-class F1 (ensemble argmax)", fontsize=9)
    ax.tick_params(axis="both", which="both", labelsize=8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower left",
        fontsize=7,
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.4,
    )

    save_figure(fig=fig, name="per_class_funnel")

    trajectory_text = "; ".join(
        f"{name}: " + " / ".join(f"{value:.3f}" for value in class_trajectories[name])
        for name in ORDINAL_CLASS_NAMES
    )
    save_caption(
        name="per_class_funnel",
        title="Per-class F1 of the full MCDN configuration across the four reported evaluation columns.",
        body=(
            "Polylines show the 10-seed ensemble argmax F1 for each ordinal damage class across the DROIDs "
            "default split (the dataset's published train/test partition) and the LOEO Michael, Mayfield and Ida "
            "holdouts. Class colors come from the FEMA-aligned ColorBrewer Set1 ordinal palette in `_common.py`; "
            "marker shapes (circle / square / triangle / diamond) provide redundant non-color encoding for "
            "greyscale print and color-vision-deficient readers. Values in column order "
            f"({', '.join(split_labels)}) - {trajectory_text}. "
            "Source: `outputs/ablation_dgx/_ensembles/all_features__<split>.json` -> "
            "`cross_variant.equal_seed_metrics.argmax.per_class_f1`."
        ),
    )


if __name__ == "__main__":
    main()
