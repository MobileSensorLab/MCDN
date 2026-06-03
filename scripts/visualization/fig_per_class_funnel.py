"""R3: per-class F1 across the canonical 10-seed baseline evaluation funnel.

Compresses the three per-class-decomposition tables in
[doc/5-results.qmd](doc/5-results.qmd) into a single polyline plot. Each line
tracks one ordinal damage class across the three holdouts (in-distribution
Spatial Block East, proximate-OOD Hurricane Michael, distant-OOD Mayfield
Tornado), so the chapter's per-class narrative reads as visual trajectories
rather than three separately-tabulated rows:

    - **No Damage** dips on Hurricane Michael, then *rises* sharply on
      Mayfield (the prevalent class is easier on the imbalanced distant-OOD
      holdout).
    - **Minor** is roughly flat across the funnel - the bottleneck class on
      every split.
    - **Major** drops on Hurricane Michael and stays flat on Mayfield (recall
      collapse on the unseen-event splits).
    - **Destroyed** holds across the hurricane funnel (slight rise on
      Michael) and *collapses* on Mayfield - the chapter's central per-class
      signature, driven by precision falling against still-strong recall as
      Major-as-Destroyed false positives concentrate on tornado debris fields.

Visual conventions:
    - Lines colored by the FEMA-aligned ColorBrewer Set1 ordinal palette
      from ``_common.py`` (green / orange / red / purple) for sign-level
      visual consistency with the other chapter figures.
    - Markers are class-specific shapes (circle, square, triangle, diamond)
      for redundant non-color encoding - the figure remains parseable in
      greyscale print and for color-vision-deficient readers.
    - Compact single-column figure footprint with a 2 x 2 in-axes legend.

Source: ``outputs/ablation/baseline/<split>/ensemble_metrics.json`` ->
``ensemble.argmax.per_class.<class>.f1``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualization._common import (
    ORDINAL_CLASS_NAMES,
    ORDINAL_CLASS_PALETTE,
    WIDTH_2COL,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_BASELINE_DIR: Final[Path] = _REPO_ROOT / "outputs" / "ablation" / "baseline"

# Funnel ordering matches R1 confusion triptych and R2 ablation deltas so
# the three figures read consistently across the chapter.
_SPLITS: Final[tuple[tuple[str, str], ...]] = (
    ("Spatial_Block_East", "E/W"),
    ("Hurricane_Michael", "Michael"),
    ("Mayfield_Tornado", "Mayfield"),
)

# Class-specific marker shapes for redundant non-color encoding. Order
# matches ORDINAL_CLASS_NAMES so the i-th marker pairs with the i-th palette
# color.
_CLASS_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")


def _load_per_class_f1(split_dir: str) -> dict[str, float]:
    """Read per-class ensemble argmax F1 for one split."""

    metrics_path = _BASELINE_DIR / split_dir / "ensemble_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Ensemble metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    per_class = payload["ensemble"]["argmax"]["per_class"]
    return {name: float(per_class[name]["f1"]) for name in ORDINAL_CLASS_NAMES}


def main() -> None:
    """Render the per-class F1 funnel polyline plot."""

    setup_publication_style()

    splits_data = [(label, _load_per_class_f1(d)) for d, label in _SPLITS]
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

    for class_name, color, marker in zip(
        ORDINAL_CLASS_NAMES, ORDINAL_CLASS_PALETTE, _CLASS_MARKERS
    ):
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

    ax.set_xlim(-0.2, len(splits_data) - 0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_labels, fontsize=8)
    ax.set_ylim(0.65, 0.95)
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

    save_caption(
        name="per_class_funnel",
        title=(
            "Per-class F1 across the canonical 10-seed baseline evaluation "
            "funnel."
        ),
        body=(
            "Polylines show the ensemble argmax F1 for each ordinal damage "
            "class as the holdout shifts from in-distribution (Spatial Block "
            "East), through proximate-OOD (Hurricane Michael LOEO), to "
            "distant-OOD (Mayfield Tornado LOEO). Class colors come from the "
            "FEMA-aligned ColorBrewer Set1 ordinal palette in `_common.py`; "
            "marker shapes (circle / square / triangle / diamond) provide "
            "redundant non-color encoding for greyscale print and "
            "color-vision-deficient readers. Four trajectories, four "
            "different stories: No Damage dips on Hurricane Michael then "
            "rises sharply on Mayfield (the prevalent class is easier on "
            "the imbalanced distant-OOD split); Minor is roughly flat - the "
            "consistent bottleneck class; Major drops on Hurricane Michael "
            "and stays flat on Mayfield (recall collapse on the unseen-event "
            "splits); and Destroyed holds across the hurricane funnel and "
            "collapses on Mayfield - the chapter's central per-class "
            "signature, driven by precision falling against still-strong "
            "recall as Major-as-Destroyed false positives concentrate on "
            "tornado debris fields. Source: "
            "`outputs/ablation/baseline/<split>/ensemble_metrics.json` -> "
            "`ensemble.argmax.per_class.<class>.f1`."
        ),
    )


if __name__ == "__main__":
    main()
