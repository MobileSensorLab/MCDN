"""F3 - Adjacency-aware label smoothing comparison.

Renders a 2 x 4 grid of bar charts showing how four label-smoothing schemes shape
the target distribution q for one interior and one extreme true class, at an
illustrative epsilon = 0.10. The four schemes are:

- Hard one-hot (no smoothing).
- Naive uniform smoothing (standard CE-style; spreads epsilon over every
  non-target class equally).
- Classical adjacency-aware [@diazSoftLabelsOrdinal2019]: full epsilon to the
  sole adjacent neighbor at extreme classes; epsilon / 2 per neighbor at
  interiors.
- Per-neighbor-rate-constant (MCDN's; reproduced verbatim from
  src/model/loss.py lines 51-77): epsilon / 2 per adjacent neighbor regardless
  of position, so the per-neighbor smoothing rate stays invariant across
  interior and extreme true classes.

The two-row layout exposes the diagnostic difference. The interior row (true =
Major) shows that classical and per-neighbor-rate-constant agree exactly on
interior cases. The extreme row (true = Destroyed) shows the asymmetry the
MCDN scheme corrects: classical doubles the per-neighbor rate at extremes
(epsilon on the sole neighbor instead of epsilon / 2), biasing extreme-class
predictions toward their adjacent class. The per-neighbor-rate-constant scheme
keeps the rate at epsilon / 2.

The training configuration uses epsilon = 0.02 (per
config/presets/ablation_all_features.yaml, training.label_smoothing); the figure
displays a larger illustrative value so the smoothing-mass differences are
visually legible at print resolution. The schemes' relative shape is preserved
at any epsilon in [0, 1).

Usage::

    uv run python -m scripts.visualization.fig_label_smoothing
"""
from __future__ import annotations

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

# Illustrative smoothing parameter for visual legibility; the training
# value is 0.02 and the schemes' shape is preserved at any value in [0, 1).
EPSILON: float = 0.10
NUM_CLASSES: int = 4

# Two true classes selected to surface the diagnostic difference: one interior
# (where adjacency-aware schemes agree) and one extreme (where classical and
# per-neighbor-rate-constant diverge).
TRUE_CLASS_INTERIOR: int = 2  # Major
TRUE_CLASS_EXTREME: int = 3   # Destroyed
TRUE_CLASSES_TO_SHOW: tuple[int, ...] = (TRUE_CLASS_INTERIOR, TRUE_CLASS_EXTREME)

SCHEMES: tuple[str, ...] = ("hard", "uniform", "classical", "per_neighbor_constant")
SCHEME_LABELS: dict[str, str] = {
    "hard": "Hard one-hot",
    "uniform": r"Uniform" + "\n" + r"($\varepsilon$ over all $K{-}1$)",
    "classical": "Classical" + "\n" + "adj.-aware",
    "per_neighbor_constant": "Per-neighbor-rate" + "\n" + "constant (ours)",
}


def hard_target(true_class: int) -> np.ndarray:
    """Return the one-hot target distribution for ``true_class``."""

    q = np.zeros(NUM_CLASSES, dtype=np.float64)
    q[true_class] = 1.0
    return q


def uniform_smoothed_target(true_class: int, epsilon: float) -> np.ndarray:
    """Return the standard CE-style smoothed target.

    Mass 1 - epsilon stays on the true class; the remaining epsilon spreads
    uniformly across the other K - 1 classes.
    """

    q = np.full(NUM_CLASSES, epsilon / (NUM_CLASSES - 1), dtype=np.float64)
    q[true_class] = 1.0 - epsilon
    return q


def classical_adjacency_target(true_class: int, epsilon: float) -> np.ndarray:
    """Return the Diaz/Marathe 2019 adjacency-aware smoothed target.

    Interior classes receive epsilon / 2 on each of two adjacent neighbors;
    extreme classes (0 or K - 1) ship the full epsilon to their sole adjacent
    neighbor, doubling the per-neighbor smoothing rate at extremes.
    """

    q = np.zeros(NUM_CLASSES, dtype=np.float64)
    if true_class == 0:
        q[0] = 1.0 - epsilon
        q[1] = epsilon
    elif true_class == NUM_CLASSES - 1:
        q[NUM_CLASSES - 1] = 1.0 - epsilon
        q[NUM_CLASSES - 2] = epsilon
    else:
        q[true_class] = 1.0 - epsilon
        q[true_class - 1] = epsilon / 2.0
        q[true_class + 1] = epsilon / 2.0
    return q


def per_neighbor_constant_target(true_class: int, epsilon: float) -> np.ndarray:
    """Return the MCDN per-neighbor-rate-constant smoothed target.

    Reproduces src/model/loss.py lines 51-77. Each adjacent neighbor receives
    epsilon / 2 regardless of whether the true class is interior or extreme.
    Extreme classes therefore retain 1 - epsilon / 2 on self (versus
    1 - epsilon for interiors), but the per-neighbor diffusion rate is invariant.
    """

    q = np.zeros(NUM_CLASSES, dtype=np.float64)
    half = epsilon / 2.0
    if true_class == 0:
        q[0] = 1.0 - half
        q[1] = half
    elif true_class == NUM_CLASSES - 1:
        q[NUM_CLASSES - 1] = 1.0 - half
        q[NUM_CLASSES - 2] = half
    else:
        q[true_class] = 1.0 - epsilon
        q[true_class - 1] = half
        q[true_class + 1] = half
    return q


def compute_target(scheme: str, true_class: int, epsilon: float) -> np.ndarray:
    """Dispatch to the named scheme's target-distribution generator.

    Args:
        scheme: One of ``"hard"``, ``"uniform"``, ``"classical"``,
            ``"per_neighbor_constant"``.
        true_class: Index in ``[0, NUM_CLASSES - 1]``.
        epsilon: Smoothing parameter (ignored for ``"hard"``).

    Raises:
        ValueError: If ``scheme`` is not one of the recognized names.
    """

    if scheme == "hard":
        return hard_target(true_class=true_class)
    if scheme == "uniform":
        return uniform_smoothed_target(true_class=true_class, epsilon=epsilon)
    if scheme == "classical":
        return classical_adjacency_target(true_class=true_class, epsilon=epsilon)
    if scheme == "per_neighbor_constant":
        return per_neighbor_constant_target(true_class=true_class, epsilon=epsilon)
    raise ValueError(f"Unknown scheme: {scheme!r}")


def _draw_panel(ax: plt.Axes, q: np.ndarray, true_class: int) -> None:
    """Draw one 4-bar target-distribution panel with redundant non-color encoding.

    The true-class bar carries a hatch pattern in addition to its FEMA color so
    the panel remains parseable for color-vision-deficient readers and under
    black-and-white print reproduction. Numeric mass annotations sit above each
    bar at print-legible precision.
    """

    bars = ax.bar(
        ORDINAL_CLASS_NAMES,
        q,
        color=ORDINAL_CLASS_PALETTE,
        edgecolor="black",
        linewidth=0.5,
    )

    bars[true_class].set_hatch("///")

    for bar, value in zip(bars, q, strict=True):
        if value >= 0.005:
            text = f"{value:.2f}"
        elif value > 0:
            text = f"{value:.3f}"
        else:
            text = "0"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.025,
            text,
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


CAPTION_TITLE: str = (
    r"Label-smoothing target distributions at $\varepsilon = 0.10$ "
    r"(illustrative)."
)
CAPTION_BODY: str = (
    "Each panel shows the target distribution $q$ produced by one smoothing "
    "scheme for a chosen true class (hatched bar). **Top row:** true class is "
    "*Major* (interior); all adjacency-aware schemes agree because both "
    "ordinal neighbors are accessible. **Bottom row:** true class is *Destroyed* "
    "(extreme); the **classical** scheme [@diazSoftLabelsOrdinal2019] ships the "
    "full $\\varepsilon$ to the sole adjacent neighbor, doubling the "
    "per-neighbor smoothing rate relative to interior cases, while the "
    "MCDN's **per-neighbor-rate-constant** scheme retains $\\varepsilon"
    " / 2$ on the adjacent neighbor and keeps $1 - \\varepsilon / 2$ on self. "
    "Bars are colored by the FEMA-aligned damage-class palette (No Damage "
    "green, Minor orange, Major red, Destroyed purple) with a hatch overlay on "
    "the true-class bar so the figure remains parseable under "
    "color-vision-deficient and black-and-white reproduction. The displayed "
    "$\\varepsilon = 0.10$ is illustrative; training uses "
    "$\\varepsilon = 0.02$ "
    "(`config/presets/ablation_all_features.yaml::training.label_smoothing`) and "
    "the schemes' relative shape is preserved at any "
    "$\\varepsilon \\in [0, 1)$."
)


def main() -> None:
    """Render the F3 label-smoothing comparison figure and its sibling caption."""

    setup_publication_style()

    n_rows = len(TRUE_CLASSES_TO_SHOW)
    n_cols = len(SCHEMES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(WIDTH_2COL, 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row_idx, true_class in enumerate(TRUE_CLASSES_TO_SHOW):
        is_extreme = true_class == 0 or true_class == NUM_CLASSES - 1
        position_label = "extreme" if is_extreme else "interior"
        for col_idx, scheme in enumerate(SCHEMES):
            ax = axes[row_idx, col_idx]
            q = compute_target(scheme=scheme, true_class=true_class, epsilon=EPSILON)
            _draw_panel(ax=ax, q=q, true_class=true_class)
            if row_idx == 0:
                ax.set_title(SCHEME_LABELS[scheme], fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(
                    f"True: {ORDINAL_CLASS_NAMES[true_class]}\n({position_label})",
                    fontsize=8,
                )

    save_figure(fig=fig, name="label_smoothing")
    save_caption(name="label_smoothing", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
