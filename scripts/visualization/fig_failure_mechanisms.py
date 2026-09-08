"""R6: per-column failure-mechanism breakdown of the full MCDN configuration (T-9).

Summarizes the error inventories written by ``scripts/mine_errors.py`` for the
four reported evaluation columns into two stacked single-column panels (unlettered;
the caption refers to them as top and bottom):

    Top: Error composition by ordinal direction and severity. Each column's
        errors are partitioned into severe under-grades (predicted two or more
        grades below the label), adjacent under-grades (one grade below),
        adjacent over-grades (one grade above), and severe over-grades, and
        drawn as a 100 % stacked horizontal bar; the column's error count and
        error rate are annotated at the bar end.
    Bottom: Bias versus variance. For each column, the share of errors on which
        all ten seeds agree (the ensemble is confidently and systematically
        wrong) is shown for all errors and for severe errors alone. High
        unanimity marks a column whose errors ensembling cannot repair; low
        unanimity marks errors that averaging across seeds does repair, which
        is where the ensemble lift concentrates (T-7).

Visual conventions:
    - under-grades in the muted ColorBrewer RdBu red, over-grades in the muted
      blue (the same sign palette as R2); severe segments are the saturated
      tone and carry a hatch as redundant non-color encoding, adjacent
      segments the lighter tone.
    - The bottom panel uses neutral greys so it reads as a different quantity
      from the top panel.

Source artifacts: ``outputs/error_mining/all_features__<split>/inventory.json``
(``summary.transitions_true_x_pred``, ``summary.n_errors``,
``summary.n_instances``, ``summary.errors_unanimous_across_seeds``,
``summary.n_severe``, ``summary.severe_unanimous_across_seeds``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch

from scripts.visualization._common import FIG_ANNOT_PT, FIG_FONT_PT, WIDTH_1COL, save_caption, save_figure, setup_publication_style

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_MINING_DIR: Final[Path] = _REPO_ROOT / "outputs" / "error_mining"
_VARIANT: Final[str] = "all_features"

# Reporting order per D-20: the DROIDs default split, then the LOEO holdouts. The
# first element is the error-mining output directory suffix (the default-split
# run was written under the short name ``default_split``).
_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("default_split", "DROIDs default"),
    ("Hurricane_Michael", "LOEO Michael"),
    ("Mayfield_Tornado", "LOEO Mayfield"),
    ("Hurricane_Ida", "LOEO Ida")
)

# Segment order left-to-right in the top panel: most severe under-grade first so the
# under-grade mass sits on the left and the over-grade mass on the right.
_SEGMENTS: Final[tuple[tuple[str, str, str, str | None], ...]] = (
    ("severe_under", "Severe under-grade (≥ 2 grades)", "#b2182b", "////"),
    ("adjacent_under", "Adjacent under-grade", "#ef8a62", None),
    ("adjacent_over", "Adjacent over-grade", "#67a9cf", None),
    ("severe_over", "Severe over-grade (≥ 2 grades)", "#2166ac", "////")
)
_UNANIMOUS_ALL_COLOR: Final[str] = "#4d4d4d"
_UNANIMOUS_SEVERE_COLOR: Final[str] = "#bababa"
_BAR_HEIGHT: Final[float] = 0.62


class ColumnErrors(NamedTuple):
    """Error-mass decomposition and seed-unanimity statistics for one evaluation column."""

    n_instances: int
    n_errors: int
    n_severe: int
    shares: dict[str, float]
    unanimous_all: float
    unanimous_severe: float


def _load_column(split: str) -> ColumnErrors:
    """Read one column's inventory and decompose its error mass by direction and severity."""

    path = _MINING_DIR / f"{_VARIANT}__{split}" / "inventory.json"
    if not path.exists():
        raise FileNotFoundError(f"Error inventory not found: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]

    transitions = np.asarray(summary["transitions_true_x_pred"], dtype=np.int64)  # [true, pred]
    true_idx, pred_idx = np.indices(transitions.shape)
    step = pred_idx - true_idx  # negative = under-grade
    counts = {
        "severe_under": int(transitions[step <= -2].sum()),
        "adjacent_under": int(transitions[step == -1].sum()),
        "adjacent_over": int(transitions[step == 1].sum()),
        "severe_over": int(transitions[step >= 2].sum())
    }
    n_errors = int(summary["n_errors"])
    if sum(counts.values()) != n_errors:
        raise ValueError(f"Transition mass {sum(counts.values())} does not match n_errors {n_errors} for {split}.")

    n_severe = int(summary["n_severe"])
    return ColumnErrors(
        n_instances=int(summary["n_instances"]), n_errors=n_errors, n_severe=n_severe,
        shares={key: value / n_errors for key, value in counts.items()},
        unanimous_all=int(summary["errors_unanimous_across_seeds"]) / n_errors,
        unanimous_severe=int(summary["severe_unanimous_across_seeds"]) / n_severe if n_severe else 0.0
    )


def _draw_composition(ax: plt.Axes, columns: list[tuple[str, ColumnErrors]]) -> None:
    """Render the 100 % stacked error-composition bars, one row per column."""

    y_positions = np.arange(len(columns))[::-1]
    for y, (_label, errors) in zip(y_positions, columns, strict=True):
        left = 0.0
        for key, _legend, color, hatch in _SEGMENTS:
            share = errors.shares[key]
            ax.barh(y, share, left=left, height=_BAR_HEIGHT, color=color, hatch=hatch, edgecolor="white", linewidth=0.6, zorder=3)
            if share >= 0.10:
                ax.text(left + share / 2, y, f"{100 * share:.0f}", ha="center", va="center", fontsize=FIG_ANNOT_PT,
                        color="white", fontweight="medium", zorder=4)
            left += share
        ax.text(1.01, y, f"n = {errors.n_errors}\n({100 * errors.n_errors / errors.n_instances:.1f} %)",
                ha="left", va="center", fontsize=FIG_ANNOT_PT, color="#333333", transform=ax.get_yaxis_transform())

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in columns], fontsize=FIG_FONT_PT)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=FIG_FONT_PT)
    ax.set_xlabel("Share of errors (%)", fontsize=FIG_FONT_PT)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _draw_unanimity(ax: plt.Axes, columns: list[tuple[str, ColumnErrors]]) -> None:
    """Render seed-unanimous shares for all errors and for severe errors, one row per column."""

    y_positions = np.arange(len(columns))[::-1]
    offset = _BAR_HEIGHT / 4
    for y, (_label, errors) in zip(y_positions, columns, strict=True):
        ax.barh(y + offset, errors.unanimous_all, height=_BAR_HEIGHT / 2, color=_UNANIMOUS_ALL_COLOR, zorder=3)
        ax.barh(y - offset, errors.unanimous_severe, height=_BAR_HEIGHT / 2, color=_UNANIMOUS_SEVERE_COLOR, zorder=3)
        ax.text(errors.unanimous_all + 0.01, y + offset, f"{100 * errors.unanimous_all:.0f}", va="center", fontsize=FIG_ANNOT_PT,
                color="#333333")
        ax.text(errors.unanimous_severe + 0.01, y - offset, f"{100 * errors.unanimous_severe:.0f} ({errors.n_severe})",
                va="center", fontsize=FIG_ANNOT_PT, color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in columns], fontsize=FIG_FONT_PT)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=FIG_FONT_PT)
    ax.set_xlabel("Errors on which all 10 seeds agree (%)", fontsize=FIG_FONT_PT)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def main() -> None:
    """Render the failure-mechanism figure and its caption."""

    setup_publication_style()
    columns = [(label, _load_column(split)) for split, label in _COLUMNS]

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(WIDTH_1COL, 4.2),
        gridspec_kw={"left": 0.27, "right": 0.84, "top": 0.88, "bottom": 0.10, "hspace": 0.65}
    )
    _draw_composition(axes[0], columns)
    _draw_unanimity(axes[1], columns)

    # No panel letters: the two panels are distinguished by their x-axis labels and
    # are referred to as top / bottom in the caption. The composition legend sits
    # above the top panel; the unanimity legend sits in the gap above the bottom one.
    composition_handles = [Patch(facecolor=color, hatch=hatch, edgecolor="white", label=legend) for _, legend, color, hatch in _SEGMENTS]
    fig.legend(handles=composition_handles, loc="upper center", ncol=2, frameon=False, fontsize=FIG_ANNOT_PT,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.6, columnspacing=1.0)
    unanimity_handles = [Patch(facecolor=_UNANIMOUS_ALL_COLOR, label="All errors"),
                         Patch(facecolor=_UNANIMOUS_SEVERE_COLOR, label="Severe errors (count)")]
    axes[1].legend(handles=unanimity_handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=False,
                   fontsize=FIG_ANNOT_PT, handlelength=1.6, borderaxespad=0.0, labelspacing=0.2)

    save_figure(fig=fig, name="failure_mechanisms")

    save_caption(
        name="failure_mechanisms",
        title="Failure mechanisms of the full MCDN configuration by evaluation column.",
        body=(
            "**Top:** composition of the 10-seed ensemble's errors (argmax rule) by ordinal direction and severity: "
            "severe under-grades (prediction two or more damage grades below the label), adjacent under-grades, adjacent "
            "over-grades, and severe over-grades, as a share of each column's errors; the error count and error rate are "
            "given at the right. **Bottom:** share of errors on which all ten seeds agree, for all errors and for severe "
            "errors alone (severe count in parentheses). The DROIDs default split is the train/test partition published "
            "with the dataset; the LOEO columns hold out one event each. The hurricane columns fail predominantly by "
            "confident, seed-unanimous under-grading — on Ida almost entirely Minor-to-No Damage, on Michael "
            "Major-to-Minor — which averaging across seeds cannot repair; the tornado holdout fails by uncertainty, with "
            "balanced under- and over-grades and few unanimous errors, and is correspondingly the column where "
            "ensembling and test-time augmentation help most. Source: "
            "``outputs/error_mining/all_features__<split>/inventory.json``."
        )
    )


if __name__ == "__main__":
    main()
