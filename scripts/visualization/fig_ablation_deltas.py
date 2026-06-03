"""R2: ablation-delta signature plot across the three single-knob ablations.

Compresses the three per-ablation tables in [doc/5-results.qmd](doc/5-results.qmd)
into a single 2 x 3 grid: rows are the two headline metrics (delta F1, delta
QWK) and columns are the three single-knob ablations (mask, typology, sensor
modality). The mask and typology columns each render three bars (Spatial
Block East, Hurricane Michael, Mayfield Tornado); the sensor column renders
two bars centered in the panel (Spatial Block East, Hurricane Michael)
because the manned-aircraft DROIDs subset contains zero orthomosaics from
the Mayfield Tornado event - that gap is documented in the chapter caption
rather than rendered as a placeholder slot in the figure.

Visual conventions:
    - Bars are colored by sign: a muted ColorBrewer-RdBu red for negative
      deltas, a muted blue for positive deltas. The signs encode the chapter's
      central claim - "geometric prior helps on hurricane data, fails on
      tornado data; typology contributes a small distant-OOD QWK lift only;
      sensor modality is a different scale entirely" - and color reinforces
      direction at a glance.
    - Per-bar whiskers show the per-seed argmax SD (sigma across the 10-seed
      pool's best-checkpoint argmax F1 / QWK) on the ablation arm, drawn in
      neutral dark grey. A delta whose magnitude lies inside its whisker is
      indistinguishable from seed-only initialization variance.
    - Per-bar value labels are placed OUTSIDE the bar at the whisker tip
      (above the whisker for positive bars, below for negative), in plain
      neutral dark text without bbox or special styling. Per-column shared
      y-axis (F1 and QWK comparable within an ablation), per-row independent
      y-axis (the sensor column's order-of-magnitude greater deltas are
      encoded in the axis numbers rather than in the bar heights). A 20
      percent y-margin is applied per column to ensure outside labels sit
      clear of the panel frame without any per-bar overflow detection or
      inside/outside flipping.

Source artifacts: `outputs/ablation/{baseline, mask, typology, resolution}/<split>/
ensemble_metrics.json` (delta vs baseline) and `aggregate_metrics.json` (per-
seed SD on the ablation arm). All deltas are computed at the canonical argmax
ensemble decoding rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualization._common import (
    WIDTH_2COL,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_ABLATION_ROOT: Final[Path] = _REPO_ROOT / "outputs" / "ablation"

# Column ordering: mask -> typology -> sensor matches the chapter's section
# ordering (Geometric Prior -> Contextual Prior -> Sensor Modality).
_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("mask", "Mask ablation"),
    ("typology", "Typology ablation"),
    ("resolution", "Sensor modality"),
)

# Split ordering matches the chapter's evaluation funnel (in-distribution ->
# proximate OOD -> distant OOD) and the R1 confusion triptych panel order so
# the two figures read consistently. The sensor column drops Mayfield because
# the manned-aircraft DROIDs subset has zero Mayfield Tornado coverage.
_SPLITS_FULL: Final[tuple[tuple[str, str], ...]] = (
    ("Spatial_Block_East", "E/W"),
    ("Hurricane_Michael", "Michael"),
    ("Mayfield_Tornado", "Mayfield"),
)
_SPLITS_NO_MAYFIELD: Final[tuple[tuple[str, str], ...]] = _SPLITS_FULL[:2]
# Shared x-axis range across all columns so bar visual widths read consistently
# regardless of how many bars a column contains. Sensor's two bars are
# centered within this range; mask and typology fill it.
_X_LIM: Final[tuple[float, float]] = (-0.5, 2.5)

# ColorBrewer RdBu (muted) for sign-based encoding. Negative bars (loss) use
# a desaturated red; positive bars (gain) use a desaturated blue. Both are
# colorblind-discriminable and print acceptably in greyscale (red darker
# than blue under standard luminance mapping).
_COLOR_NEGATIVE: Final[str] = "#d6604d"
_COLOR_POSITIVE: Final[str] = "#4393c3"
_COLOR_WHISKER: Final[str] = "#2c2c2c"
# Y-axis padding (fraction of auto-determined yrange) applied via ax.margins.
# 20% on each side comfortably accommodates outside-bar value labels at the
# whisker tip plus a 3 pt offset, across all six subplots - eliminating the
# need for per-bar overflow detection.
_Y_MARGIN: Final[float] = 0.20


def _read_ensemble_argmax(arm: str, split: str) -> dict[str, float] | None:
    """Read ensemble argmax F1/QWK for one (arm, split). Returns None if missing."""

    path = _ABLATION_ROOT / arm / split / "ensemble_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    block = payload["ensemble"]["argmax"]
    return {"macro_f1": float(block["macro_f1"]), "qwk": float(block["qwk"])}


def _read_per_seed_sds(arm: str, split: str) -> dict[str, float] | None:
    """Read per-seed argmax F1/QWK SDs for one (arm, split). Returns None if missing."""

    path = _ABLATION_ROOT / arm / split / "aggregate_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["per_seed"]
    f1 = np.asarray([r["by_rule"]["argmax"]["best"]["macro_f1"] for r in rows])
    qwk = np.asarray([r["by_rule"]["argmax"]["best_val_qwk"] for r in rows])
    return {
        "f1_sd": float(np.std(f1, ddof=1)),
        "qwk_sd": float(np.std(qwk, ddof=1)),
    }


_LABEL_FONTSIZE: Final[float] = 6.8
_LABEL_OFFSET_PTS: Final[float] = 3.0


def _draw_panel(
    ax: plt.Axes,
    *,
    x_positions: np.ndarray,
    deltas: list[float],
    sds: list[float],
    split_labels: list[str],
    y_label: str | None,
    show_x_labels: bool,
    is_top_row: bool,
    column_title: str | None,
) -> None:
    """Render one ablation-delta subplot: bars + whiskers + outside value labels."""

    bar_colors = [
        _COLOR_NEGATIVE if d < 0 else _COLOR_POSITIVE for d in deltas
    ]

    ax.bar(
        x_positions,
        deltas,
        width=0.62,
        color=bar_colors,
        edgecolor="#444444",
        linewidth=0.6,
        zorder=2,
    )

    for x, delta, sd in zip(x_positions, deltas, sds):
        ax.errorbar(
            x,
            delta,
            yerr=sd,
            fmt="none",
            ecolor=_COLOR_WHISKER,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )

    ax.axhline(0.0, color="black", linewidth=0.7, zorder=3)

    # Outside value labels at whisker tip + 3 pt offset. Placement is purely
    # rule-based (above whisker for positive bars, below for negative); the
    # 20% y-margin applied in main() ensures every label fits inside the
    # panel frame without per-bar overflow detection.
    for x, delta, sd in zip(x_positions, deltas, sds):
        if delta >= 0:
            anchor_y = delta + sd
            offset_pts = _LABEL_OFFSET_PTS
            va = "bottom"
        else:
            anchor_y = delta - sd
            offset_pts = -_LABEL_OFFSET_PTS
            va = "top"
        ax.annotate(
            f"{delta:+.4f}",
            xy=(x, anchor_y),
            xytext=(0, offset_pts),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=_LABEL_FONTSIZE,
            color="#333333",
        )

    ax.set_xlim(*_X_LIM)
    ax.set_xticks(x_positions)
    if show_x_labels:
        ax.set_xticklabels(split_labels, fontsize=8)
    else:
        ax.set_xticklabels([])

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=9)

    if is_top_row and column_title is not None:
        ax.set_title(column_title, fontsize=9, pad=4)

    ax.tick_params(axis="both", which="both", labelsize=7, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _collect_column_data(
    arm: str, splits: tuple[tuple[str, str], ...]
) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    """For one ablation arm, gather delta-F1/QWK and per-seed SDs across the given splits.

    Returns five parallel lists: delta-F1, delta-QWK, SD-F1, SD-QWK, and the
    display-label tuple. Splits with missing artifacts are silently dropped
    with a warning print so the caller's downstream rendering remains
    monotone in length.
    """

    delta_f1: list[float] = []
    delta_qwk: list[float] = []
    sd_f1: list[float] = []
    sd_qwk: list[float] = []
    labels: list[str] = []

    for split, label in splits:
        baseline = _read_ensemble_argmax("baseline", split)
        ablation = _read_ensemble_argmax(arm, split)
        sds = _read_per_seed_sds(arm, split)

        if baseline is None or ablation is None or sds is None:
            print(
                f"[fig_ablation_deltas] Dropping ({arm}, {split}): "
                f"one or more required artifacts are missing."
            )
            continue

        delta_f1.append(ablation["macro_f1"] - baseline["macro_f1"])
        delta_qwk.append(ablation["qwk"] - baseline["qwk"])
        sd_f1.append(sds["f1_sd"])
        sd_qwk.append(sds["qwk_sd"])
        labels.append(label)

    return delta_f1, delta_qwk, sd_f1, sd_qwk, labels


def _x_positions_for(n: int) -> np.ndarray:
    """Return bar x-positions centered within ``_X_LIM``."""

    if n <= 0:
        return np.empty(0, dtype=float)
    midpoint = 0.5 * (_X_LIM[0] + _X_LIM[1])
    if n == 1:
        return np.array([midpoint], dtype=float)
    half = (n - 1) / 2.0
    return midpoint + np.arange(-half, half + 0.5, 1.0)


def main() -> None:
    """Render the 2 x 3 ablation-delta signature plot."""

    setup_publication_style()

    column_data = []
    for arm, arm_label in _COLUMNS:
        splits = _SPLITS_NO_MAYFIELD if arm == "resolution" else _SPLITS_FULL
        delta_f1, delta_qwk, sd_f1, sd_qwk, labels = _collect_column_data(arm, splits)
        x_positions = _x_positions_for(len(labels))
        column_data.append({
            "title": arm_label,
            "x_positions": x_positions,
            "labels": labels,
            "delta_f1": delta_f1,
            "delta_qwk": delta_qwk,
            "sd_f1": sd_f1,
            "sd_qwk": sd_qwk,
        })

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(WIDTH_2COL, 4.0),
        sharey="col",
        gridspec_kw={
            "wspace": 0.18,
            "hspace": 0.22,
            "left": 0.085,
            "right": 0.985,
            "top": 0.92,
            "bottom": 0.10,
        },
    )

    # Single-pass render: bars + whiskers + outside labels per panel. The
    # 20% y-margin applied below guarantees outside labels fit within the
    # panel frame without per-bar overflow detection.
    for col_idx, col in enumerate(column_data):
        _draw_panel(
            axes[0, col_idx],
            x_positions=col["x_positions"],
            deltas=col["delta_f1"],
            sds=col["sd_f1"],
            split_labels=col["labels"],
            y_label="$\\Delta$ F1" if col_idx == 0 else None,
            show_x_labels=False,
            is_top_row=True,
            column_title=col["title"],
        )
        _draw_panel(
            axes[1, col_idx],
            x_positions=col["x_positions"],
            deltas=col["delta_qwk"],
            sds=col["sd_qwk"],
            split_labels=col["labels"],
            y_label="$\\Delta$ QWK" if col_idx == 0 else None,
            show_x_labels=True,
            is_top_row=False,
            column_title=None,
        )

    # Apply y-margin per column (sharey="col" propagates through the shared
    # axis so applying once per column suffices). 20% padding on each side
    # of the auto-determined data range is enough room for the outside-bar
    # value labels without further geometry work.
    for col_idx in range(3):
        axes[0, col_idx].margins(y=_Y_MARGIN)

    save_figure(fig=fig, name="ablation_deltas")

    save_caption(
        name="ablation_deltas",
        title=(
            "Ablation deltas (vs. canonical baseline) across the three "
            "single-knob ablations and the evaluation funnel."
        ),
        body=(
            "Rows are the two headline metrics (delta Macro-F1, delta QWK); "
            "columns are the three ablations (mask removal, typology removal, "
            "sensor-modality swap from sUAS to manned-aircraft imagery). The "
            "mask and typology columns each render three bars (Spatial Block "
            "East, Hurricane Michael, Mayfield Tornado); the sensor column "
            "renders two bars centered in the panel (Spatial Block East, "
            "Hurricane Michael) because the manned-aircraft DROIDs subset "
            "contains zero orthomosaics from the Mayfield Tornado event. "
            "Bar color encodes the sign of the delta (red = ablation worsens "
            "the metric, blue = ablation improves it). Per-bar whiskers show "
            "the per-seed argmax SD on the ablation arm (sigma across the "
            "10-seed pool's best-checkpoint argmax F1 / QWK); a delta whose "
            "magnitude lies inside its whisker is indistinguishable from "
            "seed-only initialization variance. Per-bar value labels are "
            "placed outside the bar at the whisker tip (above for positive "
            "bars, below for negative). Per-column shared y-axis (F1 and QWK "
            "comparable within an ablation), per-row independent y-axis (the "
            "sensor column's order-of-magnitude greater deltas are encoded "
            "in the axis numbers rather than in the bar heights). A 20 "
            "percent y-margin is applied per column to ensure outside labels "
            "sit clear of the panel frame. Source: "
            "ensemble_metrics.json for delta computation, aggregate_metrics.json "
            "for per-seed SD."
        ),
    )


if __name__ == "__main__":
    main()
