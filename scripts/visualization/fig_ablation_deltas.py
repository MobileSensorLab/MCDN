"""R2: ablation deltas of every arm against the full MCDN configuration, on the four reported columns.

Renders the component factorial and the two training-recipe arms as a pair of
horizontal grouped-bar panels (delta QWK on the left, delta Macro-F1 on the
right). Each row is one ablation arm; within a row the four bars are the four
reported evaluation columns (DROIDs default split, LOEO Michael, LOEO
Mayfield, LOEO Ida). Deltas are the arm's 10-seed ensemble argmax metric minus
the ``all_features`` ensemble's on the same column; whiskers show the per-seed
argmax SD on the ablation arm, so a delta inside its whisker is
indistinguishable from initialization variance.

The factorial rows are labeled by the mask-derived components the arm
*retains* (C = mask input channel, P = mask-weighted pooling, T = typology
FiLM); the two training arms swap the loss (CE for EMD) and remove label
smoothing while retaining all three components. The ``C + T`` cell (channel
and typology without pooling) was not trained and is absent from the grid.

Source artifacts: ``outputs/ablation_dgx/_ensembles/<arm>__<split>.json``.
The ``mask`` and ``typology`` arms were trained on the DGX for the default
split and LOEO Ida only; their LOEO Michael and Mayfield ensembles are the
v1 local runs re-summarized under ``mask_local`` / ``typology_local`` with
identical schema. Deltas are computed at the reported argmax decoding rule.
"""
from __future__ import annotations

from typing import Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch

from scripts.visualization._common import (
    REPORTED_COLUMNS,
    WIDTH_2COL,
    ensemble_rule_metrics,
    ensemble_summary_path,
    per_seed_rule_metrics,
    save_caption,
    save_figure,
    setup_publication_style,
)


class Arm(NamedTuple):
    """One ablation arm: variant directory name, display label, and its group."""

    variant: str
    label: str
    group: str


# Row order: the component factorial from most to least retained, then the training arms.
_ARMS: Final[tuple[Arm, ...]] = (
    Arm("typology", "C + P  (no typology)", "factorial"),
    Arm("pooling_typology", "P + T  (no channel)", "factorial"),
    Arm("mask_channel_only", "C only", "factorial"),
    Arm("pooling_only", "P only", "factorial"),
    Arm("mask", "T only", "factorial"),
    Arm("rgb_only", "RGB only", "factorial"),
    Arm("ce_loss", "CE loss (for EMD)", "training"),
    Arm("no_smoothing", "No label smoothing", "training")
)

# Arms whose LOEO Michael / Mayfield ensembles come from the v1 local lineage.
_LOCAL_FALLBACK: Final[dict[str, str]] = {"mask": "mask_local", "typology": "typology_local"}

# Colorblind-safe column palette (Okabe-Ito), one hue per evaluation column.
_COLUMN_COLORS: Final[tuple[str, ...]] = ("#000000", "#0072B2", "#E69F00", "#CC79A7")
_WHISKER_COLOR: Final[str] = "#555555"
_BAR_HEIGHT: Final[float] = 0.19
_GROUP_GAP: Final[float] = 0.6


def _resolve_variant(variant: str, split: str) -> str | None:
    """Return the ensemble variant name that holds ``(variant, split)``, or None if neither lineage has it."""

    if ensemble_summary_path(split, variant).exists():
        return variant
    fallback = _LOCAL_FALLBACK.get(variant)
    if fallback is not None and ensemble_summary_path(split, fallback).exists():
        return fallback
    return None


def _collect(metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[n_arms, n_columns]`` arrays of ensemble deltas and per-seed SDs for one metric (NaN where absent)."""

    deltas = np.full((len(_ARMS), len(REPORTED_COLUMNS)), np.nan)
    sds = np.full_like(deltas, np.nan)
    for col_idx, (split, _label) in enumerate(REPORTED_COLUMNS):
        reference = float(ensemble_rule_metrics(split, rule="argmax")[metric])
        for arm_idx, arm in enumerate(_ARMS):
            resolved = _resolve_variant(arm.variant, split)
            if resolved is None:
                print(f"[fig_ablation_deltas] Missing ensemble for ({arm.variant}, {split}); leaving the slot empty.")
                continue
            deltas[arm_idx, col_idx] = float(ensemble_rule_metrics(split, rule="argmax", variant=resolved)[metric]) - reference
            per_seed = np.asarray([entry[metric] for entry in per_seed_rule_metrics(split, rule="argmax", variant=resolved)])
            sds[arm_idx, col_idx] = float(per_seed.std(ddof=1))
    return deltas, sds


def _row_centers() -> np.ndarray:
    """Return the y-center of each arm row, with an extra gap between the factorial and training groups."""

    centers = []
    y = 0.0
    previous_group = _ARMS[0].group
    for arm in _ARMS:
        if arm.group != previous_group:
            y += _GROUP_GAP
            previous_group = arm.group
        centers.append(y)
        y += 1.0
    return -np.asarray(centers)  # top row first


def _draw_panel(ax: plt.Axes, *, deltas: np.ndarray, sds: np.ndarray, x_label: str, show_row_labels: bool) -> None:
    """Render one metric's grouped horizontal bars with per-seed whiskers."""

    centers = _row_centers()
    n_cols = len(REPORTED_COLUMNS)
    offsets = (np.arange(n_cols) - (n_cols - 1) / 2.0) * _BAR_HEIGHT
    for col_idx, color in enumerate(_COLUMN_COLORS):
        y = centers - offsets[col_idx]
        values = deltas[:, col_idx]
        present = ~np.isnan(values)
        ax.barh(y[present], values[present], height=_BAR_HEIGHT * 0.92, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(values[present], y[present], xerr=sds[present, col_idx], fmt="none", ecolor=_WHISKER_COLOR,
                    elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=4)

    ax.axvline(0.0, color="black", linewidth=0.8, zorder=2)
    # Separator between the factorial and the training-recipe arms.
    boundary = next(i for i, arm in enumerate(_ARMS) if arm.group == "training")
    ax.axhline((centers[boundary - 1] + centers[boundary]) / 2.0, color="#999999", linewidth=0.6, linestyle=":", zorder=1)

    ax.set_yticks(centers)
    ax.set_yticklabels([arm.label for arm in _ARMS] if show_row_labels else [], fontsize=8)
    ax.set_ylim(centers[-1] - 0.6, centers[0] + 0.6)
    ax.set_xlabel(x_label, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.5, length=2)
    ax.spines["left"].set_visible(False)
    ax.margins(x=0.12)


def main() -> None:
    """Render the ablation-delta figure and its caption."""

    setup_publication_style()

    delta_qwk, sd_qwk = _collect("qwk")
    delta_f1, sd_f1 = _collect("macro_f1")

    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH_2COL, 3.6),
        gridspec_kw={"wspace": 0.12, "left": 0.24, "right": 0.985, "top": 0.86, "bottom": 0.13}
    )
    _draw_panel(axes[0], deltas=delta_qwk, sds=sd_qwk, x_label="$\\Delta$ QWK vs. full configuration", show_row_labels=True)
    _draw_panel(axes[1], deltas=delta_f1, sds=sd_f1, x_label="$\\Delta$ Macro-F1 vs. full configuration", show_row_labels=False)

    handles = [Patch(facecolor=color, label=label) for color, (_split, label) in zip(_COLUMN_COLORS, REPORTED_COLUMNS, strict=True)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.6, 0.97),
               handlelength=1.4, columnspacing=1.2)

    save_figure(fig=fig, name="ablation_deltas")

    def _fmt(values: np.ndarray) -> str:
        return " / ".join("n/a" if np.isnan(v) else f"{v:+.3f}" for v in values)

    rows = "; ".join(f"{arm.label}: QWK {_fmt(delta_qwk[i])}, F1 {_fmt(delta_f1[i])}" for i, arm in enumerate(_ARMS))
    save_caption(
        name="ablation_deltas",
        title="Ablation deltas against the full MCDN configuration across the four reported evaluation columns.",
        body=(
            "Each row is one ablation arm; bars are the arm's 10-seed ensemble argmax metric minus the full "
            "configuration's on the same column (left: QWK, right: Macro-F1), colored by evaluation column (DROIDs "
            "default split, LOEO Michael, LOEO Mayfield, LOEO Ida). Whiskers show the per-seed argmax SD on the "
            "ablation arm; a delta inside its whisker is indistinguishable from initialization variance. Factorial "
            "rows are labeled by the mask-derived components retained (C = mask input channel, P = mask-weighted "
            "pooling, T = typology FiLM); the two rows below the dotted separator retain all three components and "
            "change the training recipe (cross-entropy in place of the EMD loss; label smoothing removed). "
            f"Values in column order (default / Michael / Mayfield / Ida) - {rows}. "
            "Source: `outputs/ablation_dgx/_ensembles/<arm>__<split>.json`; the `mask` and `typology` arms' LOEO "
            "Michael and Mayfield ensembles are the v1 local runs (`mask_local`, `typology_local`)."
        ),
    )


if __name__ == "__main__":
    main()
