"""Companion to R1: precision-normalized confusion-matrix panels.

Sibling figure to ``fig_confusion_triptych`` that renders the same four
ensemble-argmax confusion matrices (DROIDs default split, LOEO Michael,
Mayfield, Ida) but column-normalized rather than row-normalized. Where the
recall view (``confusion_triptych``) answers "given truth = X, what was
predicted?", this view answers "given prediction = X, what was the truth?".
The diagonal of each column equals the per-class **precision**.

The figure surfaces a claim that the recall view does not visualize: the
distant-OOD Mayfield Tornado holdout exhibits a **Destroyed precision
shortfall** driven by Major-as-Destroyed false positives concentrating in the
Destroyed column, while the hurricane holdouts' under-calling shows up as
impure No Damage / Minor columns instead. The caption reports the per-column
Destroyed precision values computed at render time.

Visual conventions are deliberately identical to the recall view so the
two figures read as a paired recall-precision view:
    - Cells colored by per-column percent under a shared sequential ``Blues``
      colormap, 0 to 100 percent across all panels for direct comparability.
    - Cell text shows count above per-column percent; text color flips white
      above 50 percent for contrast.
    - Diagonal cells outlined in solid black to emphasize correct predictions.
    - Per-panel title carries the column label plus ensemble Macro-F1 and QWK.

Source artifacts: ``outputs/ablation/_ensembles/all_features__<split>.json``
-> ``cross_variant.equal_seed_metrics.argmax.confusion_matrix``. Same JSON the
recall view reads.
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts.visualization._common import (
    ORDINAL_CLASS_NAMES,
    REPORTED_COLUMNS,
    WIDTH_2COL,
    ensemble_rule_metrics,
    save_caption,
    save_figure,
    setup_publication_style,
)

_SHORT_LABELS: Final[tuple[str, ...]] = ("None", "Minor", "Major", "Destr.")


def _load_panel(split_dir: str) -> dict[str, object]:
    """Read the ensemble argmax block for one split."""

    argmax = ensemble_rule_metrics(split_dir, rule="argmax")
    return {
        "confusion_matrix": np.asarray(argmax["confusion_matrix"], dtype=np.int64),
        "macro_f1": float(argmax["macro_f1"]),
        "qwk": float(argmax["qwk"]),
    }

def _draw_panel(
    ax: plt.Axes,
    *,
    cm_counts: np.ndarray,
    title: str,
    show_y_labels: bool,
    cmap: object,
    vmin: float,
    vmax: float,
) -> object:
    """Render a single column-normalized confusion-matrix panel onto ``ax``."""

    col_totals = cm_counts.sum(axis=0, keepdims=True)
    col_totals_safe = np.where(col_totals == 0, 1, col_totals)
    cm_pct = cm_counts / col_totals_safe * 100.0

    image = ax.imshow(
        cm_pct,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    n_classes = cm_counts.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(cm_counts[i, j])
            pct = float(cm_pct[i, j])
            text_color = "white" if pct > 50.0 else "black"
            label = f"{count}\n{pct:.1f}%" if count > 0 else "0"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
                linespacing=1.0,
            )

    for k in range(n_classes):
        ax.add_patch(
            Rectangle(
                (k - 0.5, k - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(_SHORT_LABELS, fontsize=7)
    ax.set_yticks(range(n_classes))
    if show_y_labels:
        ax.set_yticklabels(ORDINAL_CLASS_NAMES, fontsize=8)
        ax.set_ylabel("True class")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted class")
    ax.set_title(title, fontsize=8, pad=5)

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def main() -> None:
    """Render the precision (column-normalized) confusion-matrix panels across the four reported columns."""

    setup_publication_style()

    panels = [(_load_panel(d), label) for d, label in REPORTED_COLUMNS]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(WIDTH_2COL, 2.35),
        gridspec_kw={"wspace": 0.16, "left": 0.085, "right": 0.915, "top": 0.85, "bottom": 0.2},
    )

    cmap = plt.get_cmap("Blues")
    image = None
    for idx, ((data, label), ax) in enumerate(zip(panels, axes, strict=True)):
        cm_counts = data["confusion_matrix"]
        title = (
            f"{label}\n"
            f"F1 {data['macro_f1']:.3f}  QWK {data['qwk']:.3f}"
        )
        image = _draw_panel(
            ax,
            cm_counts=cm_counts,
            title=title,
            show_y_labels=(idx == 0),
            cmap=cmap,
            vmin=0.0,
            vmax=100.0,
        )

    cbar_ax = fig.add_axes([0.93, 0.24, 0.01, 0.55])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Column-normalized %", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)

    save_figure(fig=fig, name="precision_triptych")

    destroyed_notes = []
    for data, label in panels:
        cm = data["confusion_matrix"]
        predicted_destroyed = int(cm[:, 3].sum())
        precision = cm[3, 3] / predicted_destroyed if predicted_destroyed else float("nan")
        destroyed_notes.append(f"{label} {precision:.3f} ({int(cm[3, 3])} of {predicted_destroyed}, {int(cm[2, 3])} from true Major)")

    save_caption(
        name="precision_triptych",
        title=(
            "Ensemble argmax precision (column-normalized) confusion matrices of the full MCDN configuration "
            "across the four reported evaluation columns."
        ),
        body=(
            "Companion to the row-normalized recall view. Rows are ground-truth classes; columns are predicted "
            "classes; cells are tinted by per-COLUMN percent (the share of a given prediction that came from each "
            "true class). The diagonal of each column is the corresponding per-class precision. Diagonal cells are "
            "outlined in solid black; cell tinting follows a shared `Blues` sequential colormap on the same 0-100 "
            "percent scale as the recall view, so the two figures read as paired views over the same data. Source: "
            "`outputs/ablation/_ensembles/all_features__<split>.json` -> "
            "`cross_variant.equal_seed_metrics.argmax.confusion_matrix`. Per-panel headline F1 and QWK are the "
            "corresponding ensemble scalars from the same JSON. Destroyed-column precision by column: "
            + "; ".join(destroyed_notes) + ". The Mayfield shortfall is the Major-versus-Destroyed visual boundary "
            "on tornado debris fields; on the hurricane holdouts the impurity sits instead in the No Damage and "
            "Minor columns, where confidently under-called Minor and Major buildings accumulate."
        ),
    )


if __name__ == "__main__":
    main()
