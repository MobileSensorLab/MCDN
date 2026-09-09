"""R1: confusion-matrix heatmaps across the four reported evaluation columns.

Renders four ensemble-argmax confusion matrices side-by-side - the DROIDs
default split (the dataset's published train/test partition), then the three
leave-one-event-out holdouts (Hurricane Michael, Mayfield Tornado, Hurricane
Ida) - so the shift of off-diagonal mass across columns is visible in a
single glance.

Visual conventions:
    - Cells colored by per-row percent (probability of predicting class j given
      true class i). Row-normalization keeps the off-diagonal pattern visible
      regardless of class-support imbalance, which matters here because the
      columns have markedly different class balances.
    - Single shared sequential ``Blues`` colormap (0-100 percent) so the panels
      are directly comparable. No colorbar: every cell prints its per-row
      percent, so the tint is a reading aid rather than an encoding to decode.
    - Cell text shows count + per-row percent; text color flips white above
      50 percent for contrast.
    - Diagonal cells outlined in solid black to emphasize correct predictions
      without competing with the colormap.
    - Per-panel title carries the column label plus ensemble Macro-F1 and QWK
      so readers can read the headline metric next to the matrix that produced
      it.

Source artifacts: ``outputs/ablation/_ensembles/all_features__<split>.json``
-> ``cross_variant.equal_seed_metrics.argmax`` (the reported decoding rule;
the EV and hybrid blocks are preserved in the same JSON but not rendered here).
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts.visualization._common import (
    FIG_ANNOT_PT,
    FIG_FONT_PT,
    ORDINAL_CLASS_NAMES,
    REPORTED_COLUMNS,
    WIDTH_2COL,
    ensemble_rule_metrics,
    save_caption,
    save_figure,
    setup_publication_style,
)

# Short axis-tick labels so 4-class panels don't crowd horizontally; full
# class names appear once each as panel-row labels via the y-axis on the
# leftmost panel.
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
    """Render a single confusion-matrix panel onto ``ax`` and return the image."""

    row_totals = cm_counts.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    cm_pct = cm_counts / row_totals_safe * 100.0

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
                fontsize=FIG_ANNOT_PT,
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
    ax.set_xticklabels(_SHORT_LABELS, fontsize=FIG_FONT_PT)
    ax.set_yticks(range(n_classes))
    if show_y_labels:
        ax.set_yticklabels(ORDINAL_CLASS_NAMES, fontsize=FIG_FONT_PT)
        ax.set_ylabel("True class")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted class")
    ax.set_title(title, fontsize=FIG_FONT_PT, pad=5)

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def main() -> None:
    """Render the confusion-matrix panels across the four reported columns."""

    setup_publication_style()

    panels = [(_load_panel(d), label) for d, label in REPORTED_COLUMNS]

    # No colorbar: the per-row percent is printed in every cell, so the width it
    # occupied goes to the four panels instead.
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(WIDTH_2COL, 2.45),
        gridspec_kw={"wspace": 0.14, "left": 0.085, "right": 0.99, "top": 0.85, "bottom": 0.2},
    )

    cmap = plt.get_cmap("Blues")
    for idx, ((data, label), ax) in enumerate(zip(panels, axes, strict=True)):
        cm_counts = data["confusion_matrix"]
        title = (
            f"{label}\n"
            f"F1 {data['macro_f1']:.3f}  QWK {data['qwk']:.3f}"
        )
        _draw_panel(
            ax,
            cm_counts=cm_counts,
            title=title,
            show_y_labels=(idx == 0),
            cmap=cmap,
            vmin=0.0,
            vmax=100.0,
        )

    save_figure(fig=fig, name="confusion_triptych")

    save_caption(
        name="confusion_triptych",
        title=(
            "Ensemble argmax confusion matrices of the full MCDN configuration across the four reported "
            "evaluation columns: the DROIDs default split and the LOEO Michael, Mayfield and Ida holdouts."
        ),
        body=(
            "Rows are ground-truth classes (top to bottom: No Damage, Minor, "
            "Major, Destroyed); columns are predicted classes in the same "
            "order. Each cell carries the raw count above the per-row "
            "percent. Diagonal cells are outlined in black to emphasize "
            "correct predictions; cell tinting follows the per-row percent "
            "under a single shared `Blues` sequential colormap (0-100 %) so the four "
            "panels are directly comparable independent of column-level class "
            "support. Source: `outputs/ablation/_ensembles/all_features__<split>.json` "
            "-> `cross_variant.equal_seed_metrics.argmax.confusion_matrix`. "
            "Per-panel headline F1 and QWK are the corresponding ensemble "
            "scalars from the same JSON. The dominant residual error modes "
            "differ by column: the hurricane holdouts concentrate residuals on "
            "the interior boundaries as under-calls (Minor called No Damage on "
            "Ida, Major called Minor on Michael); the Mayfield panel spreads "
            "residuals in both directions and exposes a Destroyed-precision "
            "shortfall driven by Major-as-Destroyed false positives, consistent "
            "with the harder Major-versus-Destroyed visual boundary on tornado "
            "debris fields."
        ),
    )


if __name__ == "__main__":
    main()
