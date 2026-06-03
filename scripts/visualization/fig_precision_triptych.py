"""Companion to R1: precision-normalized confusion-matrix triptych.

Sibling figure to ``fig_confusion_triptych`` that renders the same three
ensemble-argmax confusion matrices but column-normalized rather than
row-normalized. Where the recall view (``confusion_triptych``) answers
"given truth = X, what was predicted?", this view answers "given prediction
= X, what was the truth?". The diagonal of each column equals the per-class
**precision**.

The figure surfaces a chapter-5 claim that the recall view does not visualize:
the distant-OOD Mayfield Tornado holdout exhibits a **Destroyed precision
collapse** (0.677, against 0.918 in-distribution and 0.933 on Hurricane
Michael), driven by Major-as-Destroyed false positives concentrating in the
Destroyed column. Reading down the Destroyed column on Mayfield, 130 of the
192 ensemble Destroyed predictions are correct; 57 of the remaining 62 are
true Major - exactly the chapter's prose claim, now made visually evident
at the same prominence as the recall view's per-row pattern.

Visual conventions are deliberately identical to the recall triptych so the
two figures read as a paired recall-precision view:
    - Cells colored by per-column percent under a shared sequential ``Blues``
      colormap, 0 to 100 percent across all panels for direct comparability.
    - Cell text shows count above per-column percent; text color flips white
      above 50 percent for contrast.
    - Diagonal cells outlined in solid black to emphasize correct predictions.
    - Per-panel title carries the split name plus ensemble Macro-F1 and QWK.

Source artifacts: ``outputs/ablation/baseline/<split>/ensemble_metrics.json``
-> ``ensemble.argmax.confusion_matrix``. Same JSON the recall triptych reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from scripts.visualization._common import (
    ORDINAL_CLASS_NAMES,
    WIDTH_2COL,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_BASELINE_DIR: Final[Path] = _REPO_ROOT / "outputs" / "ablation" / "baseline"

# Funnel ordering matches fig_confusion_triptych so the two figures display
# their splits in the same left-to-right order when paired in the chapter.
_PANELS: Final[tuple[tuple[str, str], ...]] = (
    ("Spatial_Block_East", "Spatial Block East"),
    ("Hurricane_Michael", "Hurricane Michael"),
    ("Mayfield_Tornado", "Mayfield Tornado"),
)

_SHORT_LABELS: Final[tuple[str, ...]] = ("None", "Minor", "Major", "Destr.")


def _load_panel(split_dir: str) -> dict[str, object]:
    """Read the argmax block from ``ensemble_metrics.json`` for one split."""

    metrics_path = _BASELINE_DIR / split_dir / "ensemble_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Ensemble metrics not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    argmax = payload["ensemble"]["argmax"]
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
                fontsize=7.5,
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
    ax.set_xticklabels(_SHORT_LABELS, fontsize=8)
    ax.set_yticks(range(n_classes))
    if show_y_labels:
        ax.set_yticklabels(ORDINAL_CLASS_NAMES, fontsize=8)
        ax.set_ylabel("True class")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted class")
    ax.set_title(title, fontsize=9, pad=6)

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def main() -> None:
    """Render the precision (column-normalized) confusion-matrix triptych."""

    setup_publication_style()

    panels = [(_load_panel(d), label) for d, label in _PANELS]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH_2COL, 2.85),
        gridspec_kw={"wspace": 0.18, "left": 0.085, "right": 0.91, "top": 0.88, "bottom": 0.18},
    )

    cmap = plt.get_cmap("Blues")
    image = None
    for idx, ((data, label), ax) in enumerate(zip(panels, axes)):
        cm_counts = data["confusion_matrix"]
        title = (
            f"{label}\n"
            f"F1 = {data['macro_f1']:.3f}    "
            f"QWK = {data['qwk']:.3f}"
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

    cbar_ax = fig.add_axes([0.925, 0.22, 0.012, 0.6])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Column-normalized %", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)

    save_figure(fig=fig, name="precision_triptych")

    save_caption(
        name="precision_triptych",
        title=(
            "Ensemble argmax precision (column-normalized) confusion matrices "
            "across the evaluation funnel."
        ),
        body=(
            "Companion to the row-normalized recall triptych. Rows are "
            "ground-truth classes; columns are predicted classes; cells are "
            "tinted by per-COLUMN percent (the share of a given prediction "
            "that came from each true class). The diagonal of each column is "
            "the corresponding per-class precision. Diagonal cells are "
            "outlined in solid black; cell tinting follows a shared `Blues` "
            "sequential colormap on the same 0-100 percent scale as the "
            "recall view, so the two figures read as paired views over the "
            "same data. Source: `outputs/ablation/baseline/<split>/"
            "ensemble_metrics.json` -> `ensemble.argmax.confusion_matrix`. "
            "Per-panel headline F1 and QWK are the corresponding ensemble "
            "scalars from the same JSON. The chapter's central per-class "
            "Mayfield signature is encoded in the Destroyed column of the "
            "right-most panel: 130 of 192 ensemble Destroyed predictions are "
            "correct (precision = 0.677), with 57 of the remaining 62 false "
            "positives drawn from true Major - the harder Major-versus-"
            "Destroyed visual boundary on tornado debris fields. The same "
            "Destroyed column on the in-distribution and proximate-OOD "
            "panels is nearly pure (precision = 0.918, 0.933 respectively)."
        ),
    )


if __name__ == "__main__":
    main()
