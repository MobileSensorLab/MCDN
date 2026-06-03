"""R1: confusion-matrix heatmap triptych across the canonical evaluation funnel.

Renders three ensemble-argmax confusion matrices side-by-side - in-distribution
Spatial Block East, proximate-OOD Hurricane Michael, distant-OOD Mayfield
Tornado - so the progression of off-diagonal mass under increasing domain
shift is visible in a single glance. Replaces the two fixed-width ASCII
confusion-matrix code blocks in [doc/5-results.qmd](doc/5-results.qmd) and
surfaces the previously prose-only Hurricane Michael matrix.

Visual conventions:
    - Cells colored by per-row percent (probability of predicting class j given
      true class i). Row-normalization keeps the off-diagonal pattern visible
      regardless of class-support imbalance, which matters here because the
      three holdouts have markedly different class balances (Mayfield's No
      Damage prevalence in particular would otherwise dominate raw-count tinting).
    - Single shared sequential ``Blues`` colormap with a single colorbar at the
      right edge so the three panels are directly comparable.
    - Cell text shows count + per-row percent; text color flips white above
      50 percent for contrast.
    - Diagonal cells outlined in solid black to emphasize correct predictions
      without competing with the colormap.
    - Per-panel title carries the split name plus ensemble Macro-F1 and QWK
      so readers can read the headline metric next to the matrix that produced
      it.

Source artifacts: ``outputs/ablation/baseline/<split>/ensemble_metrics.json``.
The argmax block under ``ensemble.argmax`` is the canonical decoding rule
established in chapter 5; per-rule blocks for EV and hybrid are preserved in
the same JSON for the ablation discussion but not rendered here.
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

# Funnel ordering: in-distribution -> proximate OOD -> distant OOD.
# Display labels use the chapter's canonical split names for cross-reference.
_PANELS: Final[tuple[tuple[str, str], ...]] = (
    ("Spatial_Block_East", "Spatial Block East"),
    ("Hurricane_Michael", "Hurricane Michael"),
    ("Mayfield_Tornado", "Mayfield Tornado"),
)

# Short axis-tick labels so 4-class panels don't crowd horizontally; full
# class names appear once each as panel-row labels via the y-axis on the
# leftmost panel.
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
    """Render the confusion-matrix triptych across the three baseline splits."""

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
    cbar.set_label("Row-normalized %", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)

    save_figure(fig=fig, name="confusion_triptych")

    save_caption(
        name="confusion_triptych",
        title=(
            "Ensemble argmax confusion matrices across the evaluation funnel: "
            "in-distribution Spatial Block East, proximate-OOD Hurricane "
            "Michael, distant-OOD Mayfield Tornado."
        ),
        body=(
            "Rows are ground-truth classes (top to bottom: No Damage, Minor, "
            "Major, Destroyed); columns are predicted classes in the same "
            "order. Each cell carries the raw count above the per-row "
            "percent. Diagonal cells are outlined in black to emphasize "
            "correct predictions; cell tinting encodes the per-row percent "
            "under a single shared `Blues` sequential colormap so the three "
            "panels are directly comparable independent of split-level class "
            "support. Source: `outputs/ablation/baseline/<split>/"
            "ensemble_metrics.json` -> `ensemble.argmax.confusion_matrix`. "
            "Per-panel headline F1 and QWK are the corresponding ensemble "
            "scalars from the same JSON. The dominant residual error modes "
            "shift across the funnel: the two hurricane-class splits "
            "concentrate residuals on the interior Minor/No Damage and "
            "Major/Minor boundaries; the Mayfield panel additionally exposes "
            "a Destroyed-precision shortfall driven by Major-as-Destroyed "
            "false positives, consistent with the harder Major-versus-"
            "Destroyed visual boundary on tornado debris fields."
        ),
    )


if __name__ == "__main__":
    main()
