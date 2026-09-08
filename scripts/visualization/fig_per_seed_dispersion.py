"""R4: per-seed F1 dispersion of the full MCDN configuration across the four reported columns.

Visualizes the per-seed initialization-variance signature behind the
"dispersion scales with domain shift" claim. Each column renders ten jittered
dots (one per seed at the best-checkpoint argmax F1, replayed from the stored
checkpoints) plus a single horizontal bar at the ensemble argmax F1 (the
softmax-mean of per-seed probabilities decoded under argmax). The bar
typically sits at or above the per-seed cluster top, encoding the
ensemble-lift mechanism: averaging across the seed pool's softmax outputs
exceeds the typical individual seed's argmax F1 because seed-specific
calibration noise cancels in expectation.

Visual conventions:
    - Per-seed dots in neutral medium grey with white edges, slight
      horizontal jitter so 10 samples don't pile on a single x-coordinate.
    - Ensemble F1 rendered as a short horizontal bar centered on each column
      position, dark grey, drawn above the dots in the layer stack.
    - Per-bar value annotation (3-decimal F1) above each ensemble bar.

Source: ``outputs/ablation_dgx/_ensembles/all_features__<split>.json`` ->
``variants[0].per_seed[*].replay_metrics.argmax.macro_f1`` (dots) and
``cross_variant.equal_seed_metrics.argmax.macro_f1`` (bar).
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualization._common import (
    REPORTED_COLUMNS,
    WIDTH_2COL,
    ensemble_rule_metrics,
    per_seed_rule_metrics,
    save_caption,
    save_figure,
    setup_publication_style,
)

_DOT_COLOR: Final[str] = "#7a7a7a"
_ENSEMBLE_COLOR: Final[str] = "#1f1f1f"
_JITTER_HALF_WIDTH: Final[float] = 0.13
_BAR_HALF_WIDTH: Final[float] = 0.27
_DOT_SIZE: Final[float] = 26.0


def _load_per_seed_argmax_f1(split_dir: str) -> np.ndarray:
    """Read the 10-seed per-seed best-checkpoint argmax F1 array."""

    return np.asarray([entry["macro_f1"] for entry in per_seed_rule_metrics(split_dir, rule="argmax")], dtype=np.float64)


def _load_ensemble_argmax_f1(split_dir: str) -> float:
    """Read the ensemble argmax F1 (softmax-mean across seeds)."""

    return float(ensemble_rule_metrics(split_dir, rule="argmax")["macro_f1"])


def main() -> None:
    """Render the per-seed F1 dispersion strip plot."""

    setup_publication_style()

    # Deterministic horizontal jitter so the figure is bit-identical
    # across regenerations.
    rng = np.random.default_rng(seed=0)

    fig, ax = plt.subplots(
        figsize=(WIDTH_2COL, 3.0),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "top": 0.94,
            "bottom": 0.14,
        },
    )

    dispersion_notes: list[str] = []
    all_values: list[float] = []
    for x_idx, (split_dir, split_label) in enumerate(REPORTED_COLUMNS):
        per_seed = _load_per_seed_argmax_f1(split_dir)
        ensemble = _load_ensemble_argmax_f1(split_dir)
        all_values.extend([*per_seed.tolist(), ensemble])
        dispersion_notes.append(
            f"{split_label}: SD {per_seed.std(ddof=1):.4f}, ensemble lift over the per-seed mean {ensemble - per_seed.mean():+.4f}"
        )

        jitter = rng.uniform(-_JITTER_HALF_WIDTH, _JITTER_HALF_WIDTH, size=per_seed.size)
        ax.scatter(
            x_idx + jitter,
            per_seed,
            s=_DOT_SIZE,
            color=_DOT_COLOR,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        ax.plot(
            [x_idx - _BAR_HALF_WIDTH, x_idx + _BAR_HALF_WIDTH],
            [ensemble, ensemble],
            color=_ENSEMBLE_COLOR,
            linewidth=2.0,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.annotate(
            f"{ensemble:.3f}",
            xy=(x_idx, ensemble),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=_ENSEMBLE_COLOR,
            fontweight="medium",
        )

    ax.set_xticks(range(len(REPORTED_COLUMNS)))
    ax.set_xticklabels([label for _, label in REPORTED_COLUMNS], fontsize=8)
    ax.set_xlim(-0.5, len(REPORTED_COLUMNS) - 0.5)
    ax.set_ylim(np.floor(min(all_values) * 50) / 50, np.ceil(max(all_values) * 50) / 50 + 0.01)
    ax.set_ylabel("Macro-F1 (best checkpoint, argmax)", fontsize=9)
    ax.tick_params(axis="both", which="both", labelsize=8, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(fig=fig, name="per_seed_dispersion")

    save_caption(
        name="per_seed_dispersion",
        title="Per-seed F1 dispersion of the full MCDN configuration across the four reported evaluation columns.",
        body=(
            "Each grey dot is one of the ten fixed-seed best-checkpoint argmax Macro-F1 values for that column "
            "(seeds 0, 11, 22, 33, 44, 55, 66, 77, 88, 99); horizontal jitter is purely visual (deterministic via "
            "``numpy.random.default_rng(seed=0)``) and carries no semantic content. The dark horizontal bar marks "
            "the 10-seed ensemble argmax Macro-F1 - the softmax-mean of the ten per-seed probability tensors decoded "
            "under argmax. Per-column dispersion and ensemble lift: " + "; ".join(dispersion_notes) + ". "
            "Source: ``outputs/ablation_dgx/_ensembles/all_features__<split>.json`` (per-seed replayed metrics for "
            "the dots, ``cross_variant.equal_seed_metrics.argmax`` for the bar)."
        ),
    )


if __name__ == "__main__":
    main()
