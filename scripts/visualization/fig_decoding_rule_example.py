"""D2: prediction-rule decoding example for one borderline Mayfield val chip.

Visualizes the chapter-6 §Prediction Rule Choice claim that ensembling
smooths the averaged softmax distribution and **widens the mode-versus-
expectation gap** that argmax recovers and EV pulls toward neighbors. A
single Mayfield val chip is rendered as:

    1. A small chip thumbnail at the top-left (with the footprint outline)
       so the reader knows what sample is being decoded.
    2. A 2x5 grid of per-seed softmax bar charts to the right of the chip,
       one panel per seed in the canonical 10-seed pool.
    3. A larger ensemble softmax bar chart along the bottom, with the truth-
       class marker, the argmax decision, and the EV decision (the EV
       expectation value drawn as a dashed vertical line) overlaid.

The exemplar (manifest index 1779 in the Mayfield Tornado val pool) is the
highest-entropy chip in the argmax-vs-EV disagreement set: ensemble probs
[0.144, 0.396, 0.192, 0.268] place the mode on Minor (the truth class) but
the heavy non-adjacent Destroyed mass at 0.268 pulls the expectation
through Major's class boundary, so EV rounds to Major (the wrong adjacent
class) while argmax recovers the correct mode. The bimodal Minor + Destroyed
ensemble shape is exactly the kind of distribution that ensembling makes
smoother than any single seed - the per-seed panels show this directly.

Source artifacts: ``outputs/ablation/baseline/Mayfield_Tornado/ensemble_probs.pt``
(per-seed and ensemble probability tensors plus targets) accessed via the
``load_canonical_val_pool`` helper added to ``_common.py``.
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from scripts.visualization._common import (
    ORDINAL_CLASS_NAMES,
    ORDINAL_CLASS_PALETTE,
    WIDTH_2COL,
    load_canonical_val_pool,
    save_caption,
    save_figure,
    setup_publication_style,
)

# Match the chapter-4 chip-exemplar mask-outline convention (white-with-halo).
_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 1.6
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 0.8

# Canonical D2 selection. Manifest index 1779 in the Mayfield Tornado val
# pool is the highest-entropy chip in the argmax-vs-EV disagreement set.
_HOLDOUT: Final[str] = "Mayfield_Tornado"
_MANIFEST_IDX: Final[int] = 1779

# Annotation color for the EV expectation-line and decision markers.
_EV_LINE_COLOR: Final[str] = "#444444"
_TRUTH_MARKER_COLOR: Final[str] = "#1a1a1a"
_ARGMAX_MARKER_COLOR: Final[str] = "#1a1a1a"

# Compact short labels for the per-seed mini-panel x-axes (avoid full class
# names crowding 10 mini panels).
_SHORT_LABELS: Final[tuple[str, ...]] = ("N", "Mn", "Mj", "D")


def _expected_value(probs: np.ndarray) -> float:
    """Compute the ordinal expectation E[c] = sum_k k * p_k for a 1-D softmax."""

    return float(np.dot(probs, np.arange(probs.shape[0], dtype=np.float64)))


def _draw_chip_panel(ax: plt.Axes, *, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Render the input chip with the footprint polygon as white-with-halo."""

    ax.imshow(rgb, interpolation="bilinear")
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=[_MASK_HALO_COLOR],
        linewidths=_MASK_HALO_WIDTH,
    )
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=[_MASK_OUTLINE_COLOR],
        linewidths=_MASK_OUTLINE_WIDTH,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_per_seed_panel(
    ax: plt.Axes,
    *,
    probs: np.ndarray,
    seed_label: str,
    show_y_ticks: bool,
) -> None:
    """Render one per-seed 4-class softmax bar chart at compact size.

    The seed label is placed inside the panel (upper-left corner) rather than
    above as a title, so the inter-row spacing in the per-seed grid does not
    have to leave room for two stacked title heights.
    """

    ax.bar(
        np.arange(len(probs)),
        probs,
        width=0.78,
        color=list(ORDINAL_CLASS_PALETTE),
        edgecolor="#444444",
        linewidth=0.4,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(len(probs)))
    ax.set_xticklabels(_SHORT_LABELS, fontsize=6.5)
    if show_y_ticks:
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=6.5, length=2)
    else:
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=2)
    ax.tick_params(axis="x", length=2)
    ax.text(
        0.04,
        0.96,
        seed_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#1a1a1a",
        family="monospace",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_ensemble_panel(
    ax: plt.Axes,
    *,
    probs: np.ndarray,
    truth_idx: int,
    argmax_idx: int,
    ev_value: float,
    ev_decoded_idx: int,
) -> None:
    """Render the ensemble bar chart with EV expectation line + corner decision summary."""

    n_classes = len(probs)
    ax.bar(
        np.arange(n_classes),
        probs,
        width=0.62,
        color=list(ORDINAL_CLASS_PALETTE),
        edgecolor="#444444",
        linewidth=0.6,
    )
    ax.set_ylim(0.0, max(0.55, probs.max() * 1.18))
    ax.set_xlim(-0.55, n_classes - 0.45)
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(ORDINAL_CLASS_NAMES, fontsize=8.5)
    ax.set_yticks([0.0, 0.2, 0.4])
    ax.tick_params(axis="both", which="both", labelsize=8, length=2)
    ax.set_ylabel("Softmax", fontsize=8.5, labelpad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # EV expectation line - the load-bearing visual annotation. Dashed,
    # drawn at the non-integer ordinal expectation so the reader sees where
    # it lands relative to the rounding boundary between Minor (x=1) and
    # Major (x=2). The dashed line is the only on-figure annotation; truth
    # and argmax decisions live in the corner decision-summary text below.
    ax.axvline(
        ev_value,
        linestyle="--",
        color=_EV_LINE_COLOR,
        linewidth=1.1,
        zorder=4,
    )
    ax.annotate(
        f"EV = {ev_value:.2f}",
        xy=(ev_value, ax.get_ylim()[1]),
        xytext=(3, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color=_EV_LINE_COLOR,
    )

    # Compact decision summary in the upper-right corner. Three lines, plain
    # text, no arrows or markers competing with the bars.
    truth_name = ORDINAL_CLASS_NAMES[truth_idx]
    argmax_name = ORDINAL_CLASS_NAMES[argmax_idx]
    ev_name = ORDINAL_CLASS_NAMES[ev_decoded_idx]
    arg_mark = "OK" if argmax_idx == truth_idx else "X"
    ev_mark = "OK" if ev_decoded_idx == truth_idx else "X"
    summary = (
        f"truth = {truth_name}\n"
        f"argmax \u2192 {argmax_name} [{arg_mark}]\n"
        f"EV (round) \u2192 {ev_name} [{ev_mark}]"
    )
    ax.text(
        0.015,
        0.96,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#1a1a1a",
        family="monospace",
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.95,
            "pad": 3.0,
            "linewidth": 0.5,
        },
    )


def main() -> None:
    """Render the D2 prediction-rule decoding example."""

    setup_publication_style()

    pool = load_canonical_val_pool(holdout=_HOLDOUT)
    dataset = pool["dataset"]
    targets = pool["targets"]
    individual_probs = pool["individual_probs"]  # [n_seeds, N, K]
    ensemble_probs = pool["ensemble_probs"]      # [N, K]
    seeds = pool["seeds"]

    sample = dataset[_MANIFEST_IDX]
    image_uint8 = sample["image"].numpy()
    rgb = image_uint8[:3].transpose(1, 2, 0).copy()
    mask = image_uint8[3].copy()

    truth_idx = int(targets[_MANIFEST_IDX])
    seed_probs = individual_probs[:, _MANIFEST_IDX, :]   # [n_seeds, K]
    ens_probs = ensemble_probs[_MANIFEST_IDX]            # [K]
    argmax_idx = int(ens_probs.argmax())
    ev_value = _expected_value(ens_probs)
    ev_decoded_idx = int(np.clip(round(ev_value), 0, ens_probs.shape[0] - 1))

    fig = plt.figure(figsize=(WIDTH_2COL, 4.5))
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 0.42],
        left=0.06,
        right=0.985,
        top=0.97,
        bottom=0.10,
        hspace=0.34,
    )

    # Top row: chip on the left (one-row tall, square cell at this figure
    # height; aspect='equal' fills the cell without lateral whitespace) +
    # ensemble bar chart on the right (the decision-relevant element).
    top = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 2.45],
        wspace=0.26,
    )
    chip_ax = fig.add_subplot(top[0, 0])
    _draw_chip_panel(chip_ax, rgb=rgb, mask=mask)

    ensemble_ax = fig.add_subplot(top[0, 1])
    _draw_ensemble_panel(
        ensemble_ax,
        probs=ens_probs,
        truth_idx=truth_idx,
        argmax_idx=argmax_idx,
        ev_value=ev_value,
        ev_decoded_idx=ev_decoded_idx,
    )

    # Bottom row: 1x10 per-seed strip - no row stacking, so the in-panel
    # seed labels never collide with adjacent rows.
    n_seeds = seed_probs.shape[0]
    bottom = gridspec.GridSpecFromSubplotSpec(
        1,
        n_seeds,
        subplot_spec=outer[1],
        wspace=0.16,
    )
    for seed_idx in range(n_seeds):
        ax = fig.add_subplot(bottom[0, seed_idx])
        seed_label = seeds[seed_idx].replace("seed_", "s")
        _draw_per_seed_panel(
            ax,
            probs=seed_probs[seed_idx],
            seed_label=seed_label,
            show_y_ticks=(seed_idx == 0),
        )

    save_figure(fig=fig, name="decoding_rule_example")

    save_caption(
        name="decoding_rule_example",
        title=(
            "Prediction-rule decoding example: ensemble smoothing widens the "
            "mode-versus-expectation gap on a borderline Mayfield Tornado chip."
        ),
        body=(
            "Top-left: input chip (manifest index 1779 in the Mayfield Tornado "
            "val pool) with the structure footprint outlined as a white-with-"
            "black-halo contour, matching the F6/F7/F8 convention. Top-right: "
            "the ten per-seed argmax-rule softmax distributions over the four "
            "ordinal damage classes (No Damage, Minor, Major, Destroyed); seed "
            "labels are shortened to 'sNN'. Bottom: the ensemble softmax (mean "
            "across the ten seeds), with three decision markers overlaid - the "
            "ground-truth class (Minor), the argmax-decoded class (Minor; "
            "matches truth), and the EV-decoded class (Major; off by one "
            "adjacent step), with the EV expectation value drawn as a dashed "
            "vertical line at the non-integer ordinal position 1.58. The "
            "ensemble distribution [0.144, 0.396, 0.192, 0.268] places its mode "
            "on Minor at 0.396 - the visible peak the argmax rule recovers - "
            "but the heavy non-adjacent Destroyed mass at 0.268 pulls the "
            "expectation through Major's class boundary, so EV rounds to "
            "Major. The figure is the chapter's mode-versus-expectation gap "
            "made visible at one chip; under ensembling the smoothed averaged "
            "softmax amplifies this gap by ~2.5x relative to per-seed "
            "decoding, which produces the +0.011 ensemble argmax F1 lead "
            "over EV reported in chapter 5. Source: "
            "outputs/ablation/baseline/Mayfield_Tornado/ensemble_probs.pt."
        ),
    )


if __name__ == "__main__":
    main()
