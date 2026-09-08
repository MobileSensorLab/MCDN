"""R5: footprint-quality dose-response of the full MCDN configuration (T-6).

Renders the controlled footprint-translation sweep produced by
``scripts/eval_mask_robustness.py`` on the 10-seed ``all_features`` ensembles for
the four reported evaluation columns (the DROIDs default split and the three
leave-one-event-out holdouts). Every perturbation acts on the footprint cache
*before* chip extraction, so a displaced footprint moves both the mask channel
and the 512 px evaluation window off the roof - the failure mode an uncorrected
public footprint cache produces in deployment.

Layout is two stacked single-column panels sharing the dose axis: delta QWK on
top, delta Macro-F1 below, both relative to the tie-point-aligned footprints.
Doses are 0, 15, 60, 75, 150, and 240 px mean per-building magnitude with
directions drawn at the circular variance (0.28) Manzini et al. report for the
raw cache. Each column's *actual* raw-cache result (Manzini et al.'s unadjusted
footprints, mean misalignment 75 px) is overlaid as a hollow marker at x = 75 so
the synthetic dose can be read against the real cache it models.

Visual conventions:
    - One series per evaluation column in the seaborn ``colorblind`` palette
      with a distinct marker shape per column (redundant non-color encoding
      for greyscale reproduction). Hollow markers of the same shape and color
      carry the raw-cache result.
    - Ensemble deltas are the plotted quantity; shaded bands show +/- 1 SD of
      the per-seed metric under the same perturbation, so a delta inside its
      band is indistinguishable from seed-initialization variance (the same
      reading rule as R2).
    - Dashed vertical guide at 75 px; a secondary top axis converts pixels to
      metres at the corpus building-weighted median sUAS GSD (3.45 cm/px).

The buffer and deletion conditions in the same artifacts are deliberately not
plotted: their deltas sit inside seed variance on every column and are reported
as a sentence in the text and as a table in the response letter.

Source artifacts: ``outputs/mask_robustness/all_features__<split>.json``; deltas
are read from the ``ensemble.argmax`` block and per-seed SDs from ``summary.argmax``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.lines import Line2D

from scripts.visualization._common import FIG_ANNOT_PT, FIG_FONT_PT, WIDTH_1COL, save_caption, save_figure, setup_publication_style

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_ROBUSTNESS_DIR: Final[Path] = _REPO_ROOT / "outputs" / "mask_robustness"
_VARIANT: Final[str] = "all_features"

# Evaluation columns in the manuscript's reporting order: the DROIDs default split
# (the train/test partition published with the dataset) first, then the three
# leave-one-event-out holdouts of our own construction, ordered by event chronology.
_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("Hurricane_Idalia+Hurricane_Michael+Mayfield_Tornado+Mussett_Bayou_Fire", "DROIDs default"),
    ("Hurricane_Michael", "LOEO Michael"),
    ("Mayfield_Tornado", "LOEO Mayfield"),
    ("Hurricane_Ida", "LOEO Ida")
)
_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")

# Translation doses (mean per-building magnitude, native mosaic pixels) and the
# condition labels the evaluator writes for them. ``aligned`` is the 0 px anchor.
_OFFSET_CONDITIONS: Final[tuple[tuple[int, str], ...]] = (
    (0, "aligned"), (15, "offset_15px"), (60, "offset_60px"), (75, "offset_75px"), (150, "offset_150px"), (240, "offset_240px")
)
_RAW_CACHE_CONDITION: Final[str] = "raw_cache"
_MANZINI_MEAN_OFFSET_PX: Final[int] = 75
# Building-weighted median GSD over the 52 post-event sUAS orthomosaics in
# data/statistics.csv (MAXAR and NOAA products excluded), rounded to the 3.4 cm/px
# the manuscript quotes. Used only for the nominal metre axis; per-event GSD ranges
# from 1.95 cm (Michael) to 12.7 cm (Idalia).
_MEDIAN_SUAS_GSD_M: Final[float] = 0.034
# Translation doses whose tick label is drawn; 15 and 60 px keep their points but
# lose their labels, which otherwise collide with 0 and 75 at column width.
_LABELED_DOSES: Final[frozenset[int]] = frozenset({0, 75, 150, 240})

_METRICS: Final[tuple[tuple[str, str], ...]] = (("qwk", r"$\Delta$ QWK vs. aligned"), ("macro_f1", r"$\Delta$ Macro-F1 vs. aligned"))
_ZERO_LINE_COLOR: Final[str] = "#9a9a9a"
_GUIDE_COLOR: Final[str] = "#4a4a4a"
_BAND_ALPHA: Final[float] = 0.16
_MARKER_SIZE: Final[float] = 4.5
_RAW_MARKER_SIZE: Final[float] = 7.0


class ConditionStats(NamedTuple):
    """Ensemble metric and per-seed dispersion for one perturbation condition."""

    ensemble: float
    seed_sd: float


def _load_conditions(split: str) -> dict[str, dict[str, ConditionStats]]:
    """Read every condition of one column's robustness artifact into ``{condition: {metric: stats}}``."""

    path = _ROBUSTNESS_DIR / f"{_VARIANT}__{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Robustness artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant = next(v for v in payload["variants"] if Path(v["variant_root"]).name == _VARIANT)

    conditions: dict[str, dict[str, ConditionStats]] = {}
    for entry in variant["conditions"]:
        ensemble_block = entry["ensemble"]["argmax"]
        summary_block = entry["summary"]["argmax"]
        conditions[entry["condition"]] = {
            metric: ConditionStats(ensemble=float(ensemble_block[metric]), seed_sd=float(summary_block[metric]["std"]))
            for metric, _ in _METRICS
        }
    return conditions


def _draw_dose_response(ax: plt.Axes, metric: str, columns: dict[str, dict[str, dict[str, ConditionStats]]],
                        palette: list[tuple[float, float, float]]) -> None:
    """Render translation dose-response lines, per-seed SD bands, and raw-cache markers for one metric."""

    doses = np.asarray([dose for dose, _ in _OFFSET_CONDITIONS], dtype=np.float64)
    for series_idx, (split, label) in enumerate(_COLUMNS):
        stats = columns[split]
        aligned = stats["aligned"][metric].ensemble
        deltas = np.asarray([stats[cond][metric].ensemble - aligned for _, cond in _OFFSET_CONDITIONS])
        sds = np.asarray([stats[cond][metric].seed_sd for _, cond in _OFFSET_CONDITIONS])
        color = palette[series_idx]
        marker = _MARKERS[series_idx]

        ax.fill_between(doses, deltas - sds, deltas + sds, color=color, alpha=_BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(doses, deltas, color=color, marker=marker, markersize=_MARKER_SIZE, linewidth=1.4,
                markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3)

        # The real raw cache (mean misalignment 75 px) as a hollow marker on the synthetic curve.
        raw_delta = stats[_RAW_CACHE_CONDITION][metric].ensemble - aligned
        ax.plot([_MANZINI_MEAN_OFFSET_PX], [raw_delta], linestyle="none", marker=marker, markersize=_RAW_MARKER_SIZE,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2, zorder=4)

    ax.axhline(0.0, color=_ZERO_LINE_COLOR, linewidth=0.8, zorder=1)
    ax.set_xlim(-6, 246)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"{int(dose)}" if int(dose) in _LABELED_DOSES else "" for dose in doses])
    ax.tick_params(axis="both", labelsize=FIG_FONT_PT, length=3)


def main() -> None:
    """Render the stacked single-column footprint dose-response figure and its caption."""

    setup_publication_style()
    palette = sns.color_palette("colorblind", n_colors=len(_COLUMNS))
    columns = {split: _load_conditions(split) for split, _ in _COLUMNS}

    fig, axes = plt.subplots(
        nrows=len(_METRICS), ncols=1, figsize=(WIDTH_1COL, 4.6), sharex=True,
        gridspec_kw={"left": 0.19, "right": 0.97, "top": 0.86, "bottom": 0.11, "hspace": 0.10}
    )

    for ax, (metric, y_label) in zip(axes, _METRICS, strict=True):
        _draw_dose_response(ax, metric, columns, palette)
        ax.set_ylabel(y_label, fontsize=FIG_FONT_PT)
    axes[-1].set_xlabel("Footprint translation (px, native GSD)", fontsize=FIG_FONT_PT)

    # Secondary metre axis on the top panel, nominal at the corpus median sUAS GSD;
    # the label carries the caveat so the conversion is not read as exact.
    top_axis = axes[0].secondary_xaxis("top", functions=(lambda px: px * _MEDIAN_SUAS_GSD_M, lambda m: m / _MEDIAN_SUAS_GSD_M))
    top_axis.set_xlabel(f"Nominal displacement (m at {100 * _MEDIAN_SUAS_GSD_M:.1f} cm/px)", fontsize=FIG_FONT_PT)
    top_axis.tick_params(labelsize=FIG_FONT_PT, length=3)
    top_axis.spines["top"].set_visible(True)

    # Legend in the empty lower-left of the top panel: column series plus one
    # neutral hollow-marker proxy for the raw-cache overlay.
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], linestyle="none", marker="o", markersize=6, markerfacecolor="white",
                          markeredgecolor=_GUIDE_COLOR, markeredgewidth=1.2))
    labels.append("Raw cache (actual)")
    # Single column on a white panel so the raw-cache guide line does not run through the text.
    legend = axes[0].legend(handles, labels, loc="lower left", frameon=True, fancybox=False, fontsize=FIG_FONT_PT, handlelength=1.8,
                            labelspacing=0.3, handletextpad=0.5, borderaxespad=0.3, borderpad=0.4)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(5)

    # Raw-cache guide: full height on the bottom panel; on the top panel it is drawn
    # from the legend's top edge upward so it stops at the box rather than running
    # beneath it. The legend extent is only known after a draw.
    fig.canvas.draw()
    legend_top = legend.get_window_extent().transformed(axes[0].transData.inverted()).y1
    axes[0].plot([_MANZINI_MEAN_OFFSET_PX, _MANZINI_MEAN_OFFSET_PX], [legend_top, axes[0].get_ylim()[1]],
                 color=_GUIDE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axes[1].axvline(_MANZINI_MEAN_OFFSET_PX, color=_GUIDE_COLOR, linewidth=0.8, linestyle="--", zorder=1)

    axes[1].annotate("mean raw-cache\nmisalignment", xy=(_MANZINI_MEAN_OFFSET_PX, axes[1].get_ylim()[0]), xytext=(4, 4),
                     textcoords="offset points", ha="left", va="bottom", fontsize=FIG_ANNOT_PT, color=_GUIDE_COLOR)

    save_figure(fig=fig, name="footprint_robustness")

    save_caption(
        name="footprint_robustness",
        title="Sensitivity of the full MCDN configuration to footprint misalignment on the DROIDs default split and the three LOEO holdouts.",
        body=(
            "Footprints are translated in the cache before chip extraction, so a displaced footprint moves both the "
            "mask channel and the 512 px evaluation window off the roof. Deltas are ensemble (10 seeds, 8-view TTA, "
            "argmax) QWK (top) and Macro-F1 (bottom) relative to the tie-point-aligned footprints, at mean per-building "
            "translation magnitudes of 0, 15, 60, 75, 150, and 240 px in native mosaic pixels with directions drawn at "
            "the circular variance (0.28) Manzini et al. report for the raw cache. The DROIDs default split is the "
            "train/test partition published with the dataset; the LOEO columns hold out one event each. Shaded bands are +/- 1 SD of the "
            "per-seed metric under the same perturbation. Hollow markers at 75 px are each column's result on Manzini "
            "et al.'s actual unadjusted footprints (mean misalignment 75 px, dashed guide); the top axis gives the "
            "nominal metre equivalent at the corpus building-weighted median sUAS GSD of 3.4 cm/px (per-event GSD spans "
            "1.95 cm on Michael to 12.7 cm on Idalia). Degradation is within seed variance through the observed "
            "raw-cache misalignment scale and becomes material only beyond roughly 150 px; the tornado holdout is the "
            "most alignment-sensitive column. The real cache sits slightly below the equal-mean synthetic dose on three "
            "columns, as expected for a convex loss under a heavy-tailed offset distribution. Source: "
            "``outputs/mask_robustness/all_features__<split>.json``."
        )
    )


if __name__ == "__main__":
    main()
