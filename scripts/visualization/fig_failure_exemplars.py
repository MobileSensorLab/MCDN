"""Misclassification exemplars, one chip per recurring failure mechanism (T-9 / work-plan 2.7).

Renders a single-column 2 x 2 grid of confidently misclassified chips drawn
from the error inventories written by ``scripts/mine_errors.py``, one per
mechanism identified in the contact-sheet categorization:

    (a) Roof-invisible damage - LOEO Ida: an intact-roofed structure standing
        in surge water, labeled Major and called No Damage. Nadir imagery
        cannot see damage to the interior or lower structure.
    (b) Large-footprint truncation - LOEO Michael: the footprint fills the
        512 px chip (2 cm GSD, ~10 m window) so the model scores an intact
        fragment of a large roof whose Major label describes damage elsewhere.
    (c) Ordinal-boundary ambiguity - LOEO Mayfield: roof gone and debris
        field, labeled Major and called Destroyed - a boundary convention the
        annotators and the model draw differently.
    (d) Structure absent - LOEO Ida: the footprint sits on open water or
        scoured ground where the structure once stood, labeled Destroyed and
        called No Damage. The model classifies the visible roof and has no
        concept of a missing building.

Selection is rule-based over the inventories (see ``_pick``) so the figure is
reproducible from the artifacts; the chosen validation indices are printed at
render time and recorded in the caption. Footprint contour and title
conventions follow the other chip figures; the panel letter sits centered
below each part per IEEE multipart-figure convention, the ensemble probability
is boxed in the lower-right corner of the chip.

Source: ``outputs/error_mining/all_features__<split>/inventory.json`` and the
per-holdout val pools via ``load_reference_val_pool``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, NamedTuple

import matplotlib.pyplot as plt

from scripts.visualization._common import (
    FIG_FONT_PT,
    ORDINAL_CLASS_NAMES,
    WIDTH_1COL,
    load_reference_val_pool,
    save_caption,
    save_figure,
    setup_publication_style,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_MINING_DIR: Final[Path] = _REPO_ROOT / "outputs" / "error_mining"

_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 2.4
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 1.1
_SHORT: Final[dict[int, str]] = {0: "No Damage", 1: "Minor", 2: "Major", 3: "Destroyed"}


class Panel(NamedTuple):
    """Selection rule for one exemplar panel."""

    letter: str
    mechanism: str
    split: str
    column_label: str
    pool: str  # "severe_errors" or "adjacent_errors"
    target: int
    pred: int
    max_mask_frac: float
    min_mask_frac: float
    allow_border: bool


_PANELS: Final[tuple[Panel, ...]] = (
    Panel("a", "Roof-invisible surge damage", "Hurricane_Ida", "LOEO Ida", "severe_errors", 2, 0, 0.70, 0.30, False),
    Panel("b", "Large-footprint truncation", "Hurricane_Michael", "LOEO Michael", "severe_errors", 2, 0, 1.01, 0.95, True),
    Panel("c", "Major / Destroyed boundary", "Mayfield_Tornado", "LOEO Mayfield", "adjacent_errors", 2, 3, 0.80, 0.20, False),
    Panel("d", "Structure absent", "Hurricane_Ida", "LOEO Ida", "severe_errors", 3, 0, 0.70, 0.30, True)
)


def _pick(panel: Panel) -> dict[str, Any]:
    """Return the highest-confidence inventory entry satisfying the panel's rule."""

    inventory = json.loads((_MINING_DIR / f"all_features__{panel.split}" / "inventory.json").read_text(encoding="utf-8"))
    candidates = [
        entry for entry in inventory[panel.pool]
        if entry["target"] == panel.target and entry["pred"] == panel.pred
        and panel.min_mask_frac <= entry["mask_frac"] <= panel.max_mask_frac
        and (panel.allow_border or not entry["mask_at_border"]) and not entry["mask_empty"] and entry["nodata_frac"] < 0.01
    ]
    if not candidates:
        raise ValueError(f"No inventory entry satisfies the rule for panel ({panel.letter}) {panel.mechanism}.")
    return max(candidates, key=lambda entry: entry["confidence"])


def main() -> None:
    """Render the 2 x 2 failure-exemplar figure and its caption."""

    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_1COL, WIDTH_1COL + 0.95),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.34, "left": 0.005, "right": 0.995, "top": 0.925, "bottom": 0.045})

    notes: list[str] = []
    for ax, panel in zip(axes.ravel(), _PANELS, strict=True):
        entry = _pick(panel)
        pool = load_reference_val_pool(holdout=panel.split)
        idx = int(entry["index"])
        if int(pool["targets"][idx]) != panel.target:
            raise ValueError(f"Inventory index {idx} label mismatch on {panel.split}.")
        image = pool["dataset"][idx]["image"].numpy()  # [4, H, W] uint8
        rgb, mask = image[:3].transpose(1, 2, 0), image[3]
        chip_px = mask.shape[0]

        ax.imshow(rgb, interpolation="bilinear")
        ax.contour(mask.astype(float), levels=[0.5], colors=[_MASK_HALO_COLOR], linewidths=_MASK_HALO_WIDTH)
        ax.contour(mask.astype(float), levels=[0.5], colors=[_MASK_OUTLINE_COLOR], linewidths=_MASK_OUTLINE_WIDTH)
        ax.set_title(f"{panel.mechanism}\n{_SHORT[panel.target]} → {_SHORT[panel.pred]}", fontsize=FIG_FONT_PT, pad=3)
        ax.text(0.97 * chip_px, 0.97 * chip_px, f"p = {entry['confidence']:.2f}", ha="right", va="bottom", fontsize=FIG_FONT_PT,
                color="white", zorder=6, bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.6, "alpha": 0.75})
        ax.text(0.5, -0.03, f"({panel.letter})", ha="center", va="top", fontsize=FIG_FONT_PT, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        note = (f"({panel.letter}) {panel.column_label}, val index {idx}, {entry['mosaic']}, label {ORDINAL_CLASS_NAMES[panel.target]}, "
                f"ensemble {ORDINAL_CLASS_NAMES[panel.pred]} at p = {entry['confidence']:.3f}, {entry['seed_agreement']}/{entry['n_seeds']} seeds, "
                f"footprint {100 * entry['mask_frac']:.0f} % of chip")
        notes.append(note)
        print(f"[fig_failure_exemplars] {note}")

    save_figure(fig=fig, name="failure_exemplars")

    save_caption(
        name="failure_exemplars",
        title="Recurring failure mechanisms of the full MCDN configuration, one confidently misclassified chip each.",
        body=(
            "Each panel shows a chip the 10-seed ensemble misclassified with high confidence, with the annotated "
            "footprint traced as a white-with-black-halo contour, the label and prediction in the title, and the "
            "ensemble probability boxed in the corner. (a) Roof-invisible damage: an intact-roofed structure standing in Hurricane Ida surge water, "
            "labeled Major - nadir imagery does not see interior or lower-structure damage. (b) Large-footprint "
            "truncation: on Hurricane Michael's 2 cm GSD the 512 px chip spans about 10 m, so the footprint fills the "
            "window and the model scores an intact fragment of a roof whose Major label describes damage outside the "
            "chip. (c) Ordinal-boundary ambiguity: a Mayfield structure with the roof gone and a surrounding debris "
            "field, labeled Major and called Destroyed. (d) Structure absent: an Ida footprint over open water where the "
            "building stood, labeled Destroyed and called No Damage - the model scores the visible roof and has no "
            "concept of a missing structure. Selection is "
            "rule-based over the error inventories (`outputs/error_mining/all_features__<split>/inventory.json`). "
            "Provenance - " + "; ".join(notes) + "."
        ),
    )


if __name__ == "__main__":
    main()
