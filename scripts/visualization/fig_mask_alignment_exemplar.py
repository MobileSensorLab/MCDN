"""D1: footprint-mask alignment exemplar (Hurricane Michael vs Mayfield Tornado).

Visualizes the Discussion's geometric-anchor claim that the mask ablation null on
the distant-OOD Mayfield Tornado holdout is a *registration* problem rather
than a capacity one: the mask channel encodes the structure's pre-event
geometry registered to the post-event image. On hurricane-class events the
structure typically remains in place and the polygon points at damaged
building material; on EF-3+ tornado events the structure routinely displaces
off its pre-event footprint and the polygon points at empty foundation pixels
while the visual evidence radiates several footprint-widths beyond.

Layout: 1x2 panels at WIDTH_2COL.

    - Left: a most-confident Destroyed exemplar from the full configuration's
      Hurricane Michael LOEO val pool, with the ground-truth footprint
      polygon overlaid as a white-with-black-halo contour. The polygon
      cleanly bounds the damaged structure's visual evidence.
    - Right: a most-confident Destroyed exemplar from the Mayfield Tornado
      LOEO val pool, same overlay convention. The polygon covers
      foundation-only pixels while the visible damage signal sits in
      surrounding debris.

Mask outline convention (white-with-black-halo) matches the chapter-4 chip
exemplars (F6), the D4 group transformations (F7), and the augmentation
gallery (F8) so this figure reads as a continuation of the chapter-4 visual
vocabulary rather than as a new convention.

Source artifacts: the cached val pools at
`outputs/ablation/_ensembles/all_features__{Hurricane_Michael, Mayfield_Tornado}_probs.pt`
plus the seed-00 `config_resolved.yaml` snapshots under `outputs/ablation/all_features/`, accessed via the
parameterized ``load_reference_val_chip(holdout=...)`` helper in
``_common.py``.
"""
from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from scripts.visualization._common import (
    WIDTH_2COL,
    load_reference_val_chip,
    save_caption,
    save_figure,
    setup_publication_style,
)

# Match the chapter-4 chip-exemplar mask-outline convention.
_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 2.6
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 1.3

# Per-panel selection. rng_seed indexes into the descending-confidence
# ranking of correct Destroyed predictions for the holdout. The chips are the
# two shown in the v1 manuscript (Michael val index 124, Mayfield val index
# 1723), chosen from a 6-candidate preview: the Michael chip because it places
# the most footprint in frame while still showing residual damaged-structure
# pixels; the Mayfield chip because it cleanly shows the foundation-only /
# debris-radiating kinematic without ambiguous standing walls. Under the DGX
# all_features ensemble the same chips sit at ranks 3 and 0 respectively
# (they were ranks 1 and 0 under the v1 ensemble).
_PANELS: Final[tuple[tuple[str, int], ...]] = (
    ("Hurricane_Michael", 3),
    ("Mayfield_Tornado", 0),
)


def _draw_panel(ax: plt.Axes, *, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Render one chip with its footprint polygon overlaid as white-with-halo.

    No on-figure title or label - per-panel identification lives in the
    chapter's Quarto caption and the .caption.md sidecar, matching the
    convention established by the chapter-4 chip exemplars.
    """

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


def main() -> None:
    """Render the 1x2 footprint-mask alignment exemplar."""

    setup_publication_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_2COL, 3.55),
        gridspec_kw={
            "wspace": 0.04,
            "left": 0.005,
            "right": 0.995,
            "top": 0.99,
            "bottom": 0.01,
        },
    )

    for ax, (holdout, rng_seed) in zip(axes, _PANELS, strict=True):
        chip = load_reference_val_chip(
            class_name="Destroyed",
            rng_seed=rng_seed,
            require_correct=True,
            holdout=holdout,
        )
        _draw_panel(ax, rgb=chip["rgb"], mask=chip["mask"])

    save_figure(fig=fig, name="mask_alignment_exemplar")

    save_caption(
        name="mask_alignment_exemplar",
        title=(
            "Footprint-mask alignment exemplar: hurricane-class kinematics "
            "(footprint co-located with damage) vs. tornado-class kinematics "
            "(footprint over empty foundation)."
        ),
        body=(
            "Both panels show the most-confident correct Destroyed-class "
            "exemplar from each holdout's 10-seed full-configuration ensemble, "
            "with the ground-truth structure footprint polygon traced as a "
            "white-with-black-halo contour - the same convention used for the "
            "chip exemplars, D4 group transformations, and "
            "augmentation gallery figures. Left: Hurricane Michael "
            "(proximate OOD). The footprint cleanly bounds the structure's "
            "remaining damaged material; mask-conditioned attention points "
            "at pixels that contain damage signal. Right: Mayfield Tornado "
            "(distant OOD). The footprint covers a near-empty foundation "
            "slab while debris and the post-event visual evidence sit in "
            "the surrounding pixels; mask-conditioned attention points at "
            "empty pixels and away from the rubble that carries the actual "
            "damage signal. The structural difference between the two "
            "kinematic patterns - the structure stays vs. the structure "
            "displaces - is exactly the latent variable that explains the "
            "mask ablation's split-dependent signature. Source artifacts: "
            "outputs/ablation/_ensembles/all_features__{Hurricane_Michael, Mayfield_Tornado}_probs.pt "
            "and the seed_00/config_resolved.yaml fold-config snapshots under outputs/ablation/all_features/."
        ),
    )


if __name__ == "__main__":
    main()
