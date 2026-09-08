"""F-MCDN - MCDN architecture overview block diagram.

Renders the Mask Centered Damage Net as a top-down block diagram for the
journal Methodology section (doc/v2/mask_centered_damage_net_rev2.tex,
figure label fig:mcdn-block).
The four MCDN-specific architectural commitments beyond a stock ConvNeXt~v2
Nano backbone - early-fusion four-channel stem, FiLM typology conditioning,
mask-weighted pooling at each consumed stage, and cross-scale fusion - are
highlighted in the steel-blue palette over a neutral input/backbone/head
spine, mirroring F2's "the architectural commitment under discussion"
convention.

The backbone is intentionally drawn as a single block: its internal stage
hierarchy is rendered in F1 (``fig_convnext_hierarchy``), so this figure
keeps focus on MCDN-specific additions - where the typology vector enters
(per-stage FiLM modulation), where the mask informs pooling readouts
beyond the four-channel stem, and how the two backbone-stage feature maps
fuse into a single head-ready embedding.

The two parallel stage-3 / stage-4 streams are drawn as two columns rather
than collapsed into a single block because the chapter argues that two
scales is the minimum defensible representation: stage-3 carries fine
texture cues and stage-4 carries envelope cues, and the cross-scale mixer
is the architectural commitment that earns the two-stream cost.

Usage::

    uv run python -m scripts.visualization.fig_mcdn_block
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.visualization._common import (
    FIG_ANNOT_PT,
    FIG_FONT_PT,
    WIDTH_1COL,
    save_caption,
    save_figure,
    setup_publication_style,
)


# ============================================================================
# Layout constants. Axis units == inches given fig size + aspect="equal".
# ============================================================================

_X_LEFT: Final[float] = 0.40
_X_RIGHT: Final[float] = 5.80
_X_TOTAL: Final[float] = _X_RIGHT - _X_LEFT
_Y_TOTAL: Final[float] = 6.30

# The canvas is wider than the single column it is placed in, so LaTeX scales it down by
# WIDTH_1COL / _X_TOTAL. Font sizes are scaled up by the inverse so labels print at
# FIG_FONT_PT and sublabels at FIG_ANNOT_PT once the figure is at column width.
_PLACEMENT_SCALE: Final[float] = _X_TOTAL / WIDTH_1COL
_LABEL_PT: Final[float] = FIG_FONT_PT * _PLACEMENT_SCALE
_SUBLABEL_PT: Final[float] = FIG_ANNOT_PT * _PLACEMENT_SCALE

# Y-positions for each band's top edge (descending). The Input and Backbone
# tops sit just above their pre-compression values; the +0.04 lift over
# the original 5.30 baseline equalises the whitespace between the
# MLP-modulation manifold (steel-blue, axis y=4.42) and the Backbone box
# bottom (axis y=4.54) with the whitespace between the steel-blue
# manifold and the backbone-fork manifold (dark grey, axis y=4.30): both
# now span 0.12 axis units.
_Y_TOP_INPUT: Final[float] = 6.14
_Y_TOP_BACKBONE: Final[float] = 5.34
_Y_TOP_FILM: Final[float] = 4.00
_Y_TOP_POOL: Final[float] = 3.20
_Y_TOP_FUSION: Final[float] = 2.30
_Y_TOP_HEAD: Final[float] = 1.50
_Y_TOP_OUTPUT: Final[float] = 0.75

# Box heights per band.
_H_INPUT: Final[float] = 0.50
_H_TYPOLOGY: Final[float] = 0.42
_H_BACKBONE: Final[float] = 0.80
_H_MLP: Final[float] = 0.55
_H_FILM: Final[float] = 0.55
_H_POOL: Final[float] = 0.55
_H_FUSION: Final[float] = 0.55
_H_HEAD: Final[float] = 0.50
_H_OUTPUT: Final[float] = 0.50

# Column x-centers. The diagram spine (FiLM-merge, Pool-merge, Fusion, Head,
# Output, backbone-fork trunk and manifold midpoint) sits at ``_X_CENTER``.
# The two stream columns ``_X_S3`` and ``_X_S4`` are placed close to
# ``_X_CENTER`` to read as a paired two-stream structure rather than as
# widely separated independent columns.
#
# The four top-row blocks (Input, Backbone, Typology, MLP) form an
# asymmetric two-column group: Input/Backbone on the left at
# ``_X_INPUT_BACKBONE`` and Typology/MLP on the right at ``_X_TYPOLOGY`` /
# ``_X_MLP``. These four x-centers are chosen so the group's visual centre
# (Backbone left edge to Typology right edge) lands on the spine
# ``_X_CENTER``, which would otherwise lean rightward because the side
# column has no symmetric left counterpart.
#
# The backbone-fork trunk leaves the Backbone box at the spine x rather
# than the Backbone-box centre, which means the trunk exits from a point
# slightly right of the Backbone-box centre. This is preferred over
# kinking the trunk into a dogleg or making the fork manifold
# asymmetric about its trunk landing point.
_X_CENTER: Final[float] = 3.10
_X_S3: Final[float] = 2.05
_X_S4: Final[float] = 4.15
_X_INPUT_BACKBONE: Final[float] = 2.10
_X_TYPOLOGY: Final[float] = 4.80
_X_MLP: Final[float] = 4.80

# Box widths.
_W_INPUT: Final[float] = 2.6
_W_TYPOLOGY: Final[float] = 1.6
_W_BACKBONE: Final[float] = 3.0
_W_MLP: Final[float] = 1.6
_W_FILM: Final[float] = 1.7
_W_POOL: Final[float] = 1.9  # sized for "Mask-Weighted Pool" at column-width 8 pt bold
_W_FUSION: Final[float] = 3.6
_W_HEAD: Final[float] = 2.4
_W_OUTPUT: Final[float] = 4.0

# Y-coordinates for the two manifold buses (backbone-fork and MLP-modulation).
# Both sit in the band between the backbone/MLP block bottoms (axis y=4.50,
# 4.75) and the FiLM block tops (axis y=4.00). The MLP-modulation bus is
# placed above the backbone-fork bus so the steel-blue conditioning pathway
# visually arches over the neutral feature-flow fork; the MLP-modulation
# legs cross the backbone-fork manifold at one point each, with palette
# disambiguating the pathways at the crossing. Manifold y-positions chosen
# so both leg sets render at length >= 0.30 axis units (the threshold below
# which arrows visually stub against the FiLM box tops).
_Y_BACKBONE_MANIFOLD: Final[float] = 4.30
_Y_MLP_MANIFOLD: Final[float] = 4.42

# X-positions where the MLP-modulation legs land on the FiLM block top
# edges. Each leg is offset 0.30 axis units toward the spine center from
# its FiLM column center (_X_S3 = 2.10, _X_S4 = 4.10), making the MLP
# pathway mirror-symmetric across the spine: FiLM_3 receives MLP from the
# right of its box center (toward spine), FiLM_4 receives MLP from the
# left of its box center (also toward spine). The backbone-fork legs land
# at the FiLM box centers themselves, so each FiLM box has the feature
# arrow at center and the conditioning arrow leaning spine-ward.
_X_MLP_LEG_S3: Final[float] = 2.40
_X_MLP_LEG_S4: Final[float] = 3.80

# Colors. Mirrors F2 / F5 conventions.
_NEUTRAL_FILL: Final[str] = "#F0F0F0"
_NEUTRAL_EDGE: Final[str] = "#666666"
_INPUT_FILL: Final[str] = "#E0E0E0"
_HIGHLIGHT_FILL: Final[str] = "#D6E4F2"
_HIGHLIGHT_EDGE: Final[str] = "#2F5C8C"
_ARROW_COLOR: Final[str] = "#444444"
_MOD_COLOR: Final[str] = _HIGHLIGHT_EDGE


@dataclass(frozen=True)
class BoxSpec:
    """Specification for one labeled block in the diagram.

    Attributes:
        x_center: Horizontal center in axis units.
        y_top: Top edge in axis units.
        width: Box width in axis units.
        height: Box height in axis units.
        label: Primary text drawn at the top-center of the box. Inline math
            is permitted via standard ``$ ... $`` delimiters.
        sublabel: Optional second line of text drawn beneath ``label`` in a
            smaller italic style. Used for shape annotations and
            sub-mechanism notes.
        highlight: When True, render in the steel-blue MCDN-commitment
            palette with a thicker edge and bold label.
        is_io: When True, render in the neutral input/output palette (a
            slightly darker grey than standard operation blocks) with a
            bold label.
    """

    x_center: float
    y_top: float
    width: float
    height: float
    label: str
    sublabel: str | None = None
    highlight: bool = False
    is_io: bool = False


def _draw_box(ax: plt.Axes, *, spec: BoxSpec) -> None:
    """Draw a rounded labeled rectangle in the project's box-diagram style."""

    if spec.highlight:
        face = _HIGHLIGHT_FILL
        edge = _HIGHLIGHT_EDGE
        edge_width = 1.4
        weight = "bold"
    elif spec.is_io:
        face = _INPUT_FILL
        edge = _NEUTRAL_EDGE
        edge_width = 0.9
        weight = "bold"
    else:
        face = _NEUTRAL_FILL
        edge = _NEUTRAL_EDGE
        edge_width = 0.7
        weight = "bold"

    box = FancyBboxPatch(
        (spec.x_center - spec.width / 2.0, spec.y_top - spec.height),
        spec.width,
        spec.height,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        facecolor=face,
        edgecolor=edge,
        linewidth=edge_width,
        zorder=3,
    )
    ax.add_patch(box)

    y_center = spec.y_top - spec.height / 2.0

    if spec.sublabel is None:
        ax.text(spec.x_center, y_center, spec.label,
                ha="center", va="center",
                fontsize=_LABEL_PT, fontweight=weight, zorder=4)
    else:
        ax.text(spec.x_center, y_center + 0.09, spec.label,
                ha="center", va="center",
                fontsize=_LABEL_PT, fontweight=weight, zorder=4)
        ax.text(spec.x_center, y_center - 0.11, spec.sublabel,
                ha="center", va="center",
                fontsize=_SUBLABEL_PT, color="#555555", style="italic",
                zorder=4)


def _draw_arrow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    arrowed: bool = True,
    color: str = _ARROW_COLOR,
    linewidth: float = 1.0,
) -> None:
    """Draw a connecting line from ``start`` to ``end`` with optional arrowhead.

    ``shrinkA=0, shrinkB=0`` is set explicitly so the shaft extends to its
    nominal endpoints rather than being retracted by the FancyArrowPatch
    default of 2 pt at each end - that retraction creates a visible gap
    between shaft and arrowhead at the figure scale this diagram targets.
    """

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>" if arrowed else "-",
        mutation_scale=9 if arrowed else 0,
        shrinkA=0,
        shrinkB=0,
        color=color,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(arrow)


def _draw_segment(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = _ARROW_COLOR,
    linewidth: float = 1.0,
) -> None:
    """Draw a non-arrowed line segment - used for trunks and bus segments."""

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=2,
    )


def _draw_backbone_fork(ax: plt.Axes) -> None:
    """Draw the backbone -> FiLM_3 / FiLM_4 fork in neutral palette."""

    backbone_bottom = _Y_TOP_BACKBONE - _H_BACKBONE
    film_top = _Y_TOP_FILM
    manifold_y = _Y_BACKBONE_MANIFOLD

    # Trunk from backbone center down to the manifold.
    _draw_segment(ax=ax,
                  start=(_X_CENTER, backbone_bottom),
                  end=(_X_CENTER, manifold_y))

    # Horizontal manifold spanning both columns.
    _draw_segment(ax=ax,
                  start=(_X_S3, manifold_y),
                  end=(_X_S4, manifold_y))

    # Legs down to FiLM_3 / FiLM_4 tops.
    _draw_arrow(ax=ax,
                start=(_X_S3, manifold_y),
                end=(_X_S3, film_top + 0.005))
    _draw_arrow(ax=ax,
                start=(_X_S4, manifold_y),
                end=(_X_S4, film_top + 0.005))


def _draw_mlp_modulation_fork(ax: plt.Axes) -> None:
    """Draw the MLP -> FiLM_3 / FiLM_4 modulation fork in highlight palette.

    The MLP-modulation manifold sits above the backbone-fork manifold so the
    steel-blue conditioning pathway visually arches over the neutral
    feature-flow fork. The legs land on the FiLM box top edges at an
    x-offset right of the backbone-fork legs so both pathways enter each
    FiLM box at distinguishable positions.
    """

    mlp_bottom = _Y_TOP_BACKBONE - _H_MLP
    film_top = _Y_TOP_FILM
    manifold_y = _Y_MLP_MANIFOLD

    # Trunk from MLP center down to the modulation manifold.
    _draw_segment(ax=ax,
                  start=(_X_MLP, mlp_bottom),
                  end=(_X_MLP, manifold_y),
                  color=_MOD_COLOR, linewidth=1.0)

    # Horizontal modulation bus from MLP column over to the leftmost leg.
    _draw_segment(ax=ax,
                  start=(_X_MLP_LEG_S3, manifold_y),
                  end=(_X_MLP, manifold_y),
                  color=_MOD_COLOR, linewidth=1.0)

    # Legs down to FiLM_3 / FiLM_4 top edges (right of backbone-fork legs).
    # The (gamma, beta) modulation parameters are not labeled on the legs
    # because the FiLM-equation sublabel inside each FiLM box already
    # carries them; doubling the annotation crowds the band between the
    # backbone manifold and the FiLM tops without adding signal.
    _draw_arrow(ax=ax,
                start=(_X_MLP_LEG_S3, manifold_y),
                end=(_X_MLP_LEG_S3, film_top + 0.005),
                color=_MOD_COLOR, linewidth=1.0)
    _draw_arrow(ax=ax,
                start=(_X_MLP_LEG_S4, manifold_y),
                end=(_X_MLP_LEG_S4, film_top + 0.005),
                color=_MOD_COLOR, linewidth=1.0)


def _draw_pool_to_fusion_merge(ax: plt.Axes) -> None:
    """Draw the Pool_3 / Pool_4 -> Fusion merge using a manifold pattern."""

    pool_bottom = _Y_TOP_POOL - _H_POOL
    fusion_top = _Y_TOP_FUSION
    manifold_y = (pool_bottom + fusion_top) / 2.0

    # Trunks from each pool block down to the merge manifold.
    _draw_segment(ax=ax,
                  start=(_X_S3, pool_bottom),
                  end=(_X_S3, manifold_y))
    _draw_segment(ax=ax,
                  start=(_X_S4, pool_bottom),
                  end=(_X_S4, manifold_y))

    # Horizontal manifold and arrow into the fusion top.
    _draw_segment(ax=ax,
                  start=(_X_S3, manifold_y),
                  end=(_X_S4, manifold_y))
    _draw_arrow(ax=ax,
                start=(_X_CENTER, manifold_y),
                end=(_X_CENTER, fusion_top + 0.005))


CAPTION_TITLE: Final[str] = (
    "MCDN architecture overview."
)
CAPTION_BODY: Final[str] = (
    "MCDN extends a ConvNeXt~v2 Nano backbone with four mask- and "
    "typology-conditioned commitments (steel-blue): a four-channel stem "
    "that fuses the rasterized footprint mask with RGB at the network "
    "entry; Feature-wise Linear Modulation (FiLM) of the stage-3 and "
    "stage-4 feature maps by a typology one-hot projected through a "
    "two-layer MLP; mask-weighted pooling at each consumed stage, "
    "concatenating four readouts ($g_{\\mathrm{avg}}$, $g_{\\mathrm{max}}$, "
    "$m_{\\mathrm{avg}}$, $m_{\\mathrm{max}}$) per stage; and a cross-"
    "scale residual mixer with a per-sample scale gate that fuses the "
    "two stage-specific embeddings into a single head-ready vector. "
    "The plain linear classifier and softmax output are deliberately "
    "conventional; ordinal structure is enforced by the squared Earth "
    "Mover's Distance loss with adjacency-aware label smoothing, not by "
    "a rank-consistent ordinal head. The backbone is rendered as a single "
    "block here; its internal stage hierarchy is shown separately in "
    "F1 (``fig_convnext_hierarchy``)."
)


def main() -> None:
    """Render the MCDN architecture overview block diagram."""

    setup_publication_style()

    fig, ax = plt.subplots(figsize=(_X_TOTAL, _Y_TOTAL),
                           constrained_layout=True)

    # ------------------------------------------------------------------
    # Top row: Input (left) and Typology one-hot (right).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_INPUT_BACKBONE,
        y_top=_Y_TOP_INPUT,
        width=_W_INPUT,
        height=_H_INPUT,
        label="RGB + Mask Input",
        sublabel=r"[B, 4, 512, 512]",
        is_io=True,
    ))
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_TYPOLOGY,
        y_top=_Y_TOP_INPUT - 0.04,
        width=_W_TYPOLOGY,
        height=_H_TYPOLOGY,
        label="Typology one-hot",
        is_io=True,
    ))

    # ------------------------------------------------------------------
    # Backbone row: ConvNeXt v2 Nano (left) and Typology MLP (right).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_INPUT_BACKBONE,
        y_top=_Y_TOP_BACKBONE,
        width=_W_BACKBONE,
        height=_H_BACKBONE,
        label="ConvNeXt v2 Nano Backbone",
        sublabel="(FCMAE+IN22k, 4-ch stem, zero-init mask)",
    ))
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_MLP,
        y_top=_Y_TOP_BACKBONE,
        width=_W_MLP,
        height=_H_MLP,
        label="MLP",
        sublabel=r"$4 \to 64 \to 128$",
    ))

    # ------------------------------------------------------------------
    # FiLM row (per stage, highlighted).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_S3,
        y_top=_Y_TOP_FILM,
        width=_W_FILM,
        height=_H_FILM,
        label="FiLM (stage 3)",
        sublabel=r"$(1+\gamma_3) \odot F_3 + \beta_3$",
        highlight=True,
    ))
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_S4,
        y_top=_Y_TOP_FILM,
        width=_W_FILM,
        height=_H_FILM,
        label="FiLM (stage 4)",
        sublabel=r"$(1+\gamma_4) \odot F_4 + \beta_4$",
        highlight=True,
    ))

    # ------------------------------------------------------------------
    # Pool row (per stage, highlighted).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_S3,
        y_top=_Y_TOP_POOL,
        width=_W_POOL,
        height=_H_POOL,
        label="Mask-Weighted Pool",
        sublabel="4 readouts",
        highlight=True,
    ))
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_S4,
        y_top=_Y_TOP_POOL,
        width=_W_POOL,
        height=_H_POOL,
        label="Mask-Weighted Pool",
        sublabel="4 readouts",
        highlight=True,
    ))

    # ------------------------------------------------------------------
    # Cross-scale fusion (highlighted).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_CENTER,
        y_top=_Y_TOP_FUSION,
        width=_W_FUSION,
        height=_H_FUSION,
        label="Cross-Scale Fusion",
        sublabel="zero-init residual mixer + per-sample scale gate",
        highlight=True,
    ))

    # ------------------------------------------------------------------
    # Linear head (neutral).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_CENTER,
        y_top=_Y_TOP_HEAD,
        width=_W_HEAD,
        height=_H_HEAD,
        label="Linear Head",
        sublabel=r"dropout 0.3 $\to$ 4 logits",
    ))

    # ------------------------------------------------------------------
    # Output (softmax + decoding rules).
    # ------------------------------------------------------------------
    _draw_box(ax=ax, spec=BoxSpec(
        x_center=_X_CENTER,
        y_top=_Y_TOP_OUTPUT,
        width=_W_OUTPUT,
        height=_H_OUTPUT,
        label="Softmax",
        sublabel="argmax (Macro-F1) / EV (QWK)",
        is_io=True,
    ))

    # ------------------------------------------------------------------
    # Connection arrows.
    # ------------------------------------------------------------------

    # Input -> Backbone (vertical, neutral). Drawn at the Input/Backbone
    # column centre so the arrow centres within both boxes; this is left of
    # the spine x because the four top blocks are shifted left as a group
    # to balance the rightward weight of the Typology/MLP side column.
    _draw_arrow(ax=ax,
                start=(_X_INPUT_BACKBONE, _Y_TOP_INPUT - _H_INPUT),
                end=(_X_INPUT_BACKBONE, _Y_TOP_BACKBONE + 0.005))

    # Typology -> MLP (vertical, neutral).
    _draw_arrow(ax=ax,
                start=(_X_TYPOLOGY, _Y_TOP_INPUT - 0.04 - _H_TYPOLOGY),
                end=(_X_MLP, _Y_TOP_BACKBONE + 0.005))

    # Backbone -> FiLM_3 / FiLM_4 (manifold-style fork, neutral).
    _draw_backbone_fork(ax=ax)

    # MLP -> FiLM_3 / FiLM_4 modulation (manifold-style fork, highlight).
    _draw_mlp_modulation_fork(ax=ax)

    # FiLM -> Pool (vertical legs, per stage).
    _draw_arrow(ax=ax,
                start=(_X_S3, _Y_TOP_FILM - _H_FILM),
                end=(_X_S3, _Y_TOP_POOL + 0.005))
    _draw_arrow(ax=ax,
                start=(_X_S4, _Y_TOP_FILM - _H_FILM),
                end=(_X_S4, _Y_TOP_POOL + 0.005))

    # Pool -> Fusion merge (manifold-style join).
    _draw_pool_to_fusion_merge(ax=ax)

    # Fusion -> Head -> Output spine.
    _draw_arrow(ax=ax,
                start=(_X_CENTER, _Y_TOP_FUSION - _H_FUSION),
                end=(_X_CENTER, _Y_TOP_HEAD + 0.005))
    _draw_arrow(ax=ax,
                start=(_X_CENTER, _Y_TOP_HEAD - _H_HEAD),
                end=(_X_CENTER, _Y_TOP_OUTPUT + 0.005))

    # Final framing. The xlim is set to ``[_X_LEFT, _X_RIGHT]`` rather than
    # to ``[0, fig_width]`` so the figure crops tightly around the diagram
    # content (Backbone left edge at x=0.60 to Typology right edge at
    # x=5.60) with a balanced 0.20 axis-unit margin on each side.
    ax.set_xlim(_X_LEFT, _X_RIGHT)
    ax.set_ylim(0.20, _Y_TOTAL)
    ax.set_aspect("equal")
    ax.set_axis_off()

    save_figure(fig=fig, name="mcdn_block")
    save_caption(name="mcdn_block", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
