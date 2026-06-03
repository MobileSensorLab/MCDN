"""F2 - ConvNeXt-V2 block internal computation.

Renders the canonical ConvNeXt-V2 block as a vertical flow diagram suitable for
the [Chapter 4 Backbone subsection](../../doc/4-modeling.qmd#sec-modeling-backbone).
The seven operations - depthwise 7x7 conv, LayerNorm, pointwise expansion,
GELU, GRN, pointwise compression, DropPath - sit on a vertical main chain
with a residual-connection arc on the right side that bypasses every
operation and rejoins at the addition node before the block output.

The Global Response Normalization (GRN) layer is highlighted (steel-blue
palette, thicker edge) as the v2-specific addition over v1, supporting the
chapter's "Why v2 over v1" justification (the v2 paper retains the v1 block
shape at identical FLOPs and parameter count but inserts GRN to address
inter-channel feature collapse). Channel-count annotations on the pointwise
operations make the 4x expansion-and-compression trade explicit.

Source for block structure: ConvNeXt-V2 paper [@wooConvNeXtV22023] section 3
and the timm reference implementation. Highlighting choice mirrors F1's
"consumed by MCDN" highlighting so a reader looking at both figures
immediately maps "steel-blue = the architectural commitment under
discussion".

Usage::

    uv run python -m scripts.visualization.fig_convnext_block
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from scripts.visualization._common import (
    WIDTH_1COL,
    save_caption,
    save_figure,
    setup_publication_style,
)


@dataclass(frozen=True)
class BlockSpec:
    """Specification for one element in the vertical block-diagram chain.

    Attributes:
        label: Primary text drawn inside the element.
        kind: One of ``"io"`` (input/output marker), ``"op"`` (standard
            operation block), ``"highlight"`` (v2-specific addition), or
            ``"add"`` (residual addition node, drawn as a circle with a plus
            symbol rather than a labeled rectangle).
        sublabel: Optional second line of text drawn beneath ``label`` inside
            the element. Used for channel-count annotations on the pointwise
            convolutions.
    """

    label: str
    kind: str
    sublabel: str | None = None


CHAIN: Final[tuple[BlockSpec, ...]] = (
    BlockSpec(label="Input feature [C, H, W]", kind="io"),
    BlockSpec(label="DW Conv 7x7", kind="op"),
    BlockSpec(label="LayerNorm", kind="op"),
    BlockSpec(label="PW Conv 1x1", kind="op", sublabel="C -> 4C (expand)"),
    BlockSpec(label="GELU", kind="op"),
    BlockSpec(label="GRN (V2 addition)", kind="highlight"),
    BlockSpec(label="PW Conv 1x1", kind="op", sublabel="4C -> C (compress)"),
    BlockSpec(label="DropPath", kind="op"),
    BlockSpec(label="+", kind="add"),
    BlockSpec(label="Output feature [C, H, W]", kind="io"),
)

_X_CENTER: Final[float] = 1.0
_BLOCK_WIDTH: Final[float] = 1.6
_BLOCK_LEFT: Final[float] = _X_CENTER - _BLOCK_WIDTH / 2.0
_BLOCK_RIGHT: Final[float] = _X_CENTER + _BLOCK_WIDTH / 2.0
_SKIP_X: Final[float] = 2.55

_HEIGHT_IO: Final[float] = 0.50
_HEIGHT_OP: Final[float] = 0.42
_HEIGHT_ADD: Final[float] = 0.32
_GAP: Final[float] = 0.18
_TOP_PADDING: Final[float] = 0.30
_BOTTOM_PADDING: Final[float] = 0.30

_NEUTRAL_FILL: Final[str] = "#F0F0F0"
_NEUTRAL_EDGE: Final[str] = "#666666"
_IO_FILL: Final[str] = "#E0E0E0"
_HIGHLIGHT_FILL: Final[str] = "#D6E4F2"
_HIGHLIGHT_EDGE: Final[str] = "#2F5C8C"
_ADD_FILL: Final[str] = "#FFFFFF"


def _height_for(kind: str) -> float:
    """Return the rendered height (axis units) for a chain element by kind."""

    if kind == "io":
        return _HEIGHT_IO
    if kind == "add":
        return _HEIGHT_ADD
    return _HEIGHT_OP


def _draw_chain_block(
    ax: plt.Axes,
    *,
    spec: BlockSpec,
    y_top: float,
) -> tuple[float, float]:
    """Draw one chain element starting at ``y_top`` and return (y_top, y_bot)."""

    height = _height_for(spec.kind)
    y_bot = y_top - height
    y_center = (y_top + y_bot) / 2.0

    if spec.kind == "add":
        radius = height / 2.0
        circle = Circle(
            (_X_CENTER, y_center),
            radius=radius,
            facecolor=_ADD_FILL,
            edgecolor=_NEUTRAL_EDGE,
            linewidth=0.9,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(_X_CENTER, y_center, spec.label,
                ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=4)
        return y_top, y_bot

    if spec.kind == "highlight":
        face_color = _HIGHLIGHT_FILL
        edge_color = _HIGHLIGHT_EDGE
        edge_width = 1.4
        text_weight = "bold"
    elif spec.kind == "io":
        face_color = _IO_FILL
        edge_color = _NEUTRAL_EDGE
        edge_width = 0.9
        text_weight = "bold"
    else:
        face_color = _NEUTRAL_FILL
        edge_color = _NEUTRAL_EDGE
        edge_width = 0.7
        text_weight = "normal"

    box = FancyBboxPatch(
        (_BLOCK_LEFT, y_bot),
        _BLOCK_WIDTH,
        height,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        facecolor=face_color,
        edgecolor=edge_color,
        linewidth=edge_width,
        zorder=3,
    )
    ax.add_patch(box)

    if spec.sublabel is None:
        ax.text(_X_CENTER, y_center, spec.label,
                ha="center", va="center", fontsize=8,
                fontweight=text_weight, zorder=4)
    else:
        ax.text(_X_CENTER, y_center + 0.06, spec.label,
                ha="center", va="center", fontsize=8,
                fontweight=text_weight, zorder=4)
        ax.text(_X_CENTER, y_center - 0.08, spec.sublabel,
                ha="center", va="center", fontsize=7,
                color="#555555", style="italic", zorder=4)

    return y_top, y_bot


def _draw_arrow(ax: plt.Axes, *, x: float, y_start: float, y_end: float) -> None:
    """Draw a vertical down-pointing arrow connecting two y-positions."""

    arrow = FancyArrowPatch(
        (x, y_start),
        (x, y_end),
        arrowstyle="-|>",
        mutation_scale=8,
        color="#444444",
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(arrow)


_SKIP_COLOR: Final[str] = "#444444"


def _draw_skip_connection(
    ax: plt.Axes,
    *,
    y_input_center: float,
    y_add_center: float,
) -> None:
    """Draw the residual-connection path as three explicit segments.

    The path leaves the input block's right edge, runs right to the skip
    column, descends to the addition-node y level, then enters the addition
    node from its right side with an arrowhead. Drawn in a neutral dark grey
    so it does not compete visually with the GRN highlight (which uses the
    steel-blue palette to mark the V2-specific addition).
    """

    add_right_edge = _X_CENTER + _HEIGHT_ADD / 2.0

    # Top horizontal: input right -> skip column.
    ax.plot(
        [_BLOCK_RIGHT, _SKIP_X],
        [y_input_center, y_input_center],
        color=_SKIP_COLOR,
        linewidth=1.0,
        solid_capstyle="round",
        zorder=2,
    )

    # Vertical descent along the skip column.
    ax.plot(
        [_SKIP_X, _SKIP_X],
        [y_input_center, y_add_center],
        color=_SKIP_COLOR,
        linewidth=1.0,
        solid_capstyle="round",
        zorder=2,
    )

    # Bottom horizontal with arrowhead: skip column -> addition node right.
    arrow = FancyArrowPatch(
        (_SKIP_X, y_add_center),
        (add_right_edge, y_add_center),
        arrowstyle="-|>",
        mutation_scale=8,
        color=_SKIP_COLOR,
        linewidth=1.0,
        zorder=2,
    )
    ax.add_patch(arrow)

    # Compact inline marker at the top-right corner of the skip path so a
    # reader who is not yet sure what the long bypass is sees the word
    # "residual" without competing with the arrow itself.
    ax.text(
        (_BLOCK_RIGHT + _SKIP_X) / 2.0,
        y_input_center + 0.08,
        "residual",
        ha="center",
        va="bottom",
        fontsize=7,
        color=_SKIP_COLOR,
        style="italic",
        zorder=3,
    )


CAPTION_TITLE: str = (
    "ConvNeXt-V2 block internal computation."
)
CAPTION_BODY: str = (
    "The seven internal operations sit on a vertical main chain (depthwise "
    "7x7 conv, LayerNorm, pointwise expansion, GELU, GRN, pointwise "
    "compression, DropPath) with a residual-connection arc on the right side "
    "that bypasses every operation and rejoins at the addition node before "
    "the block output. Channel-count annotations on the pointwise blocks "
    "make the 4x expansion-and-compression trade explicit. **Global Response "
    "Normalization (GRN)** is highlighted (steel-blue fill, thicker edge) as "
    "the V2-specific addition over V1: the V2 paper [@wooConvNeXtV22023] "
    "retains the V1 block shape at identical FLOPs and parameter count but "
    "inserts GRN to address inter-channel feature collapse, yielding "
    "approximately one ImageNet-1k top-1 percentage point at zero "
    "architectural cost - the chapter's 'v2 over v1' justification "
    "([Chapter 4 Backbone](4-modeling.qmd#sec-modeling-backbone)). The "
    "highlight palette mirrors F1's 'consumed by MCDN' palette so a reader "
    "looking at both figures maps steel-blue to 'the architectural "
    "commitment under discussion'."
)


def main() -> None:
    """Render the F2 ConvNeXt-V2 block diagram and its caption sidecar."""

    setup_publication_style()

    total_height = (
        _TOP_PADDING
        + sum(_height_for(spec.kind) for spec in CHAIN)
        + _GAP * (len(CHAIN) - 1)
        + _BOTTOM_PADDING
    )

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 5.4), constrained_layout=True)

    y_cursor = total_height - _TOP_PADDING

    y_input_center: float | None = None
    y_add_center: float | None = None

    for idx, spec in enumerate(CHAIN):
        y_top, y_bot = _draw_chain_block(ax=ax, spec=spec, y_top=y_cursor)

        if spec.kind == "io" and idx == 0:
            y_input_center = (y_top + y_bot) / 2.0
        if spec.kind == "add":
            y_add_center = (y_top + y_bot) / 2.0

        if idx < len(CHAIN) - 1:
            next_y_top = y_bot - _GAP
            arrow_start = y_bot
            arrow_end = next_y_top - 0.005
            _draw_arrow(ax=ax, x=_X_CENTER, y_start=arrow_start, y_end=arrow_end)
            y_cursor = next_y_top

    if y_input_center is not None and y_add_center is not None:
        _draw_skip_connection(
            ax=ax,
            y_input_center=y_input_center,
            y_add_center=y_add_center,
        )

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, total_height)
    ax.set_aspect("auto")
    ax.set_axis_off()

    save_figure(fig=fig, name="convnext_block")
    save_caption(name="convnext_block", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
