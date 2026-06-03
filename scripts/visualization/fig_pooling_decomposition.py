"""F5 - Mask-weighted pooling decomposition (schematic flow).

Renders the four pooled-readout branches MCDN concatenates at every backbone
stage as a top-down schematic in F2's vocabulary: a single feature map enters
at the top, splits into four parallel branches (``g_avg``, ``g_max``,
``m_avg``, ``m_max``), and reconverges into a 4C-wide concatenated embedding.
Each branch carries a mini chip thumbnail showing the spatial-weighting
contract that branch applies, plus the output shape annotation.

The mask-weighted branches (``m_avg`` and ``m_max``) are highlighted in
steel-blue, matching the chapter's visual convention that steel-blue marks
"the architectural commitment under discussion" - in F1 the MCDN-consumed
backbone stages, in F2 the V2-specific GRN, here the pooling pathway MCDN
adds beyond standard ConvNeXt.

Mini-thumbnails are rendered at small scale (1.0 inch square) on a real
Mayfield Tornado chip ("Minor" class, ``rng_seed=3`` selects an exemplar
with a centered footprint mask and gradient-argmax positions cleanly
separated between the global and mask-restricted cases). The activation
field used for the ``max`` markers is the squared central-difference
gradient magnitude with a 24-pixel edge zone suppressed - a stable
texture-energy proxy that puts the global argmax outside the building's
footprint and the mask-weighted argmax inside it, demonstrating the
``feat + (w_map - 1) * 1e4`` argmax-suppression mechanism from
``src/model/mcdn.py`` lines 54-55.

Usage::

    uv run python -m scripts.visualization.fig_pooling_decomposition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.visualization._common import (
    WIDTH_2COL,
    load_canonical_val_chip,
    save_caption,
    save_figure,
    setup_publication_style,
)


@dataclass(frozen=True)
class BranchSpec:
    """Specification for one pooling branch in the schematic.

    Attributes:
        name: Math symbol drawn at the top of the box (e.g., ``$g_{\\mathrm{max}}$``).
            The math symbol alone is sufficient for an ML reader; the chapter
            prose carries the long-form name.
        overlay: One of ``"uniform"``, ``"global_max"``, ``"mask_avg"``, or
            ``"mask_max"`` - selects which spatial-weighting visualization is
            painted on top of the chip thumbnail.
        consumed: ``True`` for the mask-weighted branches that get the
            steel-blue highlight palette.
    """

    name: str
    overlay: str
    consumed: bool


BRANCHES: Final[tuple[BranchSpec, ...]] = (
    BranchSpec(name="$g_{\\mathrm{avg}}$", overlay="uniform", consumed=False),
    BranchSpec(name="$g_{\\mathrm{max}}$", overlay="global_max", consumed=False),
    BranchSpec(name="$m_{\\mathrm{avg}}$", overlay="mask_avg", consumed=True),
    BranchSpec(name="$m_{\\mathrm{max}}$", overlay="mask_max", consumed=True),
)

# Layout (axis units; figure uses aspect="equal" so 1 axis unit = 1 inch).
_X_TOTAL: Final[float] = 7.0
_Y_TOTAL: Final[float] = 5.5

_INPUT_Y_TOP: Final[float] = 5.20
_INPUT_HEIGHT: Final[float] = 0.40
_INPUT_WIDTH: Final[float] = 3.20
_INPUT_X_CENTER: Final[float] = _X_TOTAL / 2.0

_BRANCH_Y_TOP: Final[float] = 4.10
_BRANCH_HEIGHT: Final[float] = 1.55
_BRANCH_WIDTH: Final[float] = 1.42
_BRANCH_X_CENTERS: Final[tuple[float, ...]] = (0.875, 2.625, 4.375, 6.125)

_HEADER_Y_TOP_OFFSET: Final[float] = 0.07
_THUMBNAIL_SIZE: Final[float] = 1.10
_THUMBNAIL_Y_TOP_OFFSET: Final[float] = 0.30
_FOOTER_GAP_BELOW_THUMB: Final[float] = 0.05

_CONCAT_Y_TOP: Final[float] = 1.78
_CONCAT_HEIGHT: Final[float] = 0.40
_CONCAT_WIDTH: Final[float] = 3.20

_NEUTRAL_FILL: Final[str] = "#F0F0F0"
_NEUTRAL_EDGE: Final[str] = "#666666"
_INPUT_FILL: Final[str] = "#E0E0E0"
_HIGHLIGHT_FILL: Final[str] = "#D6E4F2"
_HIGHLIGHT_EDGE: Final[str] = "#2F5C8C"
_ARROW_COLOR: Final[str] = "#444444"

_MARKER_FACE: Final[str] = "#E41A1C"
_MARKER_EDGE: Final[str] = "#FFFFFF"
_MARKER_SIZE: Final[float] = 26.0
_DIM_ALPHA: Final[float] = 0.55
_TINT_ALPHA: Final[float] = 0.30

_EDGE_SUPPRESS_PX: Final[int] = 24


def _texture_activation(rgb: np.ndarray) -> np.ndarray:
    """Edge-suppressed squared gradient magnitude on luminance (texture-energy proxy).

    Returns ``[H, W]`` float32. The ``_EDGE_SUPPRESS_PX``-pixel border is
    zeroed so the argmax is not dominated by chip-boundary discontinuities
    introduced when ``rasterio.read(boundless=True, fill_value=0)`` injects
    zeros for centroid windows that extend past orthomosaic bounds.
    """

    luminance = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                 + 0.114 * rgb[..., 2]).astype(np.float32)
    gy, gx = np.gradient(luminance)
    mag = gx ** 2 + gy ** 2
    edge = _EDGE_SUPPRESS_PX
    mag[:edge, :] = 0.0
    mag[-edge:, :] = 0.0
    mag[:, :edge] = 0.0
    mag[:, -edge:] = 0.0
    return mag


def _draw_box(
    ax: plt.Axes,
    *,
    x_center: float,
    y_top: float,
    width: float,
    height: float,
    label: str,
    fill: str,
    edge: str,
    edge_width: float,
    fontsize: float = 8.0,
    fontweight: str = "normal",
) -> None:
    """Draw a rounded rectangle with a centered label."""

    box = FancyBboxPatch(
        (x_center - width / 2.0, y_top - height),
        width,
        height,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        facecolor=fill,
        edgecolor=edge,
        linewidth=edge_width,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x_center, y_top - height / 2.0, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, zorder=4)


def _draw_thumbnail(
    ax: plt.Axes,
    *,
    x_center: float,
    y_center: float,
    size: float,
    rgb: np.ndarray,
    mask: np.ndarray,
    overlay: str,
    global_max_pos: tuple[int, int],
    mask_max_pos: tuple[int, int],
) -> None:
    """Draw a square chip thumbnail with the appropriate spatial-weighting overlay.

    ``size`` is the rendered side length in axis units (the chip is square).
    ``overlay`` selects the visualization mode per ``BranchSpec.overlay``.
    """

    extent = (
        x_center - size / 2.0,
        x_center + size / 2.0,
        y_center - size / 2.0,
        y_center + size / 2.0,
    )

    ax.imshow(rgb, extent=extent, aspect="auto",
              interpolation="bilinear", zorder=4)

    height, width = mask.shape

    def pixel_to_axis(row: int, col: int) -> tuple[float, float]:
        x = extent[0] + (col + 0.5) / width * size
        y = extent[3] - (row + 0.5) / height * size
        return x, y

    if overlay == "uniform":
        # Uniform steel-blue tint across the whole thumbnail.
        tint = np.zeros((height, width, 4), dtype=np.float32)
        tint[..., 0] = 0.184
        tint[..., 1] = 0.361
        tint[..., 2] = 0.549
        tint[..., 3] = _TINT_ALPHA
        ax.imshow(tint, extent=extent, aspect="auto",
                  interpolation="nearest", zorder=5)

    elif overlay == "global_max":
        # No spatial overlay; just a marker at the gradient argmax.
        x_marker, y_marker = pixel_to_axis(*global_max_pos)
        ax.scatter([x_marker], [y_marker],
                   s=_MARKER_SIZE, facecolors=_MARKER_FACE,
                   edgecolors=_MARKER_EDGE, linewidths=0.8, zorder=6)

    elif overlay == "mask_avg":
        # Dim everything outside the mask.
        outside_overlay = np.zeros((height, width, 4), dtype=np.float32)
        outside_overlay[mask == 0, 3] = _DIM_ALPHA
        ax.imshow(outside_overlay, extent=extent, aspect="auto",
                  interpolation="nearest", zorder=5)

    elif overlay == "mask_max":
        # Dim outside mask + marker at in-mask gradient argmax.
        outside_overlay = np.zeros((height, width, 4), dtype=np.float32)
        outside_overlay[mask == 0, 3] = _DIM_ALPHA
        ax.imshow(outside_overlay, extent=extent, aspect="auto",
                  interpolation="nearest", zorder=5)
        x_marker, y_marker = pixel_to_axis(*mask_max_pos)
        ax.scatter([x_marker], [y_marker],
                   s=_MARKER_SIZE, facecolors=_MARKER_FACE,
                   edgecolors=_MARKER_EDGE, linewidths=0.8, zorder=6)

    else:
        raise ValueError(f"Unknown overlay: {overlay!r}")


def _draw_arrow(ax: plt.Axes, *, start: tuple[float, float],
                end: tuple[float, float],
                arrowed: bool = True) -> None:
    """Draw a connecting line, optionally with an arrowhead at ``end``."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>" if arrowed else "-",
        mutation_scale=8 if arrowed else 0,
        color=_ARROW_COLOR,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(arrow)


def _draw_routing(
    ax: plt.Axes,
    *,
    input_y_bot: float,
    branch_y_top: float,
    branch_y_bot: float,
    concat_y_top: float,
) -> None:
    """Connect input -> 4 branches and 4 branches -> concat with manifold-style arrows.

    Pattern: a short vertical trunk from the input/concat block descends/ascends
    halfway to a horizontal manifold line spanning the four branch x-centers,
    with vertical legs reaching each branch top/bottom.
    """

    input_trunk_y = (input_y_bot + branch_y_top) / 2.0
    concat_trunk_y = (branch_y_bot + concat_y_top) / 2.0

    # Input -> manifold trunk.
    _draw_arrow(ax=ax,
                start=(_INPUT_X_CENTER, input_y_bot),
                end=(_INPUT_X_CENTER, input_trunk_y), arrowed=False)
    # Horizontal manifold across all branches.
    ax.plot(
        [_BRANCH_X_CENTERS[0], _BRANCH_X_CENTERS[-1]],
        [input_trunk_y, input_trunk_y],
        color=_ARROW_COLOR, linewidth=0.8, zorder=2,
    )
    # Manifold -> each branch top (with arrowhead).
    for x_branch in _BRANCH_X_CENTERS:
        _draw_arrow(ax=ax, start=(x_branch, input_trunk_y),
                    end=(x_branch, branch_y_top), arrowed=True)

    # Each branch bottom -> concat manifold.
    for x_branch in _BRANCH_X_CENTERS:
        _draw_arrow(ax=ax, start=(x_branch, branch_y_bot),
                    end=(x_branch, concat_trunk_y), arrowed=False)
    # Horizontal manifold above concat.
    ax.plot(
        [_BRANCH_X_CENTERS[0], _BRANCH_X_CENTERS[-1]],
        [concat_trunk_y, concat_trunk_y],
        color=_ARROW_COLOR, linewidth=0.8, zorder=2,
    )
    # Manifold -> concat top (with arrowhead).
    _draw_arrow(ax=ax,
                start=(_INPUT_X_CENTER, concat_trunk_y),
                end=(_INPUT_X_CENTER, concat_y_top), arrowed=True)


CAPTION_TITLE: str = (
    "Mask-weighted pooling decomposition: four parallel readouts concatenated to a 4C-wide embedding."
)
CAPTION_BODY: str = (
    "Each backbone-stage feature map ``[B, C, H, W]`` enters at the top and "
    "branches into four pooled readouts that concatenate into a single 4C-wide "
    "embedding before the projection MLP. **Global** branches "
    "($g_{\\mathrm{avg}}$, $g_{\\mathrm{max}}$) treat every spatial position "
    "equally; **mask-weighted** branches "
    "($m_{\\mathrm{avg}}$, $m_{\\mathrm{max}}$, in steel-blue) inject the "
    "footprint polygon as the spatial weight and constrain max selection to "
    "in-mask positions via the ``feat + (w_map - 1) * 1e4`` activation shift "
    "from ``src/model/mcdn.py`` lines 54-55. The mini-thumbnails show each "
    "branch's spatial-weighting contract on a real Mayfield Tornado *Minor*-"
    "class chip: $g_{\\mathrm{avg}}$ tints uniformly, $g_{\\mathrm{max}}$ "
    "marks the gradient argmax across the whole chip, $m_{\\mathrm{avg}}$ "
    "darkens everything outside the footprint, and $m_{\\mathrm{max}}$ marks "
    "the in-mask gradient argmax. The two argmax markers land in different "
    "parts of the chip - global outside the building, mask-weighted "
    "inside - which is the contract MCDN's pooling head exploits to keep "
    "structural and peripheral signal jointly available downstream. "
    "Activation values for marker placement are a squared-gradient texture "
    "proxy (not a real forward pass through the trained backbone), with the "
    "outer 24-pixel chip border suppressed to avoid boundary-discontinuity "
    "artifacts."
)


def main() -> None:
    """Render the F5 schematic-flow figure and its caption sidecar."""

    setup_publication_style()

    chip = load_canonical_val_chip(class_name="Minor", rng_seed=3)
    rgb = chip["rgb"]
    mask = chip["mask"]

    activation = _texture_activation(rgb=rgb)
    masked_activation = activation * mask.astype(np.float32)
    global_max_pos = np.unravel_index(int(activation.argmax()), activation.shape)
    mask_max_pos = np.unravel_index(int(masked_activation.argmax()), masked_activation.shape)

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, _Y_TOTAL),
                           constrained_layout=True)

    input_y_bot = _INPUT_Y_TOP - _INPUT_HEIGHT
    branch_y_bot = _BRANCH_Y_TOP - _BRANCH_HEIGHT
    concat_y_top = _CONCAT_Y_TOP

    _draw_routing(
        ax=ax,
        input_y_bot=input_y_bot,
        branch_y_top=_BRANCH_Y_TOP,
        branch_y_bot=branch_y_bot,
        concat_y_top=concat_y_top,
    )

    _draw_box(
        ax=ax,
        x_center=_INPUT_X_CENTER,
        y_top=_INPUT_Y_TOP,
        width=_INPUT_WIDTH,
        height=_INPUT_HEIGHT,
        label="Feature map  [B, C, H, W]",
        fill=_INPUT_FILL,
        edge=_NEUTRAL_EDGE,
        edge_width=0.9,
        fontsize=9,
        fontweight="bold",
    )

    for branch, x_center in zip(BRANCHES, _BRANCH_X_CENTERS, strict=True):
        fill = _HIGHLIGHT_FILL if branch.consumed else _NEUTRAL_FILL
        edge = _HIGHLIGHT_EDGE if branch.consumed else _NEUTRAL_EDGE
        edge_width = 1.4 if branch.consumed else 0.7

        # Branch outer container.
        outer = FancyBboxPatch(
            (x_center - _BRANCH_WIDTH / 2.0, branch_y_bot),
            _BRANCH_WIDTH,
            _BRANCH_HEIGHT,
            boxstyle="round,pad=0.005,rounding_size=0.04",
            facecolor=fill,
            edgecolor=edge,
            linewidth=edge_width,
            zorder=3,
        )
        ax.add_patch(outer)

        # Branch header: math-symbol-only at the top of the box. The math
        # notation alone is sufficient signal for an ML reader; chapter prose
        # carries any required long-form expansion.
        ax.text(x_center, _BRANCH_Y_TOP - _HEADER_Y_TOP_OFFSET, branch.name,
                ha="center", va="top", fontsize=11,
                fontweight="bold", zorder=5)

        # Thumbnail and footer anchored to a fixed offset from each other
        # rather than to the box's bottom edge - keeps the "thumbnail -> shape
        # annotation" pair tight regardless of branch-height changes.
        thumb_y_top = _BRANCH_Y_TOP - _THUMBNAIL_Y_TOP_OFFSET
        thumb_y_center = thumb_y_top - _THUMBNAIL_SIZE / 2.0
        thumb_y_bot = thumb_y_top - _THUMBNAIL_SIZE
        _draw_thumbnail(
            ax=ax,
            x_center=x_center,
            y_center=thumb_y_center,
            size=_THUMBNAIL_SIZE,
            rgb=rgb,
            mask=mask,
            overlay=branch.overlay,
            global_max_pos=global_max_pos,
            mask_max_pos=mask_max_pos,
        )

        ax.text(x_center, thumb_y_bot - _FOOTER_GAP_BELOW_THUMB,
                "\u2192 [B, C]",
                ha="center", va="top", fontsize=8,
                color="#444444", zorder=5)

    _draw_box(
        ax=ax,
        x_center=_INPUT_X_CENTER,
        y_top=_CONCAT_Y_TOP,
        width=_CONCAT_WIDTH,
        height=_CONCAT_HEIGHT,
        label="concat  \u2192  [B, 4C]",
        fill=_INPUT_FILL,
        edge=_NEUTRAL_EDGE,
        edge_width=0.9,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(0, _X_TOTAL)
    ax.set_ylim(_CONCAT_Y_TOP - _CONCAT_HEIGHT - 0.20, _Y_TOTAL)
    ax.set_aspect("equal")
    ax.set_axis_off()

    save_figure(fig=fig, name="pooling_decomposition")
    save_caption(name="pooling_decomposition", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
