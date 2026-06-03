"""F1 - ConvNeXt-V2 Nano spatial-channel hierarchy.

Renders six pseudo-3D feature-volume blocks left-to-right, one per level of the
backbone hierarchy on a 512-pixel input: the raw RGB-plus-mask input, the
stride-4 stem, and four backbone stages.

The **Input block** carries a real Mayfield Tornado *Minor*-class chip
thumbnail (the same exemplar shown in F6/F7/F8) on its front face, with the
white-with-black-halo footprint mask outline matching the chip-overlay
convention used in those figures. The pseudo-3D top and right faces remain
in neutral grey so the input keeps the prism vocabulary while being
visibly anchored in real data.

Each abstract block (Stem, Stage 1-4) is a beveled rectangular prism with
three pieces of texture that carry information beyond the block's overall
size:

- The **front face** carries an :math:`N \\times N` grid suggesting feature-map
  discretization at the relevant spatial resolution. The grid density scales
  log-linearly with the spatial dimension.
- The **right face** carries :math:`M` parallel depth-direction lines
  suggesting the stack of :math:`M`-binned channels at this level. The line
  density scales log-linearly with the channel count.
- The **front-face side length** is square-root scaled to spatial extent and
  the **depth extent** is square-root scaled to channel count, so the visual
  reads as "input is a tall thin slab (large spatial, three channels), Stage 4
  is a small deep cube (small spatial, six-hundred-forty channels)" - the
  canonical convolutional-network compression-and-channel-expansion trade.

The mixed convention (real chip at input, abstract prisms downstream) is
deliberate: the input *is* literally an image, while the stage outputs are
multi-channel abstract feature volumes that no single thumbnail can
honestly represent.

Stages 3 and 4 are highlighted (steel-blue palette + thicker edge + bottom
bracket) because MCDN consumes their feature maps via ``out_indices=(2, 3)``
(see ``src/model/mcdn.py`` line 113) for downstream FiLM modulation,
mask-weighted pooling, and cross-scale fusion.

Flow-operation labels (``v 4x``, ``v 2x``) sit in the gaps between blocks at
the stages that introduce a spatial downsample, making the architectural
compression event explicit.

Channel counts and per-stage block depths follow the
``convnextv2_nano.fcmae_ft_in22k_in1k`` checkpoint configuration:
``dims = [80, 160, 320, 640]``, ``depths = [2, 2, 8, 2]``.

Usage::

    uv run python -m scripts.visualization.fig_convnext_hierarchy
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon
from PIL import Image

from scripts.visualization._common import (
    WIDTH_2COL,
    load_canonical_val_chip,
    save_caption,
    save_figure,
    setup_publication_style,
)


def _downsample_array(arr: np.ndarray, *, target: int,
                      mode: Literal["bilinear", "nearest"]) -> np.ndarray:
    """Downsample a 2-D or 3-D ``[H, W, ...]`` uint8 array to ``target x target``.

    ``mode="bilinear"`` is used for RGB chips (smooth resampling); ``"nearest"``
    is used for binary masks so the polygon boundary stays sharp at the
    downsampled resolution.
    """

    resampling = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    return np.array(Image.fromarray(arr).resize((target, target), resampling))


@dataclass(frozen=True)
class StageSpec:
    """Specification for one pseudo-3D block in the hierarchy strip.

    Attributes:
        name: Header label drawn above the block (e.g., "Stage 3").
        channels: Channel count at this level. Used to scale the prism's depth
            extent and to compute the stacked-line count on the right face.
        spatial: One side of the square spatial extent, ``H == W``.
        depth: Per-stage block count. ``None`` for input and stem rows.
        consumed: ``True`` if MCDN consumes this stage's feature map.
        channel_label: Optional override for the channel-count display string.
            ``None`` falls back to ``f"{channels} ch"``. Used by the Input
            block to render ``3 + 1 ch`` (RGB + mask) instead of ``4 ch``,
            making the multi-modal-input decomposition explicit per
            remote-sensing publication convention.
    """

    name: str
    channels: int
    spatial: int
    depth: int | None
    consumed: bool
    channel_label: str | None = None


# Source: ``src/model/mcdn.py`` line 113 (``out_indices=(2, 3)``) and the
# ``convnextv2_nano.fcmae_ft_in22k_in1k`` timm checkpoint config. Input
# channels=4 reflects the post-concatenation stem input that ``mcdn.py``
# line 106 actually constructs (3 RGB + 1 mask).
HIERARCHY: Final[tuple[StageSpec, ...]] = (
    StageSpec(name="Input",   channels=4,   spatial=512, depth=None, consumed=False,
              channel_label="3 + 1 ch"),
    StageSpec(name="Stem",    channels=80,  spatial=128, depth=None, consumed=False),
    StageSpec(name="Stage 1", channels=80,  spatial=128, depth=2,    consumed=False),
    StageSpec(name="Stage 2", channels=160, spatial=64,  depth=2,    consumed=False),
    StageSpec(name="Stage 3", channels=320, spatial=32,  depth=8,    consumed=True),
    StageSpec(name="Stage 4", channels=640, spatial=16,  depth=2,    consumed=True),
)

# Flow-operation labels for the gap between HIERARCHY[k] and HIERARCHY[k+1].
# ``None`` means no label is drawn for that transition. Stem -> Stage 1 has no
# label because no spatial change occurs at that boundary; the spatial halving
# events at the start of Stages 2/3/4 are the architectural transitions worth
# annotating.
FLOW_OPS: Final[tuple[str | None, ...]] = (
    None,    # Input -> Stem: stem block already speaks for the patchify operation
    None,    # Stem -> Stage 1: no spatial change
    "v 2x",  # Stage 1 -> Stage 2 downsample
    "v 2x",  # Stage 2 -> Stage 3 downsample
    "v 2x",  # Stage 3 -> Stage 4 downsample
)
assert len(FLOW_OPS) == len(HIERARCHY) - 1

_BOX_X_SPACING: Final[float] = 1.05
_BASELINE_Y: Final[float] = 0.30
_MAX_FRONT_SIZE: Final[float] = 0.70
_MAX_DEPTH: Final[float] = 0.42
_DEPTH_ANGLE_DEG: Final[float] = 28.0

_NEUTRAL_FACES: Final[tuple[str, str, str]] = ("#E8E8E8", "#C8C8C8", "#A8A8A8")
_NEUTRAL_EDGE: Final[str] = "#666666"
_HIGHLIGHT_FACES: Final[tuple[str, str, str]] = ("#D6E4F2", "#A8C2DD", "#7FA1C7")
_HIGHLIGHT_EDGE: Final[str] = "#2F5C8C"

_TEXTURE_NEUTRAL: Final[str] = "#666666"
_TEXTURE_HIGHLIGHT: Final[str] = "#2F5C8C"
_TEXTURE_ALPHA: Final[float] = 0.35

# Mask outline colors and weights for the real-chip Input block.
_CHIP_MASK_HALO_COLOR: Final[str] = "#000000"
_CHIP_MASK_HALO_WIDTH: Final[float] = 1.4
_CHIP_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_CHIP_MASK_OUTLINE_WIDTH: Final[float] = 0.7


def _front_size(spatial: int, max_spatial: int) -> float:
    """Square-root scale a spatial dimension to a front-face side length."""

    return _MAX_FRONT_SIZE * math.sqrt(spatial / max_spatial)


def _depth_extent(channels: int, max_channels: int) -> float:
    """Square-root scale a channel count to a depth-direction extent."""

    return _MAX_DEPTH * math.sqrt(channels / max_channels)


def _grid_cell_count(spatial: int) -> int:
    """Number of front-face grid cells per side, log-scaled to spatial extent.

    Derived from ``round(log2(spatial)) - 1``, clamped to ``[2, 8]``. For the
    canonical hierarchy this yields ``[8, 6, 6, 5, 4, 3]`` for input through
    Stage 4 - a clear visual progression that matches the spatial-halving
    pattern without packing the smallest blocks too densely.
    """

    target = round(math.log2(spatial)) - 1
    return max(2, min(8, target))


def _stack_line_count(channels: int) -> int:
    """Number of right-face stacked-channel lines, log-scaled to channel count.

    Derived from ``max(0, round(log2(channels)) - 3)``. For the canonical
    hierarchy this yields ``[0, 3, 3, 4, 5, 6]`` for input through Stage 4 -
    a discrete-but-suggestive depth texture that says "more channels = more
    visible layers" without claiming literal channel counts.
    """

    return max(0, round(math.log2(channels)) - 3)


def _draw_pseudo_3d_block(
    ax: plt.Axes,
    *,
    x_center: float,
    front_size: float,
    depth_size: float,
    consumed: bool,
    n_grid_cells: int,
    n_stack_lines: int,
) -> None:
    """Draw one beveled rectangular prism with feature-map grid + channel-stack lines."""

    front_palette, top_palette, right_palette = (
        _HIGHLIGHT_FACES if consumed else _NEUTRAL_FACES
    )
    edge = _HIGHLIGHT_EDGE if consumed else _NEUTRAL_EDGE
    edge_width = 1.1 if consumed else 0.7
    texture_color = _TEXTURE_HIGHLIGHT if consumed else _TEXTURE_NEUTRAL

    angle_rad = math.radians(_DEPTH_ANGLE_DEG)
    dx = depth_size * math.cos(angle_rad)
    dy = depth_size * math.sin(angle_rad)

    half = front_size / 2.0
    x_left = x_center - half
    x_right = x_center + half
    y_bot = _BASELINE_Y - half
    y_top = _BASELINE_Y + half

    front_verts = [(x_left, y_top), (x_right, y_top), (x_right, y_bot), (x_left, y_bot)]
    top_verts = [
        (x_left, y_top),
        (x_left + dx, y_top + dy),
        (x_right + dx, y_top + dy),
        (x_right, y_top),
    ]
    right_verts = [
        (x_right, y_top),
        (x_right + dx, y_top + dy),
        (x_right + dx, y_bot + dy),
        (x_right, y_bot),
    ]

    # Faces in back-to-front render order so the front face occludes the back
    # edges of the top and right faces.
    ax.add_patch(Polygon(top_verts, closed=True, facecolor=top_palette,
                         edgecolor=edge, linewidth=edge_width, zorder=2))
    ax.add_patch(Polygon(right_verts, closed=True, facecolor=right_palette,
                         edgecolor=edge, linewidth=edge_width, zorder=2))

    # Right-face stack lines: each line marks one notional channel-stack layer
    # boundary, drawn from the front-right edge to the back-right edge.
    for k in range(1, n_stack_lines + 1):
        y_at_front = y_bot + (k / (n_stack_lines + 1)) * front_size
        y_at_back = y_at_front + dy
        ax.plot([x_right, x_right + dx], [y_at_front, y_at_back],
                color=texture_color, alpha=_TEXTURE_ALPHA, linewidth=0.5, zorder=3)

    ax.add_patch(Polygon(front_verts, closed=True, facecolor=front_palette,
                         edgecolor=edge, linewidth=edge_width, zorder=4))

    # Front-face grid: vertical and horizontal lines partitioning the face into
    # ``n_grid_cells`` x ``n_grid_cells`` cells. Stylized representation of the
    # spatial discretization at this level.
    if n_grid_cells > 1:
        for i in range(1, n_grid_cells):
            x_grid = x_left + (i / n_grid_cells) * front_size
            ax.plot([x_grid, x_grid], [y_bot, y_top],
                    color=texture_color, alpha=_TEXTURE_ALPHA,
                    linewidth=0.4, zorder=5)
            y_grid = y_bot + (i / n_grid_cells) * front_size
            ax.plot([x_left, x_right], [y_grid, y_grid],
                    color=texture_color, alpha=_TEXTURE_ALPHA,
                    linewidth=0.4, zorder=5)


def _draw_input_chip_block(
    ax: plt.Axes,
    *,
    x_center: float,
    front_size: float,
    depth_size: float,
    rgb: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Draw the Input block with a real chip thumbnail on the front face.

    Top and right faces stay as neutral-grey pseudo-3D facets so the Input
    block keeps the prism vocabulary used by the abstract feature-volume
    blocks downstream. The front face is replaced by an ``imshow`` of the
    chip RGB at the same axis-coordinate extent the abstract grid would
    have occupied; the white-with-black-halo mask contour is drawn on top
    using the same convention as F6/F7/F8.
    """

    angle_rad = math.radians(_DEPTH_ANGLE_DEG)
    dx = depth_size * math.cos(angle_rad)
    dy = depth_size * math.sin(angle_rad)

    half = front_size / 2.0
    x_left = x_center - half
    x_right = x_center + half
    y_bot = _BASELINE_Y - half
    y_top = _BASELINE_Y + half

    top_verts = [
        (x_left, y_top),
        (x_left + dx, y_top + dy),
        (x_right + dx, y_top + dy),
        (x_right, y_top),
    ]
    right_verts = [
        (x_right, y_top),
        (x_right + dx, y_top + dy),
        (x_right + dx, y_bot + dy),
        (x_right, y_bot),
    ]
    ax.add_patch(Polygon(top_verts, closed=True, facecolor=_NEUTRAL_FACES[1],
                         edgecolor=_NEUTRAL_EDGE, linewidth=0.7, zorder=2))
    ax.add_patch(Polygon(right_verts, closed=True, facecolor=_NEUTRAL_FACES[2],
                         edgecolor=_NEUTRAL_EDGE, linewidth=0.7, zorder=2))

    # Real chip on the front face. The Input block's front_size is small
    # (~0.7 axis units) so a downsampled chip avoids unnecessary base64-PNG
    # bloat in the SVG without visible quality loss at this thumbnail scale.
    extent = (x_left, x_right, y_bot, y_top)
    ax.imshow(rgb, extent=extent, aspect="auto",
              interpolation="bilinear", zorder=4)

    # Front face border for crisp bounding.
    front_verts = [(x_left, y_top), (x_right, y_top),
                   (x_right, y_bot), (x_left, y_bot)]
    ax.add_patch(Polygon(front_verts, closed=True, facecolor="none",
                         edgecolor=_NEUTRAL_EDGE, linewidth=0.7, zorder=5))

    # Mask outline overlaid on the chip via meshgrid in axis coordinates so
    # the contour aligns with the imshow extent above.
    height, width = mask.shape
    xx = np.linspace(x_left, x_right, width)
    yy = np.linspace(y_top, y_bot, height)
    grid_x, grid_y = np.meshgrid(xx, yy)
    ax.contour(grid_x, grid_y, mask.astype(float), levels=[0.5],
               colors=[_CHIP_MASK_HALO_COLOR],
               linewidths=_CHIP_MASK_HALO_WIDTH, zorder=6)
    ax.contour(grid_x, grid_y, mask.astype(float), levels=[0.5],
               colors=[_CHIP_MASK_OUTLINE_COLOR],
               linewidths=_CHIP_MASK_OUTLINE_WIDTH, zorder=7)


def _draw_consumed_bracket(
    ax: plt.Axes,
    x_centers: list[float],
    consumed_indices: list[int],
    consumed_front_sizes: list[float],
    consumed_dxs: list[float],
    *,
    y_main: float,
) -> None:
    """Draw a horizontal bracket spanning the MCDN-consumed stages.

    The bracket starts at the leftmost consumed prism's front-bottom-left edge
    and extends to the rightmost consumed prism's right-back edge, so the
    right tick wraps past the depth extension of the rightmost stage. The
    bracket therefore visually encloses the entire bounding box of the
    consumed prisms rather than only their front faces.
    """

    if not consumed_indices:
        return

    left_idx = consumed_indices[0]
    right_idx = consumed_indices[-1]
    left_half = consumed_front_sizes[0] / 2.0
    right_half = consumed_front_sizes[-1] / 2.0
    right_dx = consumed_dxs[-1]

    x_left = x_centers[left_idx] - left_half - 0.05
    x_right = x_centers[right_idx] + right_half + right_dx + 0.05
    tick_height = 0.04

    ax.plot([x_left, x_right], [y_main, y_main],
            color=_HIGHLIGHT_EDGE, linewidth=1.4, zorder=6)
    ax.plot([x_left, x_left], [y_main, y_main + tick_height],
            color=_HIGHLIGHT_EDGE, linewidth=1.4, zorder=6)
    ax.plot([x_right, x_right], [y_main, y_main + tick_height],
            color=_HIGHLIGHT_EDGE, linewidth=1.4, zorder=6)

    midpoint = (x_left + x_right) / 2.0
    ax.text(midpoint, y_main - 0.04, "consumed by MCDN",
            ha="center", va="top", fontsize=7.5,
            color=_HIGHLIGHT_EDGE, fontweight="bold")


def _draw_flow_op_labels(
    ax: plt.Axes,
    x_centers: list[float],
    front_sizes: list[float],
    dxs: list[float],
) -> None:
    """Render small flow-operation labels in the gaps between adjacent prisms.

    Each label sits at the midpoint of the visible gap between the left
    prism's right-back edge (``x_center + front_size/2 + dx``) and the right
    prism's front-left edge (``x_center - front_size/2``). Anchoring to the
    actual silhouette gap rather than the midpoint of front-face centers
    keeps the label centered in the visible empty space even when depth
    offsets push the left prism's right-back edge well past its front-face
    right edge - especially at Stage 3, whose 0.262 axis-unit depth extension
    shifts the visual gap midpoint substantially right of the front-face
    midpoint.
    """

    for transition_idx, label in enumerate(FLOW_OPS):
        if label is None:
            continue
        left_right_back = (
            x_centers[transition_idx]
            + front_sizes[transition_idx] / 2.0
            + dxs[transition_idx]
        )
        right_front_left = (
            x_centers[transition_idx + 1]
            - front_sizes[transition_idx + 1] / 2.0
        )
        midpoint_x = (left_right_back + right_front_left) / 2.0
        ax.text(midpoint_x, _BASELINE_Y, label,
                ha="center", va="center", fontsize=6.5,
                color="#777777", style="italic", zorder=1)


CAPTION_TITLE: str = (
    "ConvNeXt-V2 Nano spatial-channel hierarchy under a 512 px input."
)
CAPTION_BODY: str = (
    "**Input** is rendered as a real Mayfield Tornado *Minor*-class chip "
    "(the same exemplar shown in F6/F7/F8) with the white-with-black-halo "
    "footprint mask outlined on the chip itself. The pseudo-3D top and right "
    "faces remain in neutral grey to maintain the prism vocabulary used by "
    "the abstract feature-volume blocks downstream. The Input block is "
    "labeled \"3 + 1 ch\" to make the multi-modal-input decomposition "
    "explicit: 3 RGB channels concatenate with 1 footprint-mask channel into "
    "the 4-channel stem input that ``src/model/mcdn.py`` line 106 actually "
    "constructs. Stem and Stages 1-4 stay as abstract pseudo-3D prisms "
    "because their internal feature maps are multi-channel volumes (80 to "
    "640 channels) that no single thumbnail can honestly represent.\n\n"
    "Each abstract block's **front-face side length** is square-root scaled "
    "to spatial extent ($H \\times W$); **depth extent** is square-root "
    "scaled to channel count. **Front-face grid density** is log-scaled to "
    "spatial dimension as a stylized representation of feature-map "
    "discretization at this level (6 cells per side at the stem down to "
    "3 at Stage 4). **Right-face stacked lines** are log-scaled to channel "
    "count as a stylized representation of the channel stack "
    "(3 lines at the stem - 80 channels - up to 6 lines at Stage 4 - "
    "640 channels). The visual progression encodes the canonical "
    "convolutional-network compression-with-channel-expansion trade: the "
    "input is a flat photograph, the stem patchifies it into a tall but "
    "shallow feature volume, and the deeper stages compress spatially while "
    "expanding channel-wise into small deep cubes.\n\n"
    "**Flow-operation labels** (\"v 2x\") sit in the gaps where spatial "
    "halving happens at the start of Stages 2, 3, and 4. Stages 3 and 4 "
    "carry the steel-blue palette, thicker edge, and bottom bracket because "
    "MCDN consumes their feature maps via ``out_indices=(2, 3)`` (see "
    "``src/model/mcdn.py`` line 113). Stage 3 is the network's deepest "
    "stage at 8 ConvNeXt-V2 blocks while every other stage carries only 2; "
    "the parameter mass of the backbone is concentrated where MCDN extracts "
    "its primary feature signal. Channel counts and per-stage block depths "
    "follow the ``convnextv2_nano.fcmae_ft_in22k_in1k`` checkpoint "
    "configuration (``dims = [80, 160, 320, 640]``, "
    "``depths = [2, 2, 8, 2]``)."
)


def main() -> None:
    """Render the F1 ConvNeXt-V2 Nano hierarchy figure and its caption sidecar."""

    setup_publication_style()

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.2), constrained_layout=True)

    # Real chip for the Input block (matches the F6/F7/F8 cross-figure
    # exemplar). Downsampled to 256x256 to keep the input-block raster
    # embedding small; at the ~0.7 axis-unit (~0.7 inch) front-face display
    # size the visible quality loss is negligible.
    chip = load_canonical_val_chip(class_name="Minor", rng_seed=1)
    chip_rgb = _downsample_array(chip["rgb"], target=256, mode="bilinear")
    chip_mask = _downsample_array(chip["mask"], target=256, mode="nearest")

    n_boxes = len(HIERARCHY)
    x_centers = [(0.5 + i) * _BOX_X_SPACING for i in range(n_boxes)]
    max_spatial = max(spec.spatial for spec in HIERARCHY)
    max_channels = max(spec.channels for spec in HIERARCHY)

    # Vertical layout below the blocks. Rows are positioned tight against the
    # largest block's floor with text-height-aware spacing: spatial label
    # (~0.10 axis units of 7.5pt text), block-depth label (~0.09 of 7pt text),
    # bracket main line + tick + bracket text. Each gap absorbs the rendered
    # text height plus a small buffer so smaller blocks do not leave the
    # bracket "floating" far below them.
    block_floor = _BASELINE_Y - _MAX_FRONT_SIZE / 2.0
    y_spatial = block_floor - 0.06
    y_depth = block_floor - 0.18
    y_bracket = block_floor - 0.34
    y_bracket_text = y_bracket - 0.04

    consumed_indices: list[int] = []
    consumed_front_sizes: list[float] = []
    consumed_dxs: list[float] = []
    all_front_sizes: list[float] = []
    all_dxs: list[float] = []

    angle_rad = math.radians(_DEPTH_ANGLE_DEG)

    for idx, (spec, x_center) in enumerate(zip(HIERARCHY, x_centers, strict=True)):
        front_size = _front_size(spatial=spec.spatial, max_spatial=max_spatial)
        depth_size = _depth_extent(channels=spec.channels, max_channels=max_channels)
        dx = depth_size * math.cos(angle_rad)
        all_front_sizes.append(front_size)
        all_dxs.append(dx)

        if idx == 0:
            # Input block: real chip on the front face, neutral pseudo-3D framing.
            _draw_input_chip_block(
                ax=ax,
                x_center=x_center,
                front_size=front_size,
                depth_size=depth_size,
                rgb=chip_rgb,
                mask=chip_mask,
            )
        else:
            n_grid_cells = _grid_cell_count(spec.spatial)
            n_stack_lines = _stack_line_count(spec.channels)
            _draw_pseudo_3d_block(
                ax=ax,
                x_center=x_center,
                front_size=front_size,
                depth_size=depth_size,
                consumed=spec.consumed,
                n_grid_cells=n_grid_cells,
                n_stack_lines=n_stack_lines,
            )

        # Per-block labels (top and bottom) shift right by ``dx/2`` so each
        # block's label column tracks the prism's bounding-box x-center rather
        # than only its front-face center. The shift grows progressively with
        # depth, mirroring the prisms' rightward lean across the strip and
        # keeping each block's top and bottom labels in a single vertical
        # column rather than introducing an in-block shear.
        label_x = x_center + dx / 2.0

        channel_text = spec.channel_label or f"{spec.channels} ch"
        ax.text(label_x, 0.98, spec.name,
                ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(label_x, 0.81, channel_text,
                ha="center", va="top", fontsize=7.5, color="#444444")

        ax.text(label_x, y_spatial,
                f"{spec.spatial} x {spec.spatial}",
                ha="center", va="top", fontsize=7.5)
        if spec.depth is not None:
            depth_label = f"{spec.depth} block{'s' if spec.depth != 1 else ''}"
            ax.text(label_x, y_depth, depth_label,
                    ha="center", va="top", fontsize=7, color="#666666")

        if spec.consumed:
            consumed_indices.append(idx)
            consumed_front_sizes.append(front_size)
            consumed_dxs.append(dx)

    _draw_flow_op_labels(
        ax=ax,
        x_centers=x_centers,
        front_sizes=all_front_sizes,
        dxs=all_dxs,
    )

    _draw_consumed_bracket(
        ax=ax,
        x_centers=x_centers,
        consumed_indices=consumed_indices,
        consumed_front_sizes=consumed_front_sizes,
        consumed_dxs=consumed_dxs,
        y_main=y_bracket,
    )

    ax.set_xlim(-0.10, n_boxes * _BOX_X_SPACING + 0.10)
    ax.set_ylim(y_bracket_text - 0.06, 1.06)
    ax.set_aspect("equal")
    ax.set_axis_off()

    save_figure(fig=fig, name="convnext_hierarchy")
    save_caption(name="convnext_hierarchy", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
