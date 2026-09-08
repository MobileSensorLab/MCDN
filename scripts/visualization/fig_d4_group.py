"""F7 - D4 group transformations on a Mayfield chip.

Renders the eight elements of the dihedral-four (D4) symmetry group applied
to a single Mayfield Tornado val chip, mirroring the validation
test-time-augmentation pathway in ``src/model/trainer.py`` lines 886-894.

The trainer averages softmax probabilities across these eight views before
EV rounding for prediction and log-prob conversion for EMD loss; the figure
makes the symmetry-group structure of that averaging visible.

The figure layout maps:

- **Top row**: four rotations of the chip (0deg, 90deg, 180deg, 270deg)
  with identity (no flip).
- **Bottom row**: the same four rotations after a horizontal flip.

The vertical-flip element of D4 is implicit: it is the composition of a
180-degree rotation with a horizontal flip, which is exactly what the third
column of the bottom row visualizes. Because that composition is already in
the eight-view set, no explicit vertical flip is needed.

The mask channel transforms with the image under each D4 element (as the
trainer does), so the steel-blue mask outline rotates and flips alongside
the chip - a visual confirmation that the figure is equivariant to D4 by
construction.

Usage::

    uv run python -m scripts.visualization.fig_d4_group
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class D4Element:
    """Specification for one element of the D4 symmetry group.

    Attributes:
        rotations: Number of 90-degree counter-clockwise rotations applied
            via ``np.rot90(..., k=rotations)``.
        flip: ``True`` if a horizontal flip is composed AFTER the rotation
            (matching the trainer's ``torch.flip(rot_views, dims=[3])``
            ordering at lines 893-894).
        label: Display label drawn above the panel.
    """

    rotations: int
    flip: bool
    label: str


D4_ELEMENTS: Final[tuple[D4Element, ...]] = (
    D4Element(rotations=0, flip=False, label="rot 0$^\\circ$"),
    D4Element(rotations=1, flip=False, label="rot 90$^\\circ$"),
    D4Element(rotations=2, flip=False, label="rot 180$^\\circ$"),
    D4Element(rotations=3, flip=False, label="rot 270$^\\circ$"),
    D4Element(rotations=0, flip=True, label="rot 0$^\\circ$ + hflip"),
    D4Element(rotations=1, flip=True, label="rot 90$^\\circ$ + hflip"),
    D4Element(rotations=2, flip=True, label="rot 180$^\\circ$ + hflip"),
    D4Element(rotations=3, flip=True, label="rot 270$^\\circ$ + hflip"),
)
assert len(D4_ELEMENTS) == 8

_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 2.0
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 1.0


def _apply_d4(arr: np.ndarray, element: D4Element) -> np.ndarray:
    """Apply one D4 element to a 2-D or 3-D ``[H, W, ...]`` array.

    Mirrors ``torch.rot90(image, k=element.rotations, dims=[2, 3])`` followed
    optionally by ``torch.flip(rotated, dims=[3])`` from the trainer. The
    array layout here is ``[H, W, ...]`` (numpy-native) rather than
    ``[B, C, H, W]``, so the rot90 axis pair is ``(0, 1)`` and the flip is
    ``axis=1``.
    """

    out = np.rot90(arr, k=element.rotations, axes=(0, 1))
    if element.flip:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _draw_panel(ax: plt.Axes, *, rgb: np.ndarray, mask: np.ndarray, label: str) -> None:
    """Render one transformed chip with white-halo mask outline + transformation label."""

    ax.imshow(rgb, interpolation="bilinear")
    ax.contour(mask.astype(float), levels=[0.5],
               colors=[_MASK_HALO_COLOR], linewidths=_MASK_HALO_WIDTH)
    ax.contour(mask.astype(float), levels=[0.5],
               colors=[_MASK_OUTLINE_COLOR], linewidths=_MASK_OUTLINE_WIDTH)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(label, fontsize=8, pad=3)


CAPTION_TITLE: str = (
    "D4 group transformations applied to a Mayfield val chip."
)
CAPTION_BODY: str = (
    "The eight elements of the dihedral-four (D4) symmetry group are "
    "visualized on a single Mayfield Tornado val chip ($Minor$ class, "
    "``rng_seed=1`` - the same exemplar shown in F6). The trainer's "
    "validation pathway (``src/model/trainer.py`` lines 886-894) averages "
    "softmax probabilities over these eight views before expected-value "
    "rounding for prediction; this figure is what those eight views look "
    "like.\n\n"
    "**Top row**: four rotations of the chip (0$^\\circ$, 90$^\\circ$, "
    "180$^\\circ$, 270$^\\circ$) with no flip. **Bottom row**: the same "
    "four rotations composed with a horizontal flip. The vertical-flip "
    "element of D4 is implicitly present as ``rot 180$^\\circ$ + hflip`` "
    "in the third bottom-row panel - the composition of those two operations "
    "equals a vertical flip, so an explicit vflip view is redundant. The "
    "footprint mask transforms with the image under each D4 element (the "
    "white-with-black-halo outline rotates and flips alongside the chip), "
    "making the eight-view inference equivariance to D4 visually obvious."
)


def main() -> None:
    """Render the F7 D4 group figure and its caption sidecar."""

    setup_publication_style()

    chip = load_reference_val_chip(class_name="Minor", rng_seed=1)
    rgb_base = chip["rgb"]
    mask_base = chip["mask"]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(WIDTH_2COL, 4.0),
        constrained_layout=True,
    )

    for idx, element in enumerate(D4_ELEMENTS):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        rgb_view = _apply_d4(arr=rgb_base, element=element)
        mask_view = _apply_d4(arr=mask_base, element=element)
        _draw_panel(ax=ax, rgb=rgb_view, mask=mask_view, label=element.label)

    save_figure(fig=fig, name="d4_group")
    save_caption(name="d4_group", title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
