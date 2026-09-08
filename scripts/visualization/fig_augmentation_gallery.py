"""F8 - Augmentation gallery on a Mayfield Tornado chip.

Renders 16 independent samples of the standard training augmentation pipeline
(``src.data.transform.get_train_transforms``) applied to the same Mayfield
Tornado chip, in a 4 x 4 grid. The full pipeline runs on each panel:

- Spatial group: ``HorizontalFlip`` (p=0.5), ``VerticalFlip`` (p=0.5),
  ``RandomRotate90`` (p=0.5)
- Photometric: ``MotionBlur`` (kernel 3-11, p=0.4), ``GaussNoise``
  (std range 0.02-0.05, p=0.4), ``ColorJitter`` (b/c/s 0.3, h 0.05, p=0.8)
- Occlusion: ``CoarseDropout`` (2-6 holes covering 6-20% of chip dim each,
  p=0.5)
- Mask corruption: random morphological erosion/dilation (kernel 3/5/7) +
  random translation up to 20 pixels (p=0.5; affects the mask channel only,
  the RGB tensor stays untouched)

Each panel applies the pipeline at a deterministic per-call seed so the
gallery reproduces; the variability across the 16 panels is the figure's
intent rather than per-augmentation isolation. The chapter prose enumerates
the augmentations individually; the gallery shows what the model trains
through.

Mask corruption effects (eroded/dilated/translated footprint outlines) are
visible because the white-with-black-halo mask contour is drawn on the
augmented mask, not the original. Mask shifts of up to 20 pixels at the
chip's 5 cm GSD correspond to ~1 m of footprint-cache misalignment - a
realistic operational error mode the augmentation simulates per
``src/data/transform.py`` lines 30-65.

Chips are downsampled from the native 512x512 to 256x256 for SVG embedding
size (16 base64 PNGs at full resolution would push the figure past 5 MB
without visible quality gain at the 1.6-inch panel display size).

Usage::

    uv run python -m scripts.visualization.fig_augmentation_gallery
"""
from __future__ import annotations

import random
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from scripts.visualization._common import (
    WIDTH_2COL,
    load_reference_val_chip,
    save_caption,
    save_figure,
    setup_publication_style,
)

_NUM_PANELS: Final[int] = 16
_GRID_ROWS: Final[int] = 4
_GRID_COLS: Final[int] = 4
_DISPLAY_SIZE: Final[int] = 256

_MASK_HALO_COLOR: Final[str] = "#000000"
_MASK_HALO_WIDTH: Final[float] = 1.8
_MASK_OUTLINE_COLOR: Final[str] = "#FFFFFF"
_MASK_OUTLINE_WIDTH: Final[float] = 0.9


def _downsample_rgb(rgb: np.ndarray, target: int) -> np.ndarray:
    """Bilinear downsample an ``[H, W, 3]`` uint8 chip to ``target x target``."""

    return np.array(Image.fromarray(rgb).resize((target, target), Image.BILINEAR))


def _downsample_mask(mask: np.ndarray, target: int) -> np.ndarray:
    """Nearest-neighbor downsample a binary ``[H, W]`` uint8 mask to ``target x target``.

    Nearest-neighbor (rather than bilinear) preserves the mask's binary
    character and avoids introducing fractional values around the polygon
    boundary that would muddy the contour render.
    """

    return np.array(Image.fromarray(mask).resize((target, target), Image.NEAREST))


def _draw_panel(ax: plt.Axes, *, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Render one augmented chip with the white-halo mask outline."""

    ax.imshow(rgb, interpolation="bilinear")
    ax.contour(mask.astype(float), levels=[0.5],
               colors=[_MASK_HALO_COLOR], linewidths=_MASK_HALO_WIDTH)
    ax.contour(mask.astype(float), levels=[0.5],
               colors=[_MASK_OUTLINE_COLOR], linewidths=_MASK_OUTLINE_WIDTH)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


CAPTION_TITLE: str = (
    "Augmentation gallery: 16 samples of the training pipeline on a Mayfield val chip."
)
CAPTION_BODY: str = (
    "Each panel applies the full training augmentation pipeline "
    "(``src/data/transform.py::get_train_transforms``) to the same base "
    "Mayfield Tornado chip ($Minor$ class, ``rng_seed=1``, the same exemplar "
    "shown in F6 and F7). The pipeline composes: D4 spatial transforms "
    "(``HorizontalFlip`` / ``VerticalFlip`` / ``RandomRotate90``, each "
    "p=0.5), photometric distortions (``MotionBlur`` p=0.4, ``GaussNoise`` "
    "p=0.4, ``ColorJitter`` p=0.8), occlusion (``CoarseDropout`` 2-6 holes "
    "of 6-20% chip dim each, p=0.5), and mask corruption (random "
    "morphological erosion/dilation with random translation up to 20 px, "
    "p=0.5).\n\n"
    "Each panel uses a deterministic per-call seed (``random.seed(0)`` "
    "through ``random.seed(15)``) so the gallery reproduces; the variability "
    "across the 16 panels is the figure's intent. The white-with-black-halo "
    "mask contour is drawn on the *augmented* mask in each panel - mask "
    "corruption effects (eroded/dilated/translated footprints) are visible "
    "alongside the RGB augmentations rather than hidden behind a "
    "constant-mask outline. Mask shifts up to 20 pixels at the chip's 5 cm "
    "GSD correspond to approximately 1 m of footprint-cache misalignment, "
    "which the augmentation simulates per ``src/data/transform.py`` lines "
    "30-65 to prevent the model from over-trusting an unperturbed footprint "
    "polygon at deployment."
)


def main() -> None:
    """Render the F8 augmentation gallery and its caption sidecar."""

    # Local import keeps the visualization import-light when only style + save
    # helpers are exercised by other scripts.
    from src.data.transform import get_train_transforms

    setup_publication_style()

    chip = load_reference_val_chip(class_name="Minor", rng_seed=1)
    rgb_base = chip["rgb"]
    mask_base = chip["mask"].astype(np.uint8)

    transform = get_train_transforms()

    fig, axes = plt.subplots(
        _GRID_ROWS, _GRID_COLS,
        figsize=(WIDTH_2COL, WIDTH_2COL),
        constrained_layout=True,
    )

    for panel_idx in range(_NUM_PANELS):
        # Stock albumentations transforms draw from the Compose-owned RNG (seeded
        # here); MaskCorruption draws from the stdlib global, so seed both.
        random.seed(panel_idx)
        transform.set_random_seed(panel_idx)

        result = transform(image=rgb_base, mask=mask_base)
        rgb_aug = result["image"]
        mask_aug = result["mask"]

        rgb_display = _downsample_rgb(rgb_aug, target=_DISPLAY_SIZE)
        mask_display = _downsample_mask(mask_aug, target=_DISPLAY_SIZE)

        ax = axes[panel_idx // _GRID_COLS, panel_idx % _GRID_COLS]
        _draw_panel(ax=ax, rgb=rgb_display, mask=mask_display)

    save_figure(fig=fig, name="augmentation_gallery")
    save_caption(name="augmentation_gallery",
                 title=CAPTION_TITLE, body=CAPTION_BODY)


if __name__ == "__main__":
    main()
