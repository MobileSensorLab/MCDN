"""PyTorch Dataset implementation for mask-centered unitemporal damage assessment.

This module provides the core dataloading logic, converting orthomosaic-level
manifests into instance-level (building-specific) training chips. It generates
4-channel [R, G, B, Mask] tensors and extracts disaster typology context. When
``mask_dilation_px > 0``, the footprint mask is morphologically dilated
post-augmentation to compensate for label-source under-segmentation before being
stacked as the 4th channel.
"""

import albumentations
import cv2
import random
import rasterio
import torch

import numpy as np
import pandas as pd

from geopandas import GeoDataFrame
from pathlib import Path
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.windows import Window
from torch.utils.data import Dataset
from typing import Any, ClassVar

from src.data.geometry import VALID_ORDINAL_LABELS
from src.data.geometry import align_vector_to_raster
from src.data.geometry import apply_alignment_adjustments
from src.data.geometry import filter_polygons_valid_labels
from src.data.geometry import load_alignment_adjustments
from src.data.io import parse_crasar_json

ORDINAL_MAP = {
    "no damage": 0,
    "minor damage": 1,
    "major damage": 2,
    "destroyed": 3
}

# ImageNet normalization constants. The dataset emits uint8 [4, H, W] tensors; downstream
# consumers (trainer, ensemble inference, classical-ML probes) recover ImageNet-normalized
# floats via ``to_normalized_float``. Centralized here so the constants and the contract
# live in the same module.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def to_normalized_float(
    image: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cast a uint8 ``[..., 4, H, W]`` image tensor to float32 with ImageNet RGB normalization.

    The dataset returns uint8 to keep dataloader -> GPU bandwidth low; this helper applies
    the canonical conversion path (scale RGB by 1/255, shift by ImageNet mean, divide by std)
    on whatever device the caller supplies. The mask channel (index 3) is preserved at 0/1.

    Args:
        image: uint8 tensor with channel dim at ``-3``. Both ``[4, H, W]`` and ``[B, 4, H, W]``
            shapes are supported.
        mean: Optional pre-allocated broadcast tensor (``[3, 1, 1]`` or ``[1, 3, 1, 1]``).
            When omitted, a fresh tensor is allocated on the input's device.
        std: Optional pre-allocated broadcast tensor with the same convention as ``mean``.

    Returns:
        float32 tensor on the input's device with the RGB channels ImageNet-normalized.
        The input uint8 tensor is not modified.
    """

    out = image.float()
    if mean is None:
        mean = torch.tensor(IMAGENET_MEAN, device=out.device, dtype=torch.float32).view(3, 1, 1)
    if std is None:
        std = torch.tensor(IMAGENET_STD, device=out.device, dtype=torch.float32).view(3, 1, 1)
    rgb = out[..., :3, :, :]
    rgb.div_(255.0).sub_(mean).div_(std)
    return out


InstanceRecord = dict[str, Any]


class CRASARUnitemporalDataset(Dataset):
    """Instance-level dataset for hybrid unitemporal structural damage assessment.

    Dynamically extracts fixed-size image chips centered on individual structures,
    burns aligned structural polygons into a 4th feature channel, and generates
    one-hot disaster typology vectors to prevent geographic memorization.

    Args:
        manifest: Pandas DataFrame containing verified orthomosaic inventory.
        chip_size: Pixel dimension for the square extraction window (e.g., 512).
        transform: Optional albumentations composition for synchronized augmentations.
        is_train: If True, apply random window jitter; validation uses fixed centroids.
        cache_validation_tensors: If True and ``is_train`` is False, cache each
            validation chip tensor after the first load (speeds epoch 2+). Memory is
            roughly ``len(dataset) * 4 * chip_size**2 * 4`` bytes (float32); disable
            on large validation sets if RAM is limited.
        mask_dilation_px: Morphological dilation applied to the footprint mask itself, in
            final-chip pixel units, to compensate for systematic label-source under-segmentation.
            Applied post-augmentation (so radius is consistent under all spatial transforms).
            A value of ``0`` disables the correction and preserves legacy behavior byte-for-byte.
    """

    # Typology mappings designed to aggregate physical destructive forces
    NUM_TYPOLOGIES: ClassVar[int] = 4  # 0: Wind/Flood, 1: Kinetic, 2: Thermal, 3: Unknown
    TYPOLOGY_MAP: ClassVar[dict[str, int]] = {
        "Hurricane Ian": 0,
        "Hurricane Ida": 0,
        "Hurricane Harvey": 0,
        "Hurricane Laura": 0,
        "Hurricane Michael": 0,
        "Hurricane Idalia": 0,
        "Mayfield Tornado": 1,
        "Champlain Towers Collapse": 1,
        "Mussett Bayou Fire": 2,
        "Kilauea Eruption": 2
    }

    def __init__(self, manifest: pd.DataFrame, chip_size: int = 512, transform: albumentations.Compose | None = None,
                 is_train: bool = False, cache_validation_tensors: bool = False,
                 mask_dilation_px: int = 0) -> None:
        """Initialize dataset cache from orthomosaic manifest."""

        if mask_dilation_px < 0:
            raise ValueError("mask_dilation_px must be a non-negative integer.")

        self.manifest = manifest
        self.chip_size = chip_size
        self.transform = transform
        self.is_train = is_train
        self.cache_validation_tensors = cache_validation_tensors
        self.mask_dilation_px = mask_dilation_px

        # Pre-build the footprint-mask dilation kernel once. Elliptical kernel keeps the correction
        # isotropic so label under-segmentation is compensated uniformly around the polygon boundary.
        if self.mask_dilation_px > 0:
            k = 2 * self.mask_dilation_px + 1
            self._mask_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        else:
            self._mask_dilation_kernel = None

        # Validation only: deterministic crops; safe to cache final tensors across epochs.
        self._val_tensor_cache: dict[int, torch.Tensor] = {}

        self.instances = self._build_instance_index()

    def __getstate__(self) -> dict[str, Any]:
        """Clear validation tensor cache before pickling DataLoader workers."""

        # Build the pickled snapshot from a shallow copy and zero out the cache there.
        # Mutating ``self._val_tensor_cache`` directly would discard the parent's cache
        # on every worker spawn, which is a footgun for any caller that pickles the
        # dataset outside the ``persistent_workers=True`` path the trainer relies on.
        state = self.__dict__.copy()
        state["_val_tensor_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore dataset after unpickling in DataLoader workers."""

        self.__dict__.update(state)

    @staticmethod
    def _load_clean_polygons(image_path: Path, label_path: Path, alignment_path: Path | None) -> GeoDataFrame:
        """Parse, align, and label-filter polygon annotations for one orthomosaic.

        Args:
            image_path: Path to the orthomosaic image.
            label_path: Path to the CRASAR JSON labels.
            alignment_path: Optional path to tie-point adjustments.

        Returns:
            GeoDataFrame with valid ordinal labels in raster CRS and optional alignment shifts applied.
        """

        gdf_raw = parse_crasar_json(label_path)
        gdf_aligned = align_vector_to_raster(gdf=gdf_raw, raster_path=str(image_path))
        if alignment_path is not None and alignment_path.exists():
            adjustments = load_alignment_adjustments(str(alignment_path))
            gdf_aligned = apply_alignment_adjustments(gdf=gdf_aligned, raster_path=str(image_path), adjustments=adjustments)
        return filter_polygons_valid_labels(gdf=gdf_aligned, valid_labels=VALID_ORDINAL_LABELS)

    def _instance_from_polygon(self, image_path: Path, label_path: Path, event_name: str, transform: Affine, polygon_row: pd.Series) -> InstanceRecord:
        """Build one instance dictionary from a polygon row."""

        centroid = polygon_row.geometry.centroid
        col, row_idx = ~transform * (centroid.x, centroid.y)

        target_tensor = torch.tensor(ORDINAL_MAP[polygon_row["label"]], dtype=torch.long)
        typology_tensor = self._get_typology_vector(event_name)

        return {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "centroid_px": (int(col), int(row_idx)),
            "damage_label": polygon_row["label"],
            "event_name": event_name,
            "label_tensor": target_tensor,
            "context_tensor": typology_tensor
        }

    def _build_instance_index(self) -> list[InstanceRecord]:
        """Parse all orthomosaics and flatten to structure-level instance records."""

        self.gdf_cache = {}
        instances: list[InstanceRecord] = []
        for _, row in self.manifest.iterrows():
            img_path = Path(row["image_path"])
            lbl_path = Path(row["label_path"])
            align_path = Path(row["alignment_path"]) if pd.notna(row.get("alignment_path")) else None
            event_name = row.get("event", "Unknown")

            gdf_valid = self._load_clean_polygons(image_path=img_path, label_path=lbl_path, alignment_path=align_path)
            self.gdf_cache[str(img_path)] = gdf_valid

            with rasterio.open(img_path) as src:
                transform = src.transform

            for _, poly_row in gdf_valid.iterrows():
                instances.append(
                    self._instance_from_polygon(
                        image_path=img_path,
                        label_path=lbl_path,
                        event_name=event_name,
                        transform=transform,
                        polygon_row=poly_row
                    )
                )

        print(f"Dataset initialized: Extracted {len(instances)} unique structural instances.")
        return instances

    def _get_typology_vector(self, event_name: str) -> torch.Tensor:
        """Converts an event name string into a 4-dimensional one-hot vector."""
        idx = self.TYPOLOGY_MAP.get(event_name, 3)  # Fallback to 3 (Unknown)
        vector = torch.zeros(self.NUM_TYPOLOGIES, dtype=torch.float32)
        vector[idx] = 1.0
        return vector

    def __len__(self) -> int:
        """Return number of building-level instances in the dataset."""

        return len(self.instances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves a mask-conditioned 4-channel tensor for a specific structure."""
        if self.cache_validation_tensors and not self.is_train and idx in self._val_tensor_cache:
            cached = self._val_tensor_cache[idx]
            return {
                "image": cached.clone(),
                "label": self.instances[idx]["label_tensor"],
                "context": self.instances[idx]["context_tensor"],
            }

        instance = self.instances[idx]
        col_c, row_c = instance["centroid_px"]

        # Apply deterministic jitter during training to prevent perfect centering priors.
        # Max jitter is 25% of the chip size in any direction.
        if self.is_train:
            max_jitter = self.chip_size // 4
            col_c += random.randint(-max_jitter, max_jitter)
            row_c += random.randint(-max_jitter, max_jitter)

        half_size = self.chip_size // 2
        window = Window(
            col_off=col_c - half_size,
            row_off=row_c - half_size,
            width=self.chip_size,
            height=self.chip_size
        )

        # Open per read (no persistent handles across chips) to avoid multiprocess GDAL/rasterio issues.
        # Resilient retry: if a corrupt LZW tile (or other transient RasterioIOError) trips the
        # jittered read, fall back to a zero-jitter centered window once. The centered window touches
        # a different set of tiles in almost all cases, so single-tile corruption very rarely affects
        # both. If the fallback also fails, the exception propagates uncaught so the crash remains
        # visible rather than masked by a zero-chip silently corrupting training.
        try:
            with rasterio.open(instance["image_path"]) as src:
                rgb = src.read(window=window, indexes=(1, 2, 3), boundless=True, fill_value=0)
                win_transform = src.window_transform(window)
                win_bounds = rasterio.windows.bounds(window, src.transform)
        except RasterioIOError as err:
            print(
                f"[dataset] RasterioIOError on {Path(instance['image_path']).name} "
                f"window=({window.col_off}, {window.row_off}, {window.width}, {window.height}); "
                f"retrying with centered window. err={err.args[0] if err.args else err!r}",
                flush=True
            )
            centered_col, centered_row = instance["centroid_px"]
            window = Window(
                col_off=centered_col - half_size,
                row_off=centered_row - half_size,
                width=self.chip_size,
                height=self.chip_size
            )
            with rasterio.open(instance["image_path"]) as src:
                rgb = src.read(window=window, indexes=(1, 2, 3), boundless=True, fill_value=0)
                win_transform = src.window_transform(window)
                win_bounds = rasterio.windows.bounds(window, src.transform)

        rgb_hwc = np.moveaxis(rgb, 0, -1)  # [C, H, W] -> [H, W, C] for albumentations/numpy

        gdf = self.gdf_cache[instance["image_path"]]

        # Fast spatial intersection and in-memory rasterization
        from shapely.geometry import box
        from rasterio import features

        window_box = box(*win_bounds)
        # ``gdf.intersects(window_box)`` rebuilds the STRtree on every chip read; ``gdf.sindex``
        # caches it across calls, which compounds across the ``num_workers=8`` worker pool.
        candidate_idx = gdf.sindex.query(window_box, predicate="intersects")
        gdf_intersecting = gdf.iloc[candidate_idx]

        if not gdf_intersecting.empty:
            shapes = ((geom, 1) for geom in gdf_intersecting.geometry)
            mask_2d = features.rasterize(
                shapes=shapes,
                out_shape=(self.chip_size, self.chip_size),
                transform=win_transform,
                fill=0,
                dtype=np.uint8
            )
        else:
            mask_2d = np.zeros((self.chip_size, self.chip_size), dtype=np.uint8)

        # Synchronized Augmentations (applies spatial transforms to both RGB and Mask)
        if self.transform is not None:
            augmented = self.transform(image=rgb_hwc, mask=mask_2d)
            rgb_hwc = augmented["image"]
            mask_2d = augmented["mask"]

        # Compensate for systematic label-source under-segmentation (polygon annotations typically
        # exclude roof overhangs, eaves, and small attached structures) by morphologically dilating
        # the footprint mask. Applied post-augmentation so dilation radius is measured in final-chip
        # pixel units and remains consistent across all spatial augmentation geometry.
        if self._mask_dilation_kernel is not None:
            mask_2d = cv2.dilate(mask_2d, self._mask_dilation_kernel, iterations=1)

        rgb_chw = np.moveaxis(rgb_hwc, -1, 0)  # [H, W, C] -> [C, H, W], uint8
        mask_chw = np.expand_dims(mask_2d, axis=0)  # uint8 in {0, 1}
        tensor_nc = np.concatenate([rgb_chw, mask_chw], axis=0)  # [4, H, W] uint8

        # Stay in uint8 to cut dataloader -> GPU bandwidth ~4x. Float conversion plus
        # ImageNet normalization (RGB only; mask channel preserved at 0/1) happens
        # GPU-side in UnitemporalTrainer._prepare_batch.
        x_tensor = torch.from_numpy(tensor_nc)

        if self.cache_validation_tensors and not self.is_train:
            self._val_tensor_cache[idx] = x_tensor.clone()

        return {
            "image": x_tensor,
            "label": instance["label_tensor"],
            "context": instance["context_tensor"]
        }
