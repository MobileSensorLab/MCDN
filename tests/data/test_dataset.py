"""Tests for the CRASARUnitemporalDataset class and tensor generation logic."""

import pickle

import albumentations as A
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch

from pathlib import Path
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from src.data.dataset import CRASARUnitemporalDataset
from src.data.geometry import raster_ground_gsd_m
from tests.conftest import write_crasar_json


def test_dataset_initialization_extracts_instances(sample_manifest: pd.DataFrame) -> None:
    """Verifies that the dataset iterates over polygons, not just orthomosaics."""
    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=16)

    # We wrote two polygons with valid labels in conftest
    assert len(dataset) == 2

    # Verify the instance metadata is correctly cached
    instance1 = dataset.instances[0]
    assert instance1["damage_label"] in ["minor damage", "destroyed"]
    assert "centroid_px" in instance1
    assert instance1["event_name"] == "Hurricane Ian"


def test_dataset_typology_mapping(sample_manifest: pd.DataFrame) -> None:
    """Verifies the string-to-one-hot context vector logic."""
    dataset = CRASARUnitemporalDataset(sample_manifest)

    # Known event mappings
    assert torch.equal(dataset._get_typology_vector("Hurricane Ian"), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(dataset._get_typology_vector("Mayfield Tornado"), torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(dataset._get_typology_vector("Mussett Bayou Fire"), torch.tensor([0.0, 0.0, 1.0, 0.0]))

    # Unknown fallback
    assert torch.equal(dataset._get_typology_vector("Alien Invasion"), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_dataset_getitem_returns_valid_tensors(sample_manifest: pd.DataFrame) -> None:
    """Verifies that __getitem__ returns a correctly shaped 4-channel tensor."""
    chip_size = 32
    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=chip_size)

    item = dataset[0]

    assert "image" in item
    assert "label" in item
    assert "context" in item

    # Check tensor shapes (4 channels: R, G, B, Mask). The dataset returns uint8 to keep
    # dataloader -> GPU bandwidth low; ImageNet normalization is applied GPU-side by the
    # trainer in UnitemporalTrainer._prepare_batch.
    image = item["image"]
    assert image.shape == (4, chip_size, chip_size)
    assert image.dtype == torch.uint8

    # Check label and context
    assert item["label"].dim() == 0  # 0D Scalar for CrossEntropy
    assert item["context"].shape == (4,)


def test_dataset_rejects_negative_mask_dilation(sample_manifest: pd.DataFrame) -> None:
    """A negative dilation radius is a configuration error."""

    with pytest.raises(ValueError, match="mask_dilation_px"):
        CRASARUnitemporalDataset(sample_manifest, chip_size=32, mask_dilation_px=-1)


def test_dataset_mask_dilation_grows_footprint(sample_manifest: pd.DataFrame) -> None:
    """A positive dilation radius builds an elliptical kernel and enlarges the mask channel."""

    plain = CRASARUnitemporalDataset(sample_manifest, chip_size=32)
    dilated = CRASARUnitemporalDataset(sample_manifest, chip_size=32, mask_dilation_px=2)
    assert plain._mask_dilation_kernel is None
    assert dilated._mask_dilation_kernel.shape == (5, 5)
    assert int(dilated[0]["image"][3].sum()) > int(plain[0]["image"][3].sum())


def test_validation_tensor_cache_serves_clones_and_survives_pickling(sample_manifest: pd.DataFrame) -> None:
    """Validation chips are cached after the first read, served as clones, and dropped from pickled state."""

    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, cache_validation_tensors=True)
    assert dataset._val_tensor_cache == {}
    first = dataset[0]["image"]
    assert 0 in dataset._val_tensor_cache
    second = dataset[0]["image"]
    assert torch.equal(first, second)
    assert second.data_ptr() != dataset._val_tensor_cache[0].data_ptr()

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._val_tensor_cache == {}
    assert dataset._val_tensor_cache  # The parent keeps its cache.
    assert torch.equal(restored[0]["image"], first)


def test_dataset_retries_centered_window_on_rasterio_io_error(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    """A RasterioIOError on the first read falls back to a centered window and logs the retry."""

    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, is_train=True)
    original = CRASARUnitemporalDataset._read_chip_window
    attempts: list[int] = []

    def _flaky(self: CRASARUnitemporalDataset, src: rasterio.DatasetReader, window: object, read_px: int) -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise RasterioIOError("simulated read failure")
        return original(self, src=src, window=window, read_px=read_px)

    monkeypatch.setattr(CRASARUnitemporalDataset, "_read_chip_window", _flaky)
    item = dataset[0]
    assert item["image"].shape == (4, 32, 32)
    assert len(attempts) == 2
    assert "retrying with centered window" in capsys.readouterr().out


def test_dataset_with_albumentations_transforms(sample_manifest: pd.DataFrame) -> None:
    """Verifies that spatial transforms are applied successfully to both image and mask."""

    # A.Lambda only passes the specific target (image) to this function
    def mock_image(image: np.ndarray, **_kwargs: object) -> np.ndarray:
        return np.zeros_like(image)

    transform = A.Compose([A.Lambda(image=mock_image)])

    dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=16, transform=transform)
    item = dataset[0]

    image = item["image"]

    # The first 3 channels (RGB) are zeroed by the Lambda transform; the dataset emits
    # them as uint8 without normalization. ImageNet shift / scale is performed GPU-side
    # by the trainer, so the dataset-level expectation is just zero-bytes for RGB.
    assert image.dtype == torch.uint8
    assert image[:3].sum().item() == 0


def test_dataset_extraction_window_jitter(sample_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that spatial jitter is applied during training but not validation."""

    mock_randint_calls = []

    def mock_randint(low: int, high: int) -> int:
        mock_randint_calls.append((low, high))
        return high  # always return max jitter

    import random
    monkeypatch.setattr(random, "randint", mock_randint)

    # Validation Dataset (No Jitter)
    val_dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, is_train=False)
    _ = val_dataset[0]
    assert len(mock_randint_calls) == 0, "Jitter should not be applied when is_train=False"

    # Training Dataset (Jitter Applied)
    train_dataset = CRASARUnitemporalDataset(sample_manifest, chip_size=32, is_train=True)
    _ = train_dataset[0]
    assert len(mock_randint_calls) == 2, "Jitter should be applied twice (x and y) when is_train=True"

    # Max jitter for chip_size 32 is 32 // 4 = 8
    assert mock_randint_calls[0] == (-8, 8)
    assert mock_randint_calls[1] == (-8, 8)


# Window-geometry knobs (wide-window decimation / matched-presentation enlargement) ----

@pytest.fixture
def utm_patterned_manifest(tmp_path: Path) -> pd.DataFrame:
    """Projected-CRS (1 m/px) mosaic with a centered building and an off-window landmark.

    Geometry (256x256 px, EPSG:32616, origin 500000/4000000):
    - background value 20;
    - 16x16 bright "building" (value 200) centered at pixel (128, 128), with a matching
      16 m footprint polygon in the annotation JSON;
    - a bright column (value 255) at pixels x=[176, 180), outside a native 32 px chip
      around the building but inside a 4x-scaled (128 px) window.
    """

    img_path = tmp_path / "utm_pattern.geo.tif"
    data = np.full((3, 256, 256), 20, dtype=np.uint8)
    data[:, 120:136, 120:136] = 200
    data[:, :, 176:180] = 255
    profile = {
        "driver": "GTiff", "height": 256, "width": 256, "count": 3,
        "dtype": rasterio.uint8, "crs": "EPSG:32616",
        "transform": from_origin(500000.0, 4000000.0, 1.0, 1.0),
    }
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(data)

    # Footprint matching the bright square: cols 120-136 -> x 500120-500136,
    # rows 120-136 -> y 3999880-3999864 (row-down geotransform).
    footprint = Polygon([
        (500120.0, 3999864.0), (500136.0, 3999864.0),
        (500136.0, 3999880.0), (500120.0, 3999880.0), (500120.0, 3999864.0)
    ])
    # parse_crasar_json expects EPSG:4326 vertices; reproject the footprint corners.
    lbl_path = tmp_path / f"{img_path.name}.json"
    footprint_4326 = gpd.GeoSeries([footprint], crs="EPSG:32616").to_crs("EPSG:4326").iloc[0]
    write_crasar_json(lbl_path, [footprint_4326], labels=["destroyed"])

    return pd.DataFrame([{
        "image_path": str(img_path),
        "label_path": str(lbl_path),
        "alignment_path": None,
        "event": "Hurricane Ian"
    }])


def test_window_scale_widens_ground_context(utm_patterned_manifest: pd.DataFrame) -> None:
    """A 4x window shows the off-chip landmark and shrinks the footprint's chip area."""

    native = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32)
    wide = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_scale=4.0)

    native_item = native[0]["image"]
    wide_item = wide[0]["image"]
    assert native_item.shape == (4, 32, 32)
    assert wide_item.shape == (4, 32, 32)

    # The bright column at +48 px offset is outside the native 32 px window (max 255
    # absent) but inside the 128 px wide window after decimation. Its decimated value
    # depends on the 4x box phase (centroid reprojection drift shifts the window by a
    # pixel or two), so the assertion is presence above background, not exact recovery.
    assert native_item[:3].max().item() <= 200
    assert wide_item[:3].max().item() > 60

    # The 16 m footprint covers ~16x16 px natively but ~4x4 px after 4x decimation.
    native_mask_area = int(native_item[3].sum().item())
    wide_mask_area = int(wide_item[3].sum().item())
    assert native_mask_area >= 200
    assert 4 <= wide_mask_area <= 40
    assert set(torch.unique(wide_item[3]).tolist()) <= {0, 1}


def test_window_scale_boundless_edge_window(utm_patterned_manifest: pd.DataFrame) -> None:
    """Wide windows that overrun the mosaic bounds zero-fill without error."""

    dataset = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_scale=16.0)
    item = dataset[0]["image"]  # 512 px window on a 256 px mosaic: boundless on all sides
    assert item.shape == (4, 32, 32)
    assert item[:3, 0, 0].max().item() == 0  # zero-filled corner


def test_window_ground_m_resolves_per_mosaic(utm_patterned_manifest: pd.DataFrame) -> None:
    """Meters-based windows resolve from the CRS-aware GSD (1 m/px here)."""

    dataset = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_ground_m=64.0)
    image_path = utm_patterned_manifest.iloc[0]["image_path"]
    assert dataset._read_window_px[image_path] == 64
    assert dataset[0]["image"].shape == (4, 32, 32)


def test_window_ground_m_enlarges_when_smaller_than_chip(utm_patterned_manifest: pd.DataFrame) -> None:
    """Sub-chip ground windows are bilinearly enlarged to the chip grid."""

    dataset = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_ground_m=16.0)
    image_path = utm_patterned_manifest.iloc[0]["image_path"]
    assert dataset._read_window_px[image_path] == 16

    item = dataset[0]["image"]
    assert item.shape == (4, 32, 32)
    # The 16 m window sits (nearly) entirely inside the 16 m building: bright chip and
    # near-full mask, tolerating the 1-2 px centroid truncation/reprojection drift.
    assert item[:3].float().mean().item() > 150
    assert int(item[3].sum().item()) >= 28 * 32


def test_window_scale_jitter_uses_read_window(utm_patterned_manifest: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training jitter is 25% of the read window, not the chip size."""

    calls: list[tuple[int, int]] = []

    def mock_randint(low: int, high: int) -> int:
        calls.append((low, high))
        return 0

    import random
    monkeypatch.setattr(random, "randint", mock_randint)

    dataset = CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_scale=4.0, is_train=True)
    _ = dataset[0]
    assert calls == [(-32, 32), (-32, 32)]  # 128 px read window // 4


def test_window_param_validation(utm_patterned_manifest: pd.DataFrame) -> None:
    """Non-positive window parameters are rejected."""

    with pytest.raises(ValueError, match="window_scale must be positive"):
        CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_scale=0.0)
    with pytest.raises(ValueError, match="window_ground_m must be positive"):
        CRASARUnitemporalDataset(utm_patterned_manifest, chip_size=32, window_ground_m=-5.0)


def test_raster_ground_gsd_m_projected_and_geographic(utm_patterned_manifest: pd.DataFrame, sample_geotiff: Path) -> None:
    """GSD helper returns native step for projected CRS and meter-converted step for degrees."""

    assert raster_ground_gsd_m(utm_patterned_manifest.iloc[0]["image_path"]) == pytest.approx(1.0)

    # sample_geotiff: 0.01 deg/px at ~19.5-20 deg latitude -> ~1.0-1.1 km/px.
    gsd = raster_ground_gsd_m(sample_geotiff)
    assert 900.0 < gsd < 1200.0
