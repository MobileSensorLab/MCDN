"""One-time embedding extraction for the chapter-6 D4 typology UMAP figure.

Loads the typology-ablated baseline checkpoint at
``outputs/ablation/typology/Spatial_Block_East/seed_00/best_model.pt`` (the
cleanest test of the chapter's implicit-encoding hypothesis: FiLM is
structurally absent so the embedding is purely visual), runs forward over
the Spatial_Block_East val pool, hooks the linear head's input to capture
the fused embedding ``v_features``, and caches per-chip embeddings + event-
level typology + damage labels to ``outputs/embeddings/typology_arm_sbe_val.pt``.

The Spatial_Block_East holdout is used because its val pool has multi-event
coverage (it is a spatial split, not LOEO), so the resulting cache contains
chips from all four typology categories - exactly what the embedding-cluster
visualization needs.

Single forward pass per chip (no 8x D4 TTA) - the embedding is taken from
the identity orientation. TTA is the right protocol for *probability*
averaging, not for embedding extraction; averaging embeddings across
rotations would muddle the cluster visualization the figure is meant to
show.

Run via: ``uv run python -m scripts.visualization.extract_canonical_embeddings``.
Approximate runtime on a desktop-class GPU: ~1-3 minutes for ~1700 chips.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Final

import torch

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import CRASARUnitemporalDataset, to_normalized_float  # noqa: E402
from src.data.sampling import build_valid_manifest  # noqa: E402
from src.postproc.ensemble import (  # noqa: E402
    build_model_from_config,
    build_val_loader_from_config,
    load_resolved_config,
)


_ARM: Final[str] = "typology"
_HOLDOUT: Final[str] = "Spatial_Block_East"
_SEED: Final[str] = "seed_00"
_FOLD_DIR: Final[Path] = (
    _REPO_ROOT / "outputs" / "ablation" / _ARM / _HOLDOUT / _SEED
)
_OUT_DIR: Final[Path] = _REPO_ROOT / "outputs" / "embeddings"
_OUT_PATH: Final[Path] = _OUT_DIR / "typology_arm_sbe_val.pt"


def _build_orthomosaic_to_event_map(data_dir: Path, sensor_profile: str) -> dict[str, str]:
    """Build a dict mapping orthomosaic filename -> full event name.

    The Spatial_Block_East fold's manifest replaces per-chip ``event_name``
    with the spatial-block label (e.g. ``"Spatial_Block"``), so we recover
    the true event from the raw ``valid_manifest`` produced by
    ``src.data.sampling.build_valid_manifest`` - which carries an
    authoritative ``event`` column per orthomosaic. Map keys are file-name
    basenames (not full paths) because the dataset's ``image_path``
    instances may carry absolute paths that differ from the manifest's
    ``image_name`` column.
    """

    manifest = build_valid_manifest(
        data_dir=str(data_dir), sensor_profile=sensor_profile
    )
    if "event" not in manifest.columns or "image_name" not in manifest.columns:
        raise RuntimeError(
            f"Expected 'event' and 'image_name' columns on valid_manifest; got {list(manifest.columns)}."
        )
    mapping: dict[str, str] = {}
    for _, row in manifest.iterrows():
        ortho_name = Path(str(row["image_name"])).name
        mapping[ortho_name] = str(row["event"])
    return mapping


def _val_dataset_event_names(
    val_loader_dataset: object, ortho_to_event: dict[str, str]
) -> list[str]:
    """Return per-chip full event names aligned to dataset row indices.

    Each chip's event is looked up by its source orthomosaic filename in the
    pre-built ``ortho_to_event`` dict (constructed from
    ``src.data.sampling.build_valid_manifest`` upstream of the SBE
    spatial-manifest reformation that loses event identity). Falls back to
    ``"Unknown"`` if a chip's orthomosaic is somehow not in the manifest;
    this should be a no-op for any well-formed val pool.
    """

    if not isinstance(val_loader_dataset, CRASARUnitemporalDataset):
        raise TypeError(
            f"Expected CRASARUnitemporalDataset, got {type(val_loader_dataset).__name__}."
        )
    return [
        ortho_to_event.get(Path(str(instance["image_path"])).name, "Unknown")
        for instance in val_loader_dataset.instances
    ]


def main() -> None:
    """Extract post-FiLM-pool fused embeddings for the typology arm SBE val pool."""

    if not _FOLD_DIR.exists():
        raise FileNotFoundError(f"Fold directory not found: {_FOLD_DIR}")

    cfg = load_resolved_config(_FOLD_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[D4 extract] arm={_ARM} holdout={_HOLDOUT} seed={_SEED} device={device}")

    val_loader, holdout_label = build_val_loader_from_config(
        cfg=cfg, data_dir_override=str(_REPO_ROOT / "data")
    )
    ortho_to_event = _build_orthomosaic_to_event_map(
        data_dir=_REPO_ROOT / "data",
        sensor_profile=cfg["data"]["sensor_profile"],
    )
    event_names = _val_dataset_event_names(val_loader.dataset, ortho_to_event)
    print(f"[D4 extract] holdout label resolved to: {holdout_label!r}")
    print(f"[D4 extract] val pool size: {len(val_loader.dataset)}")
    from collections import Counter
    print(f"[D4 extract] event-name distribution: {dict(Counter(event_names))}")

    model = build_model_from_config(cfg=cfg, device=device)
    state_dict = torch.load(
        _FOLD_DIR / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    captured_embedding: dict[str, torch.Tensor] = {}

    def _capture_head_input(_module: object, head_input: tuple, _head_output: object) -> None:
        # The linear head receives v_features as a single positional input.
        captured_embedding["v_features"] = head_input[0].detach().cpu()

    handle = model.head.register_forward_hook(_capture_head_input)

    embedding_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    inference_start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images_u8 = batch["image"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            labels = batch["label"]

            images = to_normalized_float(images_u8)
            with torch.amp.autocast(autocast_device):
                _logits = model(images, context)

            embedding_chunks.append(captured_embedding["v_features"].float().cpu())
            label_chunks.append(labels.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[D4 extract] batch {batch_idx + 1} processed; "
                    f"{(batch_idx + 1) * images.shape[0]} chips done."
                )

    handle.remove()
    inference_seconds = time.time() - inference_start

    embeddings = torch.cat(embedding_chunks, dim=0)
    damage_labels = torch.cat(label_chunks, dim=0)

    if embeddings.shape[0] != len(event_names):
        raise RuntimeError(
            f"Embedding count ({embeddings.shape[0]}) != event-name count "
            f"({len(event_names)}). Loader/dataset are out of alignment."
        )

    typology_indices = torch.tensor(
        [
            CRASARUnitemporalDataset.TYPOLOGY_MAP.get(name, 3)
            for name in event_names
        ],
        dtype=torch.int64,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "typology_indices": typology_indices,
            "damage_labels": damage_labels,
            "event_names": event_names,
            "arm": _ARM,
            "holdout": _HOLDOUT,
            "seed": _SEED,
            "schema_version": 1,
        },
        _OUT_PATH,
    )

    print(
        f"[D4 extract] Saved {embeddings.shape[0]} embeddings of dim "
        f"{embeddings.shape[1]} to {_OUT_PATH} in {inference_seconds:.1f}s."
    )


if __name__ == "__main__":
    main()
