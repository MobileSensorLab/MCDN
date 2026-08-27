"""Unit tests for the seed x TTA decomposition tooling in ``src.postproc.decompose``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.postproc import decompose as dec


def _prob_vector(peak_class: int, mass: float) -> list[float]:
    """Length-4 probability vector with ``mass`` on ``peak_class`` and the rest uniform."""

    rest = (1.0 - mass) / 3.0
    return [mass if k == peak_class else rest for k in range(4)]


def _synthetic_view_stack() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Two-seed, two-view stack [2, 2, 4, 4] with hand-computable 2x2 cells.

    Targets are [0, 1, 2, 3]. Seed 0 is correct everywhere. Seed 1's identity view
    mispredicts samples 2 and 3 as class 0 (accuracy 0.5), but its augmented view is
    confident enough (0.97 peaks) that TTA averaging repairs both errors. The seeds'
    identity peaks differ (0.76 vs 0.70) so the no-TTA ensemble mean has no argmax ties
    and also repairs seed 1's errors.
    """

    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    seed0_identity = [_prob_vector(t, 0.76) for t in range(4)]
    seed0_augmented = [_prob_vector(t, 0.97) for t in range(4)]
    seed1_identity = [_prob_vector(0, 0.70), _prob_vector(1, 0.70), _prob_vector(0, 0.70), _prob_vector(0, 0.70)]
    seed1_augmented = [_prob_vector(t, 0.97) for t in range(4)]

    stack = torch.tensor([
        [seed0_identity, seed0_augmented],
        [seed1_identity, seed1_augmented]
    ], dtype=torch.float32)  # [S=2, V=2, T=4, K=4]
    return stack, targets, [0, 11]


def test_cell_probability_tensors_shapes_and_reductions() -> None:
    """Cells reduce the [S, V, T, K] stack along the documented axes."""

    generator = torch.Generator().manual_seed(7)
    raw = torch.rand((3, 8, 5, 4), generator=generator)
    view_probs = raw / raw.sum(dim=-1, keepdim=True)

    cells = dec.cell_probability_tensors(view_probs=view_probs)
    assert torch.equal(cells["single_seed_no_tta"], view_probs[:, 0])
    assert torch.allclose(cells["single_seed_tta"], view_probs.mean(dim=1))
    assert torch.allclose(cells["ensemble_no_tta"], view_probs[:, 0].mean(dim=0))
    assert torch.allclose(cells["ensemble_tta"], view_probs.mean(dim=(0, 1)))
    assert cells["single_seed_no_tta"].shape == (3, 5, 4)
    assert cells["ensemble_tta"].shape == (5, 4)


def test_cell_probability_tensors_honors_identity_view_index() -> None:
    """A non-zero identity index selects that view for the no-TTA cells."""

    generator = torch.Generator().manual_seed(11)
    raw = torch.rand((2, 4, 3, 4), generator=generator)
    view_probs = raw / raw.sum(dim=-1, keepdim=True)

    cells = dec.cell_probability_tensors(view_probs=view_probs, identity_view_index=2)
    assert torch.equal(cells["single_seed_no_tta"], view_probs[:, 2])


def test_decompose_variant_cell_metrics_match_hand_computation() -> None:
    """The 2x2 accuracies and across-seed summaries match the constructed scenario."""

    view_probs, targets, seeds = _synthetic_view_stack()
    result = dec.decompose_variant(view_probs=view_probs, targets=targets, seeds=seeds)

    assert result["seeds"] == seeds
    assert result["num_views"] == 2

    no_tta = result["cells"]["single_seed_no_tta"]
    assert [record["seed"] for record in no_tta["per_seed"]] == seeds
    assert no_tta["per_seed"][0]["metrics"]["argmax"]["accuracy"] == pytest.approx(1.0)
    assert no_tta["per_seed"][1]["metrics"]["argmax"]["accuracy"] == pytest.approx(0.5)
    assert no_tta["summary"]["argmax"]["accuracy"]["mean"] == pytest.approx(0.75)
    assert no_tta["summary"]["argmax"]["accuracy"]["std"] == pytest.approx(0.353553, abs=1e-5)

    tta = result["cells"]["single_seed_tta"]
    assert tta["summary"]["argmax"]["accuracy"]["mean"] == pytest.approx(1.0)
    assert tta["summary"]["argmax"]["accuracy"]["std"] == pytest.approx(0.0)

    assert result["cells"]["ensemble_no_tta"]["argmax"]["accuracy"] == pytest.approx(1.0)
    assert result["cells"]["ensemble_tta"]["argmax"]["accuracy"] == pytest.approx(1.0)


def test_decompose_variant_deltas_are_pairwise_cell_gains() -> None:
    """Gain entries equal the differences of the corresponding cell means."""

    view_probs, targets, seeds = _synthetic_view_stack()
    result = dec.decompose_variant(view_probs=view_probs, targets=targets, seeds=seeds)

    gains = result["deltas"]["argmax"]["accuracy"]
    assert gains["tta_gain_single_seed"] == pytest.approx(0.25)
    assert gains["ensemble_gain_no_tta"] == pytest.approx(0.25)
    assert gains["tta_gain_ensemble"] == pytest.approx(0.0)
    assert gains["ensemble_gain_tta"] == pytest.approx(0.0)
    assert gains["total_gain"] == pytest.approx(0.25)


class _TwoSampleDataset(Dataset):
    """Val-contract dataset: ``image`` uint8 [4,8,8], ``context``, ``label``."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = (torch.arange(4 * 8 * 8, dtype=torch.int64).reshape(4, 8, 8) % 251).to(torch.uint8)
        return {
            "image": base + idx,
            "context": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "label": torch.tensor(idx, dtype=torch.long),
        }


class _CornerLogitModel(nn.Module):
    """Orientation-sensitive logits so the eight views differ."""

    def forward(self, x: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        corner = x[:, 0, 0, 0]
        return torch.stack([corner, -corner, corner * 0.5, -corner * 0.5], dim=1)


def test_collect_view_probabilities_shapes_and_targets() -> None:
    """Collection preserves the view axis and concatenates batches along samples."""

    loader = DataLoader(_TwoSampleDataset(), batch_size=1)
    model = _CornerLogitModel().eval()

    view_probs, targets = dec.collect_view_probabilities(model=model, val_loader=loader, device="cpu")
    assert view_probs.shape == (8, 2, 4)
    assert torch.allclose(view_probs.sum(dim=-1), torch.ones(8, 2), atol=1e-5)
    assert targets.tolist() == [0, 1]


def _fake_record() -> dict:
    """Assembled variant record as produced by ``run_variant_decomposition``."""

    view_probs, targets, seeds = _synthetic_view_stack()
    return {
        "variant_root": "outputs/ablation/baseline",
        "holdout_event": "Spatial_Block_East",
        "seeds": seeds,
        "view_probs": view_probs,
        "targets": targets,
        "decomposition": dec.decompose_variant(view_probs=view_probs, targets=targets, seeds=seeds),
    }


def test_view_cache_roundtrip(tmp_path: Path) -> None:
    """The view cache round-trips tensors, seed labels, and view metadata."""

    record = _fake_record()
    cache_path = tmp_path / "view_probs.pt"
    dec.write_view_cache(path=cache_path, records=[record], split_name="Spatial_Block_East")

    payload = dec.load_view_cache(cache_path)
    assert payload["split"] == "Spatial_Block_East"
    assert payload["holdout_event"] == "Spatial_Block_East"
    assert payload["identity_view_index"] == dec.IDENTITY_VIEW_INDEX
    assert payload["view_order"] == list(dec.D4_VIEW_ORDER)
    assert torch.equal(payload["targets"], record["targets"])
    assert payload["variants"][0]["seeds"] == record["seeds"]
    assert torch.equal(payload["variants"][0]["view_probs"], record["view_probs"])


def test_load_view_cache_missing_file(tmp_path: Path) -> None:
    """A missing cache path raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="View cache not found"):
        dec.load_view_cache(tmp_path / "absent.pt")


def test_write_decomposition_artifact_json_structure(tmp_path: Path) -> None:
    """The JSON artifact carries split metadata and per-variant decompositions."""

    record = _fake_record()
    artifact_path = tmp_path / "decomposition.json"
    dec.write_decomposition_artifact(path=artifact_path, records=[record], split_name="Spatial_Block_East")

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["split"] == "Spatial_Block_East"
    assert payload["view_order"] == list(dec.D4_VIEW_ORDER)
    variant = payload["variants"][0]
    assert variant["variant_root"] == "outputs/ablation/baseline"
    cells = variant["decomposition"]["cells"]
    assert set(cells) == {"single_seed_no_tta", "single_seed_tta", "ensemble_no_tta", "ensemble_tta"}
    assert variant["decomposition"]["deltas"]["argmax"]["qwk"]["total_gain"] is not None


def test_print_decomposition_table_renders_all_cells(capsys: pytest.CaptureFixture[str]) -> None:
    """The console table prints every configuration row and the gain line."""

    dec.print_decomposition_table(_fake_record())
    output = capsys.readouterr().out
    assert "SEED x TTA DECOMPOSITION" in output
    assert "single seed / no TTA" in output
    assert "single seed / 8-view TTA" in output
    assert "2-seed ens / no TTA" in output
    assert "2-seed ens / 8-view TTA" in output
    assert "QWK gains:" in output


def test_main_from_view_cache_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--from-view-cache recomputes tables and the artifact without models or data."""

    cache_path = tmp_path / "view_probs.pt"
    dec.write_view_cache(path=cache_path, records=[_fake_record()], split_name="Spatial_Block_East")
    artifact_path = tmp_path / "decomposition.json"

    monkeypatch.setattr(sys, "argv", [
        "decompose",
        "--from-view-cache", str(cache_path),
        "--output-json", str(artifact_path),
    ])
    dec.main()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["split"] == "Spatial_Block_East"
    assert payload["variants"][0]["decomposition"]["cells"]["ensemble_tta"]["argmax"]["accuracy"] == pytest.approx(1.0)


def test_main_requires_variant_roots_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting both --variant-roots and --from-view-cache is a CLI error."""

    monkeypatch.setattr(sys, "argv", ["decompose"])
    with pytest.raises(SystemExit):
        dec.main()
