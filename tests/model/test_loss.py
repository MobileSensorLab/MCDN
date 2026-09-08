"""Tests for the ordinal regression loss functions."""

import pytest
import torch

import torch.nn as nn

from src.model.loss import OrdinalEarthMoversDistanceLoss, create_criterion


def test_emd_loss_penalizes_ordinal_distance() -> None:
    """Verifies EMD loss scales strictly with the severity of the misclassification."""
    criterion = OrdinalEarthMoversDistanceLoss()

    # Ground truth: Destroyed (Class 3)
    target = torch.tensor([3], dtype=torch.long)

    # Logit scenarios simulating strong confidence in different classes
    # 1. Confident it's 'Major' (Class 2) - Minor error
    logits_minor_error = torch.tensor([[0.0, 0.0, 10.0, 0.0]])

    # 2. Confident it's 'Minor' (Class 1) - Severe error
    logits_severe_error = torch.tensor([[0.0, 10.0, 0.0, 0.0]])

    # 3. Confident it's 'No Damage' (Class 0) - Catastrophic error
    logits_catastrophic_error = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    loss_minor = criterion(logits_minor_error, target).item()
    loss_severe = criterion(logits_severe_error, target).item()
    loss_catastrophic = criterion(logits_catastrophic_error, target).item()

    # The mathematical distance penalty should strictly increase
    assert loss_catastrophic > loss_severe > loss_minor


def test_emd_loss_perfect_prediction() -> None:
    """Verifies EMD loss is functionally zero for a perfect prediction."""
    criterion = OrdinalEarthMoversDistanceLoss()
    target = torch.tensor([2], dtype=torch.long)
    logits = torch.tensor([[0.0, 0.0, 15.0, 0.0]])  # Highly confident in Class 2

    loss = criterion(logits, target).item()
    assert loss < 2e-4


def test_create_criterion_returns_emd_with_wired_parameters() -> None:
    """Factory returns the ordinal EMD loss with weight and smoothing wired through."""

    weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    criterion = create_criterion(loss_name="emd", weight=weight, label_smoothing=0.02)

    assert isinstance(criterion, OrdinalEarthMoversDistanceLoss)
    assert criterion.label_smoothing == 0.02
    assert torch.equal(criterion.weight, weight)


def test_create_criterion_returns_cross_entropy_with_wired_parameters() -> None:
    """Factory returns torch cross-entropy with weight and uniform smoothing wired through."""

    weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
    criterion = create_criterion(loss_name="ce", weight=weight, label_smoothing=0.02)

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.02
    assert torch.equal(criterion.weight, weight)


def test_emd_loss_rejects_out_of_range_smoothing() -> None:
    """The smoothing rate must lie in [0, 1)."""

    with pytest.raises(ValueError, match="label_smoothing"):
        OrdinalEarthMoversDistanceLoss(label_smoothing=1.0)


def test_emd_loss_smoothing_keeps_per_neighbor_rate_constant() -> None:
    """Extreme and interior targets both push epsilon/2 onto each adjacent class.

    With smoothing active the loss against the hard one-hot prediction is strictly positive,
    and the extreme-class penalty equals half the interior-class penalty's neighbour mass
    (one neighbour versus two), which is only true under the per-neighbour-constant scheme.
    """

    criterion = OrdinalEarthMoversDistanceLoss(label_smoothing=0.2)
    sharp = torch.tensor([[50.0, 0.0, 0.0, 0.0], [0.0, 0.0, 50.0, 0.0], [0.0, 0.0, 0.0, 50.0]])
    targets = torch.tensor([0, 2, 3])
    per_sample = torch.stack([criterion(sharp[i:i + 1], targets[i:i + 1]) for i in range(3)])
    assert torch.all(per_sample > 0.0)
    # Extreme targets (0 and 3) are symmetric; the interior target moves more mass.
    assert per_sample[0] == pytest.approx(per_sample[2].item(), abs=1e-6)
    assert per_sample[1] > per_sample[0]

    unsmoothed = OrdinalEarthMoversDistanceLoss(label_smoothing=0.0)
    assert unsmoothed(sharp, targets).item() < 1e-6


def test_emd_loss_class_weights_normalize_by_batch_weight_sum() -> None:
    """Weighted reduction divides by the batch's total sample weight; all-zero weights give zero loss."""

    logits = torch.tensor([[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 3.0]])  # sample 0 wrong, sample 1 right
    targets = torch.tensor([0, 3])
    unweighted = OrdinalEarthMoversDistanceLoss()(logits, targets)
    only_first = OrdinalEarthMoversDistanceLoss(weight=torch.tensor([1.0, 0.0, 0.0, 0.0]))(logits, targets)
    per_sample_first = OrdinalEarthMoversDistanceLoss()(logits[:1], targets[:1])
    assert only_first.item() == pytest.approx(per_sample_first.item(), rel=1e-5)
    assert only_first.item() != pytest.approx(unweighted.item())

    zero_weighted = OrdinalEarthMoversDistanceLoss(weight=torch.tensor([0.0, 1.0, 1.0, 0.0]))(logits, targets)
    assert zero_weighted.item() == 0.0


def test_create_criterion_rejects_unknown_selector() -> None:
    """Factory raises for selectors outside the emd/ce pair."""

    with pytest.raises(ValueError, match="Unknown loss selector"):
        create_criterion(loss_name="focal")


def test_cross_entropy_criterion_is_ordinal_distance_invariant() -> None:
    """CE penalizes equally-confident errors identically regardless of ordinal distance.

    This is the behavioral contrast the EMD-vs-CE ablation arm measures: under EMD the same
    two errors diverge (see test_emd_loss_penalizes_ordinal_distance), under CE they tie.
    """

    criterion = create_criterion(loss_name="ce", label_smoothing=0.0)
    target = torch.tensor([3], dtype=torch.long)

    loss_adjacent = criterion(torch.tensor([[0.0, 0.0, 10.0, 0.0]]), target).item()
    loss_extreme = criterion(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), target).item()

    assert loss_adjacent == pytest.approx(loss_extreme)


