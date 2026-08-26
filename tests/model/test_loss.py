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


