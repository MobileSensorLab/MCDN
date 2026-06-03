"""Tests for the ordinal regression loss functions."""

import torch

from src.model.loss import OrdinalEarthMoversDistanceLoss


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


