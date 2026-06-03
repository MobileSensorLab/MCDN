"""Ordinal regression loss functions for damage severity classification."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class OrdinalEarthMoversDistanceLoss(nn.Module):
    """Penalizes predictions based on the ordinal severity of the error.

    Uses the Squared Earth Mover's Distance (EMD), also known as the Ranked
    Probability Score, on the cumulative distribution functions of the predictions
    and targets. A prediction of 'Minor' for a 'Destroyed' building is penalized
    proportionally more than a prediction of 'Major'.
    """

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.025) -> None:
        """Initialize the ordinal EMD loss module.

        Args:
            weight: Optional 1D tensor of shape [K] containing instance weights
                    for each ordinal class.
            label_smoothing: Amount of adjacency-aware ordinal smoothing to apply
                    to hard labels. Smoothing mass is distributed only to immediate
                    neighboring classes.
        """

        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0.0, 1.0).")
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the CDF-based ordinal loss.

        Args:
            logits: Unnormalized network outputs [B, K].
            targets: Integer class labels [B] where values are in [0, K-1].

        Returns:
            Scalar tensor containing the weighted mean ordinal loss.
        """

        probs = functional.softmax(logits, dim=-1)
        pred_cdf = torch.cumsum(probs, dim=-1)

        # Build ordinally-smoothed targets and compute the target CDF.
        num_classes = logits.shape[-1]
        targets_one_hot = functional.one_hot(targets, num_classes=num_classes).float()
        if self.label_smoothing > 0.0:
            # Per-neighbor-rate-invariant ordinal smoothing. Each ordinal neighbor receives dose ls/2
            # regardless of whether the true class sits at an ordinal extreme (single neighbor) or
            # inside (two neighbors). Under this scheme extreme classes retain (1 - ls/2) on self with
            # ls/2 on the sole neighbor; inner classes retain (1 - ls) on self with ls/2 on each of
            # two neighbors. This corrects the prior asymmetry where extreme classes transferred the
            # full ls dose to their single neighbor -- yielding a per-neighbor rate 2x that of inner
            # classes and systematically biasing extreme-class predictions toward their adjacent class.
            ls = self.label_smoothing
            half = ls * 0.5
            smoothed_targets = targets_one_hot.clone()
            m0 = targets == 0
            if m0.any():
                smoothed_targets[m0, 0] = 1.0 - half
                smoothed_targets[m0, 1] = smoothed_targets[m0, 1] + half
            m_last = targets == num_classes - 1
            if m_last.any():
                smoothed_targets[m_last, num_classes - 1] = 1.0 - half
                smoothed_targets[m_last, num_classes - 2] = smoothed_targets[m_last, num_classes - 2] + half
            mm = (targets > 0) & (targets < num_classes - 1)
            if mm.any():
                idx_b = torch.where(mm)[0]
                tc = targets[mm]
                smoothed_targets[idx_b, tc] = 1.0 - ls
                smoothed_targets[idx_b, tc - 1] = smoothed_targets[idx_b, tc - 1] + half
                smoothed_targets[idx_b, tc + 1] = smoothed_targets[idx_b, tc + 1] + half
            target_dist = smoothed_targets
        else:
            target_dist = targets_one_hot
        target_cdf = torch.cumsum(target_dist, dim=-1)

        # Squared Earth Mover's Distance (MSE between CDFs). Drop the final CDF column because it is
        # mathematically 1.0 for both predictions and targets at the final index.
        loss_unreduced = functional.mse_loss(pred_cdf[..., :-1], target_cdf[..., :-1], reduction="none")
        loss_per_sample = loss_unreduced.mean(dim=-1)

        # Normalize batch loss by the sum of sample weights in the batch to avoid magnitude bouncing.
        if self.weight is not None:
            sample_weights = self.weight[targets]
            weight_sum = sample_weights.sum()
            if weight_sum > 0:
                return (loss_per_sample * sample_weights).sum() / weight_sum
            return loss_per_sample.sum() * 0.0

        return loss_per_sample.mean()
