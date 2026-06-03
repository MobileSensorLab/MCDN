"""Neural network architectures for mask-conditioned damage assessment.

Provides the core PyTorch module that fuses 4-channel visual features (RGB + a priori footprint mask)
with disaster typology embeddings for ordinal damage classification.
"""

import timm
import torch
import torch.nn.functional as F

import torch.nn as nn

# Large negative offset so masked-out positions lose ``amax`` to in-mask values (avoids ``feat * 0 == 0``
# beating negative activations from the building).
_MASKED_POOL_MAX_PENALTY: float = 1e4

# Mean soft-mask mass on the feature grid below which the mask-weighted pooling branch falls back to
# global statistics. Internal numerical safety threshold; not a research axis - the value has never
# been varied across any trained configuration.
_MASK_POOL_MIN_AREA_FRAC: float = 1e-3


def _global_and_soft_mask_pools(
    feat: torch.Tensor,
    mask_b1hw: torch.Tensor,
    min_area_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global avg/max plus mask-weighted avg/max with fallback when mask support is tiny.

    Masked max uses ``feat + (w - 1) * penalty`` so low-weight pixels are pushed far below in-mask
    values; this avoids ``feat * w == 0`` winning ``amax`` over negative in-mask activations.

    Args:
        feat: Backbone feature map ``[B, C, H, W]``.
        mask_b1hw: Soft mask in input resolution ``[B, 1, H0, W0]`` (values in ``[0, 1]`` recommended).
        min_area_frac: If mean soft weight on the feature grid is below this, mask stats match global.

    Returns:
        Tuple ``(g_avg, g_max, m_avg, m_max)`` each ``[B, C]``.
    """

    g_avg = feat.mean(dim=(2, 3))
    g_max = torch.amax(feat, dim=(2, 3))
    _, _, h, w = feat.shape
    w_map = F.interpolate(mask_b1hw, size=(h, w), mode="bilinear", align_corners=False)
    w_map = w_map.clamp(0.0, 1.0)
    spatial = float(h * w)
    sum_w = w_map.sum(dim=(2, 3), keepdim=True)
    mean_frac = sum_w / spatial
    fb = (mean_frac < min_area_frac).to(dtype=feat.dtype)
    fb_b1 = fb.view(feat.size(0), 1)
    w_norm = w_map / (sum_w + 1e-6)
    m_avg = (feat * w_norm).sum(dim=(2, 3))
    penalty = feat.new_tensor(_MASKED_POOL_MAX_PENALTY)
    m_max = torch.amax(feat + (w_map - 1.0) * penalty, dim=(2, 3))
    m_avg = (1.0 - fb_b1) * m_avg + fb_b1 * g_avg
    m_max = (1.0 - fb_b1) * m_max + fb_b1 * g_max
    return g_avg, g_max, m_avg, m_max


class MaskConditionedDamageNet(nn.Module):
    """Fuses 4-channel imagery with disaster context for damage classification.

    Utilizes a pre-trained backbone (via timm) adapted for 4-channel input.
    The network conditions penultimate and final backbone feature maps via FiLM
    prior to pooling, then concatenates pooled multi-scale embeddings for scale
    gating and final classification.

    Args:
        backbone_name: The timm backbone to use (default: "convnext_nano").
        pretrained: Whether to load pre-trained ImageNet weights.
        num_classes: The number of ordinal damage classes (default: 4).
        context_dim: The input dimension of the typology vector (default: 4).
        mask_enabled: Whether to use the 4th-channel mask input.
        typology_enabled: Inject disaster typology via FiLM modulation over stage-3/4 feature maps
            when True. When False, the FiLM MLP and per-stage affine layers are not constructed and
            the forward pass skips conditioning entirely. Head input dim is always
            ``visual_feature_dim`` regardless of this flag.
        embedding_dim: The size of the projected context space (default: 128).
        dropout_p: The dropout probability before the final classification (default: 0.3).
        gate_strength: Dampening factor for per-sample dynamic scale gates. A value of
            0.0 enforces equal branch weights; 1.0 uses fully dynamic gate outputs.
        mask_weighted_pooling_enabled: When True (requires mask_enabled), concatenate global avg/max
            with mask-weighted avg/max (same width as global branches; fallback aligns mask stats
            with global when mask support on the feature grid is below ``_MASK_POOL_MIN_AREA_FRAC``).
        drop_path_rate: Stochastic-depth rate forwarded to the timm backbone at construction.
    """

    def __init__(self, backbone_name: str = "convnext_nano", pretrained: bool = True, num_classes: int = 4, context_dim: int = 4,
                 mask_enabled: bool = True, typology_enabled: bool = True, embedding_dim: int = 128,
                 dropout_p: float = 0.3, gate_strength: float = 0.5,
                 mask_weighted_pooling_enabled: bool = False, drop_path_rate: float = 0.1) -> None:
        """Initialize backbone, context encoder, and fusion head."""

        super().__init__()
        if not 0.0 <= gate_strength <= 1.0:
            raise ValueError("gate_strength must be in [0.0, 1.0].")
        if mask_weighted_pooling_enabled and not mask_enabled:
            raise ValueError("mask_weighted_pooling_enabled=True requires mask_enabled=True.")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0.0, 1.0).")
        self.mask_enabled = mask_enabled
        self.typology_enabled = typology_enabled
        self.mask_weighted_pooling_enabled = mask_weighted_pooling_enabled
        self.gate_strength = gate_strength
        in_channels = 3 + int(self.mask_enabled)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(2, 3),
            drop_path_rate=drop_path_rate,
        )

        # Zero-init any channel indices beyond the pretrained RGB trio (mask at index 3) on the stem
        # conv so ImageNet weights remain undisturbed at initialization.
        if self.mask_enabled and pretrained:
            with torch.no_grad():
                for module in self.backbone.modules():
                    if isinstance(module, nn.Conv2d) and module.in_channels == in_channels:
                        module.weight[:, 3, :, :] = 0.0
                        break  # Only the stem layer

        feature_channels = self.backbone.feature_info.channels()
        if len(feature_channels) < 2:
            raise ValueError(
                f"Backbone {backbone_name} must expose at least two feature stages for multi-scale pooling."
            )
        stage3_dim, stage4_dim = int(feature_channels[-2]), int(feature_channels[-1])
        pool_width = 4 if self.mask_weighted_pooling_enabled else 2
        self.stage3_feature_dim = pool_width * stage3_dim
        self.stage4_feature_dim = pool_width * stage4_dim
        pooled_branch_dim = min(self.stage3_feature_dim, self.stage4_feature_dim)
        self.stage3_proj = nn.Sequential(
            nn.Linear(self.stage3_feature_dim, pooled_branch_dim),
            nn.GELU(),
            nn.LayerNorm(pooled_branch_dim)
        )
        self.stage4_proj = nn.Sequential(
            nn.Linear(self.stage4_feature_dim, pooled_branch_dim),
            nn.GELU(),
            nn.LayerNorm(pooled_branch_dim)
        )
        # Let stage-3 and stage-4 pooled branches exchange information before scale gating.
        self.cross_scale_mixer = nn.Sequential(
            nn.Linear(2 * pooled_branch_dim, 2 * pooled_branch_dim),
            nn.GELU(),
            nn.Linear(2 * pooled_branch_dim, 2 * pooled_branch_dim)
        )
        self.stage3_mix_norm = nn.LayerNorm(pooled_branch_dim)
        self.stage4_mix_norm = nn.LayerNorm(pooled_branch_dim)
        final_mix_layer = self.cross_scale_mixer[-1]
        if isinstance(final_mix_layer, nn.Linear):
            nn.init.zeros_(final_mix_layer.weight)
            nn.init.zeros_(final_mix_layer.bias)
        # Learn a per-sample soft weighting between the two scales.
        self.scale_gate_mlp = nn.Sequential(
            nn.Linear(2 * pooled_branch_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        final_gate_layer = self.scale_gate_mlp[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.zeros_(final_gate_layer.weight)
            nn.init.zeros_(final_gate_layer.bias)
        visual_feature_dim = 2 * pooled_branch_dim
        self.visual_feature_dim = visual_feature_dim
        self.stage3_norm = nn.LayerNorm(self.stage3_feature_dim)
        self.stage4_norm = nn.LayerNorm(self.stage4_feature_dim)
        self.visual_norm = nn.LayerNorm(visual_feature_dim)

        if self.typology_enabled:
            # Project the sparse 4-dim one-hot vector into a dense representation.
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.GELU(),
                nn.Linear(64, embedding_dim),
                nn.GELU(),
            )
            self.stage3_film = nn.Linear(embedding_dim, 2 * stage3_dim)
            self.stage4_film = nn.Linear(embedding_dim, 2 * stage4_dim)
            nn.init.zeros_(self.stage3_film.weight)
            nn.init.zeros_(self.stage3_film.bias)
            nn.init.zeros_(self.stage4_film.weight)
            nn.init.zeros_(self.stage4_film.bias)
        else:
            self.context_mlp = None
            self.stage3_film = None
            self.stage4_film = None

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(visual_feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute logits from multi-channel imagery and typology context.

        The input tensor carries ``3 + mask_enabled`` channels (RGB, optional footprint mask at
        index 3). Mask-weighted pooling keys on the footprint at index 3.
        """

        # Apply mask ablation by dropping the mask channel before forwarding.
        image_tensor = x if self.mask_enabled else x[:, :3, :, :]
        feature_maps = self.backbone(image_tensor)
        if len(feature_maps) < 2:
            raise ValueError("Backbone did not return enough feature maps for multi-scale pooling.")

        # FiLM condition stage maps before pooling so context can shape spatial evidence.
        stage3_map, stage4_map = feature_maps[-2], feature_maps[-1]
        if self.typology_enabled and self.context_mlp is not None and self.stage3_film is not None and self.stage4_film is not None:
            c_features = self.context_mlp(context)
            stage3_gamma_delta, stage3_beta = torch.chunk(self.stage3_film(c_features), chunks=2, dim=1)
            stage4_gamma_delta, stage4_beta = torch.chunk(self.stage4_film(c_features), chunks=2, dim=1)
            stage3_gamma_delta = stage3_gamma_delta.unsqueeze(-1).unsqueeze(-1)
            stage3_beta = stage3_beta.unsqueeze(-1).unsqueeze(-1)
            stage4_gamma_delta = stage4_gamma_delta.unsqueeze(-1).unsqueeze(-1)
            stage4_beta = stage4_beta.unsqueeze(-1).unsqueeze(-1)
            stage3_map = stage3_map * (1.0 + stage3_gamma_delta) + stage3_beta
            stage4_map = stage4_map * (1.0 + stage4_gamma_delta) + stage4_beta

        # Pool penultimate and final feature stages; optional global + mask-weighted avg/max (concatenated).
        if self.mask_weighted_pooling_enabled:
            mask_b1hw = x[:, 3:4, :, :]
            s3_g_avg, s3_g_max, s3_m_avg, s3_m_max = _global_and_soft_mask_pools(
                stage3_map, mask_b1hw=mask_b1hw, min_area_frac=_MASK_POOL_MIN_AREA_FRAC
            )
            s4_g_avg, s4_g_max, s4_m_avg, s4_m_max = _global_and_soft_mask_pools(
                stage4_map, mask_b1hw=mask_b1hw, min_area_frac=_MASK_POOL_MIN_AREA_FRAC
            )
            stage3_features = torch.cat([s3_g_avg, s3_g_max, s3_m_avg, s3_m_max], dim=1)
            stage4_features = torch.cat([s4_g_avg, s4_g_max, s4_m_avg, s4_m_max], dim=1)
        else:
            stage3_avg = torch.mean(stage3_map, dim=[-2, -1])
            stage3_max = torch.amax(stage3_map, dim=[-2, -1])
            stage4_avg = torch.mean(stage4_map, dim=[-2, -1])
            stage4_max = torch.amax(stage4_map, dim=[-2, -1])
            stage3_features = torch.cat([stage3_avg, stage3_max], dim=1)
            stage4_features = torch.cat([stage4_avg, stage4_max], dim=1)
        stage3_features = self.stage3_norm(stage3_features)
        stage4_features = self.stage4_norm(stage4_features)
        stage3_features = self.stage3_proj(stage3_features)
        stage4_features = self.stage4_proj(stage4_features)

        mix_inputs = torch.cat([stage3_features, stage4_features], dim=1)
        mix_residual = self.cross_scale_mixer(mix_inputs)
        stage3_delta, stage4_delta = torch.chunk(mix_residual, chunks=2, dim=1)
        stage3_features = self.stage3_mix_norm(stage3_features + stage3_delta)
        stage4_features = self.stage4_mix_norm(stage4_features + stage4_delta)
        # Preserve branch energy while letting the model re-balance coarse vs fine scales per sample.
        gate_inputs = torch.cat([stage3_features, stage4_features], dim=1)
        dynamic_scale_weights = torch.softmax(self.scale_gate_mlp(gate_inputs), dim=1) * 2.0
        scale_weights = 1.0 + self.gate_strength * (dynamic_scale_weights - 1.0)
        stage3_features = stage3_features * scale_weights[:, 0:1]
        stage4_features = stage4_features * scale_weights[:, 1:2]
        v_features = torch.cat([stage3_features, stage4_features], dim=1)
        v_features = self.visual_norm(v_features)
        logits = self.head(v_features)

        return logits
