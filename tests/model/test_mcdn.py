"""Tests for the unitemporal model architecture."""

import pytest
import torch

from src.model.mcdn import MaskCenteredDamageNet


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a dummy batch matching the dataset output."""
    batch_size = 2
    # [B, C, H, W] -> 4 Channels
    images = torch.randn(batch_size, 4, 224, 224)
    # [B, Context] -> 4 Typologies
    context = torch.zeros(batch_size, 4)
    context[0, 1] = 1.0  # Simulate 'Kinetic'
    context[1, 0] = 1.0  # Simulate 'Wind/Flood'

    return images, context


def test_model_initialization() -> None:
    """Verifies the model initializes with 4 input channels correctly."""
    # Using a tiny test model to keep memory footprint low during pytest
    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False)

    # Verify the first conv layer expects 4 channels
    first_conv = list(model.backbone.modules())[1]
    assert first_conv.in_channels == 4


def test_forward_pass_shape(sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verifies the forward pass returns the correct logit shape."""
    images, context = sample_batch
    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False)

    model.eval()
    with torch.no_grad():
        logits = model(images, context)

    assert logits.shape == (2, 4)  # [Batch Size, Num Classes]


def test_gradient_flow(sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Verifies gradients flow back to both inputs (Visual and Context)."""
    images, context = sample_batch

    # Require gradients for inputs to test the backward pass
    images.requires_grad = True
    context.requires_grad = True

    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False)
    # FiLM affine layers are zero-initialized so the network starts from an identity
    # transform; perturb them so the context branch actually contributes to logits and
    # the architectural connectivity check below is meaningful.
    with torch.no_grad():
        model.stage3_film.weight.normal_(std=1e-2)
        model.stage4_film.weight.normal_(std=1e-2)
    model.train()

    logits = model(images, context)

    loss = logits.sum()
    loss.backward()

    assert images.grad is not None
    assert context.grad is not None
    assert torch.any(images.grad != 0)
    assert torch.any(context.grad != 0)


def test_model_initialization_mask_ablation_uses_three_input_channels() -> None:
    """mask_enabled=False switches backbone input channels from 4 to 3."""

    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False, mask_enabled=False)
    first_conv = list(model.backbone.modules())[1]
    assert first_conv.in_channels == 3


def test_forward_pass_shape_typology_ablation(sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """typology_enabled=False still returns valid logits shape."""

    images, context = sample_batch
    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False, typology_enabled=False)
    model.eval()
    with torch.no_grad():
        logits = model(images, context)

    assert logits.shape == (2, 4)


def test_gradient_flow_typology_ablation_ignores_context_branch(sample_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """When typology branch is disabled, gradients do not flow through context."""

    images, context = sample_batch
    images.requires_grad = True
    context.requires_grad = True
    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False, typology_enabled=False)
    model.train()

    logits = model(images, context)
    logits.sum().backward()

    assert images.grad is not None
    assert torch.any(images.grad != 0)
    assert context.grad is None


def test_typology_disabled_skips_film_modules() -> None:
    """typology_enabled=False omits the FiLM MLP and per-stage affine layers."""

    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False, typology_enabled=False)
    assert model.context_mlp is None
    assert model.stage3_film is None
    assert model.stage4_film is None


def test_typology_enabled_builds_film_modules() -> None:
    """typology_enabled=True instantiates the FiLM MLP and per-stage affine layers."""

    model = MaskCenteredDamageNet(backbone_name="resnet18", pretrained=False, typology_enabled=True)
    assert model.context_mlp is not None
    assert model.stage3_film is not None
    assert model.stage4_film is not None
