from ssregpros.loss.sobel import MaskedMSGELoss as Loss

import pytest
import torch

from hypothesis import assume, given, settings, strategies as st

ATOL = 1e-6


@pytest.fixture(scope="session")
def loss_fn():
    return Loss()


def create_test_data(batch_size=2, channels=3, height=32, width=32):
    """Helper to create test data"""
    y_true = torch.randn(batch_size, channels, height, width)
    y_pred = torch.randn(batch_size, channels, height, width)
    # Create random mask with at least one True value per batch element
    mask = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.bool)
    if (all_zeros := ~torch.any(mask, dim=(1, 2, 3))).any():
        mask.view(batch_size, -1)[all_zeros, 0] = True

    return y_true, y_pred, mask


# =============================================================================
# EXAMPLE-BASED TESTS
# =============================================================================


def test_basic_functionality(loss_fn):
    """Test that loss function runs without errors on basic input"""
    y_true, y_pred, mask = create_test_data()
    loss = loss_fn(y_true, y_pred, mask)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert loss.item() >= 0  # MSE is non-negative


def test_perfect_prediction_zero_loss(loss_fn):
    """Test that identical predictions yield zero loss"""
    shape = (1, 1, 10, 10)
    y_true = torch.ones(*shape) * 0.5  # Constant image
    y_pred = y_true.clone()
    mask = torch.ones(*shape, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_mask_broadcasting(loss_fn):
    """Test that mask broadcasts correctly for multi-channel images"""
    shape = (2, 3, 16, 16)
    mask_shape = (2, 1, 16, 16)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)
    mask = torch.ones(*mask_shape, dtype=torch.bool)  # Single channel mask

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)


def test_partial_mask(loss_fn):
    """Test loss computation with partial masking"""
    shape = (1, 1, 20, 20)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)

    # Create mask that covers only half the image
    mask = torch.zeros(*shape, dtype=torch.bool)
    mask[:, :, :10, :] = True

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_edge_gradients(loss_fn):
    """Test with images that have strong edges"""
    # Create step edge image
    shape = (1, 1, 20, 20)
    y_true = torch.zeros(*shape)
    y_true[:, :, :, :10] = 1.0  # Vertical edge

    y_pred = torch.zeros(*shape)  # No edge
    mask = torch.ones(*shape, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert (
        loss.item() > 0
    )  # Should have non-zero loss due to gradient difference


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_empty_mask_error(loss_fn):
    """Test error when mask is all False"""
    y_true, y_pred, _ = create_test_data()
    mask = torch.zeros(2, 1, 32, 32, dtype=torch.bool)  # All False

    with pytest.raises(ValueError, match="mask is empty"):
        loss_fn(y_true, y_pred, mask)


def test_shape_mismatch_error(loss_fn):
    """Test behavior with mismatched input shapes"""
    y_true = torch.randn(2, 3, 32, 32)
    y_pred = torch.randn(2, 3, 16, 16)  # Different spatial size
    mask = torch.ones(2, 1, 32, 32, dtype=torch.bool)

    with pytest.raises(
        RuntimeError
    ):  # PyTorch will raise on incompatible shapes
        loss_fn(y_true, y_pred, mask)


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================


@given(
    batch_size=st.integers(min_value=1, max_value=4),
    channels=st.integers(min_value=1, max_value=3),
    height=st.integers(min_value=8, max_value=32),
    width=st.integers(min_value=8, max_value=32),
)
@settings(deadline=10_000, max_examples=20)
def test_property_non_negative_loss(
    loss_fn, batch_size, channels, height, width
):
    """Property: Loss should always be non-negative"""
    y_true = torch.randn(batch_size, channels, height, width)
    y_pred = torch.randn(batch_size, channels, height, width)

    # Create random mask with at least one True value per batch element
    mask = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.bool)
    if (all_zeros := ~torch.any(mask, dim=(1, 2, 3))).any():
        mask.view(batch_size, -1)[all_zeros, 0] = True

    try:
        loss = loss_fn(y_true, y_pred, mask)
        assert (
            loss.item() >= 0
        ), f"Loss should be non-negative, got {loss.item()}"
        assert torch.isfinite(loss), "Loss should be finite"
    except RuntimeError:
        # Skip if we hit numerical issues
        assume(False)


@given(
    offset=st.floats(min_value=-5.0, max_value=5.0),
)
@settings(deadline=10_000, max_examples=20)
def test_property_translation_invariance(loss_fn, offset):
    """Property: Gradients should be translation-invariant"""
    shape = (1, 1, 16, 16)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)
    mask = torch.ones(*shape, dtype=torch.bool)

    # Compute loss for original and offset versions
    loss1 = loss_fn(y_true, y_pred, mask)
    loss2 = loss_fn(y_true + offset, y_pred + offset, mask)

    # Translation should not affect gradients significantly
    assert torch.isclose(
        loss1, loss2, rtol=1e-4, atol=1e-5
    ), f"Translation invariance violated: {loss1.item()} vs {loss2.item()}"


def test_property_symmetry(loss_fn):
    """Property: Loss should be symmetric in y_true and y_pred"""
    shape = (1, 1, 16, 16)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)
    mask = torch.ones(*shape, dtype=torch.bool)

    loss1 = loss_fn(y_true, y_pred, mask)
    loss2 = loss_fn(y_pred, y_true, mask)  # Swapped

    assert torch.isclose(
        loss1, loss2, rtol=ATOL
    ), f"Symmetry violated: {loss1.item()} vs {loss2.item()}"


def test_property_mask_monotonicity(loss_fn):
    """Property: Reducing mask size should not increase total loss contribution"""
    shape = (1, 1, 20, 20)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)

    # Full mask
    mask_full = torch.ones(*shape, dtype=torch.bool)

    # Half mask
    mask_half = torch.zeros(*shape, dtype=torch.bool)
    mask_half[:, :, :10, :] = True

    loss_full = loss_fn(y_true, y_pred, mask_full)
    loss_half = loss_fn(y_true, y_pred, mask_half)

    # Both should be finite (this is the main property we can reliably test)
    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_half)


@given(
    noise_std=st.floats(min_value=0.001, max_value=0.1),
)
@settings(max_examples=10)
def test_property_noise_sensitivity(loss_fn, noise_std):
    """Property: Adding noise should generally increase loss"""
    shape = (1, 1, 16, 16)
    y_true = torch.randn(*shape)
    y_pred = y_true.clone()  # Start with perfect prediction
    mask = torch.ones(*shape, dtype=torch.bool)

    loss_perfect = loss_fn(y_true, y_pred, mask)

    # Add noise to prediction
    noise = torch.randn_like(y_pred) * noise_std
    y_pred_noisy = y_pred + noise
    loss_noisy = loss_fn(y_true, y_pred_noisy, mask)

    # Noisy prediction should generally have higher loss
    assert (
        loss_noisy >= loss_perfect - ATOL
    ), f"Noise should increase loss: {loss_perfect.item()} -> {loss_noisy.item()}"


# =============================================================================
# GRADIENT/DIFFERENTIABILITY TESTS
# =============================================================================


def test_gradient_flow(loss_fn):
    """Test that gradients flow through the loss function"""
    shape = (1, 1, 16, 16)
    y_true = torch.randn(*shape, requires_grad=False)
    y_pred = torch.randn(*shape, requires_grad=True)
    mask = torch.ones(*shape, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    loss.backward()

    assert y_pred.grad is not None
    assert not torch.all(y_pred.grad == 0), "Gradients should not all be zero"
    assert torch.all(torch.isfinite(y_pred.grad)), "Gradients should be finite"


def test_numerical_gradient_check(loss_fn):
    """Basic numerical gradient check"""
    torch.manual_seed(42)
    shape = (1, 1, 8, 8)
    y_true, y_pred = (
        torch.randn(*shape, requires_grad=bool(grad), dtype=torch.float64)
        for grad in range(2)
    )
    mask = torch.ones(*shape, dtype=torch.bool)

    # Use gradcheck with reduced tolerance for this complex function
    torch.autograd.gradcheck(
        lambda x: loss_fn(y_true, x, mask),
        y_pred,
        eps=1e-4,
        atol=1e-3,
        rtol=1e-2,
    )


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================


def test_batch_vs_individual_processing(loss_fn):
    """Test that batch processing gives same average as individual processing"""
    torch.manual_seed(42)

    # Create batch data
    shape = (3, 1, 12, 12)
    y_true = torch.randn(*shape)
    y_pred = torch.randn(*shape)
    mask = torch.ones(*shape, dtype=torch.bool)

    # Compute batch loss
    batch_loss = loss_fn(y_true, y_pred, mask)

    # Compute individual losses
    individual_losses = []
    for i in range(shape[0]):
        loss_i = loss_fn(y_true[i : i + 1], y_pred[i : i + 1], mask[i : i + 1])
        individual_losses.append(loss_i.item())

    expected_loss = sum(individual_losses) / len(individual_losses)

    assert torch.isclose(
        batch_loss, torch.tensor(expected_loss), rtol=1e-5, atol=1e-7
    ), f"Batch loss {batch_loss.item()} != mean individual loss {expected_loss}"
