from ssregpros.loss import Reduction
from ssregpros.loss.ncc import MaskedNCCLoss as Loss

import pytest
import torch

from hypothesis import given, HealthCheck, settings, strategies as st

# If your tests sometimes run slow with torch on CI, removing the deadline helps.
settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")

ATOL = 1e-5


def get_available_gpu() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return None


@pytest.fixture
def loss_fn():
    """Default loss function for testing"""
    return Loss(reduction=Reduction.NONE)


@pytest.fixture
def simple_data():
    """Simple test data"""
    torch.manual_seed(42)
    y_true = torch.randn(2, 3, 4, 4)
    y_pred = torch.randn(2, 3, 4, 4)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    return y_true, y_pred, mask


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


def test_perfect_correlation_gives_zero_loss(loss_fn):
    """Perfect correlation should give loss of -1"""
    y_true = torch.randn(1, 1, 4, 4)
    y_pred = y_true.clone()  # Perfect correlation
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


def test_anti_correlation_gives_positive_one_loss(loss_fn):
    """Perfect anti-correlation should give loss of +1"""
    y_true = torch.randn(1, 1, 4, 4)
    y_pred = -y_true  # Perfect anti-correlation
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(1.0), atol=ATOL)


def test_zero_correlation_gives_naught_point_five_loss(loss_fn):
    """Orthogonal signals should give loss near 0.5"""
    # Create orthogonal signals (zero correlation)
    # NOTE: inputs are orthogonal because:
    #   • y_true = [1, -1, 1, -1], mean = 0, so centered = [1, -1, 1, -1]
    #   • y_pred = [1, 1, -1, -1], mean = 0, so centered = [1, 1, -1, -1]
    #   • Covariance = (1×1 + (-1)×1 + 1×(-1) + (-1)×(-1))/4 = (1 - 1 - 1 + 1)/4 = 0
    y_true = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32)
    y_pred = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.abs(loss - 0.5) < ATOL


def test_output_is_scalar(loss_fn, simple_data):
    """Output should be a scalar tensor"""
    y_true, y_pred, mask = simple_data
    loss = loss_fn(y_true, y_pred, mask)
    assert loss.dim() == 0  # Scalar
    assert loss.numel() == 1


def test_loss_is_differentiable(loss_fn, simple_data):
    """Loss should be differentiable w.r.t. inputs"""
    y_true, y_pred, mask = simple_data
    y_true.requires_grad_(True)
    y_pred.requires_grad_(True)

    loss = loss_fn(y_true, y_pred, mask)
    loss.backward()

    assert y_true.grad is not None
    assert y_pred.grad is not None
    assert not torch.isnan(y_true.grad).any()
    assert not torch.isnan(y_pred.grad).any()


# ============================================================================
# MATHEMATICAL PROPERTIES TESTS USING HYPOTHESIS
# ============================================================================


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    batch_size=st.integers(1, 8),
    channels=st.integers(1, 4),
    height=st.integers(2, 16),
    width=st.integers(2, 16),
)
def test_output_bounds_property(loss_fn, batch_size, channels, height, width):
    """Property test: output should always be bounded to [0, 1]"""

    y_true = torch.randn(batch_size, channels, height, width)
    y_pred = torch.randn(batch_size, channels, height, width)

    # Create random mask with at least one True value per batch element
    mask = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.bool)
    if (all_zeros := ~torch.any(mask, dim=(1, 2, 3))).any():
        mask.view(batch_size, -1)[all_zeros, 0] = True

    loss: torch.Tensor = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)
    assert 0 - ATOL <= loss <= 1.0 + ATOL


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    scale_true=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
    scale_pred=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
)
def test_scale_invariance_property(loss_fn, scale_true, scale_pred):
    """Property test: scaling should not affect NCC"""

    y_true = torch.randn(1, 1, 4, 4)
    y_pred = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss_original = loss_fn(y_true, y_pred, mask)
    loss_scaled = loss_fn(scale_true * y_true, scale_pred * y_pred, mask)

    assert torch.isclose(loss_original, loss_scaled, atol=ATOL)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    shift_true=st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    shift_pred=st.floats(
        min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
def test_translation_invariance(loss_fn, shift_true, shift_pred):
    """NCC should be invariant to translation (constant shifts)"""
    y_true = torch.randn(1, 1, 4, 4)
    y_pred = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss_original = loss_fn(y_true, y_pred, mask)
    loss_shifted = loss_fn(y_true + shift_true, y_pred + shift_pred, mask)

    assert torch.isclose(loss_original, loss_shifted, atol=ATOL)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    batch_size=st.integers(1, 8),
    channels=st.integers(1, 4),
    height=st.integers(2, 16),
    width=st.integers(2, 16),
)
def test_symmetry_property(loss_fn, batch_size, channels, height, width):
    """NCC should be symmetric: NCC(x,y) = NCC(y,x)"""
    y_true = torch.randn(batch_size, channels, height, width)
    y_pred = torch.randn(batch_size, channels, height, width)

    # Create random mask with at least one True value per batch element
    mask = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.bool)
    if (all_zeros := ~torch.any(mask, dim=(1, 2, 3))).any():
        mask.view(batch_size, -1)[all_zeros, 0] = True

    loss1 = loss_fn(y_true, y_pred, mask)
    loss2 = loss_fn(y_pred, y_true, mask)

    assert torch.isclose(loss1, loss2, atol=ATOL)


# ============================================================================
# MASKING TESTS
# ============================================================================


def test_mask_application(loss_fn):
    """Only masked pixels should contribute to the loss"""
    # Create data where masked and unmasked regions have different correlations
    y_true = torch.zeros(1, 1, 4, 4)
    y_pred = torch.zeros(1, 1, 4, 4)

    # Perfect correlation in masked region
    y_true[0, 0, :2, :2] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y_pred[0, 0, :2, :2] = torch.tensor(
        [[1, 2], [3, 4]], dtype=torch.float32
    )  # Same

    # Perfect anti-correlation in unmasked region
    y_true[0, 0, 2:, 2:] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y_pred[0, 0, 2:, 2:] = torch.tensor(
        [[-1, -2], [-3, -4]], dtype=torch.float32
    )  # Opposite

    # Mask only covers the top-left region (perfect correlation)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, :2, :2] = True

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


def test_partial_mask(loss_fn):
    """Loss should work correctly with partial masks"""
    y_true = torch.randn(1, 1, 4, 4)
    y_pred = y_true.clone()  # Perfect correlation

    # Mask only half the pixels
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, :2, :] = True  # Top half only

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


def test_mask_broadcasting(loss_fn):
    """Mask should broadcast correctly across channels"""
    y_true = torch.randn(1, 3, 4, 4)  # 3 channels
    y_pred = y_true.clone()
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # Single channel mask

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


def test_multichannel_with_multichannel_mask(loss_fn):
    """Test with multi-channel mask matching input channels"""
    y_true = torch.randn(1, 3, 4, 4)
    y_pred = y_true.clone()
    mask = torch.ones(1, 3, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


# ============================================================================
# REDUCTION SCHEME TESTS
# ============================================================================


def test_none_reduction():
    """Test NONE reduction (simple average)"""
    loss_fn = Loss(reduction=Reduction.NONE)

    # Create batch where one has perfect correlation, other has anti-correlation
    y_true = torch.zeros(2, 1, 2, 2)
    y_pred = torch.zeros(2, 1, 2, 2)

    y_true[0, 0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y_pred[0, 0] = torch.tensor(
        [[1, 2], [3, 4]], dtype=torch.float32
    )  # Perfect correlation

    y_true[1, 0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y_pred[1, 0] = torch.tensor(
        [[-1, -2], [-3, -4]], dtype=torch.float32
    )  # Anti-correlation

    mask = torch.ones(2, 1, 2, 2, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    # First batch: NCC = 1 (perfect correlation), so loss contribution = 0.5 * (1 - NCC) = 0
    # Second batch: NCC = -1 (perfect anti-correlation), so loss contribution = 0.5 * (1 - NCC) = 1
    expected = 1.0 / 2  # Average of 0 and 1

    assert torch.isclose(loss, torch.tensor(expected), atol=ATOL)


def test_average_reduction():
    """Test AVERAGE reduction (weighted by number of pixels)"""
    loss_fn = Loss(Reduction.MEAN)

    y_true = torch.zeros(2, 1, 4, 4)
    y_pred = torch.zeros(2, 1, 4, 4)

    # First batch: perfect correlation
    y_true[0, 0] = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float32,
    )
    y_pred[0, 0] = torch.tensor(
        [
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24],
            [26, 28, 30, 32],
        ],
        dtype=torch.float32,
    )  # 2x y_true

    # Second batch: perfect anti-correlation
    y_true[1, 0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).repeat(
        2, 2
    )  # Fill 4x4
    y_pred[1, 0] = torch.tensor(
        [[-2, -4], [-6, -8]], dtype=torch.float32
    ).repeat(
        2, 2
    )  # -2x y_true

    # Different mask sizes
    mask1 = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # 16 pixels
    mask2 = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask2[0, 0, :2, :2] = True  # 4 pixels
    mask = torch.cat([mask1, mask2], dim=0)

    loss = loss_fn(y_true, y_pred, mask)

    # First batch: NCC = 1 (perfect correlation), so loss contribution = 0.5 * (1 - NCC) = 0
    # Second batch: NCC = -1 (perfect anti-correlation), so loss contribution = 0.5 * (1 - NCC) = 1
    # Weighted by 16/(16+4) and 4/(16+4)
    denom = mask1.sum() + mask2.sum()  # 16+4
    expected_weight1 = mask1.sum() / denom  # 0.8
    expected_weight2 = mask2.sum() / denom  # 0.2
    expected = (0) * expected_weight1 + (1.0) * expected_weight2
    assert torch.isclose(loss, expected, atol=ATOL)


def test_sqrt_reduction():
    """Test SQRT reduction"""
    loss_fn = Loss(Reduction.SQRT)

    y_true = torch.zeros(2, 1, 4, 4)
    y_pred = torch.zeros(2, 1, 4, 4)

    # First batch: perfect correlation
    y_true[0, 0] = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float32,
    )
    y_pred[0, 0] = torch.tensor(
        [
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24],
            [26, 28, 30, 32],
        ],
        dtype=torch.float32,
    )  # 2x y_true

    # Second batch: perfect anti-correlation
    y_true[1, 0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).repeat(
        2, 2
    )  # Fill 4x4
    y_pred[1, 0] = torch.tensor(
        [[-2, -4], [-6, -8]], dtype=torch.float32
    ).repeat(
        2, 2
    )  # -2x y_true

    # Different mask sizes
    mask1 = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # 16 pixels
    mask2 = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask2[0, 0, :2, :2] = True  # 4 pixels
    mask = torch.cat([mask1, mask2], dim=0)

    loss = loss_fn(y_true, y_pred, mask)

    # First batch: NCC = 1 (perfect correlation), so loss contribution = 0.5 * (1 - NCC) = 0
    # Second batch: NCC = -1 (perfect anti-correlation), so loss contribution = 0.5 * (1 - NCC) = 1
    # Weighted by (√16)/(√16+√4) and (√4)/(√16+√4)
    denom = mask1.sum().sqrt() + mask2.sum().sqrt()  # √16+√4
    expected_weight1 = mask1.sum().sqrt() / denom  # 0.666...
    expected_weight2 = mask2.sum().sqrt() / denom  # 0.333...
    expected = (0) * expected_weight1 + (1.0) * expected_weight2
    assert torch.isclose(loss, expected, atol=ATOL)


def test_log_reduction():
    """Test LOG reduction"""
    loss_fn = Loss(Reduction.LOG)

    y_true = torch.zeros(2, 1, 4, 4)
    y_pred = torch.zeros(2, 1, 4, 4)

    # First batch: perfect correlation
    y_true[0, 0] = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=torch.float32,
    )
    y_pred[0, 0] = torch.tensor(
        [
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24],
            [26, 28, 30, 32],
        ],
        dtype=torch.float32,
    )  # 2x y_true

    # Second batch: perfect anti-correlation
    y_true[1, 0] = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).repeat(
        2, 2
    )  # Fill 4x4
    y_pred[1, 0] = torch.tensor(
        [[-2, -4], [-6, -8]], dtype=torch.float32
    ).repeat(
        2, 2
    )  # -2x y_true

    # Different mask sizes
    mask1 = torch.ones(1, 1, 4, 4, dtype=torch.bool)  # 16 pixels
    mask2 = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask2[0, 0, :2, :2] = True  # 4 pixels
    mask = torch.cat([mask1, mask2], dim=0)

    loss = loss_fn(y_true, y_pred, mask)

    # First batch: NCC = 1 (perfect correlation), so loss contribution = 0.5 * (1 - NCC) = 0
    # Second batch: NCC = -1 (perfect anti-correlation), so loss contribution = 0.5 * (1 - NCC) = 1
    # Weighted by log(16)/(log(16)+log(4)) and log(4)/(log(16)+log(4))
    denom = mask1.sum().log() + mask2.sum().log()  # log(16)+log(4)
    expected_weight1 = mask1.sum().log() / denom  # 0.666...
    expected_weight2 = mask2.sum().log() / denom  # 0.333...
    expected = (0) * expected_weight1 + (1.0) * expected_weight2
    assert torch.isclose(loss, expected, atol=ATOL)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


def test_empty_mask_raises_error(loss_fn, simple_data):
    """Empty mask should raise ValueError"""
    y_true, y_pred, _ = simple_data
    mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)  # All False

    with pytest.raises(ValueError, match="mask is empty"):
        loss_fn(y_true, y_pred, mask)


def test_constant_input_handling(loss_fn):
    """Test with constant inputs (zero variance)"""
    y_true = torch.ones(1, 1, 4, 4)  # Constant
    y_pred = torch.ones(1, 1, 4, 4) * 2.0  # Different constant
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    # Should not crash due to division by zero (epsilon protection)
    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)


def test_single_pixel_mask(loss_fn):
    """Test with single pixel mask"""
    y_true = torch.randn(1, 1, 4, 4)
    y_pred = torch.randn(1, 1, 4, 4)

    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0, 0] = True  # Only one pixel

    # Should handle single pixel case gracefully
    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)


def test_very_large_values(loss_fn):
    """Test numerical stability with large values"""
    y_true = torch.ones(1, 1, 4, 4) * 1e6
    y_pred = torch.ones(1, 1, 4, 4) * 2e6
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)
    assert torch.abs(loss) <= 1.0 + ATOL  # Should still be bounded


def test_very_small_values(loss_fn):
    """Test numerical stability with small values"""
    y_true = torch.ones(1, 1, 4, 4) * ATOL
    y_pred = torch.ones(1, 1, 4, 4) * 2 * ATOL
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isfinite(loss)


# ============================================================================
# BATCH SIZE TESTS
# ============================================================================


def test_single_batch_element(loss_fn):
    """Test with batch size 1"""
    y_true = torch.randn(1, 2, 4, 4)
    y_pred = y_true.clone()
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


def test_large_batch_size(loss_fn):
    """Test with large batch size"""
    batch_size = 32
    y_true = torch.randn(batch_size, 3, 8, 8)
    y_pred = y_true.clone()  # Perfect correlation for all
    mask = torch.ones(batch_size, 1, 8, 8, dtype=torch.bool)

    loss = loss_fn(y_true, y_pred, mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=ATOL)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_gradient_flow_in_training_loop(loss_fn):
    """Test that gradients flow properly in a training scenario"""
    model = torch.nn.Conv2d(3, 3, 1)  # Simple 1x1 conv
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    y_true = torch.randn(2, 3, 4, 4)
    x = torch.randn(2, 3, 4, 4)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)

    # Forward pass
    y_pred = model(x)
    loss = loss_fn(y_true, y_pred, mask)

    # Backward pass
    loss.backward()

    # Check that gradients exist and are finite
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

    # Optimizer step should work
    optimizer.step()


@pytest.mark.skipif(
    get_available_gpu() is None, reason="No GPU available on device."
)
def test_consistency_across_devices():
    """Test consistency between CPU and GPU (if available)"""
    loss_fn = Loss(reduction=Reduction.NONE)

    y_true = torch.randn(2, 3, 4, 4)
    y_pred = torch.randn(2, 3, 4, 4)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool)

    loss_cpu = loss_fn(y_true, y_pred, mask)

    gpu = get_available_gpu()
    assert gpu is not None

    y_true_gpu = y_true.to(gpu)
    y_pred_gpu = y_pred.to(gpu)
    mask_gpu = mask.to(gpu)
    loss_fn_gpu = loss_fn.to(gpu)

    loss_gpu = loss_fn_gpu(y_true_gpu, y_pred_gpu, mask_gpu)

    assert torch.isclose(loss_cpu, loss_gpu.cpu(), atol=ATOL)
