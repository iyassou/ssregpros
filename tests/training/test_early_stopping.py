from ssregpros.training.early_stopping import EarlyStopping, EarlyStoppingMode

import pytest


def test_min_mode_stops_when_no_improvement():
    """Test that MIN mode triggers early stopping after patience runs out."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=2,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    assert es.step(1.0) is False  # Initial value
    assert es.step(0.9) is False  # Improvement
    assert es.step(0.92) is False  # Bad step 1 (not improved enough)
    assert es.step(0.91) is False  # Bad step 2
    assert es.step(0.93) is True  # Bad step 3 - should stop


def test_max_mode_stops_when_no_improvement():
    """Test that MAX mode triggers early stopping after patience runs out."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MAX,
        patience=2,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    assert es.step(0.5) is False  # Initial value
    assert es.step(0.6) is False  # Improvement
    assert es.step(0.59) is False  # Bad step 1
    assert es.step(0.58) is False  # Bad step 2
    assert es.step(0.57) is True  # Bad step 3 - should stop


def test_min_mode_resets_patience_on_improvement():
    """Test that patience counter resets when improvement occurs."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=2,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    assert es.step(1.0) is False  # Initial
    assert es.step(1.1) is False  # Bad step 1
    assert es.step(1.2) is False  # Bad step 2
    assert es.step(0.8) is False  # Improvement - resets counter
    assert es.step(0.85) is False  # Bad step 1
    assert es.step(0.86) is False  # Bad step 2
    assert es.step(0.87) is True  # Bad step 3 - should stop


def test_absolute_delta_threshold():
    """Test that absolute delta threshold is respected."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.0,
        min_delta_abs=0.1,
    )

    assert es.step(1.0) is False  # Initial
    assert es.step(0.95) is False  # Not improved by 0.1
    assert es.step(0.89) is False  # Improved by 0.11
    assert es.step(0.85) is False  # Not improved by 0.1
    assert es.step(0.84) is True  # Patience exhausted


def test_relative_delta_threshold():
    """Test that relative delta threshold is respected."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,  # 10% improvement required
        min_delta_abs=0.0,
    )

    assert es.step(10.0) is False  # Initial, best=10.0
    # Need improvement of 10.0 * 0.1 = 1.0
    assert es.step(9.5) is False  # Not improved by 1.0
    assert es.step(8.9) is False  # Improved by 1.1, best=8.9
    # Need improvement of 8.9 * 0.1 = 0.89
    assert es.step(8.5) is False  # Not improved by 0.89
    assert es.step(8.4) is True  # Patience exhausted


def test_delta_uses_max_of_absolute_and_relative():
    """Test that delta() returns the maximum of absolute and relative thresholds."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,  # 10%
        min_delta_abs=0.5,  # Absolute 0.5
    )

    assert es.step(2.0) is False  # Initial, best=2.0
    # Relative: 2.0 * 0.1 = 0.2, Absolute: 0.5, max = 0.5
    assert es.delta() == 0.5

    assert es.step(1.4) is False  # Improved by 0.6 (> 0.5), best=1.4
    # Relative: 1.4 * 0.1 = 0.14, Absolute: 0.5, max = 0.5
    assert es.delta() == 0.5

    es_large = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.5,
    )
    assert es_large.step(10.0) is False  # Initial, best=10.0
    # Relative: 10.0 * 0.1 = 1.0, Absolute: 0.5, max = 1.0
    assert es_large.delta() == 1.0


def test_delta_scales_with_best_value():
    """Test that relative delta scales with the magnitude of best value."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.0,
    )

    es.step(1.0)  # best=1.0
    assert es.best == 1.0
    assert es.delta() == pytest.approx(0.1)

    es.step(0.8)  # best updates to 0.8
    assert es.best == 0.8
    assert es.delta() == pytest.approx(0.08)

    es.step(0.5)  # best updates to 0.5
    assert es.best == 0.5
    assert es.delta() == pytest.approx(0.05)

    # Test with larger values
    es_large = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.0,
    )
    es_large.step(10.0)  # best=10.0
    assert es_large.best == 10.0
    assert es_large.delta() == pytest.approx(1.0)

    es_large.step(5.0)  # best updates to 5.0
    assert es_large.best == 5.0
    assert es_large.delta() == pytest.approx(0.5)


def test_reset_clears_state():
    """Test that reset() clears best value and bad counter."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=2,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    es.step(1.0)
    es.step(1.1)  # Bad step
    assert es.num_bad == 1
    assert es.best == 1.0

    es.reset()
    assert es.best is None
    assert es.num_bad == 0

    # Should work as fresh start
    assert es.step(5.0) is False
    assert es.best == 5.0


def test_delta_raises_before_first_step():
    """Test that delta() raises ValueError before any step is taken."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=2,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    with pytest.raises(
        ValueError, match="cannot compute delta before first step"
    ):
        es.delta()


def test_invalid_mode_raises():
    """Test that invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="unrecognised mode"):
        EarlyStopping(
            mode="invalid",  # pyright: ignore[reportArgumentType]
            patience=2,
            min_delta_rel=0.01,
            min_delta_abs=0.0,
        )


def test_patience_zero_stops_immediately():
    """Test that patience=0 stops after first non-improvement."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=0,
        min_delta_rel=0.01,
        min_delta_abs=0.0,
    )

    assert es.step(1.0) is False  # Initial
    assert es.step(1.1) is True  # First bad step triggers stop


def test_max_mode_improvement_detection():
    """Test that MAX mode correctly detects improvements."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MAX,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.0,
    )

    assert es.step(0.5) is False  # Initial, best=0.5
    # Need improvement > 0.5 + 0.05 = 0.55
    assert es.step(0.54) is False  # Bad step 1: 0.54 < 0.55, not improved
    assert (
        es.step(0.549) is True
    )  # Bad step 2: 0.549 < 0.55, patience exhausted
    assert es.best == 0.5  # Best never changed

    # Test that actual improvement resets counter
    es2 = EarlyStopping(
        mode=EarlyStoppingMode.MAX,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.0,
    )
    assert es2.step(0.5) is False  # Initial, best=0.5
    assert es2.step(0.54) is False  # Bad step 1
    assert es2.step(0.56) is False  # Improvement! 0.56 > 0.55, best=0.56
    assert es2.best == 0.56


def test_negative_values_work_correctly():
    """Test that early stopping works with negative values."""
    es = EarlyStopping(
        mode=EarlyStoppingMode.MIN,
        patience=1,
        min_delta_rel=0.1,
        min_delta_abs=0.0,
    )

    assert es.step(-1.0) is False  # Initial, best=-1.0
    # Scale uses abs(-1.0) = 1.0, delta = 0.1
    assert es.delta() == 0.1
    assert es.step(-1.2) is False  # Improved by 0.2
    assert es.best == -1.2
