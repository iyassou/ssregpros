from ssregpros.training.scheduler import (
    CosineAnnealing,
    Linear,
    Exponential,
    Step,
    PulseWindowScheduler,
)

from unittest.mock import MagicMock

import math
import pytest


@pytest.fixture
def update_fn_tracker():
    """Create a mock update function that tracks calls."""
    tracker = MagicMock(return_value=None)
    return tracker


@pytest.fixture
def cosine_scheduler(update_fn_tracker):
    """Create a CosineAnnealing scheduler."""
    return CosineAnnealing(
        update_fn=update_fn_tracker,
        initial_weight=1.0,
        total_epochs=100,
    )


@pytest.fixture
def cosine_scheduler_with_clamp(update_fn_tracker):
    """Create a CosineAnnealing scheduler with clamping at 0.1."""
    return CosineAnnealing(
        update_fn=update_fn_tracker,
        initial_weight=1.0,
        total_epochs=100,
        clamp_at_total_epochs=0.1,
    )


@pytest.fixture
def linear_scheduler(update_fn_tracker):
    """Create a Linear scheduler."""
    return Linear(
        update_fn=update_fn_tracker,
        initial_weight=1.0,
        total_epochs=100,
    )


@pytest.fixture
def exponential_scheduler(update_fn_tracker):
    """Create an Exponential scheduler."""
    return Exponential(
        update_fn=update_fn_tracker,
        initial_weight=1.0,
        total_epochs=100,
        decay_rate=0.9,
    )


@pytest.fixture
def step_scheduler(update_fn_tracker):
    """Create a Step scheduler."""
    return Step(
        update_fn=update_fn_tracker,
        initial_weight=1.0,
        total_epochs=100,
        step_epochs=[25, 50, 75],
        gamma=0.5,
    )


class TestCosineAnnealing:
    def test_initial_value(self, cosine_scheduler):
        """Noise weight should be initial value at epoch 0."""
        assert cosine_scheduler.compute() == pytest.approx(
            cosine_scheduler.initial_weight
        )

    def test_middle_value(self, cosine_scheduler):
        """Noise weight should be ~0.5 at halfway point."""
        cosine_scheduler.current_epoch = cosine_scheduler.total_epochs / 2
        assert cosine_scheduler.compute() == pytest.approx(
            cosine_scheduler.initial_weight / 2, rel=0.01
        )

    def test_final_value(self, cosine_scheduler):
        """Noise weight should be ~0 at final epoch."""
        cosine_scheduler.current_epoch = cosine_scheduler.total_epochs
        assert cosine_scheduler.compute() == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_decrease(self, cosine_scheduler):
        """Noise weight should monotonically decrease."""
        values = []
        for i in range(cosine_scheduler.total_epochs + 1):
            cosine_scheduler.current_epoch = i
            values.append(cosine_scheduler.compute())

        for i in range(len(values) - 1):
            assert values[i + 1] <= values[i]

    def test_step_calls_update_fn(self, cosine_scheduler, update_fn_tracker):
        """step() should call update_fn with computed weight."""
        cosine_scheduler.step()
        update_fn_tracker.assert_called()
        assert (
            update_fn_tracker.call_count == 2
        )  # once in __init__, once in step()

    def test_current_epoch_increments(self, cosine_scheduler):
        """current_epoch should increment after step()."""
        assert cosine_scheduler.current_epoch == 0
        cosine_scheduler.step()
        assert cosine_scheduler.current_epoch == 1
        cosine_scheduler.step()
        assert cosine_scheduler.current_epoch == 2


class TestCosineAnnealingWithClamping:
    def test_no_clamping_before_total_epochs(self, cosine_scheduler_with_clamp):
        """Before reaching total_epochs, should follow cosine curve."""
        cosine_scheduler_with_clamp.current_epoch = (
            cosine_scheduler_with_clamp.total_epochs / 2
        )
        value = cosine_scheduler_with_clamp.compute()
        expected = (
            cosine_scheduler_with_clamp.initial_weight
            * 0.5
            * (1 + math.cos(math.pi * 0.5))
        )
        assert value == pytest.approx(expected)

    def test_clamping_at_total_epochs(self, cosine_scheduler_with_clamp):
        """At total_epochs, should return clamp_at_total_epochs value."""
        cosine_scheduler_with_clamp.current_epoch = (
            cosine_scheduler_with_clamp.total_epochs
        )
        assert cosine_scheduler_with_clamp.compute() == pytest.approx(0.1)

    def test_clamping_after_total_epochs(self, cosine_scheduler_with_clamp):
        """After total_epochs, should remain at clamp_at_total_epochs value."""
        cosine_scheduler_with_clamp.current_epoch = (
            cosine_scheduler_with_clamp.total_epochs + 1
        )
        assert cosine_scheduler_with_clamp.compute() == pytest.approx(0.1)

    def test_clamping_stays_constant(self, cosine_scheduler_with_clamp):
        """Clamped value should remain constant across multiple epochs."""
        values = []
        for epoch in range(
            cosine_scheduler_with_clamp.total_epochs,
            cosine_scheduler_with_clamp.total_epochs + 10,
        ):
            cosine_scheduler_with_clamp.current_epoch = epoch
            values.append(cosine_scheduler_with_clamp.compute())
        for val in values:
            assert val == pytest.approx(
                cosine_scheduler_with_clamp.clamp_at_total_epochs
            )


class TestLinear:
    def test_initial_value(self, linear_scheduler):
        """Noise weight should be initial value at epoch 0."""
        assert linear_scheduler.compute() == pytest.approx(
            linear_scheduler.initial_weight
        )

    def test_middle_value(self, linear_scheduler):
        """Noise weight should be ~0.5 at halfway point."""
        linear_scheduler.current_epoch = linear_scheduler.total_epochs / 2
        assert linear_scheduler.compute() == pytest.approx(
            linear_scheduler.initial_weight / 2, rel=0.01
        )

    def test_final_value(self, linear_scheduler):
        """Noise weight should be 0 at final epoch."""
        linear_scheduler.current_epoch = linear_scheduler.total_epochs
        assert linear_scheduler.compute() == pytest.approx(0.0, abs=1e-6)

    def test_beyond_total_epochs(self, linear_scheduler):
        """Noise weight should stay at 0 after total_epochs."""
        linear_scheduler.current_epoch = linear_scheduler.total_epochs + 1
        assert linear_scheduler.compute() == pytest.approx(0.0)

    def test_monotonic_decrease(self, linear_scheduler):
        """Noise weight should monotonically decrease."""
        prev_val = linear_scheduler.compute()
        for i in range(1, linear_scheduler.total_epochs + 1):
            linear_scheduler.current_epoch = i
            curr_val = linear_scheduler.compute()
            assert curr_val <= prev_val
            prev_val = curr_val


class TestExponential:
    def test_initial_value(self, exponential_scheduler):
        """Noise weight should be initial value at epoch 0."""
        assert exponential_scheduler.compute() == pytest.approx(
            exponential_scheduler.initial_weight
        )

    def test_middle_value(self, exponential_scheduler):
        """Noise weight at halfway should follow exponential decay."""
        exponential_scheduler.current_epoch = (
            exponential_scheduler.total_epochs / 2
        )
        expected = exponential_scheduler.initial_weight * (0.9**0.5)
        assert exponential_scheduler.compute() == pytest.approx(expected)

    def test_final_value(self, exponential_scheduler):
        """Noise weight at final epoch should follow exponential decay."""
        exponential_scheduler.current_epoch = exponential_scheduler.total_epochs
        expected = exponential_scheduler.initial_weight * 0.9
        assert exponential_scheduler.compute() == pytest.approx(expected)

    def test_custom_decay_rate(self, update_fn_tracker):
        """Custom decay_rate should be respected."""
        scheduler = Exponential(
            update_fn=update_fn_tracker,
            initial_weight=1.0,
            total_epochs=100,
            decay_rate=0.5,
        )
        scheduler.current_epoch = scheduler.total_epochs
        expected = scheduler.initial_weight * 0.5
        assert scheduler.compute() == pytest.approx(expected)

    def test_monotonic_decrease(self, exponential_scheduler):
        """Noise weight should monotonically decrease."""
        prev_val = exponential_scheduler.compute()
        for i in range(1, exponential_scheduler.total_epochs + 1):
            exponential_scheduler.current_epoch = i
            curr_val = exponential_scheduler.compute()
            assert curr_val <= prev_val
            prev_val = curr_val


class TestStep:
    def test_initial_value(self, step_scheduler):
        """Noise weight should be initial value before first step epoch."""
        assert step_scheduler.compute() == pytest.approx(
            step_scheduler.initial_weight
        )

    def test_after_first_step(self, step_scheduler):
        """Noise weight should be multiplied by gamma after first step epoch."""
        step = 1
        step_scheduler.current_epoch = step_scheduler.step_epochs[step - 1]
        assert step_scheduler.compute() == pytest.approx(
            step_scheduler.initial_weight * step_scheduler.gamma**step
        )

    def test_after_second_step(self, step_scheduler):
        """Noise weight should be multiplied by gamma^2 after second step epoch."""
        step = 2
        step_scheduler.current_epoch = step_scheduler.step_epochs[step - 1]
        assert step_scheduler.compute() == pytest.approx(
            step_scheduler.initial_weight * step_scheduler.gamma**step
        )

    def test_after_all_steps(self, step_scheduler):
        """Noise weight should be multiplied by gamma^3 after all step epochs."""
        step = 3
        step_scheduler.current_epoch = step_scheduler.step_epochs[step - 1]
        assert step_scheduler.compute() == pytest.approx(
            step_scheduler.initial_weight * step_scheduler.gamma**step
        )

    def test_step_epochs_are_sorted(self, update_fn_tracker):
        """step_epochs should be sorted regardless of input order."""
        scheduler = Step(
            update_fn=update_fn_tracker,
            initial_weight=1.0,
            total_epochs=100,
            step_epochs=[75, 25, 50],
            gamma=0.5,
        )
        assert scheduler.step_epochs == [25, 50, 75]


class TestSchedulerInitialization:
    def test_initial_weight_set(self, update_fn_tracker):
        """Initial weight should be passed to update_fn during init."""
        _ = Linear(
            update_fn=update_fn_tracker,
            initial_weight=0.75,
            total_epochs=100,
        )
        update_fn_tracker.assert_called_with(0.75)

    def test_current_epoch_starts_at_zero(self, linear_scheduler):
        """current_epoch should start at 0."""
        assert linear_scheduler.current_epoch == 0

    def test_initial_values_stored(self, linear_scheduler):
        """Scheduler should store initial values."""
        assert linear_scheduler.initial_weight == 1.0
        assert linear_scheduler.total_epochs == 100


class TestSchedulerStep:
    def test_step_calls_update_fn(self, update_fn_tracker):
        """step() should call update_fn with computed weight."""
        scheduler = Step(
            update_fn=update_fn_tracker,
            initial_weight=1.0,
            total_epochs=100,
            step_epochs=[25, 50, 75],
            gamma=0.5,
        )
        update_fn_tracker.reset_mock()
        scheduler.step()
        update_fn_tracker.assert_called_once()

    def test_multiple_steps(self, update_fn_tracker):
        """Multiple step() calls should progressively call update_fn."""
        scheduler = Step(
            update_fn=update_fn_tracker,
            initial_weight=1.0,
            total_epochs=100,
            step_epochs=[2, 4],
            gamma=0.5,
        )
        update_fn_tracker.reset_mock()
        for _ in range(5):
            scheduler.step()
        assert update_fn_tracker.call_count == 5


class TestPulseWindowScheduler:
    def test_returns_initial_weight_outside_windows(self, update_fn_tracker):
        """Outside pulse windows, should return initial_weight."""
        scheduler = PulseWindowScheduler(
            update_fn=update_fn_tracker,
            initial_weight=0.0,
            total_epochs=100,
            pulse_windows=[
                (20, 30, CosineAnnealing, {"clamp_at_total_epochs": None}),
            ],
        )
        scheduler.current_epoch = 10
        assert scheduler.compute() == pytest.approx(0.0)

    def test_activates_scheduler_in_window(self, update_fn_tracker):
        """Inside pulse window, should use inner scheduler."""
        scheduler = PulseWindowScheduler(
            update_fn=update_fn_tracker,
            initial_weight=0.0,
            total_epochs=100,
            pulse_windows=[
                (
                    20,
                    30,
                    CosineAnnealing,
                    {
                        "initial_weight": 1.0,
                        "clamp_at_total_epochs": None,
                    },
                ),
            ],
        )
        scheduler.current_epoch = 20
        value = scheduler.compute()
        # Should be from CosineAnnealing at its epoch 0
        assert value == pytest.approx(1.0)

    def test_multiple_pulse_windows(self, update_fn_tracker):
        """Multiple windows should activate in sequence."""
        scheduler = PulseWindowScheduler(
            update_fn=update_fn_tracker,
            initial_weight=0.0,
            total_epochs=100,
            pulse_windows=[
                (
                    10,
                    20,
                    CosineAnnealing,
                    {"initial_weight": 1.0, "clamp_at_total_epochs": None},
                ),
                (40, 50, Linear, {"initial_weight": 1.0}),
            ],
        )
        # First window
        scheduler.current_epoch = 10
        val1 = scheduler.compute()

        # Between windows
        scheduler.current_epoch = 30
        scheduler.active_scheduler = None
        val_between = scheduler.compute()

        # Second window
        scheduler.current_epoch = 40
        scheduler.active_scheduler = None
        val2 = scheduler.compute()

        assert val1 != pytest.approx(0.0)  # Active scheduler
        assert val_between == pytest.approx(0.0)  # Outside windows
        assert val2 != pytest.approx(0.0)  # Different scheduler active
