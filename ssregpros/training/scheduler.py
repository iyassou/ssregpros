from ..core.type_definitions import PositiveFloat, StrictlyPositiveInteger

from abc import ABC, abstractmethod
from typing import Callable
from typing_extensions import override

import math


class Scheduler(ABC):
    """Abstract Base Class for a scheduler."""

    def __init__(
        self,
        update_fn: Callable,
        initial_weight: PositiveFloat,
        total_epochs: StrictlyPositiveInteger,
    ):
        self.update_fn = update_fn
        self.update_fn(initial_weight)
        self.initial_weight = initial_weight
        self.total_epochs = total_epochs
        self.current_epoch = 0

    @abstractmethod
    def compute(self) -> PositiveFloat:
        """Returns the current epoch's noise weight."""
        raise NotImplementedError

    def step(self):
        noise_weight = self.compute()
        self.update_fn(noise_weight)
        self.current_epoch += 1


class CosineAnnealing(Scheduler):
    def __init__(
        self,
        *args,
        clamp_at_total_epochs: PositiveFloat | None = None,
        warmup_epochs: StrictlyPositiveInteger = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clamp_at_total_epochs = clamp_at_total_epochs
        self.warmup_epochs = warmup_epochs

    @override
    def compute(self) -> PositiveFloat:
        if self.current_epoch < self.warmup_epochs:
            return self.initial_weight
        if (
            self.clamp_at_total_epochs is not None
            and self.current_epoch >= self.total_epochs
        ):
            return self.clamp_at_total_epochs
        adjusted_epoch = self.current_epoch - self.warmup_epochs
        return (
            self.initial_weight
            * 0.5
            * (1 + math.cos(math.pi * adjusted_epoch / self.total_epochs))
        )


class Linear(Scheduler):
    @override
    def compute(self) -> PositiveFloat:
        return self.initial_weight * max(
            0, 1 - self.current_epoch / self.total_epochs
        )


class Exponential(Scheduler):
    def __init__(self, *args, decay_rate=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_rate = decay_rate

    @override
    def compute(self) -> PositiveFloat:
        return self.initial_weight * (
            self.decay_rate ** (self.current_epoch / self.total_epochs)
        )


class Step(Scheduler):
    def __init__(self, *args, step_epochs: list, gamma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_epochs = sorted(step_epochs)
        self.gamma = gamma

    @override
    def compute(self) -> PositiveFloat:
        num_steps = sum(self.current_epoch >= e for e in self.step_epochs)
        return self.initial_weight * (self.gamma**num_steps)


class StepValue(Scheduler):
    def __init__(self, *args, value_at_epochs: dict[int, float], **kwargs):
        super().__init__(*args, **kwargs)
        self.value_at_epochs = dict(sorted(value_at_epochs.items()))

    @override
    def compute(self) -> PositiveFloat:
        for epoch_threshold in reversed(self.value_at_epochs.keys()):
            if self.current_epoch >= epoch_threshold:
                return self.value_at_epochs[epoch_threshold]
        return self.initial_weight


class PulseWindowScheduler(Scheduler):
    def __init__(
        self,
        *args,
        pulse_windows: (
            list[tuple[int, int, type[Scheduler], dict]] | None
        ) = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if pulse_windows is None:
            pulse_windows = []
        self.pulse_windows = pulse_windows
        self.active_scheduler: Scheduler | None = None

    @override
    def compute(self) -> PositiveFloat:
        for start, end, scheduler_type, kwargs in self.pulse_windows:
            if start <= self.current_epoch <= end:
                if self.active_scheduler is None:
                    self.active_scheduler = scheduler_type(
                        update_fn=self.update_fn,
                        total_epochs=end - start,
                        **kwargs,
                    )
                weight = self.active_scheduler.compute()
                self.active_scheduler.current_epoch += 1
                return weight
        self.active_scheduler = None
        return self.initial_weight
