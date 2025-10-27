from enum import StrEnum


class EarlyStoppingMode(StrEnum):
    MIN = "min"
    MAX = "max"


class EarlyStopping:
    """Implementation of value-dependent early stopping."""

    def __init__(
        self,
        mode: EarlyStoppingMode,
        patience: int,
        min_delta_rel: float,
        min_delta_abs: float,
    ):
        self.mode = mode
        self.patience = patience
        self.min_delta_rel = min_delta_rel
        self.min_delta_abs = min_delta_abs
        self.best: float | None = None
        self.num_bad = 0
        if mode not in EarlyStoppingMode.__members__.values():
            raise ValueError(
                f"unrecognised {mode =!r}, must be one of {EarlyStoppingMode._member_names_}"
            )

    def delta(self) -> float:
        if self.best is None:
            raise ValueError("cannot compute delta before first step!")
        scale = abs(self.best)
        return max(self.min_delta_abs, self.min_delta_rel * scale)

    def is_better(self, cur: float) -> bool:
        if self.best is None:
            raise ValueError("cannot compare before first step!")
        if self.mode == EarlyStoppingMode.MIN:
            return cur < self.best - self.delta()
        # mode == EarlyStoppingMode.MAX
        return cur > self.best + self.delta()

    def reset(self):
        self.best = None
        self.num_bad = 0

    def step(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False
        if self.is_better(current):
            self.best = current
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad > self.patience
