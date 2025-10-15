from ...models.registration import RegistrationNetwork


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class BaseLogger(ABC):
    @abstractmethod
    def start(self): ...

    @abstractmethod
    def run_id(self) -> str: ...

    @abstractmethod
    def watch(
        self,
        model: RegistrationNetwork,
        log: str,
        log_freq_batch: int,
        **kwargs,
    ): ...

    @abstractmethod
    def log_epoch_metrics(self, metrics: dict[str, float | str], epoch: int):
        """Log only epoch-aggregated metrics (train/* val/*)."""

    @abstractmethod
    def log_validation_table(
        self, *, table_name: str, rows: Iterable[dict[str, Any]], epoch: int
    ):
        """Log a W&B Table (or equivalent) with per-sample visuals/stats for
        the whole validation set."""

    @abstractmethod
    def log_dataset_preview(
        self,
        *,
        table_name: str,
        rows: Iterable[dict[str, Any]],
        max_rows: int = 512,
    ):
        """One-time dataset preview table (images + masks)"""

    @abstractmethod
    def log_checkpoint(self, path: Path, *, name: str, aliases: list[str]): ...

    @abstractmethod
    def finish(self): ...
