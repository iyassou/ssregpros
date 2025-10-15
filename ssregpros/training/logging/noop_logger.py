from pathlib import Path
from .base_logger import BaseLogger

from typing_extensions import override


class NoOpLogger(BaseLogger):
    @override
    def start(self):
        pass

    @override
    def run_id(self) -> str:
        return "noop"

    @override
    def watch(self, model, log, log_freq_batch, **kwargs):
        pass

    @override
    def log_epoch_metrics(self, metrics, epoch):
        pass

    @override
    def log_validation_table(self, *, table_name, rows, epoch):
        pass

    @override
    def log_dataset_preview(self, *, table_name, rows, max_rows: int = 512):
        pass

    @override
    def log_checkpoint(self, path: Path, *, name: str, aliases: list[str]):
        pass

    @override
    def finish(self):
        pass
