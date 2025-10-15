from ... import LATEST_COMMIT_SHA
from ...core.type_definitions import Percentage
from ...models.registration import RegistrationNetwork
from ..data_augmentation import DataAugmentation
from ..trainer import TrainingConfig
from ..utils import (
    JSONLike,
    MatrixLike,
    to_jsonable,
    to_uint8_img,
    to_uint8_mask,
)
from . import TableIdentifiers
from .base_logger import BaseLogger

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from typing_extensions import override

import wandb


@dataclass
class WandBConfig:
    # Experiment.
    batch_size: int
    training_config: TrainingConfig
    dataset_id: str
    dataset_split: tuple[Percentage, Percentage, Percentage]
    data_augmentation: DataAugmentation
    code_rev: str = LATEST_COMMIT_SHA

    # WandB initialisation kwargs.
    project: str = "ssregpros"
    entity: str | None = None
    group: str | None = None
    job_type: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    run_name: str | None = None
    notes: str | None = None
    dir: Path | None = None

    def wandb_init_kwargs(self) -> dict[str, JSONLike]:
        kwargs: dict[str, JSONLike] = {
            "project": self.project,
            "mode": self.mode,
        }
        if self.entity:
            kwargs["entity"] = self.entity
        if self.group:
            kwargs["group"] = self.group
        if self.job_type:
            kwargs["job_type"] = self.job_type
        if self.run_name:
            kwargs["run_name"] = self.run_name
        if self.notes:
            kwargs["notes"] = self.notes
        if self.dir:
            kwargs["dir"] = to_jsonable(self.dir)

        aug: dict = to_jsonable(
            asdict(self.data_augmentation)
        )  # pyright: ignore[reportAssignmentType]
        device = aug.pop("device")
        kwargs["config"] = {
            "device": device,
            "batch_size": self.batch_size,
            "dataset_id": self.dataset_id,
            "dataset_split": to_jsonable(self.dataset_split),
            "code_rev": self.code_rev,
            "training_config": to_jsonable(asdict(self.training_config)),
            "data_augmentation": aug,
        }

        return kwargs


class WandBLogger(BaseLogger):
    def __init__(self, gradient_accumulation_steps: int, **init_kwargs):
        self.wandb = wandb
        self.run = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.init_kwargs = init_kwargs
        self._table_cache: dict[str, wandb.Table] = {}

    def _wandb_image(
        self,
        base_img: MatrixLike,
        caption: str,
        mask: dict[str, Any] | None = None,
    ):
        """
        mask = {
            "mask_data": (H, W) image,
            "class_labels": dict[int, str],
        }
        """
        img = to_uint8_img(base_img)
        if mask is not None:
            masks = {
                "prostate-binary-mask": {
                    k: (to_uint8_mask(v) if isinstance(v, MatrixLike) else v)
                    for k, v in mask.items()
                }
            }
        else:
            masks = None
        return self.wandb.Image(img, caption=caption, masks=masks)

    @override
    def start(self):
        self.run = wandb.init(**self.init_kwargs)
        # Make the optimiser step the x-axis for our metrics namespaces.
        self.run.define_metric("step")
        self.run.define_metric("train/*", step_metric="step")
        self.run.define_metric("val/*", step_metric="step")
        # Define automatic UI summaries.
        self.run.define_metric("val/loss", summary="min")
        self.run.define_metric("val/dice", summary="max")

    @override
    def run_id(self) -> str:
        if not self.run:
            raise ValueError("run hasn't commenced!")
        return f"{self.run.name}-{self.run.id}"

    @override
    def watch(
        self,
        model: RegistrationNetwork,
        log: str,
        log_freq_batch: int,
        **kwargs,
    ):
        if self.run:
            kwargs["log"] = log
            kwargs["log_freq"] = log_freq_batch
            self.wandb.watch(model, **kwargs)

    @override
    def log_epoch_metrics(self, metrics: dict[str, float | str], epoch: int):
        """Log once per epoch."""
        if not self.run:
            return
        payload = metrics | {"step": epoch * self.gradient_accumulation_steps}
        self.run.log(data=payload, commit=True)

    @override
    def log_validation_table(
        self, *, table_name: str, rows: Iterable[dict[str, Any]], epoch: int
    ):
        """`rows` is an iterable of dictionaries like:

        {
            "id": str,
            "mri": torch.Tensor,
            "mri_mask": torch.Tensor,
            "hist": torch.Tensor,
            "hist_mask": torch.Tensor,
            "checkerboard": torch.Tensor,
            "canny_band": torch.Tensor,
            "canny_mask": torch.Tensor,
        }

        """
        if not self.run:
            return
        if (table := self._table_cache.get(table_name)) is None:
            table = self.wandb.Table(
                columns=[
                    "epoch",
                    TableIdentifiers.ID.value,
                    TableIdentifiers.MRI.value,
                    TableIdentifiers.HISTOLOGY.value,
                    TableIdentifiers.CHECKERBOARD.value,
                    TableIdentifiers.CANNY_BAND.value,
                    TableIdentifiers.CANNY_MASK.value,
                ],
                log_mode="MUTABLE",
            )
            self._table_cache[table_name] = table
        # Build a row per sample.
        for row in rows:
            # Get data.
            id_: str = row[TableIdentifiers.ID]
            mri: MatrixLike = row[TableIdentifiers.MRI]
            mri_mask: MatrixLike = row[TableIdentifiers.MRI_MASK]
            hist: MatrixLike = row[TableIdentifiers.HISTOLOGY]
            hist_mask: MatrixLike = row[TableIdentifiers.HISTOLOGY_MASK]
            checkerboard: MatrixLike = row[TableIdentifiers.CHECKERBOARD]
            canny_band: MatrixLike = row[TableIdentifiers.CANNY_BAND]
            canny_mask: MatrixLike = row[TableIdentifiers.CANNY_MASK]
            # Build W&B Images.
            wb_mri = self._wandb_image(
                mri,
                caption=id_,
                mask={
                    "mask_data": mri_mask,
                    "class_labels": {0: "background", 1: "prostate"},
                },
            )
            wb_hist = self._wandb_image(
                hist,
                caption=id_,
                mask={
                    "mask_data": hist_mask,
                    "class_labels": {0: "background", 1: "prostate"},
                },
            )
            wb_checkerboard = self._wandb_image(checkerboard, caption=id_)
            wb_canny_band = self._wandb_image(canny_band, caption=id_)
            wb_canny_mask = self._wandb_image(canny_mask, caption=id_)
            # Add row to table.
            table.add_data(
                epoch,
                id_,
                wb_mri,
                wb_hist,
                wb_checkerboard,
                wb_canny_band,
                wb_canny_mask,
            )
        self.run.log(
            {table_name: table, "epoch": epoch},
            commit=False,  # commit alongside metrics
        )

    @override
    def log_dataset_preview(
        self,
        *,
        table_name: str,
        rows: Iterable[dict[str, Any]],
        max_rows: int = 512,
    ):
        if not self.run:
            return
        table = self.wandb.Table(
            columns=[
                TableIdentifiers.ID.value,
                TableIdentifiers.MRI.value,
                TableIdentifiers.MRI_MASK.value,
                TableIdentifiers.HISTOLOGY.value,
                TableIdentifiers.HISTOLOGY_MASK.value,
            ]
        )
        for i, row in enumerate(rows):
            if i >= max_rows:
                break
            # Get data.
            id_: str = row[TableIdentifiers.ID]
            mri: MatrixLike = row[TableIdentifiers.MRI]
            mri_mask: MatrixLike = row[TableIdentifiers.MRI_MASK]
            hist: MatrixLike = row[TableIdentifiers.HISTOLOGY]
            hist_mask: MatrixLike = row[TableIdentifiers.HISTOLOGY_MASK]
            # Build W&B Images.
            wb_mri = self._wandb_image(mri, caption=id_)
            wb_mri_mask = self._wandb_image(mri_mask, caption=id_)
            wb_hist = self._wandb_image(hist, caption=id_)
            wb_hist_mask = self._wandb_image(hist_mask, caption=id_)
            # Add row to table.
            table.add_data(
                id_,
                wb_mri,
                wb_mri_mask,
                wb_hist,
                wb_hist_mask,
            )
        self.run.log({table_name: table}, commit=False)

    @override
    def log_checkpoint(self, path: Path, *, name: str, aliases: list[str]):
        if not self.run:
            return
        art = self.wandb.Artifact(name=name, type="model")
        art.add_file(local_path=str(path.resolve()), name=path.name)
        self.run.log_artifact(art, aliases=aliases)

    @override
    def finish(self):
        if self.run:
            self.run.finish()
            self.run = None
