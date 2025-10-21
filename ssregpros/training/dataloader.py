from ..transforms.histology import HistologyPipelineKeys as HKeys
from ..transforms.mri import MriPipelineKeys as MKeys
from ..transforms.preprocessor import PreprocessorPipelineKeys as PKeys
from .dataset import (
    MultiModalDataset as Dataset,
    MultiModalDatasetView as DatasetView,
)

from torch.utils.data.dataloader import DataLoader, default_collate
from typing import NamedTuple

import torch


class MultiModalPersistentDataLoaderOutput(NamedTuple):
    correspondence_id: tuple[str]
    mri: torch.Tensor
    mri_mask: torch.Tensor
    haematoxylin: list[torch.Tensor]
    haematoxylin_mask: list[torch.Tensor]
    histology: list[torch.Tensor] | None
    histology_mask: list[torch.Tensor] | None


class MultiModalPersistentDataLoaderCollator:
    """Picklable collator that returns a `MultimodalPersistentDataLoaderOutput`."""

    def __init__(self, visualisation: bool):
        self.visualisation = visualisation

    def __call__(
        self, batch: list[dict]
    ) -> MultiModalPersistentDataLoaderOutput:
        """Collates dictionary inputs coming from the preprocessing pipeline
        into a standardised named tuple."""
        # Collate the correspondences representations, MRI slices,
        # and mask slices in the usual way.
        corrs: tuple[str]
        mri: torch.Tensor
        mri_mask: torch.Tensor
        corrs, mri, mri_mask = map(
            default_collate,
            zip(
                *(
                    (
                        str(d[PKeys.CORRESPONDENCE]),
                        d[MKeys.MRI_SLICE],
                        d[MKeys.MASK_SLICE],
                    )
                    for d in batch
                )
            ),  # pyright: ignore[reportArgumentType]
        )
        # Collate variable-sized haematoxylin images and their masks into lists
        # of tensors.
        haematoxylin: list[torch.Tensor]
        haematoxylin_mask: list[torch.Tensor]
        haematoxylin, haematoxylin_mask = zip(
            *[
                (d[HKeys.HAEMATOXYLIN], d[HKeys.HAEMATOXYLIN_MASK])
                for d in batch
            ]
        )  # pyright: ignore[reportAssignmentType]
        # Collate visualisation-only tensors.
        histology: list[torch.Tensor] | None
        histology_mask: list[torch.Tensor] | None
        if self.visualisation:
            histology, histology_mask = zip(
                *[(d[HKeys.HISTOLOGY], d[HKeys.HISTOLOGY_MASK]) for d in batch]
            )  # pyright: ignore[reportAssignmentType]
        else:
            histology = None
            histology_mask = None
        # Done.
        return MultiModalPersistentDataLoaderOutput(
            correspondence_id=corrs,
            mri=mri,
            mri_mask=mri_mask,
            haematoxylin=haematoxylin,
            haematoxylin_mask=haematoxylin_mask,
            histology=histology,
            histology_mask=histology_mask,
        )


class MultiModalPersistentDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset | DatasetView,
        visualisation: bool,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            collate_fn=MultiModalPersistentDataLoaderCollator(visualisation),
            **kwargs,
        )
        self.visualisation = visualisation
