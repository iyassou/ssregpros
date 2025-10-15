from ..transforms.histology import HistologyPipelineKeys as HKeys
from ..transforms.mri import MriPipelineKeys as MKeys
from ..transforms.preprocessor import PREPROCESSOR_PIPELINE_KEYS as PKeys
from .dataset import (
    MultiModalDataset as Dataset,
    MultiModalDatasetView as DatasetView,
)

from monai.data.meta_tensor import MetaTensor
from torch.utils.data.dataloader import DataLoader, default_collate
from typing import NamedTuple


class MultiModalPersistentDataLoaderOutput(NamedTuple):
    correspondence_id: tuple[str]
    mri: MetaTensor
    mri_mask: MetaTensor
    histology: list[MetaTensor]
    histology_mask: list[MetaTensor]
    histology_rgb: list[MetaTensor] | None


class MultiModalPersistentDataLoaderCollator:
    """Picklable collator that returns a `MultimodalPersistentDataLoaderOutput`."""

    def __init__(self, visualisation: bool):
        self.visualisation = visualisation

    def __call__(
        self, batch: list[dict]
    ) -> MultiModalPersistentDataLoaderOutput:
        """Collates dictionary inputs coming from the preprocessing pipeline
        into a standardised named tuple.

        Notes
        -----
        The histology mask and RGB images are cast to torch.float32 here.
        VERY important."""
        # Collate the correspondences representations, MRI slices,
        # and mask slices in the usual way.
        corrs: tuple[str]
        mri: MetaTensor
        mri_mask: MetaTensor
        corrs, mri, mri_mask = map(
            default_collate,
            zip(
                *(
                    (
                        str(d[PKeys.CORRESPONDENCE]),
                        d[MKeys.MRI_SLICE],
                        d[MKeys.MRI_MASK_SLICE],
                    )
                    for d in batch
                )
            ),  # pyright: ignore[reportArgumentType]
        )
        # Collate variable-sized histology images and their masks into lists
        # of MetaTensors.
        histology: list[MetaTensor]
        histology_mask: list[MetaTensor]
        histology, histology_mask = zip(
            *[
                (d[HKeys.HISTOLOGY], d[HKeys.HISTOLOGY_MASK].float())
                for d in batch
            ]
        )  # pyright: ignore[reportAssignmentType]
        histology_rgb: list[MetaTensor] | None
        if self.visualisation:
            histology_rgb = [
                d[HKeys.HISTOLOGY_FOR_VISUALISATION].float() for d in batch
            ]
        else:
            histology_rgb = None
        # Done.
        return MultiModalPersistentDataLoaderOutput(
            correspondence_id=corrs,
            mri=mri,
            mri_mask=mri_mask,
            histology=histology,
            histology_mask=histology_mask,
            histology_rgb=histology_rgb,
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
