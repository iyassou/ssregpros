from .. import PROCESSED_DATA_ROOT
from ..core.correspondence import Correspondence, CorrespondenceDiscoverer
from ..core.type_definitions import Percentage
from ..models.segmentor import Segmentor
from ..transforms.preprocessor import (
    Preprocessor,
    PREPROCESSOR_PIPELINE_KEYS,
)
from .segmentation_manager import SegmentationManager

from monai.data.dataset import PersistentDataset
from monai.data.meta_tensor import MetaTensor
from monai.transforms.compose import Compose

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from typing import Sequence

import hashlib
import numpy as np
import pickle
import torch


class MultiModalDataset(torch.utils.data.Dataset):
    """
    MRI and histology PyTorch dataset.

    Handles segmentation of MRIs and applying the preprocessing pipeline,
    complete with caching to avoid wasteful computation.
    """

    def __init__(
        self,
        correspondence_discoverer: CorrespondenceDiscoverer,
        segmentor: Segmentor,
        preprocessor: Preprocessor,
        cache_dir: Path | None = None,
        segmentation_progress_bar: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        # Batch segment unique masks.
        manager = SegmentationManager(
            segmentor=segmentor,
            correspondence_discoverer=correspondence_discoverer,
            cache_dir=cache_dir,
        )
        masks = manager.compute_masks(progress_bar=segmentation_progress_bar)
        # Build preprocessing pipeline inputs.
        # Some data points are invalid, and unfortunately this is where
        # I'm able to deal with it.
        data = []
        for corr in correspondence_discoverer.correspondences():
            inpt = preprocessor.build_input(corr, masks[corr.mri_filepath])
            if PREPROCESSOR_PIPELINE_KEYS.EARLY_EXIT not in inpt:
                data.append(inpt)
        self._len = len(data)
        # Build PersistentDataset for cached preprocessing.
        if cache_dir is None:
            cache_dir = (
                PROCESSED_DATA_ROOT / correspondence_discoverer.dataset_id
            )
        cache_dir /= "dataset"
        cache_dir.mkdir(exist_ok=True, parents=True)
        self.persistent_dataset = PersistentDataset(
            data=data,
            transform=preprocessor,
            cache_dir=cache_dir,
            hash_func=self._preprocessor_input_hash_func,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

    def _preprocessor_input_hash_func(self, data: dict) -> bytes:
        """
        Create a unique hash for preprocessing caching, using the
        correspondence and mask inputs.
        """
        # Retrieve arguments.
        corr: Correspondence = data[PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE]
        mask: MetaTensor = data[PREPROCESSOR_PIPELINE_KEYS.MRI_MASK]
        # Compute hash.
        corr_signature = corr.hash_string().encode("utf-8")
        mask_signature = mask.numpy().tobytes()
        md5_hash = hashlib.md5(corr_signature + mask_signature).hexdigest()
        return md5_hash.encode("utf-8")

    def __len__(self):
        return self._len

    def __getitem__(self, j: int) -> dict:
        data: dict = self.persistent_dataset[
            j
        ]  # pyright: ignore[reportAssignmentType]
        return {
            k: (v.to(self.device) if isinstance(v, MetaTensor) else v)
            for k, v in data.items()
        }

    def full_patient_ids(self) -> list[str]:
        """Return a list of each datapoint's full patient identifier.
        Useful for avoiding data leakage when splitting the dataset."""
        ids: list[str] = []
        for j in range(len(self)):
            data: dict = self.persistent_dataset[
                j
            ]  # pyright: ignore[reportAssignmentType]
            corr: Correspondence = data[
                PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE
            ]
            ids.append(corr.full_patient_id())
        return ids


class MultiModalDatasetView(torch.utils.data.Dataset):
    """
    A view over the base dataset that optionally applies an (un-cached)
    augmentation transform.
    """

    def __init__(
        self,
        dataset: MultiModalDataset,
        indices: Sequence[int],
        transform: Compose | None = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        data = self.dataset[self.indices[idx]]
        if self.transform is not None:
            data: dict = self.transform(
                data
            )  # pyright: ignore[reportAssignmentType]
        return data


def patient_stratified_split(
    dataset: MultiModalDataset,
    train: Percentage,
    val: Percentage,
    test: Percentage,
    seed: int | None = None,
    train_transform: Compose | None = None,
) -> tuple[MultiModalDatasetView, MultiModalDatasetView, MultiModalDatasetView]:
    """Splits a dataset into train, validation, and testing subsets with the
    (approximate) desired percentages while prioritising grouping by patient
    unique ID to avoid data leakage.

    Notes
    -----
    Worth examining the outputted split instead of purely relying on the
    desired percentages.

    Returns
    -------
    tuple[MultiModalDatasetView, MultiModalDatasetView, MultiModalDatasetView]
        Training, validation, and testing set views"""
    # Sanity check.
    if not np.isclose(total := (train + val + test), 1):
        raise ValueError(f"percentages add up to {total = } ≠ 1")
    N: int = len(dataset)
    # Create groups using patient unique identifiers.
    groups = np.array(dataset.full_patient_ids())
    # 1) Split off test set.
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=train + val,
        test_size=test,
        random_state=seed,
    )
    trainval_idx, test_idx = next(
        gss1.split(X=np.arange(N), y=None, groups=groups)
    )
    # 2) Split train and validation sets within the remaining indices.
    rel_train = train / (train + val)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=rel_train,
        test_size=1 - rel_train,
        random_state=seed,
    )
    train_idx, val_idx = next(
        gss2.split(X=trainval_idx, y=None, groups=groups[trainval_idx])
    )
    # Map indices from `trainval` subset back to original indices.
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]
    # Done!
    return (
        MultiModalDatasetView(dataset, train_idx, transform=train_transform),
        MultiModalDatasetView(dataset, val_idx),
        MultiModalDatasetView(dataset, test_idx),
    )
