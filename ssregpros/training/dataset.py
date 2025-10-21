from .. import PROCESSED_DATA_ROOT
from ..core.correspondence import Correspondence, CorrespondenceDiscoverer
from ..core.type_definitions import Percentage
from ..models.segmentor import Segmentor
from ..transforms.preprocessor import (
    Preprocessor,
    PreprocessorPipelineKeys,
)
from .segmentation_manager import SegmentationManager
from .utils import pformat_transform

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
        # Save preprocessor signatures.
        self.preprocessor_signature = (
            pformat_transform(preprocessor._build_mri_pipeline())
            + pformat_transform(preprocessor._build_histology_pipeline())
        ).encode("utf-8")
        self._pp_cfg_sig = preprocessor.config.hash_string().encode("utf-8")
        # Batch segment unique masks.
        manager = SegmentationManager(
            segmentor=segmentor,
            correspondence_discoverer=correspondence_discoverer,
            cache_dir=cache_dir,
        )
        masks = manager.compute_masks(progress_bar=segmentation_progress_bar)
        # Build preprocessing pipeline inputs.
        data = [
            preprocessor.build_input(corr, masks[corr.mri_filepath])
            for corr in correspondence_discoverer.correspondences()
        ]
        # Build PersistentDataset for cached preprocessing.
        if cache_dir is None:
            cache_dir = (
                PROCESSED_DATA_ROOT / correspondence_discoverer.dataset_id
            )
        cache_dir /= "dataset"
        verbose = not cache_dir.exists()
        self.persistent_dataset = PersistentDataset(
            data=data,
            transform=preprocessor,
            cache_dir=cache_dir,
            hash_func=self._preprocessor_input_hasher,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        # Some data points are invalid, unfortunately this is where I find out.
        self.valid_data = []
        out: dict
        for (
            out
        ) in self.persistent_dataset:  # pyright: ignore[reportAssignmentType]
            if (reason := out.get(PreprocessorPipelineKeys.EARLY_EXIT)) is None:
                self.valid_data.append(
                    {
                        k: (
                            v.to(self.device)
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in out.items()
                    }
                )
            elif verbose:
                print(
                    f"{out[PreprocessorPipelineKeys.CORRESPONDENCE]} | {reason}"
                )

    def _preprocessor_input_hasher(self, data: dict) -> bytes:
        """
        Create a unique hash for preprocessing caching.
        """
        # Retrieve arguments.
        corr: Correspondence = data[PreprocessorPipelineKeys.CORRESPONDENCE]
        mask: MetaTensor = data[PreprocessorPipelineKeys.MRI_MASK]
        # Compute unique hash.
        # NOTE: .digest() exists, but PersistentDataset UTF-8 -decodes this
        #       hash to determine the output file name.
        corr_signature = corr.hash_string().encode("utf-8")
        mask_signature = mask.numpy().tobytes()
        md5_hash = hashlib.md5(
            corr_signature
            + mask_signature
            + self.preprocessor_signature
            + self._pp_cfg_sig
        ).hexdigest()
        return md5_hash.encode("utf-8")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, j: int) -> dict:
        return self.valid_data[j]

    def full_patient_ids(self) -> list[str]:
        """Return a list of each datapoint's full patient identifier.
        Useful for avoiding data leakage when splitting the dataset."""
        ids: list[str] = []
        data: dict
        for data in self.valid_data:
            corr: Correspondence = data[PreprocessorPipelineKeys.CORRESPONDENCE]
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
