from .mri_axis import MRIAxis

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import override

import hashlib


@dataclass(frozen=True)
class Correspondence:
    dataset_id: str
    patient_id: str

    mri_filepath: Path
    mri_slice_index: int
    mri_slice_axis: MRIAxis
    histology_filepath: Path

    def full_patient_id(self) -> str:
        """Unique patient identifier built from dataset and patient IDs"""
        return f"{self.dataset_id} > {self.patient_id}"

    def __str__(self) -> str:
        """Compact string representation for debugging and logging purposes.

        Notes
        -----
        Histology identifier is constructed around the assumption that the
        histology file is named e.g. "<PATIENT ID>_<SOMETHING SPECIFIC>.ndpi",
        from which we can obtain the "<SOMETHING SPECIFIC>" bit.
        """
        histology: str = self.histology_filepath.stem.removeprefix(
            f"{self.patient_id}_"
        )
        return f"[{self.full_patient_id()}] ({self.mri_slice_axis}:index={self.mri_slice_index}, {histology})"

    def hash_string(self) -> str:
        string = "-".join(
            map(
                str,
                (
                    self.dataset_id,
                    self.patient_id,
                    self.mri_filepath.resolve(),
                    self.mri_slice_index,
                    self.mri_slice_axis,
                    self.histology_filepath.resolve(),
                ),
            )
        )
        return string

    @override
    def __hash__(self) -> int:
        md5_hash = hashlib.md5(self.hash_string().encode("utf-8")).hexdigest()
        return int(md5_hash, 16)


class CorrespondenceDiscoverer(ABC):
    """
    Abstract Base Class for a raw (MRI, Histology) dataset.
    Implementations should concretise the `discover_correspondences` method,
    which should return a list of `Correspondence`s.
    """

    def __init__(self, dataset_id: str, root_dir: Path):
        self.dataset_id = dataset_id
        self.root_dir = Path(root_dir)
        self._correspondences = self.discover_correspondences()

    @abstractmethod
    def discover_correspondences(self) -> list[Correspondence]:
        raise NotImplementedError

    def correspondences(self) -> list[Correspondence]:
        return self._correspondences

    def __len__(self) -> int:
        return len(self._correspondences)

    def __getitem__(self, idx: int):
        return self._correspondences[idx]

    def __iter__(self):
        return iter(self._correspondences)
