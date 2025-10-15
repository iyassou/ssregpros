from ... import RAW_DATA_ROOT
from ...core.correspondence import Correspondence, CorrespondenceDiscoverer
from ...core.mri_axis import MRIAxis

from typing_extensions import override

import random


class FakeHistoMri(CorrespondenceDiscoverer):
    """
    Mock implementation of a Histo-MRI clinical trial dataset subset's loader.
    Generates random correspondences.
    """

    def __init__(self):
        super().__init__("Fake Histo-MRI", RAW_DATA_ROOT / "Fake Histo-MRI")

    @override
    def discover_correspondences(self) -> list[Correspondence]:
        # Identify patients.
        patient_dirs = list(
            filter(
                lambda x: x.is_dir() and x.name.startswith("HMU"),
                self.root_dir.iterdir(),
            )
        )
        # Create random correspondences for each histology slide.
        mri_slice_axis = MRIAxis.AXIAL
        mri_slice_index_range = list(range(8, 17))
        correspondences = []
        random.seed(0xDEADBEEF)
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            mri_filepath = patient_dir / f"{patient_id}_T2W.nii.gz"
            assert mri_filepath.exists()
            histology_filepaths = filter(
                lambda x: x.suffix == ".ndpi", patient_dir.iterdir()
            )
            for histology_filepath in histology_filepaths:
                mri_slice_index = random.choice(mri_slice_index_range)
                correspondences.append(
                    Correspondence(
                        dataset_id=self.dataset_id,
                        patient_id=patient_id,
                        mri_filepath=mri_filepath,
                        mri_slice_index=mri_slice_index,
                        mri_slice_axis=mri_slice_axis,
                        histology_filepath=histology_filepath,
                    )
                )
        return correspondences


if __name__ == "__main__":
    A = FakeHistoMri().correspondences()
    B = FakeHistoMri().correspondences()
    print(len(A), "correspondences discovered.")
    if A == B:
        print("Random correspondences are deterministic.")
    else:
        print("ruh roh [random correspondences not deterministic!]")
