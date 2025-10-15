from ... import RAW_DATA_ROOT
from ...core.correspondence import Correspondence, CorrespondenceDiscoverer
from ...core.mri_axis import MRIAxis

from typing_extensions import override


class SampleHistoMri(CorrespondenceDiscoverer):
    """
    Implementation of a subset of the Histo-MRI clinical trial dataset's
    loader. Returns few, but real correspondences.
    """

    def __init__(self):
        super().__init__("Sample Histo-MRI", RAW_DATA_ROOT / "Sample Histo-MRI")

    @override
    def discover_correspondences(self) -> list[Correspondence]:
        # Actual correspondences.
        dID = self.dataset_id
        root_dir = self.root_dir
        axis = MRIAxis.AXIAL
        return [
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-21,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A4.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-19,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A5.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-18,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A6.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-16,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A7.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-14,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A8.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-12,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A9.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_411_KO",
                mri_filepath=root_dir / "HMU_411_KO" / "HMU_411_KO_T2W.nii.gz",
                mri_slice_index=-10,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_411_KO"
                / "HMU_411_KO_A10.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_415_RW",
                mri_filepath=root_dir / "HMU_415_RW" / "HMU_415_RW_T2W.nii.gz",
                mri_slice_index=-21,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_415_RW"
                / "HMU_415_RW_A4.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_415_RW",
                mri_filepath=root_dir / "HMU_415_RW" / "HMU_415_RW_T2W.nii.gz",
                mri_slice_index=-19,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_415_RW"
                / "HMU_415_RW_A5.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_415_RW",
                mri_filepath=root_dir / "HMU_415_RW" / "HMU_415_RW_T2W.nii.gz",
                mri_slice_index=-17,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_415_RW"
                / "HMU_415_RW_A6.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_415_RW",
                mri_filepath=root_dir / "HMU_415_RW" / "HMU_415_RW_T2W.nii.gz",
                mri_slice_index=-15,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_415_RW"
                / "HMU_415_RW_A7.ndpi",
            ),
            Correspondence(
                dataset_id=dID,
                patient_id="HMU_415_RW",
                mri_filepath=root_dir / "HMU_415_RW" / "HMU_415_RW_T2W.nii.gz",
                mri_slice_index=-14,
                mri_slice_axis=axis,
                histology_filepath=root_dir
                / "HMU_415_RW"
                / "HMU_415_RW_A8.ndpi",
            ),
        ]


if __name__ == "__main__":
    print(
        len(SampleHistoMri().correspondences()), "correspondences discovered."
    )
