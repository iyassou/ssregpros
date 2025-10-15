from ... import RAW_DATA_ROOT
from ...core.mri_axis import MRIAxis
from ...core.correspondence import Correspondence, CorrespondenceDiscoverer
from . import HISTO_MRI_BASE_CORRESPONDENCES

from pathlib import Path
from typing_extensions import override


class HistoMri(CorrespondenceDiscoverer):
    """
    Implementation of the Histo-MRI clinical trial's correspondence discoverer.

    Assumes the root directory contains folders (1) "MRI" and (2) "Histology"
    and works from there, and that all T2W slices are axial.
    """

    def __init__(self, root_dir: Path | None = None):
        if root_dir is None:
            root_dir = RAW_DATA_ROOT / "Histo-MRI"
        super().__init__("Histo-MRI", root_dir)

    @override
    def discover_correspondences(self) -> list[Correspondence]:
        # Look for relevant directories.
        # > DICOM
        mri_dir = self.root_dir / "MRI"
        if not mri_dir.exists():
            raise FileNotFoundError("could not find 'MRI' directory!")
        dicom_dir = mri_dir / "DICOM"
        if not dicom_dir.exists():
            raise FileNotFoundError("could not find 'DICOM' directory!")
        # > NDPI
        hist_dir = self.root_dir / "Histology"
        if not hist_dir.exists():
            raise FileNotFoundError("could not find 'Histology' directory!")
        ndpi_dir = hist_dir / "NDPI"
        if not ndpi_dir.exists():
            raise FileNotFoundError("could not find 'NDPI' directory!")

        # Build full correspondences using Notion page exports.
        corrs = []
        for (
            patient_id,
            histology_id,
            mri_slice_index,
        ) in HISTO_MRI_BASE_CORRESPONDENCES:
            # > MRI filepath.
            mri_dir = dicom_dir / patient_id / "T2W"
            if not mri_dir.exists():
                raise FileNotFoundError(
                    f"could not find T2W directory for {patient_id=}!"
                )
            # NOTE: should be the only DICOM file in the directory
            dicoms = list(
                filter(lambda f: f.suffix == ".dcm", mri_dir.iterdir())
            )
            if (num_dicoms := len(dicoms)) != 1:
                raise ValueError(
                    f"expected only 1 DICOM file in patient directory, found {num_dicoms}!"
                )
            mri_filepath = dicoms[0]
            # > Histology filepath.
            histology_dir = ndpi_dir / patient_id
            if not histology_dir.exists():
                raise FileNotFoundError(
                    f"could not find NDPI directory for {patient_id=}!"
                )
            histology_filepath = (
                histology_dir / f"{patient_id}_{histology_id}.ndpi"
            )
            if not histology_filepath.exists():
                raise FileNotFoundError(
                    f"could not find NDPI file for {patient_id=}!"
                )
            # > Make correspondence.
            corrs.append(
                Correspondence(
                    dataset_id=self.dataset_id,
                    patient_id=patient_id,
                    mri_filepath=mri_filepath,
                    mri_slice_index=mri_slice_index,
                    mri_slice_axis=MRIAxis.AXIAL,
                    histology_filepath=histology_filepath,
                )
            )
        return corrs
