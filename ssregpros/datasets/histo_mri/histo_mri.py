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
        trial_dir = mri_dir / "HISTO_MR"
        if not trial_dir.exists():
            raise FileNotFoundError("could not find 'HISTO_MR' directory!")
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
            t2w_dir = next(  # type: ignore[assignment]
                filter(
                    lambda x: x.name.startswith("T2W"),
                    (trial_dir / patient_id).iterdir(),
                ),
                None,
            )
            if t2w_dir is None:
                raise FileNotFoundError(
                    f"could not find T2W directory for {patient_id=}!"
                )
            mri_filepath = t2w_dir / "DICOM"
            if not mri_filepath.exists():
                raise FileNotFoundError(
                    f"could not find 'DICOM' directory for {patient_id=}!"
                )
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
                    mri_slice_index=-mri_slice_index,
                    mri_slice_axis=MRIAxis.AXIAL,
                    histology_filepath=histology_filepath,
                )
            )
        return corrs
