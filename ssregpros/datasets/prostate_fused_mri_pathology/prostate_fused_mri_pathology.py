"""
Status of this is:
    - I had a look at some of the T2 AXIAL SM FOV scans and they
      look like absolute dogshit, like you can barely make out
      the prostate from the rest of it
    - I've decided to give this a skip for now

[WEEKS LATER]

TODO:   implement the histology filepaths discoverer and run with
        the ball guess
"""

from ... import RAW_DATA_ROOT
from ...core.correspondence import Correspondence, CorrespondenceDiscoverer
from ...core.mri_axis import MRIAxis

from pathlib import Path
from typing_extensions import override

import pandas as pd


def standardise_subject_name(name: str) -> str:
    """
    Standardises subject names into a consistent format.

    Parameters
    ----------
    name: str
        Subject name, non-standardised.

    Notes
    -----
    There are assumed typos in the subject names in the form of the
    wrong number of leading zeros in the subject's numerical ID, e.g.
          'aab010', 'aab026', 'aab011', 'aab027'
    are assumed to be
          'aab0010', 'aab0026', 'aab0011', 'aab0027'
    This function ensures the subject ID has exactly 2 leading zeros.

    Returns
    -------
    str
    """
    first_digit_index = next(
        (i for i, x in enumerate(name) if x.isdigit()), None
    )
    if first_digit_index is None:
        raise ValueError(f"missing subject ID numerical identifier: {name!r}")
    number = str(int(name[first_digit_index:]))
    number = number.zfill(len(number) + 2)
    return name[:first_digit_index] + number


class ProstateFusedMriPathology(CorrespondenceDiscoverer):
    """
    Implementation of the dataset interface for the 'Prostate Fused-MRI-Pathology'
    dataset from The Cancer Imaging Archive.
    """

    # NOTE: I'd gladly take "T2 AXIAL PELVIS" and "t2tseax" but there is some
    #       ambiguity around the correspondences and the T2W modality.
    #       I believe the correspondences refer to the "T2 AXIAL SM FOV" scans,
    #       but there's no explicit metadata confirming this to be the case.
    #       I'm convinced by the abundance of "T2 AXIAL SM FOV" scans.

    ACCEPTABLE_MODALITY = "T2 AXIAL SM FOV"

    def __init__(self, root_dir: Path | None = None):
        if root_dir is None:
            root_dir = RAW_DATA_ROOT / "Prostate Fused-MRI-Pathology"
        super().__init__("Prostate Fused-MRI-Pathology", root_dir)

    @override
    def discover_correspondences(self) -> list[Correspondence]:
        """
        One spreadsheet describes the layout of the MRI data.

        Another spreadsheet describes correspondences for a given patient in
        individual sheets and multiple patients in one final big sheet.
        """
        # Obtain the MRI filepaths from the data layout spreadsheet.
        mri_filepaths = self._obtain_mri_filepaths()

        # Obtain the histology filepaths from the data layout spreadsheet.
        hist_filepaths: dict[tuple[str, str], Path] = (
            self._obtain_histology_filepaths()
        )

        # Build the list of (MRI, Histology) pairs for each patient from the
        # correspondences spreadsheet, keeping only matches confirmed with
        # certainty i.e. no Maybe.
        patient_pairs = self._obtain_pairs_from_spreadsheet()

        # Build the correspondences.
        correspondences: list[Correspondence] = []
        for patient_id, corrs in patient_pairs.items():
            mri_filepath = mri_filepaths[patient_id]
            for slice_index, hist_letter in corrs:
                histology_filepath = hist_filepaths[(patient_id, hist_letter)]
                correspondences.append(
                    Correspondence(
                        dataset_id=self.dataset_id,
                        patient_id=patient_id,
                        mri_filepath=mri_filepath,
                        mri_slice_index=slice_index,
                        mri_slice_axis=MRIAxis.AXIAL,
                        histology_filepath=histology_filepath,
                    )
                )

        return correspondences

    def _obtain_mri_filepaths(self) -> dict[str, Path]:
        """
        Obtain the file paths from the MRI data layout spreadsheet.
        """
        # Load metadata.csv.
        manifest = next(
            (
                x
                for x in self.root_dir.iterdir()
                if x.is_dir() and x.name.startswith("manifest")
            ),
            None,
        )
        if manifest is None:
            raise FileNotFoundError("could not find manifest directory")
        metadata = pd.read_csv(manifest / "metadata.csv")
        # Identify acceptable modalities.
        filtered = metadata[
            metadata["Series Description"].isin([self.ACCEPTABLE_MODALITY])
        ]
        # Correct assumed typos in subject IDs.
        filtered.loc[:, "Subject ID"] = filtered["Subject ID"].apply(
            standardise_subject_name
        )

        # Patient "aab019" (presumptuously corrected to "aab0019") has
        # two "T2 AXIAL SM FOV" scans, both with 24 slices, whose binary
        # contents differ.
        # I don't know how to choose which is which, so I'm dropping it all.
        filtered = filtered[filtered["Subject ID"] != "aab0019"]

        # Build layout.
        layout: dict[str, Path] = {}
        for patient_id, file_location in filtered[
            ["Subject ID", "File Location"]
        ].itertuples(index=False, name=None):
            file_path = manifest / Path(file_location)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"could not locate T2W scans for patient {patient_id!r}!"
                )
            layout[patient_id] = file_path

        return layout

    def _obtain_histology_filepaths(self) -> dict[tuple[str, str], Path]:
        filepaths: dict[tuple[str, str], Path] = {}
        raise NotImplementedError
        return filepaths

    def _obtain_pairs_from_spreadsheet(
        self,
    ) -> dict[str, list[tuple[int, str]]]:
        """
        Build the list of (MRI, Histology) pairs for each patient from the
        correspondences spreadsheet, keeping only matches confirmed with
        certainty i.e. no 'Maybe'.

        Returns
        -------
        dict[str, list[tuple[int, str]]]
            For each patient ID, a list of correspondences between T2W MRI
            slice index and the name of the histology slide
        """
        sheets = pd.read_excel(
            self.root_dir / "histo_MR_Correspondence.xlsx", sheet_name=None
        )
        patient_pairs: dict[str, list[tuple[int, str]]] = {}
        patient_suffix = "_Correspondence"
        for sheet_name, sheet_df in sheets.items():
            if sheet_name.endswith(patient_suffix):
                # Individual patient's spreadsheet.
                patient_id = standardise_subject_name(
                    sheet_name.removesuffix(patient_suffix)
                )
                mri_slice_key = next(
                    filter(
                        lambda x: x.lower().startswith("t2slice"),
                        sheet_df.columns,
                    )
                )
                histology_key = next(
                    filter(
                        lambda x: x.lower().startswith("heslice"),
                        sheet_df.columns,
                    )
                )
                blank_lines = sheet_df.isna().all(axis=1)
                if blank_lines.any():
                    first_blank_line_idx = blank_lines.idxmax()
                else:
                    first_blank_line_idx = len(sheet_df)
                pairs = sheet_df.iloc[:first_blank_line_idx].loc[
                    sheet_df[mri_slice_key].notna(),
                    [mri_slice_key, histology_key],
                ]
                if len(pairs):
                    patient_pairs[patient_id] = list(
                        pairs.itertuples(index=False, name=None)
                    )
                continue
            # Last spreadsheet with multiple correspondences.
            # > pandas read the first patient name as the header: undo that
            #   and turn the sheet into a Series.
            s = pd.Series([sheet_df.columns[0], *sheet_df.iloc[:, 0].tolist()])
            # > Obtain the patient identifiers and pairs.
            blank_mask = s.isna()
            header_mask = ~blank_mask & blank_mask.shift(fill_value=True)
            patient_ids = s.where(header_mask).ffill()
            mask = ~header_mask & ~blank_mask
            # > Add pairs.
            for patient_id, co in zip(patient_ids[mask], s[mask]):
                if any(
                    x in co.lower().strip()
                    for x in ("don't include", "don't consider")
                ):
                    continue
                patient_id = standardise_subject_name(
                    patient_id.strip().split()[0]
                )
                histology, mri = co.strip().split("-")
                patient_pairs.setdefault(patient_id, []).append(
                    (int(mri), histology)
                )

        return patient_pairs
