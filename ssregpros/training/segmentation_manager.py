from .. import PROCESSED_DATA_ROOT

from ..core.correspondence import CorrespondenceDiscoverer
from ..models.segmentor import Segmentor

from monai.data.meta_tensor import MetaTensor

from pathlib import Path

import hashlib
import torch


class SegmentationManager:
    """
    Handles segmentation of unique MRI volumes and caching of results.
    """

    def __init__(
        self,
        segmentor: Segmentor,
        correspondence_discoverer: CorrespondenceDiscoverer,
        cache_dir: Path | None = None,
    ):
        self.segmentor = segmentor
        self.correspondence_discoverer = correspondence_discoverer
        if cache_dir is None:
            self.cache_dir = (
                PROCESSED_DATA_ROOT
                / correspondence_discoverer.dataset_id
                / "masks"
            )
        else:
            self.cache_dir = cache_dir / "masks"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get_segmentation_cache_path(
        self, patient_id: str, mri_filepath: Path
    ) -> Path:
        """Generate cache file path for a given MRI volume."""
        filepath_hash = hashlib.md5(
            str(mri_filepath.resolve()).encode("utf-8")
        ).hexdigest()
        cache_key = f"[{patient_id}]{filepath_hash}"
        return self.cache_dir / f"{cache_key}_mask.pt"

    def compute_masks(
        self, progress_bar: bool = True
    ) -> dict[Path, MetaTensor]:
        """
        Load cached masks or compute them for all unique MRI volumes.
        Returns a mapping from MRI filepath to segmentation mask.
        """
        # Find unique MRI filepaths
        correspondences = self.correspondence_discoverer.correspondences()
        unique_mri_paths = list(
            set(
                (corr.patient_id, corr.mri_filepath) for corr in correspondences
            )
        )
        # Identify MRIs requiring segmenting.
        masks_dict = {}
        to_segment: list[tuple[str, Path]] = []
        for patient_id, mri_path in unique_mri_paths:
            cache_path = self.get_segmentation_cache_path(patient_id, mri_path)
            if cache_path.exists():
                masks_dict[mri_path] = torch.load(
                    cache_path,
                    weights_only=False,
                    map_location=self.segmentor.device,
                )
            else:
                to_segment.append((patient_id, mri_path))
        # Perform segmentation.
        if to_segment:
            masks = self.segmentor.segment(
                *(path for _, path in to_segment), progress_bar=progress_bar
            )
            for (patient_id, mri_path), mask in zip(to_segment, masks):
                cache_path = self.get_segmentation_cache_path(
                    patient_id, mri_path
                )
                torch.save(mask, cache_path)
                masks_dict[mri_path] = mask
        return masks_dict
