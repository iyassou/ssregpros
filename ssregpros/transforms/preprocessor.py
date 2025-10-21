from ..core.correspondence import Correspondence
from ..core.type_definitions import Percentage, StrictlyPositiveFloat
from .mri import (
    MriMetadataKeys,
    MriPipelineKeys,
    CalculateNewSliceIndexd,
    MRIAndMaskSlicerd,
    SufficientProstatePixelsInMaskd,
    ResampleToIsotropicSpacingd,
    CenterCropMriOnMaskd,
)
from .histology import (
    HistologyPipelineKeys,
    DynamicOpenSlideReader,
    EstimateTissueForStainDeconvolutiond,
    DeconvolveHaematoxylinInRGBSpaced,
    BinaryMasksFromHaematoxylind,
    ResizeAxesToIsotropicMPPd,
)
from .shared import (
    SharedPipelineKeys,
    MetaTensorLoggingd,
    SkipIfKeysPresentd,
    StandardiseIntensityMaskedd,
)

from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import (
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.transforms.spatial.dictionary import Orientationd

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import ceil

import json
import torch


class PreprocessorPipelineKeys(StrEnum):
    CORRESPONDENCE = "correspondence"
    MRI_MASK = MriPipelineKeys.MASK_VOLUME
    EARLY_EXIT = SharedPipelineKeys.EARLY_EXIT


@dataclass(frozen=True)
class PreprocessorConfig:
    # Consistent MRI orientation.
    mri_orientation: str = "RAS"

    # The default combination of MRI slice size (128 by 128) and pixel
    # spacing (0.5mm, 0.5mm) result in a 6.4cm by 6.4cm window, sufficient
    # to encompass the prostate, which is typically the size of a walnut.
    mri_slice_height: int = 128
    mri_slice_width: int = 128
    pixel_spacing_millimetres: float = 0.5

    @property
    def pixel_spacing_micrometres(self) -> float:
        return self.pixel_spacing_millimetres * 1_000

    # If the prostate158 segmentor is unable to generate a 3D mask for which
    # the 2D slice-of-interest does not have at least N foreground pixels,
    # we exit the rest of the MRI preprocessing pipeline early.
    # While the underlying implementation uses a pixel count, we set that
    # threshold in square millimetres here, as that is more direct and
    # clinically meaningful.
    mri_mask_prostate_area_square_millimetres_threshold: float = 1600

    # Maximum dimension (width or height, aspect-ratio-dependent) of
    # the thumbnail generated to estimate the fraction of pixels that
    # correspond to tissue in the image.
    histology_tissue_fraction_estimation_thumbnail_max_size: int = 1024

    # For Macenko stain normalisation to make sense, there needs to be
    # a sufficient number of pixels in order to obtain a statistically
    # robust eigenvalue decomposition, hence a minimum pixel count
    # threshold is set to select only from levels that possess high
    # enough tissue pixel count.
    # A tissue pixel fraction is obtained on a smaller thumbnail using
    # Otsu's algorithm, and this fraction is used to estimate the tissue
    # pixel count for each level.
    # A default value of `50_000` seems sensible (~224-by-224 image).
    histology_macenko_pixel_count_threshold: int = 50_000

    # Traditional computer vision techniques are used to preprocess certain
    # elements in the histology pipeline.
    # Morphological closing is performed in two stages using an elliptical
    # structural element. Its kernel size is determined dynamically as a
    # percentage of the image limiting dimension.
    # Contour-based filling is used to obtain a solid binary mask for the
    # tissue. While the largest contour in terms of area is deemed to be the
    # overall tissue-delimiting mask, smaller contours contained within it
    # whose areas are within a certain ratio of the larger contour are used
    # to cut into the tissue mask, in an attempt to identify large gaps in the
    # tissue.
    histology_tissue_estimate_closing_se_kernel_size_percentage: Percentage = (
        0.005
    )
    histology_tissue_estimate_minimum_hole_area_ratio: Percentage = 0.0005
    histology_haematoxylin_mask_closing_se_kernel_size_percentage: (
        Percentage
    ) = 0.05
    histology_haematoxylin_mask_minimum_hole_area_ratio: (
        StrictlyPositiveFloat
    ) = 0.005

    def hash_string(self) -> str:
        return json.dumps(dict(sorted(asdict(self).items())))


class Preprocessor(MapTransform):
    """
    Takes in a `Correspondence` and its 3D MRI binary mask and applies MRI and
    histology preprocessing transforms.
    """

    def __init__(self, config: PreprocessorConfig):
        super().__init__(
            keys=(
                PreprocessorPipelineKeys.CORRESPONDENCE,
                MriPipelineKeys.MASK_VOLUME,
            ),
            allow_missing_keys=False,
        )
        self.config = config
        self.mri_pipeline = self._build_mri_pipeline()
        self.histology_pipeline = self._build_histology_pipeline()

    def _build_mri_pipeline(self) -> Compose:
        """
        Builds the MRI preprocessing pipeline. Steps:
            (1) load the MRI and mask
            (2) scale intensity between 1st and 99th percentiles to [0, 1]
            (3) reorient to a standard orientation
            (4) obtain the 2D MRI and mask slices
            (5) check foreground pixel count threshold for early exit
            (6) resample MRI and mask to isotropic pixel spacing
            (7) center-crop prostate on MRI using mask
            (8) z-score prostate using mask
        """
        early_exit_pixel_threshold = ceil(
            self.config.mri_mask_prostate_area_square_millimetres_threshold
            / self.config.pixel_spacing_millimetres**2
        )
        if_threshold_met = Compose(
            [
                ResampleToIsotropicSpacingd(
                    self.config.pixel_spacing_millimetres
                ),
                CenterCropMriOnMaskd(
                    height=self.config.mri_slice_height,
                    width=self.config.mri_slice_width,
                ),
                EnsureChannelFirstd(
                    keys=[
                        MriPipelineKeys.MRI_SLICE,
                        MriPipelineKeys.MASK_SLICE,
                    ]
                ),
                StandardiseIntensityMaskedd(
                    keys=[
                        (MriPipelineKeys.MRI_SLICE, MriPipelineKeys.MASK_SLICE)
                    ]
                ),
            ]
        )
        return Compose(
            [
                # Load into consistent shape.
                LoadImaged(keys=[MriPipelineKeys.MRI_VOLUME]),
                EnsureChannelFirstd(
                    keys=[
                        MriPipelineKeys.MRI_VOLUME,
                        MriPipelineKeys.MASK_VOLUME,
                    ]
                ),
                # Log data.
                MetaTensorLoggingd(  # log original shape, without channel dimension
                    keys=[MriPipelineKeys.MRI_VOLUME],
                    fn=lambda meta_tensor: meta_tensor.shape[1:],
                    key=MriMetadataKeys.INPUT_MRI_SHAPE,
                ),
                MetaTensorLoggingd(  # log original affine matrix
                    keys=[MriPipelineKeys.MRI_VOLUME],
                    fn=lambda meta_tensor: meta_tensor.affine,
                    key=MriMetadataKeys.INPUT_MRI_AFFINE_MATRIX,
                ),
                # Standardise MRI volume's intensity.
                ScaleIntensityRangePercentilesd(
                    keys=[MriPipelineKeys.MRI_VOLUME],
                    lower=1,
                    upper=99,
                    b_min=0,
                    b_max=1,
                    clip=True,
                    allow_missing_keys=False,
                ),
                # Load into consistent orientation.
                Orientationd(
                    keys=[
                        MriPipelineKeys.MRI_VOLUME,
                        MriPipelineKeys.MASK_VOLUME,
                    ],
                    axcodes=self.config.mri_orientation,
                ),
                # Obtain 2D slices.
                CalculateNewSliceIndexd(),
                MRIAndMaskSlicerd(keepdim=False),
                EnsureTyped(
                    keys=[MriPipelineKeys.MASK_SLICE], dtype=torch.float32
                ),
                # Check mask validity.
                SufficientProstatePixelsInMaskd(
                    pixel_threshold=early_exit_pixel_threshold
                ),
                # Preprocess.
                SkipIfKeysPresentd(
                    keys=[SharedPipelineKeys.EARLY_EXIT],
                    transform=if_threshold_met,
                ),
            ]
        )

    def _build_histology_pipeline(self) -> Compose:
        """
        Builds the histology preprocessing pipeline. Steps:
            (1) load the level whose average MPP is closest to our target
                value, while also containing sufficient pixels for the
                next stage
            (2) use Otsu's algorithm to obtain a rough binary mask delineating
                tissue
            (3) apply Macenko H&E stain deconvolution using the rough tissue
                mask to separate out the haematoxylin channel
            (4) obtain masks from haematoxylin channel: whole and punctured
            (5) scale intensity to [0, 1]
            (6) independently resize X and Y axes of all elements to target
                isotropic MPP
            (7) z-score prostate using mask
        """
        obtain_haematoxylin_and_masks = Compose(
            [
                # Deconvolve haematoxylin stain.
                DeconvolveHaematoxylinInRGBSpaced(
                    save_as=HistologyPipelineKeys.HAEMATOXYLIN
                ),
                # Obtain binary masks.
                BinaryMasksFromHaematoxylind(
                    save_as=(
                        HistologyPipelineKeys.HISTOLOGY_MASK,  # Punctured
                        HistologyPipelineKeys.HAEMATOXYLIN_MASK,  # Whole
                    ),
                    closing_se_kernel_size_percentage=self.config.histology_haematoxylin_mask_closing_se_kernel_size_percentage,
                    minimum_hole_area_ratio=self.config.histology_haematoxylin_mask_minimum_hole_area_ratio,
                ),
            ]
        )
        scale_match_mri_and_postprocess = Compose(
            [
                # Scale haematoxylin and RGB histology from [0, 255] to [0, 1].
                ScaleIntensityRanged(
                    keys=[
                        HistologyPipelineKeys.HAEMATOXYLIN,
                        HistologyPipelineKeys.HISTOLOGY,
                    ],
                    a_min=0,
                    a_max=255,
                    b_min=0,
                    b_max=1,
                ),
                # Match MRI's isotropic scaling.
                ResizeAxesToIsotropicMPPd(
                    keys=[
                        HistologyPipelineKeys.HAEMATOXYLIN,
                        HistologyPipelineKeys.HAEMATOXYLIN_MASK,
                        HistologyPipelineKeys.HISTOLOGY,
                        HistologyPipelineKeys.HISTOLOGY_MASK,
                    ],
                    modes=["area", "nearest-exact", "area", "nearest-exact"],
                    target_microns_per_pixel=self.config.pixel_spacing_micrometres,
                ),
                # Add channel dimensions.
                EnsureChannelFirstd(
                    keys=[
                        HistologyPipelineKeys.HAEMATOXYLIN,
                        HistologyPipelineKeys.HAEMATOXYLIN_MASK,
                        HistologyPipelineKeys.HISTOLOGY,
                        HistologyPipelineKeys.HISTOLOGY_MASK,
                    ]
                ),
                # Standardise intensity within mask.
                StandardiseIntensityMaskedd(
                    keys=[
                        (
                            HistologyPipelineKeys.HAEMATOXYLIN,
                            HistologyPipelineKeys.HAEMATOXYLIN_MASK,
                        )
                    ]
                ),
            ]
        )
        return Compose(
            [
                # Load appropriate image level.
                LoadImaged(
                    keys=[HistologyPipelineKeys.HISTOLOGY],
                    reader=DynamicOpenSlideReader(
                        pixel_count_threshold=self.config.histology_macenko_pixel_count_threshold,
                        target_microns_per_pixel=self.config.pixel_spacing_micrometres,
                        thumbnail_max_size=self.config.histology_tissue_fraction_estimation_thumbnail_max_size,
                    ),  # pyright: ignore[reportArgumentType]
                    dtype="uint8",
                ),
                # Prepare for stain deconvolution: estimate tissue mask.
                EstimateTissueForStainDeconvolutiond(
                    save_as=HistologyPipelineKeys.STAIN_DECONVOLUTION_TISSUE_MASK,
                    closing_se_kernel_size_percentage=self.config.histology_tissue_estimate_closing_se_kernel_size_percentage,
                    minimum_hole_area_ratio=self.config.histology_tissue_estimate_minimum_hole_area_ratio,
                ),
                SkipIfKeysPresentd(
                    keys=[SharedPipelineKeys.EARLY_EXIT],
                    transform=Compose(
                        [
                            obtain_haematoxylin_and_masks,
                            # Preprocessor further.
                            SkipIfKeysPresentd(
                                keys=[SharedPipelineKeys.EARLY_EXIT],
                                transform=scale_match_mri_and_postprocess,
                            ),
                        ]
                    ),
                ),
            ]
        )

    def build_mri_pipeline_input(
        self, corr: Correspondence, mask: MetaTensor
    ) -> dict:
        """
        Builds a given correspondence's input dictionary to the MRI
        preprocessing pipeline.
        """
        return {
            MriPipelineKeys.MRI_VOLUME: corr.mri_filepath,
            MriPipelineKeys.MASK_VOLUME: mask,
            MriPipelineKeys.MRI_SLICE_INDEX: corr.mri_slice_index,
            MriPipelineKeys.MRI_SLICE_AXIS: corr.mri_slice_axis,
        }

    def build_histology_pipeline_input(self, corr: Correspondence) -> dict:
        """
        Builds a given correspondence's input dictionary to the histology
        preprocessing pipeline.
        """
        return {
            HistologyPipelineKeys.HISTOLOGY: corr.histology_filepath,
        }

    def build_input(self, corr: Correspondence, mask: MetaTensor) -> dict:
        return {
            PreprocessorPipelineKeys.CORRESPONDENCE: corr,
            PreprocessorPipelineKeys.MRI_MASK: mask,
        }

    def __call__(self, data: dict) -> dict:
        # Retrieve relevant arguments.
        corr: Correspondence = data[PreprocessorPipelineKeys.CORRESPONDENCE]
        mask: MetaTensor = data[PreprocessorPipelineKeys.MRI_MASK]
        # Prepare output.
        output: dict = {PreprocessorPipelineKeys.CORRESPONDENCE: corr}
        # Build modality-specific pipeline inputs.
        mri_dict = self.build_mri_pipeline_input(corr, mask)
        histology_dict = self.build_histology_pipeline_input(corr)
        # Run pipelines.
        mri_output: dict = self.mri_pipeline(
            mri_dict
        )  # pyright: ignore[reportAssignmentType]
        if (
            reason := mri_output.get(ee := PreprocessorPipelineKeys.EARLY_EXIT)
        ) is not None:
            return output | {ee: reason}
        histology_output: dict = self.histology_pipeline(
            histology_dict
        )  # pyright: ignore[reportAssignmentType]
        if (
            reason := histology_output.get(
                ee := PreprocessorPipelineKeys.EARLY_EXIT
            )
        ) is not None:
            return output | {ee: reason}
        # Combine results.
        mri_keep = MriPipelineKeys.MRI_SLICE, MriPipelineKeys.MASK_SLICE
        hist_keep = (
            HistologyPipelineKeys.HAEMATOXYLIN,
            HistologyPipelineKeys.HAEMATOXYLIN_MASK,
            HistologyPipelineKeys.HISTOLOGY,
            HistologyPipelineKeys.HISTOLOGY_MASK,
        )
        output.update(
            {k: v for k, v in mri_output.items() if k in mri_keep}
            | {k: v for k, v in histology_output.items() if k in hist_keep}
        )
        # Return result.
        return output
