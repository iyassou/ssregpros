from ..core.correspondence import Correspondence
from ..core.type_definitions import Percentage, StrictlyPositiveFloat
from .mri import (
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
    BinaryMaskForStainDeconvolutiond,
    ObtainHaematoxylinUsingMacenkoHEStainDeconvolutiond,
    BinaryMaskFromHaematoxylind,
    ResizeAxesToIsotropicMPPd,
)
from .shared import (
    SharedPipelineKeys,
    MetaTensorLoggingd,
    SkipIfKeysPresentd,
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
    CopyItemsd,
    EnsureChannelFirstd,
)
from monai.transforms.spatial.dictionary import Orientationd

from dataclasses import dataclass
from enum import StrEnum
from math import ceil

import torch


class PREPROCESSOR_PIPELINE_KEYS(StrEnum):
    CORRESPONDENCE = "correspondence"
    MRI_MASK = MriPipelineKeys.MASK_VOLUME
    EARLY_EXIT = SharedPipelineKeys.EARLY_EXIT


@dataclass
class PreprocessorConfig:
    # Consistent MRI orientation.
    mri_orientation: str = "RAS"

    # The default combination of MRI input patch size (128) and pixel
    # spacing (0.5mm, 0.5mm) result in a 6.4cm by 6.4cm window, sufficient
    # to encompass the prostate, which is typically the size of a walnut.
    mri_patch_size: int = 128
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
    mri_mask_prostate_area_square_millimetres_threshold: float = 200

    # Maximum dimension (width or height, aspect-ratio-dependent) of
    # the thumbnail generated to estimate the fraction of pixels that
    # correspond to tissue in the image.
    tissue_fraction_estimation_thumbnail_max_size: int = 1024

    # For Macenko stain normalisation to make sense, there needs to be
    # a sufficient number of pixels in order to obtain a statistically
    # robust eigenvalue decomposition, hence a minimum pixel count
    # threshold is set to select only from levels that possess high
    # enough tissue pixel count.
    # A tissue pixel fraction is obtained on a smaller thumbnail using
    # Otsu's algorithm, and this fraction is used to estimate the tissue
    # pixel count for each level.
    # A default value of `50_000` seems sensible (~224-by-224 image).
    macenko_pixel_count_threshold: int = 50_000

    # Traditional computer vision techniques are used to preprocess certain
    # elements of the histology preprocessing pipeline.
    # Morphological closing is performed in two stages using an elliptical
    # structural element. Its kernel size is determined dynamically as a
    # percentage of the image limiting dimension.
    # Contour-based filling is used to obtain a solid binary mask for the
    # tissue. While the largest contour in terms of area is deemed to be the
    # overall delimiting tissue mask, smaller contours contained within that
    # whose areas are within a certain percentage of the larger contour are
    # used to cut into the overall mask, in an attempt to identify large gaps
    # in the tissue.
    # Gaussian blurring is performed to soften the binary tissue mask used
    # to select pixels for Macenko stain estimation. Its kernel size is also
    # dynamically determined in a similar manner.
    histology_closing_se_kernel_size_percentage: Percentage = 0.05
    histology_minimum_hole_area_ratio: StrictlyPositiveFloat = 0.005
    histology_gaussian_blur_kernel_size_percentage: Percentage = 0.03
    histology_darker_is_higher: bool = False


class Preprocessor(MapTransform):
    """
    Takes in a `Correspondence` and its 3D MRI binary mask and applies MRI and
    histology preprocessing transforms.
    """

    def __init__(self, config: PreprocessorConfig):
        super().__init__(
            keys=(
                PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE,
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
            (1) load the MRI and mask into a consistent orientation
            (2) obtain the 2D MRI and mask slices
            (3) check foreground pixel count threshold for early exit
            (4) resample MRI and mask to isotropic pixel spacing
            (5) center-crop prostate on MRI using mask
            (6) scale intensity between 1st and 99th percentiles to [0, 1]
            (7) standardise to average ImageNet mean and standard deviation
            (8) convert mask to boolean
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
                CenterCropMriOnMaskd(self.config.mri_patch_size),
                EnsureChannelFirstd(
                    keys=[
                        MriPipelineKeys.MRI_SLICE,
                        MriPipelineKeys.MRI_MASK_SLICE,
                    ]
                ),
                EnsureTyped(
                    keys=[MriPipelineKeys.MRI_MASK_SLICE],
                    dtype=torch.bool,
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
                    key=MriPipelineKeys._INPUT_MRI_SHAPE,
                ),
                MetaTensorLoggingd(  # log original affine matrix
                    keys=[MriPipelineKeys.MRI_VOLUME],
                    fn=lambda meta_tensor: meta_tensor.affine,
                    key=MriPipelineKeys._INPUT_MRI_AFFINE_MATRIX,
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
            (2) duplicate the RGB image for downstream visualisation purposes
            (3) use Otsu's algorithm to obtain a rough binary mask delineating
                tissue
            (4) apply Macenko H&E stain deconvolution using the rough tissue
                mask to separate out the haematoxylin channel
            (5) morphologically close and apply contour-based filling to the
                haematoxylin channel to obtain largest (fuller) binary mask
            (6) independently resize X and Y axes of all elements to target
                isotropic MPP
            (7) scale intensity to [0, 1]
            (8) standardise to average ImageNet mean and standard deviation
        """
        if_threshold_met = Compose(
            [
                ResizeAxesToIsotropicMPPd(
                    keys=[
                        HistologyPipelineKeys.HISTOLOGY,
                        HistologyPipelineKeys.HISTOLOGY_MASK,
                        HistologyPipelineKeys.HISTOLOGY_FOR_VISUALISATION,
                        HistologyPipelineKeys.MASK_FOR_VISUALISATION,
                    ],
                    modes=["area", "nearest-exact", "area", "area"],
                    target_microns_per_pixel=self.config.pixel_spacing_micrometres,
                ),
                EnsureChannelFirstd(
                    keys=[
                        HistologyPipelineKeys.HISTOLOGY,
                        HistologyPipelineKeys.HISTOLOGY_MASK,
                        HistologyPipelineKeys.HISTOLOGY_FOR_VISUALISATION,
                    ]
                ),
                ScaleIntensityRanged(
                    keys=[HistologyPipelineKeys.HISTOLOGY],
                    a_min=0,
                    a_max=255,
                    b_min=0,
                    b_max=1,
                    allow_missing_keys=False,
                ),
                EnsureTyped(
                    keys=[HistologyPipelineKeys.HISTOLOGY_MASK],
                    dtype=torch.bool,
                ),
            ]
        )
        return Compose(
            [
                # Load appropriate image level.
                LoadImaged(
                    keys=[HistologyPipelineKeys.HISTOLOGY],
                    reader=DynamicOpenSlideReader(
                        pixel_count_threshold=self.config.macenko_pixel_count_threshold,
                        target_microns_per_pixel=self.config.pixel_spacing_micrometres,
                        thumbnail_max_size=self.config.tissue_fraction_estimation_thumbnail_max_size,
                    ),  # pyright: ignore[reportArgumentType]
                    dtype="uint8",
                ),
                # Create copy of RGB histology for downstream qualitative evaluation.
                CopyItemsd(
                    keys=[HistologyPipelineKeys.HISTOLOGY],
                    names=[HistologyPipelineKeys.HISTOLOGY_FOR_VISUALISATION],
                ),
                # Deconvolve haematoxylin stain.
                BinaryMaskForStainDeconvolutiond(
                    closing_se_kernel_size_percentage=self.config.histology_closing_se_kernel_size_percentage,
                    gaussian_blur_kernel_size_percentage=self.config.histology_gaussian_blur_kernel_size_percentage,
                ),
                ObtainHaematoxylinUsingMacenkoHEStainDeconvolutiond(
                    darker_is_higher=self.config.histology_darker_is_higher
                ),
                # Obtain binary mask.
                BinaryMaskFromHaematoxylind(
                    darker_is_higher=self.config.histology_darker_is_higher,
                    closing_se_kernel_size_percentage=self.config.histology_closing_se_kernel_size_percentage,
                    minimum_hole_area_ratio=self.config.histology_minimum_hole_area_ratio,
                ),
                # Preprocess.
                SkipIfKeysPresentd(
                    keys=[SharedPipelineKeys.EARLY_EXIT],
                    transform=if_threshold_met,
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
            PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE: corr,
            PREPROCESSOR_PIPELINE_KEYS.MRI_MASK: mask,
        }

    def __call__(self, data: dict) -> dict:
        # Retrieve relevant arguments.
        corr: Correspondence = data[PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE]
        mask: MetaTensor = data[PREPROCESSOR_PIPELINE_KEYS.MRI_MASK]
        # Build modality-specific pipeline inputs.
        mri_dict = self.build_mri_pipeline_input(corr, mask)
        histology_dict = self.build_histology_pipeline_input(corr)
        # Run pipelines.
        mri_output: dict = self.mri_pipeline(
            mri_dict
        )  # pyright: ignore[reportAssignmentType]
        if SharedPipelineKeys.EARLY_EXIT in mri_output:
            # No need to go any further.
            histology_output = {}
        else:
            histology_output: dict = self.histology_pipeline(
                histology_dict
            )  # pyright: ignore[reportAssignmentType]
        # Combine outputs.
        return (
            {PREPROCESSOR_PIPELINE_KEYS.CORRESPONDENCE: corr}
            | mri_output
            | histology_output
        )
