from ..core.mri_axis import MRIAxis
from .shared import SharedPipelineKeys

from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform

from copy import deepcopy
from enum import StrEnum
from typing import NamedTuple


from nibabel.affines import voxel_sizes as nib_voxel_sizes
import torch


class MriPipelineKeys(StrEnum):
    # Loading in.
    MRI_VOLUME = "mri_volume"
    MASK_VOLUME = "mask_volume"
    # Calculating RAS index.
    _INPUT_MRI_AFFINE_MATRIX = "input_mri_affine_matrix"
    _INPUT_MRI_SHAPE = "input_mri_shape"
    MRI_SLICE_INDEX = "mri_slice_index"
    MRI_SLICE_AXIS = "mri_slice_axis"
    # 2D slices.
    MRI_SLICE = "mri_slice"
    MRI_MASK_SLICE = "mri_mask_slice"


class CalculateNewSliceIndexd(MapTransform):
    """
    Responsible for calculating the new slice index in a newly oriented volume
    given the affine matrix of the original slice index and axis, and the newly
    oriented volume's affine matrix.
    Assumes the original MRI's shape and affine matrix have been logged under

                • `MRI_PIPELINE_KEYS._INPUT_MRI_SHAPE`
                • `MRI_PIPELINE_KEYS._INPUT_MRI_AFFINE_MATRIX`

    respectively.
    """

    def __init__(self):
        super().__init__(
            keys=(
                MriPipelineKeys.MRI_VOLUME,
                MriPipelineKeys.MRI_SLICE_INDEX,
                MriPipelineKeys.MRI_SLICE_AXIS,
            ),
            allow_missing_keys=False,
        )

    def __call__(self, data: dict) -> dict:
        # Retrieve arguments we'll be operating on.
        reoriented_volume: MetaTensor = data[MriPipelineKeys.MRI_VOLUME]
        original_affine = reoriented_volume.meta[
            MriPipelineKeys._INPUT_MRI_AFFINE_MATRIX
        ]
        new_affine = reoriented_volume.affine
        original_shape: tuple[int, int, int] = reoriented_volume.meta[
            MriPipelineKeys._INPUT_MRI_SHAPE
        ]
        assert len(original_shape) == 3
        original_slice_index: int = data[MriPipelineKeys.MRI_SLICE_INDEX]
        slice_axis: MRIAxis = data[MriPipelineKeys.MRI_SLICE_AXIS]

        # Convert affines to Float.
        original_affine = original_affine.float()
        new_affine = new_affine.float()

        # Determine the NumPy axis for the slice in the original orientation.
        original_numpy_axis = slice_axis.get_numpy_axis(original_affine)
        # Define a representative point in the center of the original slice plane.
        source_voxel_coord = torch.tensor(
            [
                *(
                    (
                        (dim - 1) / 2.0
                        if j != original_numpy_axis
                        else original_slice_index
                    )
                    for j, dim in enumerate(original_shape)
                ),
                1.0,
            ],
            dtype=torch.float32,
        )
        # Map the voxel coordinate to its physical location.
        physical_coord = torch.matmul(original_affine, source_voxel_coord)
        # Map the physical coordinate back into the voxel shape of the new volume.
        new_affine_inv = torch.linalg.inv(new_affine)
        new_voxel_coord = torch.matmul(new_affine_inv, physical_coord)
        # Determine the NumPy axis for the slice in the new orientation.
        new_numpy_axis = slice_axis.get_numpy_axis(new_affine)
        # Extract the coordinate along the new axis and round to the nearest integer.
        new_slice_index = int(round(new_voxel_coord[new_numpy_axis].item()))
        # Record transform.
        data[MriPipelineKeys.MRI_SLICE_INDEX] = new_slice_index
        return data


class MRIAndMaskSlicerd(MapTransform):
    """Transform for slicing into the MRI and binary mask volumes."""

    def __init__(self, keepdim: bool):
        super().__init__(
            keys=(
                MriPipelineKeys.MRI_VOLUME,
                MriPipelineKeys.MASK_VOLUME,
                MriPipelineKeys.MRI_SLICE_INDEX,
                MriPipelineKeys.MRI_SLICE_AXIS,
            ),
            allow_missing_keys=False,
        )
        self.keepdim = keepdim

    def __call__(self, data: dict) -> dict:
        # Get relevant arguments.
        mri_volume: MetaTensor = data[MriPipelineKeys.MRI_VOLUME]
        mask_volume: MetaTensor = data[MriPipelineKeys.MASK_VOLUME]
        slice_index: int = data[MriPipelineKeys.MRI_SLICE_INDEX]
        slice_axis: MRIAxis = data[MriPipelineKeys.MRI_SLICE_AXIS]
        # Get slice index.
        numpy_slice = slice_axis.get_numpy_slice(
            mri_volume.affine, slice_index, self.keepdim
        )
        # Slice and store.
        data[MriPipelineKeys.MRI_SLICE] = mri_volume.squeeze(0)[numpy_slice]
        data[MriPipelineKeys.MRI_MASK_SLICE] = mask_volume.squeeze(0)[
            numpy_slice
        ]
        return data


class SufficientProstatePixelsInMaskd(MapTransform):
    """
    NOTE:   At this point in the pipeline, the binary mask for a prostate,
            while likely to not be all-background at this stage, could contain
            all-background slices that might be sampled by the specific slice
            index.
            If the binary mask has no nonzero elements at this stage, the rest
            of the pipeline doesn't make any sense, so this sample needs to be
            dropped.
            I'm setting a flag if the mask's foreground pixels are fewer than
            a certain threshold. A custom threshold instead of 0 because the
            the segmentor network either (a) fails to identify the prostate, or
            (b) identifies a prostate that is too small and thus would result
            in a poor sample for the network to learn from. While the threshold
            is set in pixels here, the preprocessing pipeline grounds this
            value in actual physical units which are then converted into pixels
            for the given preprocessing parameters.
    """

    def __init__(self, pixel_threshold: int):
        super().__init__(
            keys=[MriPipelineKeys.MRI_MASK_SLICE],
            allow_missing_keys=False,
        )
        self.pixel_threshold = pixel_threshold

    def __call__(self, data: dict) -> dict:
        # Obtain mask slice.
        mask_slice: MetaTensor = data[MriPipelineKeys.MRI_MASK_SLICE]
        # Are we exiting early?
        prostate_pixels = int((mask_slice != 0).sum().item())
        if prostate_pixels <= self.pixel_threshold:
            data[SharedPipelineKeys.EARLY_EXIT] = (
                f"{prostate_pixels} ≤ {self.pixel_threshold}"
            )
        return data


class ResampleToIsotropicSpacingd(MapTransform, InvertibleTransform):
    """
    Resamples the MRI and mask slices to have a fixed pixel spacing.
    The MRI slice is resampled using bilinear interpolation.
    The mask slice is resampled using the `nearest-exact` algorithm.
    """

    def __init__(self, pixel_spacing_millimeters: float):
        MapTransform.__init__(
            self,
            keys=(
                MriPipelineKeys.MRI_SLICE,
                MriPipelineKeys.MRI_MASK_SLICE,
                MriPipelineKeys.MRI_VOLUME,
            ),
            allow_missing_keys=False,
        )
        InvertibleTransform.__init__(self)
        self.pixel_spacing_millimeters = pixel_spacing_millimeters

    def __call__(self, data: dict) -> dict:
        # Retrieve arguments we'll be operating on.
        mri_slice: MetaTensor = data[MriPipelineKeys.MRI_SLICE]
        mask_slice: MetaTensor = data[MriPipelineKeys.MRI_MASK_SLICE]
        original_volume: MetaTensor = data[MriPipelineKeys.MRI_VOLUME]
        # Clone original inputs for invertability.
        original_mri_slice = mri_slice.clone()
        original_mask_slice = mask_slice.clone()
        # Obtain spacing parameters.
        original_affine = original_volume.meta["affine"].numpy()
        original_shape = original_volume.squeeze(
            0
        ).shape  # NOTE: remove channel dimension
        original_spacing = nib_voxel_sizes(original_affine)[:2]
        # Calculate scale factors and new shape.
        scale_factor_y = original_spacing[0] / self.pixel_spacing_millimeters
        scale_factor_x = original_spacing[1] / self.pixel_spacing_millimeters
        new_shape_y = round(original_shape[0] * scale_factor_y)
        new_shape_x = round(original_shape[1] * scale_factor_x)
        new_shape = (new_shape_y, new_shape_x)
        are_we_downsampling = all(
            x < y
            for x, y in zip(
                original_spacing,
                (
                    self.pixel_spacing_millimeters,
                    self.pixel_spacing_millimeters,
                ),
            )
        )
        # Interpolate.
        mri_slice = torch.nn.functional.interpolate(
            mri_slice.unsqueeze(0).unsqueeze(0),
            size=new_shape,
            mode="bilinear",
            antialias=are_we_downsampling,
            align_corners=False,
        ).squeeze(0, 1)
        mask_slice = torch.nn.functional.interpolate(
            mask_slice.unsqueeze(0).unsqueeze(0),
            size=new_shape,
            mode="nearest-exact",
        ).squeeze(0, 1)
        # Store resampled data and original inputs.
        data[MriPipelineKeys.MRI_SLICE] = mri_slice
        self.push_transform(
            data,
            MriPipelineKeys.MRI_SLICE,
            extra_info={MriPipelineKeys.MRI_SLICE: original_mri_slice},
        )
        data[MriPipelineKeys.MRI_MASK_SLICE] = mask_slice
        self.push_transform(
            data,
            MriPipelineKeys.MRI_MASK_SLICE,
            extra_info={MriPipelineKeys.MRI_MASK_SLICE: original_mask_slice},
        )
        return data

    def inverse(self, data: dict) -> dict:
        data = deepcopy(data)
        # Retrieved stored original values.
        for key in (
            MriPipelineKeys.MRI_SLICE,
            MriPipelineKeys.MRI_MASK_SLICE,
        ):
            transform_info = self.pop_transform(data, key)
            data[key] = transform_info["extra_info"][key]
        return data


class CenterCropMriOnMaskd(MapTransform, InvertibleTransform):
    """
    Responsible for center-cropping an MRI slice using its corresponding
    mask slice, adding padding as necessary.
    """

    def __init__(self, patch_size: int):
        MapTransform.__init__(
            self,
            keys=(
                MriPipelineKeys.MRI_SLICE,
                MriPipelineKeys.MRI_MASK_SLICE,
            ),
            allow_missing_keys=False,
        )
        InvertibleTransform.__init__(self)
        self.patch_size = patch_size

    def __call__(self, data: dict) -> dict:
        # Obtain relevant arguments.
        mri_slice: MetaTensor = data[MriPipelineKeys.MRI_SLICE]
        mask_slice: MetaTensor = data[MriPipelineKeys.MRI_MASK_SLICE]
        mri_height, mri_width, _ = (
            data[MriPipelineKeys.MRI_VOLUME].squeeze(0).shape
        )  # NOTE: works because RAS-oriented at this stage
        patch_size = self.patch_size
        # Store inputs for invertability.
        original_mri_slice = mri_slice.clone()
        original_mask_slice = mask_slice.clone()
        # Determine crop coordinates from bounding box.
        bbox = Bbox.from_binary_mask(mask_slice)
        crop_y_start = round(bbox.center_y - patch_size / 2)
        crop_x_start = round(bbox.center_x - patch_size / 2)
        crop_y_end = crop_y_start + patch_size
        crop_x_end = crop_x_start + patch_size
        # Calculate and apply padding.
        pad_left = -min(0, crop_x_start)
        pad_right = max(0, crop_x_end - mri_width)
        pad_top = -min(0, crop_y_start)
        pad_bottom = max(0, crop_y_end - mri_height)
        padded_mri_slice = torch.nn.functional.pad(
            mri_slice,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
        padded_mask_slice = torch.nn.functional.pad(
            mask_slice,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )
        # Crop padded slices.
        final_crop_y_start = crop_y_start + pad_top
        final_crop_x_start = crop_x_start + pad_left
        mri_patch = padded_mri_slice[
            final_crop_y_start : final_crop_y_start + patch_size,
            final_crop_x_start : final_crop_x_start + patch_size,
        ]
        mask_patch = padded_mask_slice[
            final_crop_y_start : final_crop_y_start + patch_size,
            final_crop_x_start : final_crop_x_start + patch_size,
        ]
        # Record transforms.
        data[MriPipelineKeys.MRI_SLICE] = mri_patch
        self.push_transform(
            data,
            MriPipelineKeys.MRI_SLICE,
            extra_info={MriPipelineKeys.MRI_SLICE: original_mri_slice},
        )
        data[MriPipelineKeys.MRI_MASK_SLICE] = mask_patch
        self.push_transform(
            data,
            MriPipelineKeys.MRI_MASK_SLICE,
            extra_info={MriPipelineKeys.MRI_MASK_SLICE: original_mask_slice},
        )
        return data

    def inverse(self, data: dict) -> dict:
        data = deepcopy(data)
        # Retrieve original stored values.
        for key in (
            MriPipelineKeys.MRI_SLICE,
            MriPipelineKeys.MRI_MASK_SLICE,
        ):
            transform_info = self.pop_transform(data, key)
            data[key] = transform_info["extra_info"][key]
        return data


class Bbox(NamedTuple):
    center_x: float
    center_y: float
    height: int
    width: int

    @staticmethod
    def from_binary_mask(mask: torch.Tensor) -> "Bbox":
        prostate_pixels = torch.nonzero(mask)
        y_min, x_min = torch.min(prostate_pixels, dim=0).values
        y_max, x_max = torch.max(prostate_pixels, dim=0).values
        # NOTE: +1 because inclusive indices
        bbox_height: int = (
            y_max.item() - y_min.item() + 1
        )  # pyright: ignore[reportAssignmentType]
        bbox_width: int = (
            x_max.item() - x_min.item() + 1
        )  # pyright: ignore[reportAssignmentType]
        center_y = y_min.item() + bbox_height / 2
        center_x = x_min.item() + bbox_width / 2
        return Bbox(
            center_x=center_x,
            center_y=center_y,
            height=bbox_height,
            width=bbox_width,
        )
