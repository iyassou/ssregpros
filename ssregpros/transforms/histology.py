from ..core.type_definitions import Percentage, StrictlyPositiveFloat
from .shared import SharedPipelineKeys

from monai.apps.pathology.transforms.stain.array import ExtractHEStains
from monai.config.type_definitions import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.data.wsi_reader import WSIReader
from monai.transforms.transform import MapTransform


from copy import deepcopy
from enum import StrEnum

import cv2
import openslide
import numpy as np
import torch
import torchvision.transforms.functional as F


class HistologyMetadataKeys(StrEnum):
    ORIGINAL_MPP_X = "original_mpp_x"
    ORIGINAL_MPP_Y = "original_mpp_y"
    MPP = "mpp"


class HistologyPipelineKeys(StrEnum):
    HISTOLOGY = "histology"
    HISTOLOGY_MASK = "histology_mask"

    MACENKO_STAIN_DECONVOLUTION_MASK = "macenko_stain_deconvolution_mask"

    HISTOLOGY_FOR_VISUALISATION = "histology_for_visualisation"
    MASK_FOR_VISUALISATION = "histology_mask_for_visualisation"


class DynamicOpenSlideReader(WSIReader):
    """
    Loads in an OpenSlide image at a level which:
        (a) possesses tissue pixels above a certain threshold
        (b) possesses a microns-per-pixel ratio that is closest
            to the set target
    """

    def __init__(
        self,
        pixel_count_threshold: int,
        target_microns_per_pixel: float,
        thumbnail_max_size: int,
        **kwargs,
    ):
        super().__init__(backend="openslide", **kwargs)
        self.pixel_count_threshold = pixel_count_threshold
        self.target_microns_per_pixel = target_microns_per_pixel
        self.thumbnail_max_size = thumbnail_max_size

    def get_data(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, slide: openslide.OpenSlide
    ):
        # Determine tissue fraction.
        thumb = slide.get_thumbnail(
            (self.thumbnail_max_size, self.thumbnail_max_size)
        )
        grey = np.array(thumb.convert("L"), dtype=np.uint8)
        _, tissue_mask = cv2.threshold(
            grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        tissue_fraction = tissue_mask.mean() / 255.0
        # Filter levels by tissue pixel count.
        candidates: list[tuple[int, float, float]] = []  # (level, MPPx, MPPy)
        base_mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        base_mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        for L in range(slide.level_count):
            width, height = slide.level_dimensions[L]
            if width * height * tissue_fraction >= self.pixel_count_threshold:
                downsample = slide.level_downsamples[L]
                candidates.append(
                    (
                        L,
                        base_mpp_x * downsample,
                        base_mpp_y * downsample,
                    )
                )
        # Determine the level whose average MPP is closest to the target MPP.
        level, mpp_x, mpp_y = min(
            candidates,
            key=lambda cand: abs(
                (cand[1] + cand[2]) / 2.0 - self.target_microns_per_pixel
            ),
        )
        # Load image and metadata.
        img, meta = super().get_data(slide, level=level)
        # Record MPP for selected level.
        meta.update(
            {
                HistologyMetadataKeys.ORIGINAL_MPP_X: mpp_x,
                HistologyMetadataKeys.ORIGINAL_MPP_Y: mpp_y,
            }
        )
        # Done.
        return img, meta


class BinaryMaskForStainDeconvolutiond(MapTransform):
    """
    Compute a binary mask using Otsu's algorithm to roughly identify tissue
    pixels in the histology. This is useful for the subsequent Macenko stain
    deconvolution stage, which will be able to obtain a statistically robust
    sample relying mostly on tissue rather than all available pixels i.e.
    including white background.
    """

    def __init__(
        self,
        closing_se_kernel_size_percentage: Percentage,
        gaussian_blur_kernel_size_percentage: Percentage,
    ):
        super().__init__(
            keys=(HistologyPipelineKeys.HISTOLOGY,),
            allow_missing_keys=False,
        )
        self.closing_se_kernel_size_percentage = (
            closing_se_kernel_size_percentage
        )
        self.gaussian_blur_kernel_size_percentage = (
            gaussian_blur_kernel_size_percentage
        )

    def __call__(self, data: dict) -> dict:
        # Obtain histology.
        hist: MetaTensor = data[HistologyPipelineKeys.HISTOLOGY]
        # Apply Otsu's algorithm.
        grey = F.rgb_to_grayscale(hist, num_output_channels=1)
        grey = grey.squeeze(0)  # remove channel dimension
        _, mask = cv2.threshold(
            grey.numpy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        # Create and store binary mask.
        data[HistologyPipelineKeys.MACENKO_STAIN_DECONVOLUTION_MASK] = (
            torch.from_numpy(mask.astype(bool))
        )  # shape: (H, W)
        # Create visualisation binary mask.
        # (1) Morphologically close
        morph_size = max(
            3, round(self.closing_se_kernel_size_percentage * min(mask.shape))
        )
        morph_size += ~morph_size & 1  # ensure odd
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # (2) Feather with Gaussian blur
        feather_size = max(
            3,
            round(self.gaussian_blur_kernel_size_percentage * min(mask.shape)),
        )
        feather_size += ~feather_size & 1
        feathered_mask = np.clip(
            cv2.GaussianBlur(
                closed_mask.astype(np.float32), (feather_size, feather_size), 0
            )
            / 255.0,
            0,
            1,
        )
        # Done, create MetaTensor for later resizing capability.
        mask_visual = MetaTensor(
            torch.from_numpy(feathered_mask).unsqueeze(0)
        ).copy_meta_from(hist)
        data[HistologyPipelineKeys.MASK_FOR_VISUALISATION] = mask_visual
        return data


class ObtainHaematoxylinUsingMacenkoHEStainDeconvolutiond(MapTransform):
    """
    Deconvolves haematoxylin channel from an RGB histology input image using
    Macenko stain deconvolution and a binary mask to guide its statistical
    sampling for better colour accuracy.
    """

    def __init__(self, darker_is_higher: bool):
        super().__init__(
            keys=(
                HistologyPipelineKeys.HISTOLOGY,
                HistologyPipelineKeys.MACENKO_STAIN_DECONVOLUTION_MASK,
            ),
            allow_missing_keys=False,
        )
        self.extractor = ExtractHEStains()
        self.darker_is_higher = darker_is_higher

    def __call__(self, data: dict) -> dict:
        # Obtain histology and binary mask.
        hist: MetaTensor = data[HistologyPipelineKeys.HISTOLOGY]
        mask: torch.Tensor = data[
            HistologyPipelineKeys.MACENKO_STAIN_DECONVOLUTION_MASK
        ]
        # Obtain stain matrix S.
        hist_np = hist.detach().cpu().permute(1, 2, 0).numpy()
        H, W, num_channels = hist_np.shape
        if num_channels != 3:
            raise ValueError(
                f"expected 3-channel RGB image, received {num_channels}-channel image"
            )
        S = self.extractor(hist_np[mask])
        # Convert image from RGB space to optical density (OD) space.
        ε = np.finfo(np.float32).eps
        I0 = 255
        OD = hist_np.astype(np.float32)
        OD = np.clip(OD, ε, 255)  # avoiding log(0)
        OD = -np.log(OD / I0)
        OD_vec = OD.reshape(H * W, num_channels).T
        # Obtain pseudoinverse P of S.
        P = np.linalg.inv(S.T @ S) @ S.T
        # Obtain concentrations C.
        C = P @ OD_vec  # shape: (2, H*W)
        # Determine stain indices by sorting stain vectors on a
        # blue-red axis: haematoxylin is a "bluer" stain and eosin is
        # a "redder" stain.
        red, _, blue = S
        red_blue_ratio = red / (blue + ε)
        haematoxylin_index = np.argmin(red_blue_ratio)
        concentration = C[haematoxylin_index, :].reshape(1, H, W)
        # Convert from OD space to RGB space.
        concentration = np.clip(I0 * np.exp(-concentration), 0, 255)
        if not self.darker_is_higher:
            concentration = (
                255 - concentration
            )  # makes higher values appear lighter
        concentration = concentration.astype(np.uint8)
        # Construct MetaTensor from relevant concentration.
        meta = deepcopy(hist.meta)
        tensor = MetaTensor(concentration, meta=meta)
        data[HistologyPipelineKeys.HISTOLOGY] = tensor
        return data


class BinaryMaskFromHaematoxylind(MapTransform):
    """
    Builds a larger and morphologically-closed binary tissue mask from the
    deconvolved haemtoxylin channel image.
    """

    def __init__(
        self,
        darker_is_higher: bool,
        closing_se_kernel_size_percentage: Percentage,
        minimum_hole_area_ratio: StrictlyPositiveFloat,
    ):
        super().__init__(
            keys=(HistologyPipelineKeys.HISTOLOGY), allow_missing_keys=False
        )
        self.darker_is_higher = darker_is_higher
        self.closing_se_kernel_size_percentage = (
            closing_se_kernel_size_percentage
        )
        self.minimum_hole_area_ratio = minimum_hole_area_ratio

    def __call__(self, data: dict) -> dict:
        # Obtain haematoxylin channel.
        # NOTE: assumed to have already been written to the HISTOLOGY key,
        #       hence the first channel is the haematoxylin channel.
        hist: MetaTensor = data[HistologyPipelineKeys.HISTOLOGY]
        assert hist.ndim == 3
        haematoxylin = hist[0, :, :].numpy()  # shape: (H, W)
        # > Stage 1: Morphologically close for initial cleanup.
        _, raw_mask = cv2.threshold(
            haematoxylin,
            0,
            255,
            (
                cv2.THRESH_BINARY
                if not self.darker_is_higher
                else cv2.THRESH_BINARY_INV
            )
            | cv2.THRESH_OTSU,
        )
        morph_size = max(
            3,
            round(self.closing_se_kernel_size_percentage * min(raw_mask.shape)),
        )
        morph_size += ~morph_size & 1  # ensure odd
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        # > Stage 2: Contour-based hole filling.
        contours, hierarchy = cv2.findContours(
            closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            data[SharedPipelineKeys.EARLY_EXIT] = "no contours found"
            return data
        # (a) Find largest contour and fill it.
        largest_contour_idx = max(
            range(len(contours)), key=lambda i: cv2.contourArea(contours[i])
        )
        largest_contour = contours[largest_contour_idx]
        filled_mask = np.zeros_like(raw_mask)
        cv2.drawContours(
            filled_mask,
            [largest_contour],
            -1,
            color=255,  # type: ignore
            thickness=cv2.FILLED,
        )
        # (b) Find child contours and erode them away
        if self.minimum_hole_area_ratio > 0:
            largest_area = cv2.contourArea(largest_contour)
            if hierarchy is not None:
                hierarchy = hierarchy[
                    0
                ]  # Shape: (N, 4) where columns are [Next, Previous, First_Child, Parent]
                # Find all child contours of the largest contour
                for i, h in enumerate(hierarchy):
                    parent_idx = h[3]  # Parent index
                    if parent_idx == largest_contour_idx:
                        hole_area = cv2.contourArea(contours[i])
                        if (
                            hole_area / largest_area
                            >= self.minimum_hole_area_ratio
                        ):
                            # Cut this hole out
                            cv2.drawContours(
                                filled_mask,
                                [contours[i]],
                                -1,
                                color=0,  # type: ignore
                                thickness=cv2.FILLED,
                            )
        # Done!
        data[HistologyPipelineKeys.HISTOLOGY_MASK] = MetaTensor(
            torch.from_numpy(filled_mask.astype(bool)).unsqueeze(
                0
            )  # shape: (1, H, W)
        ).copy_meta_from(hist)
        return data


class ResizeAxesToIsotropicMPPd(MapTransform):
    """
    Independently resizes an image's axes to the desired isotropic resolution.
    """

    def __init__(
        self,
        keys: KeysCollection,
        modes: list[str],
        target_microns_per_pixel: float,
    ):
        super().__init__(keys=keys, allow_missing_keys=False)
        self.modes = modes
        self.target_microns_per_pixel = target_microns_per_pixel

    def __call__(self, data: dict) -> dict:
        for key, mode in self.key_iterator(data, self.modes):
            # Get data.
            tensor: MetaTensor = data[key]
            # Determine attributes.
            _, height, width = tensor.shape
            batched = tensor.unsqueeze(0)
            # Calculate indepedent scale factors.
            mpp_x = tensor.meta[HistologyMetadataKeys.ORIGINAL_MPP_X]
            mpp_y = tensor.meta[HistologyMetadataKeys.ORIGINAL_MPP_Y]
            scale_x = mpp_x / self.target_microns_per_pixel
            scale_y = mpp_y / self.target_microns_per_pixel
            # Calculate the new dimensions.
            new_height = round(height * scale_y)
            new_width = round(width * scale_x)
            # Resize to new dimensions.
            if tensor.dtype is torch.uint8:
                mi = 0
                ma = 255
            elif tensor.dtype is torch.bool:
                mi = 0
                ma = 1
            elif tensor.dtype is torch.float32:
                mi = tensor.min()
                ma = tensor.max()
            else:
                raise TypeError(f"unexpected tensor type: {tensor.dtype}")
            resized_hist = torch.nn.functional.interpolate(
                batched.type(torch.float32),
                size=(new_height, new_width),
                mode=mode,
            ).squeeze(0)
            if not tensor.dtype.is_floating_point:
                resized_hist = resized_hist.round()
            resized_hist = resized_hist.clamp(mi, ma).type(tensor.dtype)
            resized_hist.meta[HistologyMetadataKeys.MPP] = (
                self.target_microns_per_pixel
            )
            # Record transform.
            data[key] = resized_hist
        return data
