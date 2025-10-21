from ..core.type_definitions import Percentage, StrictlyPositiveFloat
from .shared import SharedPipelineKeys

from monai.apps.pathology.transforms.stain.array import ExtractHEStains
from monai.config.type_definitions import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.data.wsi_reader import WSIReader
from monai.transforms.intensity.array import ScaleIntensityRangePercentiles
from monai.transforms.transform import MapTransform


from enum import StrEnum
from typing import Hashable

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
    # Reading in the NDPI image.
    HISTOLOGY = "histology"
    # Binary mask for stain deconvolution.
    STAIN_DECONVOLUTION_TISSUE_MASK = "stain_deconvolution_tissue_mask"
    # Deconvolve haematoxylin image.
    HAEMATOXYLIN = "haematoxylin"
    # Obtain binary mask.
    HISTOLOGY_MASK = "histology_mask"
    HAEMATOXYLIN_MASK = "haematoxylin_mask"


def clean_up_mask(
    raw_mask: np.ndarray,
    ellipse_axes_size_percentage: tuple[Percentage, Percentage],
    minimum_hole_area_ratio: StrictlyPositiveFloat,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean up 'artefacts' from a binary mask using morphological operations.

    Parameters
    ----------
    raw_mask: np.ndarray
    ellipse_axes_size_percentage: tuple[Percentage, Percentage]
        Ellipse axes specified as percentage of limiting dimension
    minimum_hole_area_ratio: StrictlyPositiveFloat
        Cut out contours whose area is larger than this fraction of the largest
        contour, otherwise leave them filled in.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        "Cleaner" mask, "cleaner" mask with no holes cut out
    """
    # Stage 1: Morphologically close for initial cleanup.
    morph_size = tuple(
        max(3, (m := round(ax * min(raw_mask.shape))) + (~m & 1))
        for ax in ellipse_axes_size_percentage
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_size)
    closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    # Stage 2: Contour-based hole filling.
    contours, hierarchy = cv2.findContours(
        closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("no contours found")
    #   2.a) Find largest contour and fill it.
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
    #   2.b) Find child contours and erode them away
    punctured_mask = np.copy(filled_mask)
    if minimum_hole_area_ratio > 0:
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
                    if hole_area / largest_area >= minimum_hole_area_ratio:
                        # Cut this hole out
                        cv2.drawContours(
                            punctured_mask,
                            [contours[i]],
                            -1,
                            color=0,  # type: ignore
                            thickness=cv2.FILLED,
                        )
    return punctured_mask, filled_mask


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


class EstimateTissueForStainDeconvolutiond(MapTransform):
    """
    Compute a binary mask using Otsu's algorithm and some morphological ops to
    to roughly identify tissue pixels in the histology. This is useful for the
    subsequent Macenko stain deconvolution stage, which will be able to obtain
    a statistically robust sample relying mostly on tissue rather than all
    available pixels i.e. including white background.
    """

    def __init__(
        self,
        save_as: Hashable,
        closing_se_kernel_size_percentage: Percentage,
        minimum_hole_area_ratio: StrictlyPositiveFloat,
    ):
        super().__init__(
            keys=(HistologyPipelineKeys.HISTOLOGY,),
            allow_missing_keys=False,
        )
        self.save_as = save_as
        self.closing_se_kernel_size_percentage = (
            closing_se_kernel_size_percentage
        )
        self.minimum_hole_area_ratio = minimum_hole_area_ratio

    def __call__(self, data: dict) -> dict:
        # Obtain histology.
        hist: MetaTensor = data[HistologyPipelineKeys.HISTOLOGY]
        # Apply Otsu's algorithm.
        grey = F.rgb_to_grayscale(hist, num_output_channels=1)
        grey = grey.squeeze(0)  # remove channel dimension
        grey_np = grey.numpy()
        _, mask = cv2.threshold(
            grey_np, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        # Clean mask up.
        try:
            mask, _ = clean_up_mask(
                raw_mask=mask,
                ellipse_axes_size_percentage=(
                    self.closing_se_kernel_size_percentage,
                    self.closing_se_kernel_size_percentage,
                ),
                minimum_hole_area_ratio=self.minimum_hole_area_ratio,
            )
        except ValueError:
            data[SharedPipelineKeys.EARLY_EXIT] = (
                "(Histology) No contours found."
            )
            return data
        # Create and store binary mask.
        data[self.save_as] = mask.astype(bool)  # shape: (H, W)
        # Done.
        return data


class DeconvolveHaematoxylinInRGBSpaced(MapTransform):
    """
    Deconvolves haematoxylin channel from an RGB histology input image using
    Macenko stain deconvolution and a binary mask to guide its statistical
    sampling for better colour accuracy.
    """

    OD_MAX = 2
    # NOTE: OD  = -log_{10}(I / I0) = 2
    #    => I   = exp(-2 * log(10)) * I0
    #    => I   ~ 0.01 * I0
    #    => 99% absorption, so mega heavy staining

    def __init__(self, save_as: Hashable):
        super().__init__(
            keys=(
                HistologyPipelineKeys.HISTOLOGY,
                HistologyPipelineKeys.STAIN_DECONVOLUTION_TISSUE_MASK,
            ),
            allow_missing_keys=False,
        )
        self.save_as = save_as
        self.extractor = ExtractHEStains()

    def __call__(self, data: dict) -> dict:
        # Obtain histology and binary mask.
        hist: MetaTensor = data[HistologyPipelineKeys.HISTOLOGY]
        mask: torch.Tensor = data[
            HistologyPipelineKeys.STAIN_DECONVOLUTION_TISSUE_MASK
        ]
        # Obtain stain matrix S.
        hist_np = hist.detach().cpu().permute(1, 2, 0).numpy()
        H, W, num_channels = hist_np.shape
        if num_channels != 3:
            raise ValueError(
                f"expected 3-channel RGB image, received {num_channels}-channel image"
            )
        S = self.extractor(hist_np[mask])  # shape: (mask.sum(), 3)
        # Convert image from RGB space to optical density (OD) space.
        ε = np.finfo(np.float32).eps
        I0 = 255
        OD = hist_np.astype(np.float32)
        OD = np.clip(OD, ε, 255)  # avoiding log(0)
        OD = -np.log10(OD / I0)
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
        # Obtain haematoxylin image in OD space.
        haematoxylin = C[haematoxylin_index, :].reshape(1, H, W)
        # Clip.
        haematoxylin = np.clip(haematoxylin, 0, self.OD_MAX)
        # Turn to RGB.
        haematoxylin = np.clip(I0 - I0 * np.exp(-haematoxylin), 0, I0)
        breakpoint()
        # Done!
        tensor = MetaTensor(haematoxylin).copy_meta_from(hist)
        data[self.save_as] = tensor
        return data


class BinaryMasksFromHaematoxylind(MapTransform):
    """
    Builds a larger and morphologically-closed binary tissue mask from the
    deconvolved haematoxylin image, as well as a version with holes cut out
    for visualisation purposes.
    """

    def __init__(
        self,
        save_as: tuple[Hashable, Hashable],
        closing_se_kernel_size_percentage: Percentage,
        minimum_hole_area_ratio: StrictlyPositiveFloat,
    ):
        super().__init__(
            keys=(HistologyPipelineKeys.HAEMATOXYLIN), allow_missing_keys=False
        )
        self.save_as = save_as
        self.closing_se_kernel_size_percentage = (
            closing_se_kernel_size_percentage
        )
        self.minimum_hole_area_ratio = minimum_hole_area_ratio
        self.otsu_prep = ScaleIntensityRangePercentiles(
            lower=1,
            upper=99,
            b_min=0,
            b_max=255,
            clip=True,
            dtype=np.uint8,
        )

    def __call__(self, data: dict) -> dict:
        # Prep haematoxylin for Otsu.
        haematoxylin = data[HistologyPipelineKeys.HAEMATOXYLIN]
        haematoxylin_uint8: MetaTensor = self.otsu_prep(
            haematoxylin
        )  # pyright: ignore[reportAssignmentType]
        haematoxylin_np = haematoxylin_uint8.squeeze().numpy()
        # Obtain mask.
        _, raw_mask = cv2.threshold(haematoxylin_np, 0, 255, cv2.THRESH_OTSU)
        # Clean up.
        try:
            masks = clean_up_mask(
                raw_mask=raw_mask,
                ellipse_axes_size_percentage=(
                    self.closing_se_kernel_size_percentage,
                    self.closing_se_kernel_size_percentage,
                ),
                minimum_hole_area_ratio=self.minimum_hole_area_ratio,
            )
        except ValueError:
            data[SharedPipelineKeys.EARLY_EXIT] = (
                "(Histology) No contours found."
            )
            return data
        # Convert masks to [0, 1] float32 and save.
        for mask, save_as in zip(masks, self.save_as):
            mask = mask.astype(np.float32) / 255
            data[save_as] = MetaTensor(mask).copy_meta_from(haematoxylin)
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
            *_, height, width = tensor.shape
            batched = tensor
            while batched.ndim < 4:
                batched = batched.unsqueeze(0)
            # Calculate indepedent scale factors.
            mpp_x = tensor.meta[HistologyMetadataKeys.ORIGINAL_MPP_X]
            mpp_y = tensor.meta[HistologyMetadataKeys.ORIGINAL_MPP_Y]
            scale_x = mpp_x / self.target_microns_per_pixel
            scale_y = mpp_y / self.target_microns_per_pixel
            # Calculate the new dimensions.
            new_height = round(height * scale_y)
            new_width = round(width * scale_x)
            # Resize to new dimensions.
            if tensor.dtype is torch.float32:
                mi = tensor.min()
                ma = tensor.max()
            else:
                raise TypeError(f"unexpected tensor type: {tensor.dtype}")
            resized_hist = torch.nn.functional.interpolate(
                batched.type(torch.float32),
                size=(new_height, new_width),
                mode=mode,
            ).squeeze(0)
            resized_hist = resized_hist.clamp(mi, ma).type(tensor.dtype)
            resized_hist.meta[HistologyMetadataKeys.MPP] = (
                self.target_microns_per_pixel
            )
            # Done.
            data[key] = resized_hist
        return data
