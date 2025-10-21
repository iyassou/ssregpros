from ..models.registration import RegistrationNetwork, RegistrationNetworkOutput

from enum import StrEnum
from typing import NamedTuple

import kornia
import kornia.morphology as morphology
import math
import torch

RGB = tuple[float, float, float]


class GateEdgesMode(StrEnum):
    BAND = "band"
    MASK = "mask"


class CannyTolerantOverlayOutput(NamedTuple):
    overlay: torch.Tensor
    mri_coverage: torch.Tensor
    histology_coverage: torch.Tensor
    symmetric_coverage: torch.Tensor


class QualitativeVisuals(NamedTuple):
    warped_histology: torch.Tensor
    # Warped RGB histology.

    warped_histology_mask: torch.Tensor
    # Warped histology mask, has holes.

    checkerboard: torch.Tensor
    # Checkerboard overlay

    canny_band: CannyTolerantOverlayOutput
    # Band-gated Canny overlay

    canny_mask: CannyTolerantOverlayOutput
    # Mask-gated Canny overlay


def generate_qualitative_visuals(
    model: RegistrationNetwork,
    pred: RegistrationNetworkOutput,
    mri: torch.Tensor,
    mri_mask: torch.Tensor,
    histology: list[torch.Tensor],
    histology_mask: list[torch.Tensor],
    canny_band_sigma: float,
    display_in_clinical_convention: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CannyTolerantOverlayOutput,
    CannyTolerantOverlayOutput,
]:
    """
    Generate qualitative visualisations using "red" for MRI and "green" for histology.

    Notes
    -----
    Reminder: `histology_mask` is the mask with punctures as identified by the
    histology preprocessing pipeline.
    The histology mask is used for the checkerboard overlay, but the haematoxylin,
    read "whole", mask is used for the gated Canny overlays.

    Returns
    -------
    QualitativeVisuals
    """
    # Generate warped RGB histology and its mask for visuals.
    warped_histology = model.apply_rigid_transform(
        pred.thetas,
        histology,
        grid_sampling_mode="bilinear",
    )
    warped_histology_mask = model.apply_rigid_transform(
        pred.thetas,
        histology_mask,
        grid_sampling_mode="nearest",
    )
    # Checkerboard overlay.
    checkerboard = checkerboard_overlay(
        mri_batch=mri,
        mri_mask_batch=mri_mask,
        warped_histology_rgb_batch=warped_histology,
        warped_histology_mask_batch=warped_histology_mask,
        patch_size=16,
        mri_kernel_size=5,
        mri_border_colour=(1, 0, 0),
        histology_kernel_size=3,
        histology_border_colour=(0, 1, 0),
        display_in_clinical_convention=display_in_clinical_convention,
    )
    # Band-gated Canny overlay.
    canny_band = canny_tolerant_overlay(
        mri_batch=mri,
        mri_mask_batch=mri_mask,
        warped_haematoxylin_batch=pred.warped_haematoxylin,
        warped_haematoxylin_mask_batch=pred.warped_haematoxylin_mask,
        sigma_px=canny_band_sigma,
        mode=GateEdgesMode.BAND,
        display_in_clinical_convention=display_in_clinical_convention,
    )
    # Mask-gated Canny overlay.
    canny_mask = canny_tolerant_overlay(
        mri_batch=mri,
        mri_mask_batch=mri_mask,
        warped_haematoxylin_batch=pred.warped_haematoxylin,
        warped_haematoxylin_mask_batch=pred.warped_haematoxylin_mask,
        sigma_px=canny_band_sigma,
        mode=GateEdgesMode.MASK,
        display_in_clinical_convention=display_in_clinical_convention,
    )
    # Done!
    return QualitativeVisuals(
        warped_histology=warped_histology,
        warped_histology_mask=warped_histology_mask,
        checkerboard=checkerboard,
        canny_band=canny_band,
        canny_mask=canny_mask,
    )


def checkerboard_overlay(
    mri_batch: torch.Tensor | torch.Tensor,
    mri_mask_batch: torch.Tensor | torch.Tensor,
    warped_histology_rgb_batch: torch.Tensor,
    warped_histology_mask_batch: torch.Tensor,
    patch_size: int,
    mri_kernel_size: int,
    mri_border_colour: RGB,
    histology_kernel_size: int,
    histology_border_colour: RGB,
    display_in_clinical_convention: bool,
) -> torch.Tensor:
    """
    Creates a checkerboard overlay of the fixed MRI and warped histology.

    Parameters
    ----------
    mri_batch: torch.Tensor
        Shape (B, 1, H, W), values in [0, 1], torch.float32
    mri_mask_batch: torch.Tensor
        Shape (B, 1, H, W), values in {0, 1}, torch.float32
    warped_histology_rgb_batch: torch.Tensor
        Shape (B, 3, H, W), values in [0, 1], torch.float32
    warped_histology_mask_batch: torch.Tensor
        Shape (B, 1, H, W), values in {0, 1}, torch.float32
    patch_size: int
        The size of each square in the checkerboard.
    mri_kernel_size: int
        Roughly translates to MRI border thickness
    mri_border_colour: RGB
        MRI border colour
    histology_kernel_size: int
        Roughly translates to histology border thickness
    histology_border_colour: RGB
        Histology border colour
    display_in_clinical_convention: bool
        Are we displaying in clinical convention?

    Returns
    -------
    torch.Tensor
        Shape (B, 3, H, W), values in [0, 1], torch.float32
    """
    # Sanity check devices.
    device = mri_batch.device
    if (ecived := warped_histology_rgb_batch.device) != device:
        raise ValueError(
            f"input tensors reside on different devices! {device} and {ecived}"
        )
    if (ecived := warped_histology_mask_batch.device) != device:
        raise ValueError(
            f"input tensors reside on different devices! {device} and {ecived}"
        )
    # Create checkerboard mask.
    _, _, H, W = mri_batch.shape
    x_coords = torch.arange(W, device=device)
    y_coords = torch.arange(H, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    mask_2d = (
        grid_x // patch_size + grid_y // patch_size
    ) % 2 == 1  # shape: (H, W)
    checkerboard_mask = mask_2d.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
    # Convert MRI to 3-channel grayscale.
    mri_rgb = mri_batch.repeat(1, 3, 1, 1)
    # Create alpha mask from tissue mask.
    alpha_mask = warped_histology_mask_batch.repeat(1, 3, 1, 1)
    # Three-way combination logic.
    # NOTE: where the checkerboard is "off" for histology, blend
    #       the histology and MRI using the tissue alpha mask
    blended = (
        alpha_mask * warped_histology_rgb_batch + (1 - alpha_mask) * mri_rgb
    )
    overlay = torch.where(
        ~checkerboard_mask, blended, mri_rgb
    )  # shape: (B, 3, H, W)
    # =========================================================================
    # Draw MRI border.
    overlay = _mask_border_overlay(
        canvas_batch=overlay,
        mask_batch=mri_mask_batch,
        kernel_size=mri_kernel_size,
        colour=mri_border_colour,
    )
    # Draw histology border.
    overlay = _mask_border_overlay(
        canvas_batch=overlay,
        mask_batch=warped_histology_mask_batch,
        kernel_size=histology_kernel_size,
        colour=histology_border_colour,
    )
    # =========================================================================
    if display_in_clinical_convention:
        overlay = clinical_convention(overlay)
    overlay = overlay.detach().cpu()
    return overlay


def canny_tolerant_overlay(
    mri_batch: torch.Tensor,
    mri_mask_batch: torch.Tensor,
    warped_haematoxylin_batch: torch.Tensor,
    warped_haematoxylin_mask_batch: torch.Tensor,
    sigma_px: float,
    mode: GateEdgesMode,
    band_factor: float = 1.96,
    tol_factor: float = 1.96,
    display_in_clinical_convention: bool = False,
) -> CannyTolerantOverlayOutput:
    """
    Creates a tolerance-aware Canny overlay between the two modalities:
    yellow means overlap, green means histology, red means MRI.

    Parameters
    ----------
    mri_batch: torch.Tensor
        Shape (B, 1, H, W), values in [0, 1], torch.float32
    mri_mask_batch: torch.Tensor
        Shape (B, 1, H, W), values in {0, 1}, torch.float32
    warped_haematoxylin_batch: torch.Tensor
        Shape (B, 1, H, W), values in [0, 1], torch.float32
    warped_haematoxylin_mask_batch: torch.Tensor
        Shape (B, 1, H, W), values in [0, 1], torch.float32
    sigma_px: float
        Boundary uncertainty model sigma
    mode: GateEdgesMode, default GateEdgesMode.BAND
        Gating mode: band or mask
    band_factor: float, default 1.96
        95% half-width
    tol_factor: float, default 1.96
        95% half-width
    display_in_clinical_convention: bool, default False
        Are we displaying in clinical convention?

    Returns
    -------
    CannyTolerantOverlayOutput
        torch.Tensor
            Overlay, shape: (B, 3, H, W)
        torch.Tensor
            MRI edges explained by histology, shape (B,)
        torch.Tensor
            Histology edges explained by MRI, shape (B,)
        torch.Tensor
            Symmetric coverage
    """
    # Normalise images for edge detection.
    mri = mri_batch / mri_batch[mri_mask_batch > 0.5].max()
    hist = (
        warped_haematoxylin_batch
        / warped_haematoxylin_batch[warped_haematoxylin_mask_batch > 0.5].max()
    )
    # Obtain Canny edges.
    canny = kornia.filters.Canny(low_threshold=0.2, high_threshold=0.8)
    mri_edges, _ = canny(mri)
    hist_edges, _ = canny(hist)
    # Calculate radii from sigma.
    r_band = max(1, math.ceil(band_factor * sigma_px))
    r_tol = max(1, math.ceil(tol_factor * sigma_px))
    # Gate edges.
    gated_mri_edges = _gate_edges(mri_edges, mri_mask_batch, r_band, mode)
    gated_hist_edges = _gate_edges(
        hist_edges, warped_haematoxylin_mask_batch, r_band, mode
    )
    # Perform tolerance-aware matching in both directions.
    mri_matches = _matches_within_tolerance(
        gated_mri_edges, gated_hist_edges, r_tol
    )
    hist_matches = _matches_within_tolerance(
        gated_hist_edges, gated_mri_edges, r_tol
    )
    matched = mri_matches | hist_matches
    # Create overlay: yellow = match, red/green = separate
    red = gated_mri_edges.clone()
    green = gated_hist_edges.clone()
    blue = torch.zeros_like(red)
    red[matched] = 1.0
    green[matched] = 1.0
    overlay = torch.stack(
        [red.squeeze(1), green.squeeze(1), blue.squeeze(1)], dim=1
    )  # shape: (B, 3, H, W)
    if display_in_clinical_convention:
        overlay = clinical_convention(overlay)
    overlay = overlay.detach().cpu()
    # Calculate coverage.
    eps = torch.finfo(torch.float32).eps
    mri_cov = mri_matches.float().sum(dim=(1, 2, 3)) / (
        gated_mri_edges.sum(dim=(1, 2, 3)) + eps
    )  # MRI edges explained by histology
    hist_cov = hist_matches.float().sum(dim=(1, 2, 3)) / (
        gated_hist_edges.sum(dim=(1, 2, 3)) + eps
    )  # Histology edges explained by MRI
    sym_cov = (2 * mri_cov * hist_cov) / (mri_cov + hist_cov + eps)
    # Done.
    return CannyTolerantOverlayOutput(
        overlay=overlay,
        mri_coverage=mri_cov,
        histology_coverage=hist_cov,
        symmetric_coverage=sym_cov,
    )


def clinical_convention(
    x: torch.Tensor | torch.Tensor,
) -> torch.Tensor | torch.Tensor:
    """
    Displays the assumed RAS orientation tensor in clinical convention.

    Parameters
    ----------
    x: torch.Tensor | torch.Tensor
        Shape (*_, H, W)

    Notes
    -----
    90º counter-clockwise rotation followed by a horizontal flip.
    """
    return torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])


def _mask_border_overlay(
    canvas_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    kernel_size: int,
    colour: RGB,
) -> torch.Tensor:
    """
    Draws a border from a binary mask onto the canvas.

    Parameters
    ----------
    canvas_batch: torch.Tensor
        Shape (B, 3, H, W), values in [0, 1], torch.float32
    mask_batch: torch.Tensor
        Shape (B, 1, H, W), values in {0, 1}, torch.float32
    kernel_size: int
        Roughly translates to border thickness.
    colour: RGB
        Border colour

    Returns
    -------
    torch.Tensor
        Shape (B, 3, H, W), values in [0, 1], torch.float32
    """
    # Create the border mask using morphological dilation.
    dev = canvas_batch.device
    kernel = torch.ones(kernel_size, kernel_size, device=dev)
    dilated_mask = morphology.dilation(mask_batch.float(), kernel)
    border_mask = (dilated_mask - mask_batch.float()).clamp(
        0.0, 1.0
    )  # Shape (B, 1, H, W)
    # Paint the border on the canvas.
    overlay = torch.where(
        border_mask.repeat(1, 3, 1, 1) == 1,
        torch.tensor(data=colour, dtype=torch.float32, device=dev).view(
            1, 3, 1, 1
        ),
        canvas_batch,
    )
    # Done!
    return overlay


def _disk_kernel(radius: int, device: torch.device) -> torch.Tensor:
    k = 2 * radius + 1
    yy, xx = torch.meshgrid(
        torch.arange(k, device=device),
        torch.arange(k, device=device),
        indexing="ij",
    )
    se = ((yy - radius) ** 2 + (xx - radius) ** 2) <= (radius**2)
    return se.float()  # shape (k, k)


def _gate_edges(
    edges: torch.Tensor, mask: torch.Tensor, r_band: int, mode: GateEdgesMode
) -> torch.Tensor:
    """
    Returns
    -------
    torch.Tensor
        torch.float32"""
    if mode == GateEdgesMode.BAND:
        # Build annulus.
        se = _disk_kernel(r_band, device=mask.device)
        dil = morphology.dilation(mask, se)
        ero = morphology.erosion(mask, se)
        band = (dil > 0.5) & (ero <= 0.5)
        band = band.float()
        # Gate.
        return (edges > 0.5).float() * band
    elif mode == GateEdgesMode.MASK:
        # Use mask directly.
        return (edges > 0.5).float() * (mask > 0.5).float()
    else:
        raise ValueError(f"unknown mode: {mode=}")


def _matches_within_tolerance(
    edges_A: torch.Tensor, edges_B: torch.Tensor, r_tol: int
) -> torch.Tensor:
    """
    Returns
    -------
    torch.Tensor
        Shape (B, 1, H, W), torch.bool"""
    se = _disk_kernel(r_tol, device=edges_A.device)
    B_dil = morphology.dilation((edges_B > 0.5).float(), se)
    return (edges_A > 0.5) & (B_dil > 0.5)  # shape: (B, 1, H, W)
