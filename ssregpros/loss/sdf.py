from kornia.contrib import distance_transform

import torch


def differentiable_signed_distance_transform(mask: torch.Tensor):
    """
    Differentiable approximation of the signed distance transform.

    Notes
    -----
    SciPy's `distance_transform_edt` and Kornia's operate in opposite ways,
    hence why the distance to the foreground and background here are computed
    using `mask` and `1 - mask` respectively, whereas the SciPy version's
    use `1 - mask` and `mask` respectively.
    """
    binary = (mask > 0.5).float()
    distance_to_foreground = distance_transform(binary)
    distance_to_background = distance_transform(1.0 - binary)
    sdt = distance_to_foreground - distance_to_background
    return sdt


class SignedDistanceFieldMSELoss(torch.nn.modules.loss._Loss):
    """
    Computes the Mean Squared Error between two signed distance fields.
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self, mri_mask_sdt: torch.Tensor, warped_haematoxylin_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mri_mask_sdt: torch.Tensor
            SDT of the MRI's binary mask.
            Shape (B, 1, H, W)
        warped_haematoxylin_mask: torch.Tensor
            Soft, warped haemaotxylin (read "whole") mask, values in [0, 1].
            Shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            A single scalar loss value
        """
        warped_haematoxylin_mask_sdt = differentiable_signed_distance_transform(
            warped_haematoxylin_mask
        )
        loss = self.mse_loss(mri_mask_sdt, warped_haematoxylin_mask_sdt)
        return loss
