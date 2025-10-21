from . import Reduction

import torch


class MaskedNCCLoss(torch.nn.modules.loss._Loss):
    """
    Computes the masked normalised cross-correlation between two tensors.
    NCC is a similarity metric (higher is better), so we've set the loss to
    0.5 * (1 - NCC) to obtain "0 good, 1 bad".
    Calculations are performed only on the pixels where the mask is nonzero.

    The NCC can be weighted by one of:
        - nothing, so just an average over each batch
        - the number of pixels per mask
        - sqrt(the number of pixels per mask)
        - log(the number of pixels per mask)
    """

    def __init__(
        self,
        reduction: Reduction = Reduction.MEAN,
        dtype: torch.dtype = torch.float32,
        anticorrelated: bool = False,
    ):
        super().__init__(reduction=reduction)
        self.dtype = dtype
        self.epsilon_value = 1e-8
        self.anticorrelated = anticorrelated

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_true: torch.Tensor
            Ground truth, shape (B, C, H, W)
        y_pred: torch.Tensor
            Model prediction, shape (B, C, H, W)
        mask: torch.Tensor
            Binary or soft mask, shape (B, 1, H, W), dtype torch.float32

        Returns
        -------
        torch.Tensor
            A single scalar loss value, unless reduction is Reduction.NONE
        """
        # Ensure mask is broadcastable and contains at least one nonzero entry.
        if not torch.any(mask):
            raise ValueError("mask is empty!")
        if mask.shape[1] != y_true.shape[1]:
            mask = mask.expand(-1, y_true.shape[1], -1, -1)

        # Calculate the number of masked pixels per channel.
        num_pixels = mask.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        num_pixels_safe = torch.clamp_min(num_pixels, 1.0)

        # Calculate the masked mean.
        sum_true = (mask * y_true).sum(dim=(2, 3), keepdim=True)
        sum_pred = (mask * y_pred).sum(dim=(2, 3), keepdim=True)

        mu_true = sum_true / num_pixels_safe
        mu_pred = sum_pred / num_pixels_safe

        # Calculate masked centered values.
        centered_true = mask * (y_true - mu_true)
        centered_pred = mask * (y_pred - mu_pred)

        # Calculate covariance and variances in one pass
        # This avoids computing centered values multiple times
        cov = (centered_true * centered_pred).sum(dim=(2, 3), keepdim=True)
        var_true = (centered_true * centered_true).sum(dim=(2, 3), keepdim=True)
        var_pred = (centered_pred * centered_pred).sum(dim=(2, 3), keepdim=True)

        # Compute NCC with improved numerical stability
        # Use the property that NCC = cov / (std_true * std_pred)
        # But compute it as: cov / sqrt(var_true * var_pred)
        # with careful handling of the sqrt
        # Method 1: Clamp before sqrt (more stable)
        denominator = torch.sqrt(
            torch.clamp_min(var_true, self.epsilon_value)
            * torch.clamp_min(var_pred, self.epsilon_value)
        )

        # Add epsilon to denominator for safety
        ncc = cov / (denominator + self.epsilon_value)

        # Clamp to valid NCC range
        ncc = torch.clamp(ncc, -1.0, 1.0)

        # Squeeze spatial dimensions but keep channel dimension for averaging
        ncc = ncc.squeeze(-1).squeeze(-1)  # (B, C)
        # Average across channels
        ncc = ncc.mean(dim=1)  # (B,)

        # Apply reduction based on support size
        weight_mass = num_pixels[:, 0, 0, 0]  # (B,)

        if self.reduction == Reduction.NONE:
            if self.anticorrelated:
                return 0.5 * (1.0 + ncc)
            return 0.5 * (1.0 - ncc)

        if self.reduction == Reduction.MEAN:
            reduction_weights = weight_mass
        elif self.reduction == Reduction.SQRT:
            reduction_weights = torch.sqrt(weight_mass)
        elif self.reduction == Reduction.LOG:
            reduction_weights = torch.log(torch.clamp_min(weight_mass, 1.0))
        else:
            raise ValueError("unknown reduction")

        # Normalize weights
        reduction_weights = reduction_weights / (
            reduction_weights.sum() + 1e-12
        )

        if self.anticorrelated:
            return 0.5 * (1.0 + (ncc * reduction_weights).sum())
        return 0.5 * (1.0 - (ncc * reduction_weights).sum())
