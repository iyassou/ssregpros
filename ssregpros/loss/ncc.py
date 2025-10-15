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
        self.epsilon_value = torch.finfo(dtype).eps
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
            Binary or soft mask, shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor
            A single scalar loss value
        """
        # Ensure mask is broadcastable and contains at least one nonzero entry.
        if not torch.any(mask):
            raise ValueError("mask is empty!")
        # > Build soft weights in [0,1], broadcast to channels.
        if mask.dtype == torch.bool:
            soft_mask = mask.to(self.dtype)
        else:
            soft_mask = mask.to(self.dtype).clamp(0.0, 1.0)
        if soft_mask.shape[1] != y_true.shape[1]:
            soft_mask = soft_mask.expand(-1, y_true.shape[1], -1, -1)

        # Weighted sums
        # > Calculate the number of masked pixels per channel.
        num_pixels = soft_mask.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        # > Avoid division by zero with out-of-place operation
        num_pixels_safe = num_pixels + (num_pixels == 0).to(self.dtype)

        # > Calculate the masked mean.
        mu_true = (soft_mask * y_true).sum(
            dim=(2, 3), keepdim=True
        ) / num_pixels_safe
        mu_pred = (soft_mask * y_pred).sum(
            dim=(2, 3), keepdim=True
        ) / num_pixels_safe

        # > Calculate masked centered values.
        centered_true = y_true - mu_true
        centered_pred = y_pred - mu_pred

        # > Calculate masked standard deviation.
        # > var = E[(X - µ)^2]
        var_true = (soft_mask * centered_true.pow(2)).sum(
            dim=(2, 3), keepdim=True
        ) / num_pixels_safe
        var_pred = (soft_mask * centered_pred.pow(2)).sum(
            dim=(2, 3), keepdim=True
        ) / num_pixels_safe

        # > Calculate NCC numerator:
        # > E[(X - µ_x) * (Y - µ_y)]
        numerator = (soft_mask * centered_true * centered_pred).sum(
            dim=(2, 3), keepdim=True
        ) / num_pixels_safe

        # > Calculate NCC denominator:
        # > σ_x × σ_y + epsilon
        denominator = (
            var_true.clamp_min(0) * var_pred.clamp_min(0)
        ).sqrt() + self.epsilon_value

        # > Calculate NCC per channel.
        ncc = numerator / denominator
        # > Squeeze spatial dimensions but keep channel dimension for averaging
        ncc = ncc.squeeze(-1).squeeze(-1)  # (B, C)
        # > Average across channels
        ncc = ncc.mean(dim=1)  # (B,)
        # huh
        if self.anticorrelated:
            ncc = -1 * ncc

        # Apply reduction based on support size (out-of-place operations only)
        # Use the total weight mass per sample
        weight_mass = num_pixels[:, 0, 0, 0]  # (B,)

        if self.reduction == Reduction.NONE:
            return 0.5 * (1.0 - ncc.mean())

        if self.reduction == Reduction.MEAN:
            reduction_weights = weight_mass
        elif self.reduction == Reduction.SQRT:
            reduction_weights = weight_mass.sqrt()
        elif self.reduction == Reduction.LOG:
            reduction_weights = weight_mass.clamp_min(1.0).log()
        else:
            raise ValueError("unknown reduction")

        # > Normalize weights (out-of-place division)
        reduction_weights_sum = reduction_weights.sum() + 1e-12
        reduction_weights = reduction_weights / reduction_weights_sum

        return 0.5 * (1.0 - (ncc * reduction_weights).sum())
