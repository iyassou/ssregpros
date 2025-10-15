import kornia
import torch


class MaskedMeanSquaredGradientErrorLoss(torch.nn.modules.loss._Loss):
    """
    Computes the Mean Squared Error of the Sobel gradients of two images
    along a binary mask.
    """

    def __init__(self):
        super().__init__()

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
            Ground truth, shape (N, C, H, W)
        y_pred: torch.Tensor
            Model prediction, shape (N, C, H, W)
        mask: torch.Tensor
            Mask, shape (N, 1, H, W)

        Returns
        -------
        torch.Tensor
            A single scalar loss value"""
        # Ensure mask is broadcastable and contains at least one nonzero entry.
        if not torch.any(mask):
            raise ValueError("mask is empty!")
        if mask.shape[1] != y_true.shape[1]:
            mask = mask.expand_as(y_true)

        # Obtain Sobel gradient magnitudes.
        dtype = torch.float32
        eps = torch.finfo(dtype).eps
        grad_true = kornia.filters.sobel(
            y_true.type(dtype), normalized=True, eps=eps
        )  # shape: (N, C, H, W)
        grad_pred = kornia.filters.sobel(
            y_pred.type(dtype), normalized=True, eps=eps
        )  # shape: (N, C, H, W)

        # Calculate squared error, apply mask.
        sq_err = (grad_true - grad_pred).pow(2)  # shape: (N, C, H, W)
        if mask.dtype is torch.bool:
            mask = mask.float()
        sq_err *= mask

        # Compute mean over masked region per-sample, then batch mean
        denom = mask.sum(dim=(1, 2, 3)).type(sq_err.dtype)
        loss_per_sample = sq_err.sum(dim=(1, 2, 3)) / denom  # shape: (N,)
        loss = loss_per_sample.mean()  # shape: ()

        return loss


MaskedMSGELoss = MaskedMeanSquaredGradientErrorLoss
