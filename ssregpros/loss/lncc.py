import torch
import torch.nn.functional as F


class MaskedLNCCLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        window_size: int = 11,
        min_coverage: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.window_size = window_size
        self.min_coverage = min_coverage
        self.dtype = dtype

    def epsilon(self, device: torch.device) -> torch.Tensor:
        return torch.Tensor([torch.finfo(self.dtype).eps]).to(device)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Local (over window) normalized cross correlation loss with masking.

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

        # Set window size (default 9x9)
        win_size = self.window_size
        pad_no = win_size // 2

        # Create sum filter
        sum_filt = torch.ones(
            [1, 1, win_size, win_size],
            device=y_true.device,
            dtype=self.dtype,
        )

        # Apply mask to inputs
        Ii = y_true * soft_mask
        Ji = y_pred * soft_mask

        # Compute squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # Sum over local windows using 2D convolution
        I_sum = F.conv2d(Ii, sum_filt, stride=(1, 1), padding=(pad_no, pad_no))
        J_sum = F.conv2d(Ji, sum_filt, stride=(1, 1), padding=(pad_no, pad_no))
        I2_sum = F.conv2d(I2, sum_filt, stride=(1, 1), padding=(pad_no, pad_no))
        J2_sum = F.conv2d(J2, sum_filt, stride=(1, 1), padding=(pad_no, pad_no))
        IJ_sum = F.conv2d(IJ, sum_filt, stride=(1, 1), padding=(pad_no, pad_no))

        # Sum of mask over windows (to account for valid pixels in each window)
        mask_sum = F.conv2d(
            soft_mask, sum_filt, stride=(1, 1), padding=(pad_no, pad_no)
        )
        mask_sum = torch.clamp(mask_sum, min=1e-5)

        # Compute means
        u_I = I_sum / mask_sum
        u_J = J_sum / mask_sum

        # Compute cross correlation and variances
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * mask_sum
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * mask_sum
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * mask_sum

        # Compute normalized cross correlation
        cc = cross * cross / (I_var * J_var + 1e-5)

        # Apply mask to correlation map for final averaging
        cc_masked = cc * soft_mask

        return -torch.mean(cc_masked)
