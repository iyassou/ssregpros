from .regression import RegressionTransformedParameters

import torch
import torch.nn.functional as F


class SpatialTransformerNetwork(torch.nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.device = device

    def output_size(self, batch_size: int, num_channels: int) -> torch.Size:
        return torch.Size((batch_size, num_channels, self.height, self.width))

    def build_theta(
        self, params: RegressionTransformedParameters
    ) -> torch.Tensor:
        """
        Builds the (2, 3) affine matrix `theta` from the regressed
        transformed parameters.

        Parameters
        ----------
        params: RegressionTransformedParameters
            Predicted similarity transform parameters, each with shape (B,)

        Returns
        -------
        torch.Tensor
            Affine transformation matrix `theta`, shape (B, 2, 3)
        """
        # Build A = R S, where R is the rotation matrix and S the scaling matrix.
        scale = params.scale
        cos = params.cos
        sin = params.sin
        B = scale.size(0)  # wlog
        A = torch.zeros(B, 2, 2, device=self.device)
        A[:, 0, 0] = scale * cos
        A[:, 0, 1] = scale * -sin
        A[:, 1, 0] = scale * sin
        A[:, 1, 1] = scale * cos
        # Create the translation vector.
        t = torch.stack((params.tx, params.ty), dim=1)  # shape: (B, 2)
        # θ := [A | t]
        return torch.cat([A, t.unsqueeze(-1)], dim=2)  # shape: (B, 2, 3)

    def forward(
        self, x: torch.Tensor, theta: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """Takes in the images to be warped, as well as their affine
        transformation matrices.

        Parameters
        ----------
        x: torch.Tensor
            Input images, shape (B, C, H, W), numeric type
        theta: torch.Tensor
            Affine transformation matrices, shape (B, 2, 3)

        Returns
        -------
        torch.Tensor
            Warped batch of images resampled to the fixed output size,
            shape (B, C, self.height, self.width)
        """
        # Calculate output size.
        assert x.ndim == 4
        batch_size = x.size(0)
        num_channels = x.size(1)
        output_size = self.output_size(batch_size, num_channels)
        # Generate the sampling grid.
        grid = F.affine_grid(
            theta,
            output_size,  # pyright: ignore[reportArgumentType]
            align_corners=False,
        )
        # Sample the input image using the grid.
        warped_x = F.grid_sample(
            x, grid, mode=mode, padding_mode="zeros", align_corners=False
        )
        return warped_x
