import torch
import torch.nn as nn
import torch.nn.functional as F


class PrincipalAxisLoss(nn.Module):
    """
    Computes a loss that encourages two masks to have the same principal orientation.

    This is achieved by calculating the orientation angle of the primary eigenvector
    of the spatial covariance matrix of each mask's coordinates and penalizing
    the difference in orientation.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def _get_orientation_vector(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates a 2D vector representing the principal orientation of a mask.
        """
        # Ensure mask is float [0, 1]
        mask = mask.float()
        # Get shape and create coordinate grids
        n, _, h, w = mask.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=mask.device, dtype=mask.dtype),
            torch.arange(w, device=mask.device, dtype=mask.dtype),
            indexing="ij",
        )
        # Calculate spatial moments (zeroth and first order)
        m00 = torch.sum(mask, dim=[1, 2, 3]) + self.epsilon  # Area
        m10 = torch.sum(x_coords * mask, dim=[1, 2, 3])
        m01 = torch.sum(y_coords * mask, dim=[1, 2, 3])
        # Calculate centroids
        cx = m10 / m00
        cy = m01 / m00
        # Calculate second-order central moments (covariance matrix components)
        # Reshape cx, cy for broadcasting
        cx = cx.view(n, 1, 1, 1)
        cy = cy.view(n, 1, 1, 1)

        mu20 = torch.sum(((x_coords - cx) ** 2) * mask, dim=[1, 2, 3])
        mu02 = torch.sum(((y_coords - cy) ** 2) * mask, dim=[1, 2, 3])
        mu11 = torch.sum(
            (x_coords - cx) * (y_coords - cy) * mask, dim=[1, 2, 3]
        )
        # Calculate orientation angle theta using the central moments
        # The angle is 0.5 * atan2(2*mu11, mu20 - mu02)
        angle = 0.5 * torch.atan2(2 * mu11, mu20 - mu02)
        # Return a 2D unit vector representing this orientation
        orientation_vector = torch.stack(
            [torch.cos(angle), torch.sin(angle)], dim=1
        )
        return orientation_vector

    def forward(
        self, y_true_mask: torch.Tensor, y_pred_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_true_mask (torch.Tensor): The target mask (e.g., MRI).
            y_pred_mask (torch.Tensor): The predicted mask (e.g., warped histology).
        """
        axis_true = self._get_orientation_vector(y_true_mask)
        axis_pred = self._get_orientation_vector(y_pred_mask)

        # Calculate cosine similarity. We want the axes to be parallel, so
        # the similarity should be close to 1 or -1. We penalize the deviation
        # of the absolute similarity from 1.
        # Loss = 1 - |cos(theta)|
        cosine_sim_abs = F.cosine_similarity(axis_true, axis_pred, dim=1).abs()

        return torch.mean(1.0 - cosine_sim_abs)
