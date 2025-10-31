from ..core.type_definitions import (
    PositiveFloat,
    StrictlyPositiveFloat,
    assert_annotated_type,
)

import torch


class HingedMaskLeakageLoss(torch.nn.modules.loss._Loss):
    """
    Calculates a hinged leakage loss using a signed distance field (SDF).
    Penalises a predicted mask for "leaking" outside a target region defined
    by an SDF, but only when the leakage exceeds a specified tolerance.
    """

    def __init__(self, epsilon: StrictlyPositiveFloat, delta: PositiveFloat):
        super().__init__()
        assert_annotated_type(
            epsilon, StrictlyPositiveFloat, ValueError(f"{epsilon=}")
        )
        assert_annotated_type(delta, PositiveFloat, ValueError(f"{delta=}"))
        self.epsilon = epsilon
        self.delta = delta

    def forward(
        self,
        mri_mask_sdt: torch.Tensor,
        warped_haematoxylin_mask: torch.Tensor,
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
        # Calculate the distance-weighted leakage.
        # NOTE: non-zero only where the histology mask exists & SDT > 0.
        leakage_map = warped_haematoxylin_mask * mri_mask_sdt
        # Calculate the mean leakage over all pixels in each batch sample.
        # NOTE: mean per-sample, later averaged over the batch.
        A_out = torch.mean(leakage_map, dim=[1, 2, 3])  # shape (B,)
        # Apply the hinge loss function.
        loss_per_sample = torch.zeros_like(A_out)
        if self.delta > 0:
            # NOTE: piecewise function, 0 then quadratic then linear.
            # Calculate quadratic region's loss.
            quadratic_region = (A_out > self.epsilon) & (
                A_out <= self.epsilon + self.delta
            )
            loss_per_sample[quadratic_region] = (
                A_out[quadratic_region] - self.epsilon
            ).pow(2) / (2 * self.delta)
        # Calculate linear region's loss.
        linear_region = A_out > self.epsilon + self.delta
        loss_per_sample[linear_region] = (
            A_out[linear_region] - self.epsilon - (self.delta / 2)
        )
        # Done! Return batch mean.
        return torch.mean(loss_per_sample)
