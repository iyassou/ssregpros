from ..core.type_definitions import Percentage
from ..models.regression import (
    shrinkage_range_to_STN_scale_range,
    RegressionTransformedParameters,
)

from dataclasses import dataclass

import torch


@dataclass
class RigidTransformRegularisationLossConfig:
    translation_weight: float
    scale_weight: float
    scale_log_prior_confidence_interval: Percentage
    regression_head_shrinkage_range: tuple[Percentage, Percentage]


class RigidTransformRegularisationLoss(torch.nn.modules.loss._Loss):
    """Penalises rigid transformations with excessive translations and scale
    predictions outside of the hypothesised range."""

    NORMAL_GAUSSIAN = torch.distributions.Normal(loc=0, scale=1)

    def __init__(self, config: RigidTransformRegularisationLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self, parameters: RegressionTransformedParameters
    ) -> torch.Tensor:
        """
        Computes the regularisation coefficient for the rigid transformation
        parameters outputted by the network for a particular prediction.

        Parameters
        ----------
        parameters: RegressionTransformedParameters
            Model's predicted parameters

        Notes
        -----
        We regularise the translation components to penalise large deviations
        from center of the image in normalised image coordinates.

        We regularise log(scale) as we're performing MAP with a log-normal
        prior on the scale.
        The prior is on log(scale) because that matches multiplicative
        physics and should then give better gradients when training.
        Our scale range is configured using the literature. For a scale range
        of [a, b] and CI 95%, we
            (1) set the centre to the geometric mean:
                    µ = log√(ab) = 0.5 * (log(a) + log(b))
            (2) set the spread so that `a` and `b` are the 2.5% and 97.5%
                points:
                    σ = (log(b) - log(a)) / (Φ^{-1}(0.975) - Φ^{-1}(0.025))
                      = (log(b) - log(a)) / (2 * Φ^{-1}(0.975)) because symmetry
                      ≈ (log(b) - log(a)) / 3.92

        We apply no regularisation to the rotation component since the
        predicted values already have unit norm.

        Returns
        -------
        torch.Tensor
            A single scalar value"""
        # Regularise translation components with a zero-value assumption.
        tx = parameters.tx
        ty = parameters.ty
        translation_reg = (tx**2).mean() + (ty**2).mean()  # shape: (B,)
        # Regularise scale component with a log prior.
        scale = parameters.scale
        a, b = shrinkage_range_to_STN_scale_range(
            self.config.regression_head_shrinkage_range
        )
        a = torch.Tensor([a])
        b = torch.Tensor([b])
        mu = 0.5 * (torch.log(a) + torch.log(b))
        percentile = (
            self.config.scale_log_prior_confidence_interval
            + (1 - self.config.scale_log_prior_confidence_interval) / 2
        )
        sigma = (torch.log(b) - torch.log(a)) / (
            2 * self.NORMAL_GAUSSIAN.icdf(torch.Tensor([percentile]))
        )
        log_scale = torch.log(scale)
        eps = torch.finfo(torch.float32).eps
        z = (log_scale - mu) / (sigma + eps)
        scale_reg: torch.Tensor = (z**2).mean()  # shape: (B,)
        # Combine.
        dev = parameters.tx.device  # wlog
        translation_weight, scale_weight = (
            self.config.translation_weight,
            self.config.scale_weight,
        )
        return (
            translation_weight * translation_reg + scale_weight * scale_reg
        ).to(dev)
