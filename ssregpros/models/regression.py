from ..core.type_definitions import (
    Percentage,
    PositiveFloat,
    ScalingFactor,
    assert_annotated_type,
)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import NamedTuple


def shrinkage_range_to_STN_scale_range(
    shrinkage_range: tuple[Percentage, Percentage],
) -> tuple[ScalingFactor, ScalingFactor]:
    """
    Converts the physical shrinkage range to the scale range used by the
    Spatial Transformer Network.
    """
    # Calculate STN scale ~ 1 / (1 - shrinkage).
    pmin, pmax = shrinkage_range
    s_min = 1.0 / (1.0 - pmin)
    s_max = 1.0 / (1.0 - pmax)
    assert_annotated_type(s_min, ScalingFactor, ValueError(f"{s_min = }"))
    assert_annotated_type(s_max, ScalingFactor, ValueError(f"{s_max = }"))
    return s_min, s_max


@dataclass
class RegressionHeadConfig:
    num_input_features: int
    bottleneck_layer_size: int
    shrinkage_range: tuple[Percentage, Percentage]

    noise_weight: PositiveFloat = 1.0
    translation_noise_std: PositiveFloat = 0.005
    rotation_noise_std: PositiveFloat = math.radians(1.5)
    log_scale_noise_std: PositiveFloat = 0.005

    device: torch.device = torch.device("cpu")


class RegressionTransformedParameters(NamedTuple):
    tx: torch.Tensor
    ty: torch.Tensor
    scale: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


class RegressionHead(nn.Module):
    def __init__(self, config: RegressionHeadConfig):
        super().__init__()
        self.config = config
        self.regressor = nn.Sequential(
            nn.Linear(
                in_features=config.num_input_features,
                out_features=config.bottleneck_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=config.bottleneck_layer_size,
                out_features=len(RegressionTransformedParameters._fields),
            ),
        ).to(config.device)
        self.initialise_weights()

    def initialise_weights(self):
        """
        Makes the final layer initially predict a sensible transform.

        Notes
        -----
        "Sensible" here means predicting centred coordinates, a scale that's
        exactly the midpoint of the given range, and no rotation.

        Actually insane how crucial you are.
        """
        # Zero out weights.
        self.regressor[-1].weight.data.zero_()  # type: ignore
        # Initialise biases for "sensible" transform.
        with torch.no_grad():
            bias: torch.Tensor = self.regressor[-1].bias.data  # type: ignore
            bias[0] = 0.0  # tx = 0
            bias[1] = 0.0  # ty = 0
            bias[2] = 0.0  # scale = midpoint
            bias[3] = 1.0  # cos = 1
            bias[4] = 0.0  # sin = 0

    def _inject_noise(
        self, parameters: RegressionTransformedParameters
    ) -> RegressionTransformedParameters:
        if (not self.training) or (not (weight := self.config.noise_weight)):
            return parameters
        B = parameters.cos.size(0)  # wlog
        dev = self.config.device
        # Translation.
        if (std_t := self.config.translation_noise_std) > 0:
            std_t *= weight
            tx = parameters.tx + torch.randn(B, device=dev) * std_t
            ty = parameters.ty + torch.randn(B, device=dev) * std_t
        else:
            tx = parameters.tx
            ty = parameters.ty
        # Rotation
        if (std_rot := self.config.rotation_noise_std) > 0:
            std_rot *= weight
            delta = torch.randn(B, device=dev) * std_rot
            cos_delta, sin_delta = torch.cos(delta), torch.sin(delta)
            cos = cos_delta * parameters.cos - sin_delta * parameters.sin
            sin = sin_delta * parameters.cos + cos_delta * parameters.sin
        else:
            cos = parameters.cos
            sin = parameters.sin
        # Scale
        if (std_log_scale := self.config.log_scale_noise_std) > 0:
            std_log_scale *= weight
            eps = torch.randn(B, device=dev) * std_log_scale
            scale = parameters.scale * torch.exp(eps)
        else:
            scale = parameters.scale
        # Done.
        return parameters._replace(tx=tx, ty=ty, scale=scale, cos=cos, sin=sin)

    def forward(
        self, feature_vector: torch.Tensor
    ) -> RegressionTransformedParameters:
        """Regression head forward pass, complete with parameter transformation.

        Parameters
        ----------
        feature_vector: torch.Tensor
            Shape (B, self.config.num_input_features)

        Notes
        -----
        The network is outputting five parameters:
            - translation along the X axis
            - translation along the Y axis
            - a scaling factor
            - an unconstrained 2D rotation vector i.e. 2 components

        The translations along the X and Y axis are given in a normalised
        coordinate space where `[-1, 1]` corresponds to the edges of the image.
        To this end, both translation values are `torch.tanh`d to normalise
        them to said interval. Note that `tanh(0) = 0`, desired behaviour.

        TODO: write about the scaling factor here

        The unconstrained 2D rotation vector is normalised to have unit length as
        per the work of Zhou et al. (2018) "On the Continuity of Rotation
        Representations in Neural Networks", to ensure the rotation vector is
        constrained to the unit circle.

        Returns
        -------
        RegressionTransformedParameters"""
        dev = self.config.device
        # Obtain raw parameters.
        raw_parameters = self.regressor(feature_vector.to(dev))  # shape: (B, 5)
        # Retrieve raw translation, rotation, and scaling parameters.
        raw_tx, raw_ty, raw_scale = raw_parameters[:, :3].T
        # Transform translation.
        tx = torch.tanh(raw_tx.to(dev))
        ty = torch.tanh(raw_ty.to(dev))
        # Transform scale.
        mi, ma = shrinkage_range_to_STN_scale_range(self.config.shrinkage_range)
        scale = mi + torch.sigmoid(raw_scale) * (ma - mi)
        scale = scale.to(dev)
        # Normalise the unconstrained 2D rotation vector and extract sine, cosine.
        rot_vec = raw_parameters[:, 3:]  # shape: (N, 2)
        rot_vec_normalised = F.normalize(rot_vec, p=2, dim=1)
        cos, sin = rot_vec_normalised.T
        cos = cos.to(dev)
        sin = sin.to(dev)
        # Optionally inject noise onto output.
        parameters = self._inject_noise(
            RegressionTransformedParameters(
                tx=tx,
                ty=ty,
                scale=scale,
                cos=cos,
                sin=sin,
            )
        )
        # Done.
        return parameters
