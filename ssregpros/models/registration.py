from ..core import set_deterministic_seed
from ..core.type_definitions import Percentage
from .feature_encoder import (
    ResNetFeatureEncoder,
    swap_batchnorm2d_for_instancenorm2d,
)
from .regression import (
    RegressionHead,
    RegressionHeadConfig,
    RegressionTransformedParameters,
)
from .spatial_transformer import SpatialTransformerNetwork

from monai.data.meta_tensor import MetaTensor


import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import NamedTuple


class RegistrationNetworkOutput(NamedTuple):
    warped_histology_batch: MetaTensor
    warped_histology_mask_batch: MetaTensor
    parameters: RegressionTransformedParameters
    thetas: torch.Tensor


@dataclass
class RegistrationNetworkConfig:
    """Registration network configuration dataclass"""

    seed: int
    height: int = 128
    width: int = 128
    model_name: str = "resnet18"
    regression_head_bottleneck_layer_size: int = 256
    regression_head_shrinkage_range: tuple[Percentage, Percentage] = (
        0.05,
        0.35,
    )
    device: torch.device = torch.device("cpu")


class RegistrationNetwork(nn.Module):
    """Predicts histology-to-MRI rigid transformation parameters."""

    def __init__(self, config: RegistrationNetworkConfig):
        set_deterministic_seed(config.seed)
        super().__init__()

        dev = config.device
        self.config = config

        # A)    Feature Encoders with fixed-sized outputs.
        #       NOTE:   swap out histology encoder's `BatchNorm2d`
        #               layers for `InstanceNorm2d` layers.
        self.mri_encoder = ResNetFeatureEncoder(
            model_name=config.model_name
        ).to(dev)
        self.histology_encoder: ResNetFeatureEncoder = (
            swap_batchnorm2d_for_instancenorm2d(
                ResNetFeatureEncoder(model_name=config.model_name)
            )
        ).to(
            dev
        )  # pyright: ignore[reportAttributeAccessIssue]

        # B)    Regression Head
        self.regression_head = RegressionHead(
            RegressionHeadConfig(
                num_input_features=(
                    self.mri_encoder.output_size()
                    + self.histology_encoder.output_size()
                ),
                bottleneck_layer_size=self.config.regression_head_bottleneck_layer_size,
                shrinkage_range=self.config.regression_head_shrinkage_range,
                device=dev,
            )
        ).to(dev)

        # C)    Spatial Transformer Network
        self.stn = SpatialTransformerNetwork(
            height=self.config.height,
            width=self.config.width,
            device=self.config.device,
        ).to(dev)

    def predict_rigid_transform_parameters(
        self, mri_batch: MetaTensor, histology_list: list[MetaTensor]
    ) -> RegressionTransformedParameters:
        r"""Takes in the batched MRI images and the corresponding histology
        images and returns the rigid transform parameters.

        Parameters
        ----------
        mri_batch: MetaTensor
            Batch MRI tensors, shape: (B, 1, `self.config.height`, `self.config.width`)
        histology_list: list[MetaTensor]
            List of variable-sized histology images with shape (C, $H_i$, $W_i$)
            and length B

        Returns
        -------
        RegressionTransformedParameters"""
        # Predict parameters.
        mri_feature_vector: torch.Tensor = self.mri_encoder(mri_batch)
        histology_feature_vector = torch.cat(
            [
                self.histology_encoder(
                    hist.unsqueeze(0)  # adding batch dimension
                )
                for hist in histology_list
            ],
            dim=0,
        )
        combined_feature_vector = torch.cat(
            (mri_feature_vector, histology_feature_vector), dim=1
        )  # shape: (B, mri_encoder.output_size() + histology_encoder.output_size())
        parameters: RegressionTransformedParameters = self.regression_head(
            combined_feature_vector
        )  # shape ~ (B, 5)
        # Done!
        return parameters

    def apply_rigid_transform(
        self,
        thetas: torch.Tensor,
        tensors: list[MetaTensor],
        grid_sampling_mode: str,
    ) -> MetaTensor:
        """Applies a rigid transformation to the given tensors and returns
        their batched output.

        Parameters
        ----------
        thetas: torch.Tensor
            Shape: (B, 2, 3)
        tensors: list[MetaTensor]
            Length B, each variable-sized with shape (C, $H_i$, $W_i$),
            dtype torch.float32
        grid_sampling_mode: str
            Grid sampling mode used by the spatial transformer network

        Notes
        -----
        Assumes `thetas` was provided by `self.predict_rigid_transform_theta`
        called on the same tensors.

        Returns
        -------
        MetaTensor
            Warped batch, shape (B, C, `self.config.height`, `self.config.weight`)
        """
        if not tensors:
            raise ValueError("received empty list of tensors")
        # Loop through histology and apply the spatial transformer network.
        warped_histology = []
        for i, hist in enumerate(tensors):
            hist = hist.unsqueeze(0)  # shape: (1, C, H_i, W_i)
            theta = thetas[i : i + 1, :, :]  # shape: (1, 2, 3)
            warped_hist = self.stn(
                hist, theta, mode=grid_sampling_mode
            )  # shape: (1, 1, config.height, config.width)
            warped_histology.append(warped_hist)
        # Batch warped histology images together.
        warped_histology_batch = torch.cat(
            warped_histology, dim=0
        )  # shape: (B, 1, config.height, config.width)
        # Done!
        return warped_histology_batch  # pyright: ignore[reportReturnType]

    def forward(
        self,
        mri_batch: MetaTensor,
        histology_list: list[MetaTensor],
        histology_mask_list: list[MetaTensor],
    ) -> RegistrationNetworkOutput:
        """
        Takes in the batched MRI tensor, the associated histology list, the
        corresponding histology mask list, and returns the network's
        prediction, including the batched warped histology and histology masks.

        Parameters
        ----------
        mri_batch: MetaTensor
            Batched MRI tensors, shape: (B, 1, `self.config.height`, `self.config.width`)
        histology_list: list[MetaTensor]
            Length B list of histology samples with varying dimensions, each
            with shape (1, H_i, W_i), dtype torch.float32
        histology_mask_list: list[MetaTensor]
            Length B list of histology masks with varying dimensions, each with
            shape (1, H_i, W_i), dtype torch.float32

        Returns
        -------
        RegistrationNetworkOutput
        """
        # Obtain rigid transform parameters.
        parameters = self.predict_rigid_transform_parameters(
            mri_batch, histology_list
        )
        # Build (B, 2, 3) rigid transform matrices.
        thetas = self.stn.build_theta(parameters)
        # Apply rigid transform matrices.
        warped_histology_batch = self.apply_rigid_transform(
            thetas, tensors=histology_list, grid_sampling_mode="bilinear"
        )
        warped_histology_mask_batch = self.apply_rigid_transform(
            thetas, tensors=histology_mask_list, grid_sampling_mode="bilinear"
        )
        return RegistrationNetworkOutput(
            warped_histology_batch=warped_histology_batch,
            warped_histology_mask_batch=warped_histology_mask_batch,
            parameters=parameters,
            thetas=thetas,
        )
