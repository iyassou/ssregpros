from ..core import set_deterministic_seed
from ..core.type_definitions import Percentage
from .feature_encoder import (
    ResNetVariant,
    ResNetFeatureEncoder,
    swap_batchnorm2d_for_instancenorm2d,
)
from .regression import (
    RegressionHead,
    RegressionHeadConfig,
    RegressionTransformedParameters,
)
from .spatial_transformer import SpatialTransformerNetwork


import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import NamedTuple


class RegistrationNetworkOutput(NamedTuple):
    warped_haematoxylin: torch.Tensor
    warped_haematoxylin_mask: torch.Tensor
    parameters: RegressionTransformedParameters
    thetas: torch.Tensor


@dataclass
class RegistrationNetworkConfig:
    seed: int
    height: int
    width: int
    resnet: ResNetVariant
    regression_head_bottleneck_layer_size: int
    regression_head_shrinkage_range: tuple[Percentage, Percentage]
    device: torch.device = torch.device("cpu")


class RegistrationNetwork(nn.Module):
    """Predicts histology-to-MRI rigid transformation parameters."""

    def __init__(self, config: RegistrationNetworkConfig):
        set_deterministic_seed(config.seed)
        super().__init__()

        dev = config.device
        self.config = config

        # A)    Feature Encoders with fixed-sized outputs.
        #       NOTE:   swap out haematoxylin encoder's `BatchNorm2d`
        #               layers for `InstanceNorm2d` layers.
        self.mri_encoder = ResNetFeatureEncoder(var=config.resnet).to(dev)
        self.haematoxylin_encoder: ResNetFeatureEncoder = (
            swap_batchnorm2d_for_instancenorm2d(
                ResNetFeatureEncoder(var=config.resnet)
            )
        ).to(
            dev
        )  # pyright: ignore[reportAttributeAccessIssue]

        # B)    Regression Head
        self.regression_head = RegressionHead(
            RegressionHeadConfig(
                num_input_features=(
                    self.mri_encoder.output_size()
                    + self.haematoxylin_encoder.output_size()
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
        self,
        mri_batch: torch.Tensor,
        haematoxylin_list: list[torch.Tensor],
    ) -> RegressionTransformedParameters:
        """
        Takes in the batched MRI images and the corresponding haematoxylin
        images and returns the rigid transform parameters.

        Parameters
        ----------
        mri_batch: torch.Tensor
            Batched MRIs, shape: (B, 1, `self.config.height`, `self.config.width`)
        haematoxylin_list: list[torch.Tensor]
            List of variable-sized haematoxylin images with shape (C, H_i, W_i)
            and length B

        Returns
        -------
        RegressionTransformedParameters
        """
        # Build concatenated feature vector using both modalities.
        mri_feature_vector: torch.Tensor = self.mri_encoder(mri_batch)
        haematoxylin_feature_vector = torch.cat(
            [
                self.haematoxylin_encoder(h.unsqueeze(0))  # adding batch dim
                for h in haematoxylin_list
            ],
            dim=0,
        )
        combined_feature_vector = torch.cat(
            (mri_feature_vector, haematoxylin_feature_vector),
            dim=1,
        )  # shape: (B, mri_encoder.output_size() + haematoxylin_encoder.output_size())
        # Predict parameters.
        parameters: RegressionTransformedParameters = self.regression_head(
            combined_feature_vector
        )  # shape ~ (B, 5)
        # Done!
        return parameters

    def apply_rigid_transform(
        self,
        thetas: torch.Tensor,
        tensors: list[torch.Tensor],
        grid_sampling_mode: str,
    ) -> torch.Tensor:
        """Applies a rigid transformation to the given tensors and returns
        their batched output.

        Parameters
        ----------
        thetas: torch.Tensor
            Shape: (B, 2, 3)
        tensors: list[torch.Tensor]
            Length B, each variable-sized with shape (1, H_i, W_i),
            dtype torch.float32
        grid_sampling_mode: str
            Grid sampling mode used by the spatial transformer network

        Notes
        -----
        Assumes `thetas` was provided by `self.predict_rigid_transform_theta`
        called on the same tensors.

        Returns
        -------
        torch.Tensor
            Warped batch, shape (B, 1, `self.config.height`, `self.config.weight`)
        """
        if not tensors:
            raise ValueError("received empty list of tensors")
        # Loop through tensors and apply the spatial transformer network.
        warped_tensors = []
        for i, tensor in enumerate(tensors):
            tensor = tensor.unsqueeze(0)  # shape: (1, 1, H_i, W_i)
            theta = thetas[i : i + 1, :, :]  # shape: (1, 2, 3)
            warped_tensor = self.stn(
                tensor, theta, mode=grid_sampling_mode
            )  # shape: (1, 1, config.height, config.width)
            warped_tensors.append(warped_tensor)
        # Batch warped tensors together.
        warped_tensors_batch = torch.cat(
            warped_tensors, dim=0
        )  # shape: (B, 1, config.height, config.width)
        # Done!
        return warped_tensors_batch  # pyright: ignore[reportReturnType]

    def forward(
        self,
        mri_batch: torch.Tensor,
        haematoxylin_list: list[torch.Tensor],
        haematoxylin_mask_list: list[torch.Tensor],
    ) -> RegistrationNetworkOutput:
        """
        Takes in the batched MRI tensor, the associated haematoxylin images
        list, the corresponding haematoxylin mask list, and returns the
        network's prediction, including the batched warped haematoxylin images
        and masks.

        Parameters
        ----------
        mri_batch: torch.Tensor
            Batched MRI tensors, shape: (B, 1, `self.config.height`, `self.config.width`)
        haematoxylin_list: list[torch.Tensor]
            Length B list of haematoxylin images with varying dimensions, each
            with shape (1, H_i, W_i), dtype torch.float32
        haematoxylin_mask_list: list[torch.Tensor]
            Length B list of haematoxylin masks with varying dimensions, each
            with shape (1, H_i, W_i), dtype torch.float32

        Returns
        -------
        RegistrationNetworkOutput
        """
        # Obtain rigid transform parameters.
        parameters = self.predict_rigid_transform_parameters(
            mri_batch=mri_batch,
            haematoxylin_list=haematoxylin_list,
        )
        # Build (B, 2, 3) rigid transform matrices.
        thetas = self.stn.build_theta(parameters)
        # Apply rigid transform matrices.
        warped_haematoxylin = self.apply_rigid_transform(
            thetas, tensors=haematoxylin_list, grid_sampling_mode="bilinear"
        )
        warped_haematoxylin_mask = self.apply_rigid_transform(
            thetas,
            tensors=haematoxylin_mask_list,
            grid_sampling_mode="bilinear",
        )
        return RegistrationNetworkOutput(
            warped_haematoxylin=warped_haematoxylin,
            warped_haematoxylin_mask=warped_haematoxylin_mask,
            parameters=parameters,
            thetas=thetas,
        )
