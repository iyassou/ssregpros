from monai.networks.nets.resnet import ResNet
from torchvision.models import (
    ResNet as TV_ResNet,
    resnet18 as tv_resnet18,
    resnet34 as tv_resnet34,
    resnet50 as tv_resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

from functools import partial
from typing import Callable

import torch.nn as nn


MONAI_RESNET_FEATURE_ENCODER_SPECIFIC_PARAMS: dict[str, dict] = {
    "resnet18": {
        "block": "basic",
        "layers": [2, 2, 2, 2],
    },
    "resnet34": {
        "block": "basic",
        "layers": [3, 4, 6, 3],
    },
    "resnet50": {
        "block": "bottleneck",
        "layers": [3, 4, 6, 3],
    },
}

TORCHVISION_PRETRAINED_MODEL_FACTORIES: dict[str, Callable[..., TV_ResNet]] = {
    "resnet18": partial(
        tv_resnet18,
        weights=ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet34": partial(
        tv_resnet34,
        weights=ResNet34_Weights.IMAGENET1K_V1,
    ),
    "resnet50": partial(
        tv_resnet50,
        weights=ResNet50_Weights.IMAGENET1K_V1,
    ),
}


def swap_batchnorm2d_for_instancenorm2d(
    module: nn.Module,
) -> nn.Module:
    """Recursively swaps `BatchNorm2d` layers for `InstanceNorm2d` layers."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(
                module,
                name,
                nn.InstanceNorm2d(
                    num_features=child.num_features,
                    eps=child.eps,
                    momentum=child.momentum or 0.1,
                    affine=child.affine,
                    track_running_stats=True,
                ),
            )
        else:
            swap_batchnorm2d_for_instancenorm2d(child)
    return module


class ResNetFeatureEncoder(ResNet):
    def __init__(
        self,
        model_name: str,
    ):
        """Creates a 2D ResNet feature encoder with ImageNet V1 weights.

        Parameters
        ----------
        model_name: str
            The model's name, `resnetXX`

        Notes
        -----
        The MONAI ResNet architecture is created with `bias_downsample=False`,
        as those modules are missing from the pretrained ImageNet V1 weights.
        As such, the only expected discrepancy between the MONAI ResNet's state
        dictionary and the torchvision ResNet state dictionary are the weights
        in the first convolutional layer, `conv1.weight`. This is because the
        first convolutional layer in this network has 1 input channel, which is
        different from the typical 3 expected by a ResNet architecture.
        The first convolutional layer's weights are randomly initialised.
        """
        # Initialise MONAI ResNet feature encoder.
        super().__init__(
            **MONAI_RESNET_FEATURE_ENCODER_SPECIFIC_PARAMS[model_name],
            spatial_dims=2,
            feed_forward=False,
            bias_downsample=False,
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=1,
        )
        # Retrieve torchvision weights.
        pretrained_state_dict = TORCHVISION_PRETRAINED_MODEL_FACTORIES[
            model_name
        ]().state_dict()
        # Check for missing keys.
        state_dict = self.state_dict()
        filtered_pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if k in state_dict and state_dict[k].shape == v.shape
        }
        if (
            missing := len(state_dict) - len(filtered_pretrained_state_dict)
        ) != 1:
            raise ValueError(
                f"expected 1 mismatched key, actually missing {missing}"
            )
        if (
            diff := state_dict.keys() - filtered_pretrained_state_dict.keys()
        ) != {key := "conv1.weight"}:
            raise ValueError(
                f"expected only {key!r} to be missing, actually missing {diff}"
            )
        # Load in weights.
        state_dict.update(filtered_pretrained_state_dict)
        self.load_state_dict(state_dict)

    def output_size(self) -> int:
        """The output feature vector size is determined by examining the fourth
        layer's second batch normalisation module.

        Notes
        -----
        Swapping out `BatchNorm2d` layers for `InstanceNorm2d` layers doesn't
        affect this method since (1) the names of the layers don't change, and
        (2) both `BatchNorm2d` and `InstanceNorm2d` have `num_features` as an
        attribute."""
        return self.layer4[
            -1
        ].bn2.num_features  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]
