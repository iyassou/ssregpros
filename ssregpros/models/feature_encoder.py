from monai.networks.nets.resnet import ResNet

from typing import Callable, Literal, get_args

import functools
import torch
import torch.nn as nn
import torchvision.models as tv_models

ResNetVariant = Literal["resnet18", "resnet34", "resnet50", "resnet101"]


def _monai_resnet_params(var: ResNetVariant) -> dict:
    if var not in get_args(ResNetVariant):
        raise ValueError(f"unrecognised/unsupported ResNet variant: {var!r}")
    params: dict
    if var == "resnet18":
        params = {
            "block": "basic",
            "layers": [2, 2, 2, 2],
        }
    elif var == "resnet34":
        params = {
            "block": "basic",
            "layers": [3, 4, 6, 3],
        }
    elif var == "resnet50":
        params = {
            "block": "bottleneck",
            "layers": [3, 4, 6, 3],
        }
    elif var == "resnet101":
        params = {
            "block": "bottleneck",
            "layers": [3, 4, 23, 3],
        }
    return params


def _torchvision_resnet_imagenet_v1_weights_factory(
    var: ResNetVariant,
) -> Callable[..., tv_models.ResNet]:
    if var not in get_args(ResNetVariant):
        raise ValueError(f"unrecognised/unsupported ResNet variant: {var!r}")
    func: Callable[..., tv_models.ResNet]
    weights: (
        tv_models.ResNet18_Weights
        | tv_models.ResNet34_Weights
        | tv_models.ResNet50_Weights
        | tv_models.ResNet101_Weights
    )
    if var == "resnet18":
        func = tv_models.resnet18
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
    elif var == "resnet34":
        func = tv_models.resnet34
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1
    elif var == "resnet50":
        func = tv_models.resnet50
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
    elif var == "resnet101":
        func = tv_models.resnet101
        weights = tv_models.ResNet101_Weights.IMAGENET1K_V1
    return functools.partial(func, weights=weights)


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
                    track_running_stats=False,
                ),
            )
        else:
            swap_batchnorm2d_for_instancenorm2d(child)
    return module


class ResNetFeatureEncoder(ResNet):
    def __init__(self, var: ResNetVariant):
        """Creates a 2D ResNet feature encoder with ImageNet V1 weights.

        Parameters
        ----------
        var: ResNetVariant
            The model's name, `resnetXX`

        Notes
        -----
        The MONAI ResNet architecture is created with `bias_downsample=False`,
        as those modules are missing from the pretrained ImageNet V1 weights.
        As such, the only expected discrepancy between the MONAI ResNet's state
        dictionary and the torchvision ResNet state dictionary are the
        classifier, and the weights in the first convolutional layer,
        `conv1.weight`. This is because the first convolutional layer in this
        network has 1 input channel, which is different from the typical 3
        expected by a ResNet architecture. The first convolutional layer's
        weights are randomly initialised.
        """
        # Initialise MONAI ResNet feature encoder.
        monai_params = _monai_resnet_params(var)
        super().__init__(
            **monai_params,
            spatial_dims=2,
            feed_forward=False,
            bias_downsample=False,
            block_inplanes=[64, 128, 256, 512],
            n_input_channels=1,
        )
        # Retrieve torchvision weights.
        weights_factory = _torchvision_resnet_imagenet_v1_weights_factory(var)
        tv_state_dict: dict[str, torch.Tensor] = weights_factory().state_dict()
        # Check for shape mismatches.
        state_dict: dict[str, torch.Tensor] = self.state_dict()
        shape_mismatches: dict[str, tuple[torch.Size, torch.Size]] = {}
        filtered: dict[str, torch.Tensor] = {}
        for k, v1 in tv_state_dict.items():
            if (v2 := state_dict.get(k)) is not None:
                if v1.shape == v2.shape:
                    filtered[k] = v1
                elif not k.startswith("conv1"):
                    shape_mismatches[k] = (v1.shape, v2.shape)
        if mms := len(shape_mismatches):
            raise ValueError(
                f"Shape mismatches = {mms}, expected vs actual\n"
                + "\n".join(
                    f"\t{k} - {v1} vs {v2}"
                    for k, (v1, v2) in shape_mismatches.items()
                )
            )
        # Check that `conv1.weight` is the only missing weight on the MONAI
        # ResNet's side.
        expected_missing = {"conv1.weight"}
        missing_keys = state_dict.keys() - filtered.keys()
        if missing_keys != expected_missing:
            raise ValueError(
                f"Expected to be missing {expected_missing!r}, got: {missing_keys!r}"
            )
        # Check that `fc.something` weights are the only missing weights on
        # the torchvision side.
        pretrained_extras = tv_state_dict.keys() - state_dict.keys()
        unexpected_extras = {
            k for k in pretrained_extras if not k.startswith("fc")
        }
        if unexpected_extras:
            raise ValueError(
                f"Unexpected additional pretrained keys (not fc.*): {sorted(unexpected_extras)}"
            )
        # Load in weights.
        state_dict.update(filtered)
        self.load_state_dict(state_dict)

    def output_size(self) -> int:
        """The output feature vector size is determined by examining the fourth
        layer's last normalisation module.

        Notes
        -----
        Swapping out `BatchNorm2d` layers for `InstanceNorm2d` layers doesn't
        affect this method since (1) the names of the layers don't change, and
        (2) both `BatchNorm2d` and `InstanceNorm2d` have `num_features` as an
        attribute."""
        fourth_layer = self.layer4
        last_module = fourth_layer[-1]
        norms = [
            (n, m)
            for n, m in last_module.named_children()
            if n.startswith("bn")
        ]
        last_norm: nn.BatchNorm2d | nn.InstanceNorm2d
        _, last_norm = max(
            norms, key=lambda tup: int(tup[0].removeprefix("bn"))
        )  # pyright: ignore[reportAssignmentType]
        return last_norm.num_features
