from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import math
import numpy as np
import torch
import torchvision.utils


class MetricAverager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, **metrics: float):
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (v != v)):  # NaN guard
                continue
            self.sums[k] += float(v)
            self.counts[k] += 1

    def mean(self) -> dict[str, float]:
        return {k: v / self.counts[k] for k, v in self.sums.items()}


JSONLike = (
    None
    | bool
    | int
    | float
    | str
    | Sequence["JSONLike"]
    | Mapping[str, "JSONLike"]
)
MatrixLike = torch.Tensor | np.ndarray


def to_jsonable(x: Any) -> JSONLike:
    """Make values W&B/JSON-friendly"""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return list(map(to_jsonable, x))
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    # Last resort: string.
    return str(x)


def to_uint8_img(x: MatrixLike) -> np.ndarray:
    """Accepts (C, H, W), [0, 1] or uint8, and returns (H, W, C) uint8."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype.is_floating_point:
            # Convert range from [0, 1] to {0, ..., 255}
            x = (x * 255.0).clamp(0, 255).round()
        x = x.to(torch.uint8)
        x = x.numpy()
    # x is now a numpy (C, H, W) or (H, W, C) or (H, W)
    if x.ndim == 2:
        x = np.expand_dims(x, 0)
    if x.shape[0] in (1, 3):
        x = np.moveaxis(x, 0, -1)
    return x  # (H, W, C), uint8


def to_uint8_mask(mask: MatrixLike) -> np.ndarray:
    """Accepts (H, W) or (1, H, W) binary masks and returns (H, W) uint8."""
    m: np.ndarray = (
        mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    )
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = m.astype(np.uint8)
    return m  # (H, W), uint8


def validation_image_filename(
    name: str, epoch: int, batch: int, suffix: str
) -> str:
    return f"{name}_batch{batch:03d}_epoch{epoch:04d}_{suffix}.png"


def save_image_grid(
    tensor: torch.Tensor,
    filepath: Path,
    padding: int = 15,
    pad_value: int = 1,
    **kwargs,
):
    """Creates and saves an image grid from a batched tensor."""
    if (ndim := tensor.ndim) != 4:
        raise ValueError(
            f"expected {ndim}D tensor with shape (B, C, H, W), got {ndim=}"
        )
    batch_size = tensor.size(0)
    torchvision.utils.save_image(
        tensor=tensor,
        fp=filepath,
        nrow=batch_size,
        padding=padding,
        pad_value=pad_value,
        **kwargs,
    )


def split_weight_decay(
    named_params: Iterator[tuple[str, torch.nn.parameter.Parameter]],
) -> tuple[
    list[torch.nn.parameter.Parameter], list[torch.nn.parameter.Parameter]
]:
    """
    Sorts named model parameters by whether or not they should be affected
    by weight decay.
    """
    decay, no_decay = [], []
    keywords = "bias", "bn", "norm"
    for n, p in named_params:
        if not p.requires_grad:
            continue
        name = n.lower()
        if any(k in name for k in keywords):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def steps_to_epochs(
    steps: int,
    len_dataloader: int,
    gradient_accumulation_steps: int,
) -> int:
    """Converts number of optimiser steps into number of epochs."""
    steps_per_epoch = max(
        1, math.ceil(len_dataloader / max(1, gradient_accumulation_steps))
    )
    return max(1, math.ceil(steps / steps_per_epoch))
