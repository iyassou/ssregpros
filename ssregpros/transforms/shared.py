from monai.config.type_definitions import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import MapTransform, Transform

from enum import StrEnum
from typing import Callable, Collection, Hashable


class SharedPipelineKeys(StrEnum):
    EARLY_EXIT = "early_exit"


class MetaTensorLogging(Transform):
    def __init__(self, key: str, fn: Callable):
        self.key = key
        self.fn = fn

    def __call__(self, tensor: MetaTensor) -> MetaTensor:
        tensor.meta[self.key] = self.fn(tensor)
        return tensor


class MetaTensorLoggingd(MapTransform):
    def __init__(self, keys: KeysCollection, key: str, fn: Callable):
        super().__init__(keys=keys, allow_missing_keys=False)
        self.key = key
        self.fn = fn

    def __call__(self, data: dict) -> dict:
        for key in self.key_iterator(data):
            # NOTE: data[key] is the tensor.
            data[key].meta[self.key] = self.fn(data[key])
        return data


class SkipIfKeysPresentd(MapTransform):
    """Conditional execution if certain keys are not present."""

    def __init__(
        self, keys: KeysCollection, transform: MapTransform | Transform
    ):
        super().__init__(keys=keys, allow_missing_keys=True)
        self.transform = transform

    def __call__(self, data: dict) -> dict:
        for _ in self.key_iterator(data):
            return data
        return self.transform(data)


class StandardiseIntensityMaskedd(MapTransform):
    """Standardises intensity using masked statistics."""

    def __init__(self, keys: Collection[Hashable]):
        for pair in keys:
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise TypeError(f"expect iterable of key pairs, not: {pair!r}")
        super().__init__(keys=keys, allow_missing_keys=True)

    def __call__(self, data: dict) -> dict:
        for tensor_key, mask_key in self.key_iterator(data):
            tensor = data[tensor_key]
            mask = data[mask_key]
            mean = tensor[mask].mean()
            std = tensor[mask].std()
            tensor = (tensor - mean) / std
            data[tensor_key] = tensor
        return data
