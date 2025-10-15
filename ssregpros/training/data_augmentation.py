from ..core.type_definitions import Percentage
from ..transforms.histology import (
    HistologyPipelineKeys as HIST_KEYS,
)
from ..transforms.mri import MriPipelineKeys as MRI_KEYS

from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRicianNoised,
)
from monai.transforms.utility.dictionary import Lambdad

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence
from yaml import safe_load

import pprint
import torch


@dataclass
class DataAugmentation:
    """
    Dataclass for data augmentations to be applied during runtime.
    Save for the probabilities and device, dataclass default values follow
    MONAI defaults.
    """

    # Device should be set at runtime ideally to both leverage hardware
    # acceleration and avoid "tensors live on different devices" errors.
    device: torch.device = torch.device("cpu")

    # [Intensity Augmentations]
    # >>> MRI Augmentations <<<
    # > RandBiasFieldd
    mri_bias_field_prob: Percentage = 0.0
    mri_bias_field_degree: int = 3
    mri_bias_field_coeff_range: tuple[float, float] = (0, 0.1)
    # > RandAdjustContrastd
    mri_adjust_contrast_prob: Percentage = 0.0
    mri_adjust_contrast_gamma: tuple[float, float] | float = (0.95, 1.05)
    # > RandGaussianSmoothd
    mri_gaussian_smooth_prob: Percentage = 0.0
    mri_gaussian_smooth_sigma_x: tuple[float, float] = (0.25, 1.5)
    mri_gaussian_smooth_sigma_y: tuple[float, float] = (0.25, 1.5)
    mri_gaussian_smooth_approx: str = "erf"
    # > RandRicianNoised
    mri_rician_noise_prob: Percentage = 0.0
    mri_rician_noise_mean: Sequence[float] | float = 0.0
    mri_rician_noise_std: Sequence[float] | float = 0.01
    # >>> Histology Augmentations <<<
    # > RandAdjustContrastd
    hist_adjust_contrast_prob: Percentage = 0.0
    hist_adjust_contrast_gamma: tuple[float, float] | float = (0.95, 1.05)
    # > RandGaussianSmoothd
    hist_gaussian_smooth_prob: Percentage = 0.0
    hist_gaussian_smooth_sigma_x: tuple[float, float] = (0.25, 1.5)
    hist_gaussian_smooth_sigma_y: tuple[float, float] = (0.25, 1.5)
    hist_gaussian_smooth_approx: str = "erf"
    # > RandGaussianNoised
    hist_gaussian_noise_prob: Percentage = 0.0
    hist_gaussian_noise_mean: float = 0.0
    hist_gaussian_noise_std: float = 0.01

    def transform(self) -> Compose:
        """
        Returns the full data augmentation pipeline as prescribed by this
        configuration struct.

        Notes
        -----
        This pipeline is designed to be applied to a data dictionary containing
        normalised, floating-point tensors. It makes the model robust to
        variations in scanner contrast, staining intensity, and sensor noise.

        Returns
        -------
        Compose"""
        transforms = []

        # MRI Augmentations
        if self.mri_bias_field_prob > 0:
            assert self.mri_bias_field_degree is not None
            assert self.mri_bias_field_coeff_range is not None
            transforms.append(
                RandBiasFieldd(
                    keys=(MRI_KEYS.MRI_SLICE,),
                    degree=self.mri_bias_field_degree,
                    coeff_range=self.mri_bias_field_coeff_range,
                    prob=self.mri_bias_field_prob,
                    allow_missing_keys=False,
                )
            )
        if self.mri_adjust_contrast_prob > 0:
            assert self.mri_adjust_contrast_gamma is not None
            transforms.append(
                RandAdjustContrastd(
                    keys=(MRI_KEYS.MRI_SLICE,),
                    prob=self.mri_adjust_contrast_prob,
                    gamma=self.mri_adjust_contrast_gamma,
                    allow_missing_keys=False,
                )
            )
        if self.mri_gaussian_smooth_prob > 0:
            assert self.mri_gaussian_smooth_sigma_x is not None
            assert self.mri_gaussian_smooth_sigma_y is not None
            assert self.mri_gaussian_smooth_approx is not None
            transforms.append(
                RandGaussianSmoothd(
                    keys=(MRI_KEYS.MRI_SLICE,),
                    sigma_x=self.mri_gaussian_smooth_sigma_x,
                    sigma_y=self.mri_gaussian_smooth_sigma_y,
                    approx=self.mri_gaussian_smooth_approx,
                    prob=self.mri_gaussian_smooth_prob,
                    allow_missing_keys=False,
                )
            )
        if self.mri_rician_noise_prob > 0:
            assert self.mri_rician_noise_mean is not None
            assert self.mri_rician_noise_std is not None
            transforms.append(
                RandRicianNoised(
                    keys=(MRI_KEYS.MRI_SLICE,),
                    prob=self.mri_rician_noise_prob,
                    mean=self.mri_rician_noise_mean,
                    std=self.mri_rician_noise_std,
                    allow_missing_keys=False,
                )
            )

        # Histology Augmentations
        if self.hist_adjust_contrast_prob > 0:
            assert self.hist_adjust_contrast_gamma is not None
            transforms.append(
                RandAdjustContrastd(
                    keys=(HIST_KEYS.HISTOLOGY,),
                    prob=self.hist_adjust_contrast_prob,
                    gamma=self.hist_adjust_contrast_gamma,
                    allow_missing_keys=False,
                )
            )
        if self.hist_gaussian_smooth_prob > 0:
            assert self.hist_gaussian_smooth_sigma_x is not None
            assert self.hist_gaussian_smooth_sigma_y is not None
            assert self.hist_gaussian_smooth_approx is not None
            transforms.append(
                RandGaussianSmoothd(
                    keys=(HIST_KEYS.HISTOLOGY,),
                    sigma_x=self.hist_gaussian_smooth_sigma_x,
                    sigma_y=self.hist_gaussian_smooth_sigma_y,
                    approx=self.hist_gaussian_smooth_approx,
                    prob=self.hist_gaussian_smooth_prob,
                    allow_missing_keys=False,
                )
            )
        if self.hist_gaussian_noise_prob > 0:
            assert self.hist_gaussian_noise_mean is not None
            assert self.hist_gaussian_noise_std is not None
            transforms.append(
                RandGaussianNoised(
                    keys=(HIST_KEYS.HISTOLOGY,),
                    prob=self.hist_gaussian_noise_prob,
                    mean=self.hist_gaussian_noise_mean,
                    std=self.hist_gaussian_noise_std,
                )
            )

        # Clamp values between 0 and 1.
        transforms.append(
            Lambdad(
                keys=(MRI_KEYS.MRI_SLICE, HIST_KEYS.HISTOLOGY),
                func=lambda tensor: torch.clamp(tensor, 0.0, 1.0),
                overwrite=True,
            )
        )

        return Compose(transforms)

    @staticmethod
    def from_yaml(
        filepath: Path, device: torch.device = torch.device("cpu")
    ) -> "DataAugmentation":
        with open(filepath, "rb") as handle:
            config = safe_load(handle)
        if "device" not in config:
            config["device"] = device
        # Extra fields?
        given_fields = config.keys()
        expected_fields = set(x.name for x in fields(DataAugmentation))
        if diff := (given_fields - expected_fields):
            raise ValueError(
                f"{len(diff)} unrecognised fields from {filepath.name}:\n{pprint.pformat(diff)}"
            )
        return DataAugmentation(**config)
