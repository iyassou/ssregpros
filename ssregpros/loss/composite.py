from ..core.type_definitions import (
    PositiveFloat,
    RegularisationCoefficient,
    StrictlyPositiveFloat,
)
from ..models.registration import RegistrationNetworkOutput
from ..regularisation.rigid_transform import (
    RigidTransformRegularisationLoss,
    RigidTransformRegularisationLossConfig,
)
from . import Reduction
from .mask_leakage import HingedMaskLeakageLoss
from .ncc import MaskedNCCLoss

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import torch


class CompositeLossInput(NamedTuple):
    mri: torch.Tensor
    mri_mask: torch.Tensor
    mri_mask_sdt: torch.Tensor
    prediction: RegistrationNetworkOutput


class CompositeLossKeys(StrEnum):
    LOSS = "composite_loss"

    NCC = "-ncc_loss"
    HINGED_MASK_LEAKAGE = "hinged_mask_leakage"
    PARAMS_REG = "params_reg"

    NCC_WEIGHT = "ncc_weight"
    HINGED_MASK_LEAKAGE_WEIGHT = "hinged_mask_leakage_weight"
    PARAMS_REG_WEIGHT = "params_reg_weight"


@dataclass
class CompositeLossConfig:
    ncc_weight: RegularisationCoefficient
    hinged_mask_leakage_weight: RegularisationCoefficient
    transformation_parameters_weight: RegularisationCoefficient

    hinged_mask_leakage_epsilon: StrictlyPositiveFloat
    hinged_mask_leakage_delta: PositiveFloat
    transformation_regularisation_config: RigidTransformRegularisationLossConfig


class CompositeLoss(torch.nn.modules.loss._Loss):
    def __init__(self, config: CompositeLossConfig):
        super().__init__()
        self.config = config
        self.ncc = MaskedNCCLoss(reduction=Reduction.NONE, anticorrelated=True)
        self.leakage = HingedMaskLeakageLoss(
            epsilon=config.hinged_mask_leakage_epsilon,
            delta=config.hinged_mask_leakage_delta,
        )

        self.params_reg = RigidTransformRegularisationLoss(
            config=config.transformation_regularisation_config
        )

        self._latest: dict[CompositeLossKeys, float | None] = {}
        self.reset_latest_cache()

    def __str__(self) -> str:
        comps: list[str] = []
        if self.config.ncc_weight:
            comps.append("-NCC")
        if self.config.hinged_mask_leakage_weight:
            comps.append("MaskLeakage")
        if self.config.transformation_parameters_weight:
            comps.append("Reg")
        return f"Composite({', '.join(comps)})"

    def reset_latest_cache(self):
        self._latest = {
            # Composite loss
            CompositeLossKeys.LOSS: None,
            # Components
            CompositeLossKeys.NCC: None,
            CompositeLossKeys.HINGED_MASK_LEAKAGE: None,
            CompositeLossKeys.PARAMS_REG: None,
            # Weights
            CompositeLossKeys.NCC_WEIGHT: None,
            CompositeLossKeys.HINGED_MASK_LEAKAGE_WEIGHT: None,
            CompositeLossKeys.PARAMS_REG_WEIGHT: None,
        }

    def latest(self) -> dict[str, float]:
        return {k.value: v for k, v in self._latest.items() if v is not None}

    def forward(self, x: CompositeLossInput) -> torch.Tensor:
        """Composite loss that combines the selected components."""
        loss = torch.zeros(1, device=x.mri.device)
        if ncc_weight := self.config.ncc_weight:
            self._latest[CompositeLossKeys.NCC_WEIGHT] = ncc_weight
            ncc = self.ncc.forward(
                y_true=x.mri,
                y_pred=x.prediction.warped_haematoxylin,
                mask=x.mri_mask,
            ).mean()
            self._latest[CompositeLossKeys.NCC] = ncc.item()
            loss += ncc_weight * ncc
        if leakage_weight := self.config.hinged_mask_leakage_weight:
            self._latest[CompositeLossKeys.HINGED_MASK_LEAKAGE_WEIGHT] = (
                leakage_weight
            )
            leakage = self.leakage.forward(
                mri_mask_sdt=x.mri_mask_sdt,
                warped_haematoxylin_mask=x.prediction.warped_haematoxylin_mask,
            )
            self._latest[CompositeLossKeys.HINGED_MASK_LEAKAGE] = leakage.item()
            loss += leakage_weight * leakage
        if params_reg_weight := self.config.transformation_parameters_weight:
            self._latest[CompositeLossKeys.PARAMS_REG_WEIGHT] = (
                params_reg_weight
            )
            params_reg = self.params_reg.forward(
                parameters=x.prediction.parameters
            )
            self._latest[CompositeLossKeys.PARAMS_REG] = params_reg.item()
            loss += params_reg_weight * params_reg
        self._latest[CompositeLossKeys.LOSS] = loss.item()
        return loss
