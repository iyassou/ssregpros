from ..core.type_definitions import RegularisationCoefficient
from ..models.registration import RegistrationNetworkOutput
from ..regularisation.rigid_transform import (
    RigidTransformRegularisationLoss,
    RigidTransformRegularisationLossConfig,
)
from . import Reduction
from .boundary_heatmap import BoundaryHeatmapMSELoss
from .ncc import MaskedNCCLoss
from .sobel import MaskedMeanSquaredGradientErrorLoss

from monai.data.meta_tensor import MetaTensor

from dataclasses import dataclass
from enum import StrEnum

import torch


class CompositeLossKeys(StrEnum):
    LOSS = "composite_loss"

    NCC = "-ncc_loss"
    SOBEL = "sobel_loss"
    BHM = "bhm_loss"
    PARAMS_REG = "params_reg"

    NCC_WEIGHT = "ncc_weight"
    SOBEL_WEIGHT = "sobel_weight"
    BHM_WEIGHT = "bhm_weight"
    PARAMS_REG_WEIGHT = "params_reg_weight"


@dataclass
class CompositeLossConfig:
    ncc_weight: RegularisationCoefficient
    sobel_weight: RegularisationCoefficient
    boundary_heatmap_weight: RegularisationCoefficient
    transformation_parameters_weight: RegularisationCoefficient

    transformation_regularisation_config: RigidTransformRegularisationLossConfig


class CompositeLoss(torch.nn.modules.loss._Loss):
    def __init__(self, config: CompositeLossConfig):
        super().__init__()
        self.config = config
        self.ncc = MaskedNCCLoss(reduction=Reduction.NONE, anticorrelated=True)
        self.sobel = MaskedMeanSquaredGradientErrorLoss()
        self.boundary_heatmap = BoundaryHeatmapMSELoss()

        self.params_reg = RigidTransformRegularisationLoss(
            config=config.transformation_regularisation_config
        )

        self._latest: dict[CompositeLossKeys, float | None] = {}
        self.reset_latest_cache()

    def __str__(self) -> str:
        comps: list[str] = []
        if self.config.ncc_weight:
            comps.append("-NCC")
        if self.config.sobel_weight:
            comps.append("Sobel")
        if self.config.boundary_heatmap_weight:
            comps.append("BHM")
        if self.config.transformation_parameters_weight:
            comps.append("Reg")
        return f"Composite({', '.join(comps)})"

    def reset_latest_cache(self):
        self._latest = {
            # Composite loss
            CompositeLossKeys.LOSS: None,
            # Components
            CompositeLossKeys.NCC: None,
            CompositeLossKeys.SOBEL: None,
            CompositeLossKeys.BHM: None,
            CompositeLossKeys.PARAMS_REG: None,
            # Weights
            CompositeLossKeys.NCC_WEIGHT: None,
            CompositeLossKeys.SOBEL_WEIGHT: None,
            CompositeLossKeys.BHM_WEIGHT: None,
            CompositeLossKeys.PARAMS_REG_WEIGHT: None,
        }

    def latest(self) -> dict[str, float]:
        return {k.value: v for k, v in self._latest.items() if v is not None}

    def forward(
        self,
        y_true: torch.Tensor,
        pred: RegistrationNetworkOutput,
        mask: torch.Tensor | MetaTensor,
    ) -> torch.Tensor:
        """Composite loss that combines the selected components."""
        loss = torch.zeros(1, device=y_true.device)
        if ncc_weight := self.config.ncc_weight:
            self._latest[CompositeLossKeys.NCC_WEIGHT] = ncc_weight
            ncc = self.ncc(y_true, pred.warped_haematoxylin, mask).mean()
            self._latest[CompositeLossKeys.NCC] = ncc.item()
            loss += ncc_weight * ncc
        if sobel_weight := self.config.sobel_weight:
            self._latest[CompositeLossKeys.SOBEL_WEIGHT] = sobel_weight
            sobel = self.sobel(y_true, pred.warped_haematoxylin, mask)
            self._latest[CompositeLossKeys.SOBEL] = sobel.item()
            loss += sobel_weight * sobel
        if bhm_weight := self.config.boundary_heatmap_weight:
            self._latest[CompositeLossKeys.BHM_WEIGHT] = bhm_weight
            bhm = self.boundary_heatmap(mask, pred.warped_haematoxylin_mask)
            self._latest[CompositeLossKeys.BHM] = bhm.item()
            loss += bhm_weight * bhm
        if params_reg_weight := self.config.transformation_parameters_weight:
            self._latest[CompositeLossKeys.PARAMS_REG_WEIGHT] = (
                params_reg_weight
            )
            params_reg = self.params_reg(pred.parameters)
            self._latest[CompositeLossKeys.PARAMS_REG] = params_reg.item()
            loss += params_reg_weight * params_reg
        self._latest[CompositeLossKeys.LOSS] = loss.item()
        return loss
