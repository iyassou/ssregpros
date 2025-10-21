from ..models.registration import RegistrationNetwork

import torch.nn as nn


class UnfreezeController:
    """Gradually unfreezes the twin ResNet encoders' layers."""

    def __init__(self, model: RegistrationNetwork, plan: list[str] | None):
        self.model = model
        if plan is None:
            self.plan = []
        else:
            self.plan = plan
        self.index = -1
        # Start frozen.
        self.freeze()

    def freeze(self):
        """Freeze feature encoders and MRI encoder's BatchNorm2d layers."""
        # Feature encoders
        for name, param in self.model.mri_encoder.named_parameters():
            if name != "conv1":
                param.requires_grad = False
        for name, param in self.model.haematoxylin_encoder.named_parameters():
            if name != "conv1":
                param.requires_grad = False
        # MRI encoder `BatchNorm2d` (histology has none)
        for m in self.model.mri_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    def status(self) -> dict[str, bool]:
        return {
            f"unfrozen/{name}": i <= self.index
            for i, name in enumerate(self.plan)
        }

    def melted(self) -> bool:
        if not self.plan:
            return True
        return all(self.status().values())

    def next(self) -> str | None:
        """Attempt to unfreeze the next layer in the plan."""
        self.index += 1
        if self.index >= len(self.plan):
            return None
        block = self.plan[self.index]
        mri_target: nn.Module
        haem_target: nn.Module
        mri_target, haem_target = map(
            lambda x: getattr(x, block),
            (self.model.mri_encoder, self.model.haematoxylin_encoder),
        )
        for p in mri_target.parameters():
            p.requires_grad = True
        for q in haem_target.parameters():
            q.requires_grad = True
        return block
