from . import Reduction

import kornia
import math
import torch


class BoundaryHeatmapMSELoss(torch.nn.modules.loss._Loss):
    """
    Turns both input masks into soft boundary heatmaps using Sobel gradient
    magnitude and Gaussian blurring, and computes their mean-squared error.
    Default sigma value is ~ 1.27.
    """

    def __init__(
        self,
        gaussian_blur_sigma: float = 3 / (2 * math.sqrt(2 * math.log(2))),
    ):
        super().__init__()
        if gaussian_blur_sigma < 0:
            raise ValueError(
                f"cannot use negative sigma value: {gaussian_blur_sigma}"
            )
        self.sigma = gaussian_blur_sigma
        radius = math.ceil(3 * gaussian_blur_sigma)  # 3 sigma so 99.7%
        k = 2 * radius + 1
        self.kernel_size: tuple[int, int] = (k, k)
        self.loss = torch.nn.MSELoss(reduction=Reduction.MEAN)

    def boundary_heatmap(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary heatmap.

        Parameters
        ----------
        mask: torch.Tensor
            Shape (B, 1, H, W), torch.float32

        Returns
        -------
        torch.Tensor
            Shape (B, 1, H, W), torch.float32
        """
        # Sobel edge detection.
        eps = torch.finfo(mask.dtype).eps
        edges = kornia.filters.sobel(
            mask, normalized=False, eps=eps
        )  # (B, 1, H, W)
        # Gaussian blurring.
        blurred = kornia.filters.gaussian_blur2d(
            edges,
            kernel_size=self.kernel_size,
            sigma=(self.sigma, self.sigma),
        )
        # Per-sample normalisation.
        batch_max = blurred.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        blurred = blurred / batch_max
        return blurred

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Yep.

        Parameters
        ----------
        y_true: torch.Tensor
            Shape (B, 1, H, W), torch.bool
        y_pred: torch.Tensor
            Shape (B, 1, H, W), torch.bool

        Returns
        -------
        torch.Tensor
            A single scalar value
        """
        # Sanity checks.
        if y_true.shape != y_pred.shape:
            raise ValueError("masks have different shapes!")
        if (ndim := y_true.ndim) != 4:
            raise ValueError(f"masks should be 4D, got {ndim=}")
        if (C := y_true.size(1)) != 1:
            raise ValueError(f"expected number of channels to be 1, got {C=}")
        # Obtain boundary heatmaps.
        # NOTE: converting to torch.float32 is critical for gradients to flow.
        heatmap_true = self.boundary_heatmap(y_true.float())
        heatmap_pred = self.boundary_heatmap(y_pred.float())

        # =====
        if False:
            import matplotlib.pyplot as plt

            batch_size = y_true.size(0)
            fig, all_axes = plt.subplots(
                batch_size, 4, figsize=(2 * batch_size, 6)
            )
            all_axes = all_axes.flatten()
            for B in range(batch_size):
                axes = all_axes[B * 4 : (B + 1) * 4]
                imgs = (
                    y_true[B, 0].clone().detach().numpy(),
                    heatmap_true[B, 0].clone().detach().numpy(),
                    y_pred[B, 0].clone().detach().numpy(),
                    heatmap_pred[B, 0].clone().detach().numpy(),
                )
                cmaps = "gray", "hot", "gray", "hot"
                titles = map(
                    lambda x: f"[b={B}] " + x,
                    (
                        "MRI Mask",
                        "MRI Heatmap",
                        "Histology Mask",
                        "Histology Heatmap",
                    ),
                )
                for j, (ax, img, cmap, title) in enumerate(
                    zip(axes, imgs, cmaps, titles)
                ):
                    ax.set_axis_off()
                    im = ax.imshow(img, cmap=cmap)
                    ax.set_title(title)
                    if j % 2:
                        plt.colorbar(im, ax=ax)
            fig.tight_layout()
            plt.show()
            breakpoint()
        # =====

        # Compute loss.
        loss = self.loss(heatmap_true, heatmap_pred)
        # Done!
        return loss


if __name__ == "__main__":

    def main():
        import matplotlib.pyplot as plt
        import numpy as np

        size = 128
        center = size // 2
        radius = 30
        # Create coordinate grid
        y, x = np.ogrid[:size, :size]
        mask = ((x - center) ** 2 + (y - center) ** 2) <= radius**2
        mask = mask.astype(bool)
        # Convert to torch tensor with batch and channel dims
        mask_tensor = (
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 128, 128)
        # Create boundary heatmap.
        loss = BoundaryHeatmapMSELoss()
        boundary = loss.boundary_heatmap(mask_tensor.float())
        # Plot.
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = axes.flatten()
        imgs = mask, boundary.squeeze().numpy()
        cmaps = "gray", "hot"
        titles = "Original Binary Mask", "Boundary Heatmap (Sobel + Blur)"
        for j, (ax, img, title, cmap) in enumerate(
            zip(axes, imgs, titles, cmaps)
        ):
            ax.set_axis_off()
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            if j:
                plt.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.show()

    main()
