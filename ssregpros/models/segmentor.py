from .. import WEIGHTS_ROOT
from ..transforms.shared import MetaTensorLogging

from monai.data.meta_tensor import MetaTensor
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.unet import UNet

from monai.transforms.compose import Compose
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.spatial.array import Spacing, Orientation
from monai.transforms.intensity.array import ScaleIntensity, NormalizeIntensity
from monai.transforms.post.array import KeepLargestConnectedComponent
from monai.utils.enums import GridSampleMode

from nibabel.nifti1 import Nifti1Image
from nibabel.processing import resample_from_to

from pathlib import Path

import torch
import tqdm

ORIGINAL_VOLUME_SHAPE_KEY = "original_volume_shape"


class Segmentor:
    """
    Hardcoded [prostate158](https://github.com/kbressem/prostate158) default
    anatomy segmentation U-ResNet, intended for inference only.
    Configuration values are representative of the sample `anatomy.yaml`
    file in the aforementioned repository.
    Output mask is binarised to: prostate, NOT prostate.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        # Initialise model.
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=[16, 32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            num_res_units=4,
            act=Act.PRELU,
            norm=Norm.BATCH,
            dropout=0.15,
        ).to(device)
        with open(WEIGHTS_ROOT / "prostate158-anatomy.pt", "rb") as handle:
            self.model.load_state_dict(
                torch.load(handle, map_location=device, weights_only=True)
            )
        self.model.eval()
        # Build inferer. Inferer is necessary for processing arbitrary input
        # sizes that aren't necessarily perfect multiples of the network's
        # input size. Default parameters mirror the prostate158 default.
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5,
            progress=False,
            sw_device=device,
            device=device,
        )
        # Preprocessing transforms.
        self.input_transform = Compose(
            [
                LoadImage(),
                MetaTensorLogging(
                    ORIGINAL_VOLUME_SHAPE_KEY,
                    lambda meta_tensor: meta_tensor.shape,
                ),
                EnsureChannelFirst(),
                Spacing(
                    pixdim=(0.5, 0.5, 0.5),
                    mode=GridSampleMode.BILINEAR,
                ),
                Orientation(axcodes="RAS"),
                ScaleIntensity(minv=0, maxv=1),
                NormalizeIntensity(),
            ]
        )
        # Post-processing transform.
        # NOTE: modified for binary mask, else `applied_labels=[1, 2]`.
        self.klcc = KeepLargestConnectedComponent(applied_labels=[1])

    def _segment_one(self, filepath: Path | str) -> MetaTensor:
        """Runs the segmentation for a given input volume."""
        # Prepare input volume.
        input_volume: MetaTensor = self.input_transform(
            filepath
        )  # pyright: ignore[reportAssignmentType]
        input_volume = input_volume.unsqueeze(
            0
        )  # add batch dimension # pyright: ignore[reportAssignmentType]
        input_volume = input_volume.to(
            self.device
        )  # move Tensor to correct device # pyright: ignore[reportAssignmentType]
        # Run inference.
        with torch.no_grad():
            logits: MetaTensor = self.inferer(
                inputs=input_volume, network=self.model
            )  # pyright: ignore[reportAssignmentType]
            logits = logits.squeeze(
                0
            )  # shape: (3, D, H, W) # pyright: ignore[reportAssignmentType]
        # Resample model's predictions into the input image's space.
        # NOTE: I am aware MONAI supplies the `Invert` transform, but
        #       I couldn't get it to work, so I've resorted to manually
        #       implementing it for now and will raise an issue later.
        raw_mask = torch.argmax(logits, dim=0).cpu().type(torch.uint8).numpy()
        resampled_mask = resample_from_to(
            Nifti1Image(
                raw_mask,
                affine=input_volume.meta[  # pyright: ignore[reportAttributeAccessIssue]
                    "affine"
                ],
            ),
            (
                input_volume.meta[  # pyright: ignore[reportAttributeAccessIssue]
                    ORIGINAL_VOLUME_SHAPE_KEY
                ],
                input_volume.meta[  # pyright: ignore[reportAttributeAccessIssue]
                    "original_affine"
                ],
            ),
            order=0,
            mode="nearest",
        )
        # NOTE: setting `affine=input_volume.affine` isn't necessary
        #       as the affine matrix is read from `input_volume.meta`
        amended_metadata = input_volume.meta.copy()
        amended_metadata["affine"] = input_volume.meta["original_affine"]
        del (
            amended_metadata["original_affine"],
            amended_metadata[ORIGINAL_VOLUME_SHAPE_KEY],
        )  # no longer needed, avoid confusion
        mask = MetaTensor(
            resampled_mask.get_fdata(),
            meta=amended_metadata,
            applied_operations=input_volume.applied_operations,
        )
        # Binarise mask.
        # NOTE: assumes the background is the dominant class
        labels, counts = torch.unique(mask, return_counts=True)
        background_label = labels[torch.argmax(counts)]
        mask = torch.where(mask == background_label, 0, 1)
        # Keep largest connected component.
        mask = (
            self.klcc(mask.unsqueeze(0))
            .squeeze(0)
            .type(torch.uint8)  # pyright: ignore[reportAttributeAccessIssue]
        )
        return mask  # pyright: ignore[reportReturnType]

    def segment(
        self, *filepaths: Path | str, progress_bar: bool
    ) -> MetaTensor | list[MetaTensor]:
        """Segments one or more volumes."""
        loader = (
            tqdm.tqdm(filepaths, desc="Segmenting Volumes")
            if progress_bar
            else filepaths
        )
        return list(map(self._segment_one, loader))
