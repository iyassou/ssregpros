# mypy: disable-error-code="assignment,union-attr"
from .. import CACHE_ROOT
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

import datetime
import hashlib
import httpx
import json
import torch
import tqdm

ORIGINAL_VOLUME_SHAPE_KEY = "original_volume_shape"


def fetch_zenodo_record() -> dict:
    """Retrieves the "Models for Prostate158" Zenodo record."""
    TTL = datetime.timedelta(days=7)
    ZENODO_RECORD_ID = 7040585
    cached = CACHE_ROOT / f"zenodo-record-{ZENODO_RECORD_ID}.json"
    if cached.exists():
        creation_time = datetime.datetime.fromtimestamp(
            cached.stat().st_mtime, tz=datetime.timezone.utc
        )
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        if now - creation_time < TTL:
            with open(cached, "r") as handle:
                return json.load(handle)
    # Un-cached or stale: fetch from Zenodo.
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    r = httpx.get(url)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        e.add_note("Failed to retrieve `prostate158` Zenodo record!")
        raise
    record = r.json()
    assert record["title"] == "Models for Prostate158"
    with open(cached, "w") as handle:
        json.dump(record, handle)
    return record


def retrieve_model_weights() -> Path:
    """
    Downloads the `prostate158` anatomy model's weights if not already
    downloaded, and returns the file path.
    """
    # Retrieve Zenodo record.
    record = fetch_zenodo_record()
    # Retrieve anatomy model's metadata.
    key = "anatomy.pt"
    try:
        anatomy_model_metadata = next(
            x for x in record["files"] if x["key"] == key
        )
    except StopIteration:
        raise FileNotFoundError(f"Could not locate {key}!")
    # Retrieve checksum.
    md5_checksum: str = anatomy_model_metadata["checksum"].removeprefix("md5:")
    # Check if weights have already been downloaded.
    filepath = CACHE_ROOT / f"prostate158-{key}"
    if filepath.exists():
        # Compare checksums.
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                md5.update(chunk)
        actual = md5.hexdigest()
        if actual != md5_checksum:
            raise ValueError(
                f"checksum mismatch: cached={actual!r} does not match expected={md5_checksum!r}"
            )
        # Ready to go!
        return filepath
    # Download weights, compute checksum.
    download_url = anatomy_model_metadata["links"]["self"]
    md5 = hashlib.md5()
    with httpx.stream(
        "GET", download_url, follow_redirects=False, timeout=None
    ) as r:
        r.raise_for_status()
        total_bytes = int(r.headers.get("content-length", 0))
        chunk_size = 1 << 20
        with (
            open(filepath, "wb") as f,
            tqdm.tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading Prostate158 Model Weights",
                disable=(total_bytes == 0),
            ) as pbar,
        ):
            try:
                for chunk in r.iter_bytes(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    md5.update(chunk)
                    pbar.update(len(chunk))
            except KeyboardInterrupt:
                # Delete partial download.
                f.close()
                filepath.unlink()
                raise
    # Compare checksums.
    actual = md5.hexdigest()
    if actual != md5_checksum:
        raise ValueError(
            f"checksum mismatch: downloaded={actual!r} does not match expected={md5_checksum!r}"
        )
    # Done!
    return filepath


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
        with open(retrieve_model_weights(), "rb") as handle:
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
