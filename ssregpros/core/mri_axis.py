from enum import StrEnum
from monai.data.meta_tensor import MetaTensor
from nibabel.orientations import aff2axcodes
import torch

AffineType = MetaTensor | torch.Tensor


class MRIAxis(StrEnum):
    AXIAL = "AXIAL"
    CORONAL = "CORONAL"
    SAGITTAL = "SAGITTAL"

    def get_numpy_axis(self, affine: AffineType) -> int:
        """Returns the NumPy axis corresponding to this MRI axis given the
        NumPy volume's affine matrix."""
        orientation = aff2axcodes(affine)
        if self is MRIAxis.AXIAL:
            # Axial is the Superior-Inferior axis.
            return orientation.index("S" if "S" in orientation else "I")
        if self is MRIAxis.CORONAL:
            # Coronal is the Anterior-Posterior axis.
            return orientation.index("A" if "A" in orientation else "P")
        # self is MRI_AXIS.SAGITTAL
        # Sagittal is the Left-Right axis.
        return orientation.index("L" if "L" in orientation else "R")

    def get_numpy_slice(
        self, affine: AffineType, slice_index: int, keepdim: bool
    ) -> (
        tuple[slice, slice, slice]
        | tuple[int | slice, int | slice, int | slice]
    ):
        """Computes the 3D slice indices for a volume along this slice axis,
        given its affine matrix, slice index, slice axis, and affine matrix."""
        numpy_axis = self.get_numpy_axis(affine)
        axcodes = aff2axcodes(affine)
        assert len(axcodes) == 3
        slicer: (
            tuple[slice, slice, slice]
            | tuple[int | slice, int | slice, int | slice]
        ) = tuple(
            (
                (
                    slice(slice_index, slice_index + 1)
                    if keepdim
                    else slice_index
                )
                if j == numpy_axis
                else slice(None)
            )
            for j in range(len(axcodes))
        )  # pyright: ignore[reportAssignmentType]
        return slicer
