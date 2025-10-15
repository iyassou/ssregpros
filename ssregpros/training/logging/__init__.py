from enum import StrEnum


class TableIdentifiers(StrEnum):
    ID = "ID"
    MRI = "MRI"
    MRI_MASK = "MRI Mask"
    HISTOLOGY = "Histology"
    HISTOLOGY_MASK = "Histology Mask"
    CHECKERBOARD = "Checkerboard"
    CANNY_BAND = "Canny Band"
    CANNY_MASK = "Canny Mask"
