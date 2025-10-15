from enum import StrEnum


class Reduction(StrEnum):
    NONE = "none"
    MEAN = "mean"
    SQRT = "sqrt"
    LOG = "log"
    SUM = "sum"
