from annotated_types import Ge, Gt, Le
from typing import Annotated, get_args, get_origin, Protocol

Percentage = Annotated[float, Ge(0), Le(1)]
PositiveFloat = Annotated[float, Ge(0)]
PositiveInteger = Annotated[int, Ge(0)]
RegularisationCoefficient = Annotated[float, Ge(0)]
StrictlyPositiveFloat = Annotated[float, Gt(0)]
StrictlyPositiveInteger = Annotated[int, Ge(1)]
ScalingFactor = Annotated[float, Ge(0)]


class Scheduler(Protocol):
    def step(self) -> None: ...


def assert_annotated_type(value, annotated_type: Annotated, error: Exception):
    """Asserts a type annotation, raises an exception on failure."""
    origin: type = get_origin(annotated_type)
    if origin is not Annotated:
        raise TypeError(f"{annotated_type} is not an Annotated type")
    base_type, *constraints = get_args(annotated_type)
    if not isinstance(value, base_type):
        if not (base_type is float and value == float(value)):
            error.add_note(f"\tincorrect base type: expected {base_type!r}")
            raise error
    for constraint in constraints:
        if isinstance(constraint, Ge) and not (value >= constraint.ge):
            error.add_note(f"\texpected ({value=}) ≥ {constraint.ge}")
            raise error
        if isinstance(constraint, Gt) and not (value > constraint.gt):
            error.add_note(f"\texpected ({value=}) > {constraint.gt}")
            raise error
        if isinstance(constraint, Le) and not (value <= constraint.le):
            error.add_note(f"\texpected ({value=}) ≤ {constraint.le}")
            raise error
