"""I/O -> inversion boundary contract: target/background spectra + solar angles.

Canonical forms (see conventions):
- target/background spectra: dims (y, x, band), float32 or float64, with a `band` coordinate
- solar angles:              dims (y, x),       float32 or float64

Dimension ORDER is part of the contract — the C++ inversion kernel indexes
arrays positionally, so a transposed array is as wrong as a missing dimension.
`validate_*` raises ContractError listing every violation (missing/extra dims,
wrong order, wrong dtype, missing coordinate) so a producer gets one actionable
error.
"""

__all__ = [
    "validate_target_spectra",
    "validate_background_spectra",
    "validate_solar_angles",
]

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_present,
    check_dims_present,
    check_dims_order,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)


def _validate_spectra(da, contract_name):
    violations = []
    violations += check_dims_present(da, c.SPECTRA_DIMS)
    violations += check_no_extra_dims(da, c.SPECTRA_DIMS)
    violations += check_dims_order(da, c.SPECTRA_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)
    violations += check_coords_present(da, ("band",))
    raise_if_violations(contract_name, violations)


def validate_target_spectra(da):
    """Validate mixed target reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "target_spectra")


def validate_background_spectra(da):
    """Validate background (R_0) reflectance spectra. Raises ContractError."""
    _validate_spectra(da, "background_spectra")


def validate_solar_angles(da):
    """Validate per-pixel solar zenith angles. Raises ContractError."""
    violations = []
    violations += check_dims_present(da, c.SOLAR_ANGLE_DIMS)
    violations += check_no_extra_dims(da, c.SOLAR_ANGLE_DIMS)
    violations += check_dims_order(da, c.SOLAR_ANGLE_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)
    # solar angles are 2-D: a band dimension is a violation
    if "band" in da.dims:
        violations.append(
            f"unexpected dimension 'band' for solar angles (dims: {tuple(da.dims)})"
        )
    raise_if_violations("solar_angles", violations)
