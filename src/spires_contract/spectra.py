"""I/O -> inversion boundary contract: target/background spectra + solar angles.

Canonical forms (see conventions):
- target/background spectra: dims (y, x, band), float64, with a `band` coordinate
- solar angles:              dims (y, x),       float64

`validate_*` raises ContractError listing every violation. `conform_*`
(see below) transposes/casts a nearly-conforming array into canonical form.
"""

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_present,
    check_dims_present,
    check_dtype,
    raise_if_violations,
)


def _validate_spectra(da, contract_name):
    violations = []
    violations += check_dims_present(da, c.SPECTRA_DIMS)
    violations += check_dtype(da, c.REQUIRED_DTYPE)
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
    violations += check_dtype(da, c.REQUIRED_DTYPE)
    # solar angles are 2-D: a band dimension is a violation
    if "band" in da.dims:
        violations.append(
            f"unexpected dimension 'band' for solar angles (dims: {tuple(da.dims)})"
        )
    raise_if_violations("solar_angles", violations)


def _conform_spectra(da, contract_name):
    # A missing dimension cannot be repaired by transpose/cast — fail clearly.
    missing = check_dims_present(da, c.SPECTRA_DIMS)
    raise_if_violations(contract_name, missing)
    return da.transpose(*c.SPECTRA_DIMS).astype(c.REQUIRED_DTYPE)


def conform_target_spectra(da):
    """Return target spectra transposed to (y, x, band) and cast to float64."""
    return _conform_spectra(da, "target_spectra")


def conform_background_spectra(da):
    """Return background spectra transposed to (y, x, band) and cast to float64."""
    return _conform_spectra(da, "background_spectra")


def conform_solar_angles(da):
    """Return solar angles transposed to (y, x) and cast to float64."""
    missing = check_dims_present(da, c.SOLAR_ANGLE_DIMS)
    raise_if_violations("solar_angles", missing)
    return da.transpose(*c.SOLAR_ANGLE_DIMS).astype(c.REQUIRED_DTYPE)
