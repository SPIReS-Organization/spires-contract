"""LUT -> inversion boundary contract: Mie-theory reflectance lookup table.

Canonical form (see conventions):
- reflectances: dims (band, solar_angle, lap_concentration, grain_size),
  float32, with a coordinate present for each of the four dimensions.

Dimension ORDER is part of the contract — the C++ interpolator indexes the LUT
positionally, so a transposed array is as wrong as a missing dimension. The
interpolator also reads each dimension's coordinate values to locate query
points, so all four coordinates must be present. `validate_lut` raises
ContractError listing every violation (missing/extra dims, wrong order, wrong
dtype, missing coordinate) so a producer gets one actionable error.
"""

__all__ = ["validate_lut"]

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_present,
    check_dims_present,
    check_dims_order,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)


def validate_lut(da):
    """Validate a reflectance lookup table DataArray. Raises ContractError."""
    violations = []
    violations += check_dims_present(da, c.LUT_DIMS)
    violations += check_no_extra_dims(da, c.LUT_DIMS)
    violations += check_dims_order(da, c.LUT_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)
    violations += check_coords_present(da, c.LUT_DIMS)
    raise_if_violations("lut", violations)
