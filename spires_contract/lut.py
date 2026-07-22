"""Canonical reflectance- and albedo-LUT boundary contracts."""

import re

import numpy as np
import xarray as xr

from spires_contract import conventions as c
from spires_contract._validate import (
    ContractError,
    check_coords_present,
    check_dims_order,
    check_dims_present,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)

__all__ = [
    "canonical_lut_unit",
    "validate_albedo_lut",
    "validate_lut",
    "validate_reflectance_lut",
]


_UNIT_ALIASES = {
    "degrees": {
        "degree",
        "degrees",
        "deg",
    },
    "ppm": {
        "ppm",
        "partpermillion",
        "partspermillion",
        "parts_per_million",
    },
    "um^0.5": {
        "um^0.5",
        "um^(0.5)",
        "um^(1/2)",
        "um^1/2",
        "um**0.5",
        "sqrt(um)",
        "sqrt_um",
        "micron^0.5",
        "microns^0.5",
        "micrometer^0.5",
        "micrometers^0.5",
        "micrometre^0.5",
        "micrometres^0.5",
        "µm^0.5",
        "µm^(1/2)",
        "μm^0.5",
        "μm^(1/2)",
    },
    "1": {
        "1",
        "fraction",
        "dimensionless",
        "unitless",
    },
    "km": {
        "km",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
    },
}


def canonical_lut_unit(axis_name, units):
    """Resolve a supported spelling to the canonical unit for an LUT axis.

    This helper compares unit spellings only. It never converts coordinate
    values, so scaled units such as metres for a canonical kilometre axis are
    rejected rather than silently rescaled.
    """
    if axis_name not in c.LUT_AXIS_UNITS:
        raise ContractError(f"unknown canonical LUT axis {axis_name!r}")
    if not isinstance(units, str) or not units.strip():
        raise ContractError(f"LUT axis {axis_name!r} is missing a units string")

    canonical = c.LUT_AXIS_UNITS[axis_name]
    normalized = _normalize_unit(units)
    aliases = {_normalize_unit(alias) for alias in _UNIT_ALIASES[canonical]}
    if normalized not in aliases:
        raise ContractError(
            f"LUT axis {axis_name!r} units are {units!r}; expected units "
            f"equivalent to {canonical!r}"
        )
    return canonical


def validate_reflectance_lut(dataset, *, expected_lap_type=None):
    """Validate a canonical reflectance LUT Dataset without modifying it."""
    _validate_dataset_lut(
        dataset,
        variable_name=c.REFLECTANCE_LUT_VARIABLE,
        required_dims=c.REFLECTANCE_LUT_DIMS,
        optional_dims=(),
        contract_name="reflectance_lut",
        expected_lap_type=expected_lap_type,
    )


def validate_albedo_lut(dataset, *, expected_lap_type=None):
    """Validate a canonical albedo LUT Dataset without modifying it."""
    _validate_dataset_lut(
        dataset,
        variable_name=c.ALBEDO_LUT_VARIABLE,
        required_dims=c.ALBEDO_LUT_REQUIRED_DIMS,
        optional_dims=c.ALBEDO_LUT_OPTIONAL_DIMS,
        contract_name="albedo_lut",
        expected_lap_type=expected_lap_type,
    )


def validate_lut(da):
    """Validate the legacy normalized MATLAB reflectance DataArray.

    This compatibility validator is retained only while `.mat` LUTs remain in
    supported runtime paths. Remove it when MATLAB LUT support ends. New code
    must use `validate_reflectance_lut` or `validate_albedo_lut`.
    """
    if not isinstance(da, xr.DataArray):
        raise_if_violations(
            "legacy_lut",
            [f"legacy LUT must be an xarray.DataArray, got {type(da).__name__}"],
        )

    violations = []
    violations += check_dims_present(da, c.LUT_DIMS)
    violations += check_no_extra_dims(da, c.LUT_DIMS)
    violations += check_dims_order(da, c.LUT_DIMS)
    violations += check_dtype(da, c.ACCEPTED_DTYPES)
    violations += check_coords_present(da, c.LUT_DIMS)
    raise_if_violations("legacy_lut", violations)


def _validate_dataset_lut(
    dataset,
    *,
    variable_name,
    required_dims,
    optional_dims,
    contract_name,
    expected_lap_type,
):
    if not isinstance(dataset, xr.Dataset):
        raise_if_violations(
            contract_name,
            [f"LUT must be an xarray.Dataset, got {type(dataset).__name__}"],
        )

    violations = []
    if variable_name not in dataset.data_vars:
        violations.append(
            f"missing required data variable {variable_name!r} "
            f"(found data variables: {tuple(dataset.data_vars)})"
        )
        raise_if_violations(contract_name, violations)

    array = dataset[variable_name]
    present_optional = tuple(dim for dim in optional_dims if dim in array.dims)
    expected_dims = tuple(required_dims) + present_optional

    violations += check_dims_present(array, required_dims)
    violations += check_no_extra_dims(array, tuple(required_dims) + tuple(optional_dims))
    violations += check_dims_order(array, expected_dims)
    violations += check_dtype(array, c.ACCEPTED_DTYPES)
    violations += check_coords_present(array, expected_dims)

    for dim in expected_dims:
        if dim not in array.coords:
            continue
        coordinate = array.coords[dim]
        violations += _check_coordinate(dim, coordinate)

    lap_coordinate = array.coords.get("lap_concentration")
    if lap_coordinate is not None:
        violations += _check_lap_type(lap_coordinate, expected_lap_type)

    raise_if_violations(contract_name, violations)


def _check_coordinate(axis_name, coordinate):
    violations = []
    if tuple(coordinate.dims) != (axis_name,):
        violations.append(
            f"coordinate {axis_name!r} must be one-dimensional on its own axis"
        )
        return violations

    if axis_name in c.LUT_AXIS_UNITS:
        try:
            canonical_lut_unit(axis_name, coordinate.attrs.get("units"))
        except ContractError as exc:
            violations.append(str(exc))

    values = np.asarray(coordinate.values)
    if axis_name == "band":
        if np.unique(values).size != values.size:
            violations.append("coordinate 'band' must contain unique identifiers")
        return violations

    if not np.issubdtype(values.dtype, np.number):
        violations.append(
            f"coordinate {axis_name!r} dtype is {values.dtype}, expected numeric"
        )
        return violations
    if np.any(~np.isfinite(values)):
        violations.append(f"coordinate {axis_name!r} contains nonfinite value(s)")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        violations.append(
            f"coordinate {axis_name!r} must be strictly increasing for interpolation"
        )
    if axis_name == "skyview" and np.any((values < 0) | (values > 1)):
        violations.append("coordinate 'skyview' values must lie within [0, 1]")
    return violations


def _check_lap_type(coordinate, expected_lap_type):
    lap_type = coordinate.attrs.get(c.LUT_LAP_TYPE_ATTR)
    if not isinstance(lap_type, str) or not lap_type.strip():
        return [
            "coordinate 'lap_concentration' must have a non-empty "
            f"{c.LUT_LAP_TYPE_ATTR!r} attribute"
        ]

    normalized = _normalize_lap_type(lap_type)
    if expected_lap_type is not None:
        if not isinstance(expected_lap_type, str) or not expected_lap_type.strip():
            return ["expected_lap_type must be a non-empty string when supplied"]
        expected = _normalize_lap_type(expected_lap_type)
        if normalized != expected:
            return [
                f"LUT lap_type is {lap_type!r}, expected {expected_lap_type!r}"
            ]
    return []


def _normalize_unit(units):
    return re.sub(r"\s+", "", units.strip().lower())


def _normalize_lap_type(lap_type):
    return re.sub(r"[\s_-]+", "_", lap_type.strip().lower())
