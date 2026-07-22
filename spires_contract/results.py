"""Canonical inversion-result contract for downstream SPIReS stages."""

import numpy as np
import xarray as xr

from spires_contract import conventions as c
from spires_contract._validate import (
    check_binary_mask,
    check_coords_match,
    check_coords_present,
    check_data_vars_present,
    check_dims_order,
    check_dims_present,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)

__all__ = ["validate_results"]


def validate_results(results, *, scene=None, eligibility_mask=None):
    """Validate canonical base inversion results without modifying them."""
    if not isinstance(results, xr.Dataset):
        raise_if_violations(
            "results",
            [f"results must be an xarray.Dataset, got {type(results).__name__}"],
        )

    violations = check_data_vars_present(results, c.RESULT_VARIABLES)
    base_arrays = {
        name: results[name] for name in c.RESULT_VARIABLES if name in results.data_vars
    }

    reference_name = next(iter(base_arrays), None)
    reference = base_arrays.get(reference_name) if reference_name is not None else None
    for name, array in base_arrays.items():
        violations += _check_result_array(name, array)
        if reference is not None and name != reference_name:
            violations += check_coords_match(
                reference,
                array,
                c.SPATIAL_DIMS,
                reference_name=reference_name,
                candidate_name=name,
            )

    if scene is not None:
        if not isinstance(scene, xr.Dataset):
            violations.append(
                f"scene must be an xarray.Dataset, got {type(scene).__name__}"
            )
        else:
            violations += check_coords_present(scene, c.SPATIAL_DIMS)
            for name, array in base_arrays.items():
                violations += check_coords_match(
                    scene,
                    array,
                    c.SPATIAL_DIMS,
                    reference_name="scene",
                    candidate_name=name,
                )

    mask_is_valid = False
    if eligibility_mask is not None:
        if not isinstance(eligibility_mask, xr.DataArray):
            violations.append(
                "eligibility_mask must be an xarray.DataArray, "
                f"got {type(eligibility_mask).__name__}"
            )
        else:
            mask_violations = _check_eligibility_mask(eligibility_mask)
            violations += mask_violations
            mask_is_valid = not mask_violations
            if reference is not None:
                alignment_violations = check_coords_match(
                    reference,
                    eligibility_mask,
                    c.SPATIAL_DIMS,
                    reference_name=reference_name,
                    candidate_name="eligibility_mask",
                )
                violations += alignment_violations
                mask_is_valid = mask_is_valid and not alignment_violations

    if mask_is_valid:
        eligible = np.asarray(eligibility_mask.values, dtype=bool)
        outside = ~eligible
        for name, array in base_arrays.items():
            values = np.asarray(array.values)
            if tuple(array.dims) != c.RESULT_DIMS or values.shape != eligible.shape:
                continue
            if not np.issubdtype(values.dtype, np.floating):
                continue
            if np.any(~np.isnan(values[outside])):
                violations.append(
                    f"{name} must be NaN outside the supplied eligibility mask"
                )

    raise_if_violations("results", violations)


def _check_result_array(name, array):
    violations = []
    violations += check_dims_present(array, c.RESULT_DIMS)
    violations += check_no_extra_dims(array, c.RESULT_DIMS)
    violations += check_dims_order(array, c.RESULT_DIMS)
    violations += check_dtype(array, c.ACCEPTED_DTYPES)
    violations += check_coords_present(array, c.SPATIAL_DIMS)
    violations += _check_result_metadata(name, array)

    values = np.asarray(array.values)
    if not np.issubdtype(values.dtype, np.floating):
        return violations
    if np.any(np.isinf(values)):
        violations.append(f"{name} contains infinite value(s)")

    finite = values[np.isfinite(values)]
    if name in {"fsnow", "fshade"} and np.any((finite < 0) | (finite > 1)):
        violations.append(f"finite {name} values must lie within [0, 1]")
    if name in {"lap_concentration", "grain_radius"} and np.any(finite < 0):
        violations.append(f"finite {name} values must be nonnegative")
    return violations


def _check_result_metadata(name, array):
    violations = []
    expected_long_name = c.RESULT_LONG_NAMES[name]
    if array.attrs.get("long_name") != expected_long_name:
        violations.append(
            f"{name} attribute 'long_name' must be {expected_long_name!r}"
        )

    units = array.attrs.get("units")
    if name == "grain_radius":
        if units not in c.GRAIN_RADIUS_UNIT_ALIASES:
            violations.append(
                f"grain_radius attribute 'units' must be one of "
                f"{c.GRAIN_RADIUS_UNIT_ALIASES!r}"
            )
    elif units != c.RESULT_UNITS[name]:
        violations.append(
            f"{name} attribute 'units' must be {c.RESULT_UNITS[name]!r}"
        )

    if name == "lap_concentration":
        lap_type = array.attrs.get("lap_type")
        if lap_type != c.INITIAL_LAP_TYPE:
            violations.append(
                "lap_concentration attribute 'lap_type' must be "
                f"{c.INITIAL_LAP_TYPE!r}"
            )
    return violations


def _check_eligibility_mask(mask):
    violations = []
    violations += check_dims_present(mask, c.SPATIAL_DIMS)
    violations += check_no_extra_dims(mask, c.SPATIAL_DIMS)
    violations += check_dims_order(mask, c.SPATIAL_DIMS)
    violations += check_coords_present(mask, c.SPATIAL_DIMS)
    violations += check_binary_mask(mask)
    return violations
