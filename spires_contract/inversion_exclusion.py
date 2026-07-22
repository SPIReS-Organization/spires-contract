"""Packed provenance for single-scene inversion exclusions."""

import numpy as np
import xarray as xr

from spires_contract import conventions as c
from spires_contract._validate import (
    check_binary_mask,
    check_coords_match,
    check_coords_present,
    check_dims_order,
    check_dims_present,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)

__all__ = [
    "inversion_exclusion_metadata",
    "validate_inversion_exclusion",
]


_LONG_NAMES = {
    c.INVERSION_EXCLUSION_FLAGS_VARIABLE: "Inversion Exclusion Flags",
    c.INVERSION_EXCLUSION_ASSESSED_VARIABLE: "Inversion Exclusion Conditions Assessed",
}


def inversion_exclusion_metadata(variable_name):
    """Return canonical metadata for one packed inversion-exclusion layer."""
    if variable_name not in _LONG_NAMES:
        expected = tuple(_LONG_NAMES)
        raise ValueError(
            f"unknown inversion-exclusion variable {variable_name!r}; "
            f"expected one of {expected!r}"
        )

    meanings = c.INVERSION_EXCLUSION_REASONS
    if variable_name == c.INVERSION_EXCLUSION_ASSESSED_VARIABLE:
        meanings = tuple(f"{reason}_assessed" for reason in meanings)

    return {
        "long_name": _LONG_NAMES[variable_name],
        "flag_masks": np.asarray(
            tuple(c.INVERSION_EXCLUSION_BITS.values()),
            dtype=c.INVERSION_EXCLUSION_DTYPE,
        ),
        "flag_meanings": " ".join(meanings),
        c.INVERSION_EXCLUSION_SCHEMA_ATTR: c.INVERSION_EXCLUSION_SCHEMA_VERSION,
    }


def validate_inversion_exclusion(scene):
    """Validate the optional atomic exclusion-provenance set on a scene."""
    if not isinstance(scene, xr.Dataset):
        raise_if_violations(
            "inversion_exclusion",
            [f"scene must be an xarray.Dataset, got {type(scene).__name__}"],
        )

    present = tuple(
        name for name in c.INVERSION_EXCLUSION_VARIABLES if name in scene.data_vars
    )
    if not present:
        return

    violations = []
    missing = tuple(
        name for name in c.INVERSION_EXCLUSION_VARIABLES if name not in scene.data_vars
    )
    if missing:
        violations.append(
            "inversion exclusion provenance is atomic: expected all of "
            f"{c.INVERSION_EXCLUSION_VARIABLES!r} or none; missing {missing!r}"
        )

    flags = scene.get(c.INVERSION_EXCLUSION_FLAGS_VARIABLE)
    assessed = scene.get(c.INVERSION_EXCLUSION_ASSESSED_VARIABLE)
    valid_mask = scene.get(c.VALID_INVERSION_MASK_VARIABLE)

    for name, array in (
        (c.INVERSION_EXCLUSION_FLAGS_VARIABLE, flags),
        (c.INVERSION_EXCLUSION_ASSESSED_VARIABLE, assessed),
    ):
        if array is None:
            continue
        violations += _check_array_structure(array)
        violations += check_dtype(array, c.INVERSION_EXCLUSION_DTYPE)
        violations += _check_metadata(name, array)
        violations += _check_unknown_bits(name, array)

    if valid_mask is not None:
        violations += _check_array_structure(valid_mask)
        violations += check_binary_mask(valid_mask)

    reference = next(
        (array for array in (flags, assessed, valid_mask) if array is not None),
        None,
    )
    if reference is not None:
        for name, array in (
            (c.INVERSION_EXCLUSION_FLAGS_VARIABLE, flags),
            (c.INVERSION_EXCLUSION_ASSESSED_VARIABLE, assessed),
            (c.VALID_INVERSION_MASK_VARIABLE, valid_mask),
        ):
            if array is None or array is reference:
                continue
            violations += check_coords_match(
                reference,
                array,
                c.SPATIAL_DIMS,
                reference_name=reference.name or "inversion exclusion reference",
                candidate_name=name,
            )

        if "reflectance" in scene.data_vars:
            for name, array in (
                (c.INVERSION_EXCLUSION_FLAGS_VARIABLE, flags),
                (c.INVERSION_EXCLUSION_ASSESSED_VARIABLE, assessed),
                (c.VALID_INVERSION_MASK_VARIABLE, valid_mask),
            ):
                if array is None:
                    continue
                violations += check_coords_match(
                    scene["reflectance"],
                    array,
                    c.SPATIAL_DIMS,
                    reference_name="scene reflectance",
                    candidate_name=name,
                )

    if _arrays_are_comparable(flags, assessed):
        flag_values = np.asarray(flags.values, dtype=c.INVERSION_EXCLUSION_DTYPE)
        assessed_values = np.asarray(
            assessed.values, dtype=c.INVERSION_EXCLUSION_DTYPE
        )
        flags_without_assessment = np.bitwise_and(
            flag_values,
            np.bitwise_not(assessed_values),
        )
        if np.any(flags_without_assessment):
            violations.append(
                "inversion_exclusion_flags contains set bit(s) whose matching "
                "inversion_exclusion_assessed bit is not set"
            )

    if _arrays_are_comparable(flags, valid_mask):
        flag_values = np.asarray(flags.values, dtype=c.INVERSION_EXCLUSION_DTYPE)
        valid_values = np.asarray(valid_mask.values, dtype=bool)
        expected_valid = flag_values == 0
        if not np.array_equal(valid_values, expected_valid):
            violations.append(
                "valid_inversion_mask must be true exactly where "
                "inversion_exclusion_flags is zero"
            )

    raise_if_violations("inversion_exclusion", violations)


def _check_array_structure(array):
    violations = []
    violations += check_dims_present(array, c.SPATIAL_DIMS)
    violations += check_no_extra_dims(array, c.SPATIAL_DIMS)
    violations += check_dims_order(array, c.SPATIAL_DIMS)
    violations += check_coords_present(array, c.SPATIAL_DIMS)
    return violations


def _check_metadata(name, array):
    expected = inversion_exclusion_metadata(name)
    violations = []
    for attr_name in ("long_name", "flag_meanings", c.INVERSION_EXCLUSION_SCHEMA_ATTR):
        if array.attrs.get(attr_name) != expected[attr_name]:
            violations.append(
                f"{name} attribute {attr_name!r} must be {expected[attr_name]!r}"
            )

    actual_masks = array.attrs.get("flag_masks")
    actual_masks_array = (
        None if actual_masks is None else np.asarray(actual_masks)
    )
    if (
        actual_masks_array is None
        or actual_masks_array.dtype != c.INVERSION_EXCLUSION_DTYPE
        or not np.array_equal(actual_masks_array, expected["flag_masks"])
    ):
        violations.append(
            f"{name} attribute 'flag_masks' must equal "
            f"{expected['flag_masks'].tolist()!r}"
        )
    return violations


def _check_unknown_bits(name, array):
    if np.dtype(array.dtype) != c.INVERSION_EXCLUSION_DTYPE:
        return []
    values = np.asarray(array.values, dtype=c.INVERSION_EXCLUSION_DTYPE)
    unknown_mask = np.asarray(
        np.iinfo(c.INVERSION_EXCLUSION_DTYPE).max
        ^ c.INVERSION_EXCLUSION_KNOWN_MASK,
        dtype=c.INVERSION_EXCLUSION_DTYPE,
    )
    unknown_values = np.bitwise_and(values, unknown_mask)
    if np.any(unknown_values):
        return [f"{name} contains reserved or unknown set bit(s)"]
    return []


def _arrays_are_comparable(first, second):
    return (
        first is not None
        and second is not None
        and tuple(first.dims) == c.SPATIAL_DIMS
        and tuple(second.dims) == c.SPATIAL_DIMS
        and first.shape == second.shape
    )
