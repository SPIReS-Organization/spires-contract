"""Validation for the complete I/O-to-inversion data boundary."""

from spires_contract import conventions as c
from spires_contract._validate import (
    ContractError,
    check_binary_mask,
    check_coords_present,
    check_data_vars_present,
    check_dims_order,
    check_dims_present,
    check_no_extra_dims,
    raise_if_violations,
)
from spires_contract.alignment import validate_spatial_alignment
from spires_contract.clusters import validate_clusters
from spires_contract.data import validate_spires_data
from spires_contract.spectra import (
    validate_background_spectra,
    validate_solar_angles,
    validate_target_spectra,
)

__all__ = ["validate_for_inversion"]


def validate_for_inversion(data):
    """Validate the complete single-scene input boundary for inversion."""
    validate_spires_data(data)
    scene = data.scene
    violations = check_data_vars_present(scene, c.REQUIRED_SCENE_VARIABLES)

    target = scene.get("reflectance")
    if target is not None:
        _collect_contract_error(violations, validate_target_spectra, target)
        violations += check_coords_present(target, c.SPECTRA_DIMS)

    solar = scene.get("solar_zenith")
    if solar is not None:
        _collect_contract_error(violations, validate_solar_angles, solar)
        violations += check_coords_present(solar, c.SPATIAL_DIMS)

    valid_mask = scene.get("valid_inversion_mask")
    if valid_mask is not None:
        violations += check_dims_present(valid_mask, c.SPATIAL_DIMS)
        violations += check_no_extra_dims(valid_mask, c.SPATIAL_DIMS)
        violations += check_dims_order(valid_mask, c.SPATIAL_DIMS)
        violations += check_coords_present(valid_mask, c.SPATIAL_DIMS)
        violations += check_binary_mask(valid_mask)

    if data.background is None:
        violations.append("background is required before inversion")
    else:
        _collect_contract_error(
            violations, validate_background_spectra, data.background
        )
        violations += check_coords_present(data.background, c.SPECTRA_DIMS)

    _collect_contract_error(violations, validate_spatial_alignment, data)
    _collect_contract_error(violations, validate_clusters, data)
    raise_if_violations("inversion_input", violations)


def _collect_contract_error(violations, validator, value):
    try:
        validator(value)
    except ContractError as exc:
        violations.append(str(exc))
