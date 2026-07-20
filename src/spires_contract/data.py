"""Neutral data container shared by the SPIReS scientific stages."""

from dataclasses import dataclass, replace
from typing import Optional

import xarray as xr

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
from spires_contract.spectra import (
    validate_background_spectra,
    validate_solar_angles,
    validate_target_spectra,
)

__all__ = ["SpiresData", "validate_for_inversion", "validate_spires_data"]


@dataclass(frozen=True)
class SpiresData:
    """Immutable envelope carrying one single scene through SPIReS stages.

    The object deliberately permits partially populated lifecycle states.
    Package-boundary validators enforce stage-specific requirements.
    """

    scene: xr.Dataset
    background: Optional[xr.DataArray] = None
    ancillary: Optional[xr.Dataset] = None
    results: Optional[xr.Dataset] = None

    def __post_init__(self):
        violations = []
        if not isinstance(self.scene, xr.Dataset):
            violations.append(
                f"scene must be an xarray.Dataset, got {type(self.scene).__name__}"
            )
        if self.background is not None and not isinstance(
            self.background, xr.DataArray
        ):
            violations.append(
                "background must be an xarray.DataArray or None, "
                f"got {type(self.background).__name__}"
            )
        if self.ancillary is not None and not isinstance(self.ancillary, xr.Dataset):
            violations.append(
                "ancillary must be an xarray.Dataset or None, "
                f"got {type(self.ancillary).__name__}"
            )
        if self.results is not None and not isinstance(self.results, xr.Dataset):
            violations.append(
                "results must be an xarray.Dataset or None, "
                f"got {type(self.results).__name__}"
            )
        if violations:
            bullets = "\n".join(f"  - {violation}" for violation in violations)
            raise TypeError(f"invalid SpiresData field type(s):\n{bullets}")

    def assign_scene(self, scene: xr.Dataset) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``scene``."""
        return replace(self, scene=scene.copy(deep=False))

    def assign_background(self, background: xr.DataArray) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``background``."""
        return replace(self, background=background.copy(deep=False))

    def assign_ancillary(self, ancillary: xr.Dataset) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``ancillary``."""
        return replace(self, ancillary=ancillary.copy(deep=False))

    def assign_results(self, results: xr.Dataset) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``results``."""
        return replace(self, results=results.copy(deep=False))

    @property
    def target_spectra(self) -> xr.DataArray:
        """Return the canonical target-reflectance variable."""
        return self.scene["reflectance"]

    @property
    def solar_zenith(self) -> xr.DataArray:
        """Return the canonical solar-zenith variable."""
        return self.scene["solar_zenith"]

    @property
    def valid_mask(self) -> Optional[xr.DataArray]:
        """Return the optional inversion-validity mask when present."""
        return self.scene.get("valid_inversion_mask")


def validate_spires_data(data):
    """Validate the shared container type without requiring a lifecycle stage."""
    if not isinstance(data, SpiresData):
        raise_if_violations(
            "spires_data",
            [f"data must be a SpiresData instance, got {type(data).__name__}"],
        )


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

    from spires_contract.alignment import validate_spatial_alignment
    from spires_contract.clusters import validate_clusters

    _collect_contract_error(violations, validate_spatial_alignment, data)
    _collect_contract_error(violations, validate_clusters, data)
    raise_if_violations("inversion_input", violations)


def _collect_contract_error(violations, validator, value):
    try:
        validator(value)
    except ContractError as exc:
        violations.append(str(exc))
