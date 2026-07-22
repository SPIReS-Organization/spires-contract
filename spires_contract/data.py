"""Neutral data container shared by the SPIReS scientific stages."""

from dataclasses import dataclass, replace
from typing import Optional

import xarray as xr

from spires_contract._validate import raise_if_violations

__all__ = ["SpiresData", "validate_spires_data"]


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


def validate_spires_data(data):
    """Validate the shared container and optional atomic scene contracts."""
    if not isinstance(data, SpiresData):
        raise_if_violations(
            "spires_data",
            [f"data must be a SpiresData instance, got {type(data).__name__}"],
        )

    # Local import avoids making the neutral exclusion validator depend on the
    # SpiresData class while preserving a simple public validation entry point.
    from spires_contract.inversion_exclusion import validate_inversion_exclusion

    validate_inversion_exclusion(data.scene)
