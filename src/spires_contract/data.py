"""Neutral data container shared by the SPIReS scientific stages."""

from dataclasses import dataclass, replace
from typing import Optional

import xarray as xr

__all__ = ["SpiresData"]


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

    def with_scene(self, scene: xr.Dataset) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``scene``."""
        return replace(self, scene=scene.copy(deep=False))

    def with_background(self, background: xr.DataArray) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``background``."""
        return replace(self, background=background.copy(deep=False))

    def with_ancillary(self, ancillary: xr.Dataset) -> "SpiresData":
        """Return a replacement carrying a shallow copy of ``ancillary``."""
        return replace(self, ancillary=ancillary.copy(deep=False))

    def with_results(self, results: xr.Dataset) -> "SpiresData":
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
    def valid_mask(self) -> xr.DataArray:
        """Return the canonical inversion-validity mask."""
        return self.scene["valid_inversion_mask"]
