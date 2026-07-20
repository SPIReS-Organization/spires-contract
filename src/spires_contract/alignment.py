"""Coordinate-alignment checks shared by SPIReS boundary validators."""

from spires_contract import conventions as c
from spires_contract._validate import (
    check_coords_match,
    check_coords_present,
    raise_if_violations,
)
from spires_contract.data import SpiresData

__all__ = ["validate_coordinate_alignment", "validate_spatial_alignment"]


def validate_coordinate_alignment(
    reference,
    candidate,
    required_coords,
    *,
    reference_name="reference",
    candidate_name="candidate",
    contract_name="coordinate_alignment",
):
    """Require exact coordinate alignment without changing either object."""
    violations = check_coords_match(
        reference,
        candidate,
        required_coords,
        reference_name=reference_name,
        candidate_name=candidate_name,
    )
    raise_if_violations(contract_name, violations)


def validate_spatial_alignment(data):
    """Validate exact spatial and spectral alignment within ``SpiresData``."""
    if not isinstance(data, SpiresData):
        raise_if_violations(
            "spatial_alignment",
            [f"data must be a SpiresData instance, got {type(data).__name__}"],
        )

    scene = data.scene
    if "reflectance" not in scene.data_vars:
        raise_if_violations(
            "spatial_alignment",
            ["scene is missing 'reflectance', the coordinate reference variable"],
        )

    reference = scene["reflectance"]
    violations = check_coords_present(reference, c.SPECTRA_DIMS)

    for name in ("solar_zenith", "valid_inversion_mask"):
        if name in scene.data_vars:
            violations += check_coords_match(
                reference,
                scene[name],
                c.SPATIAL_DIMS,
                reference_name="scene reflectance",
                candidate_name=name,
            )

    if data.background is not None:
        violations += check_coords_match(
            reference,
            data.background,
            c.SPECTRA_DIMS,
            reference_name="scene reflectance",
            candidate_name="background",
        )

    for field_name in ("ancillary", "results"):
        dataset = getattr(data, field_name)
        if dataset is None:
            continue
        for variable_name, array in dataset.data_vars.items():
            spatial_dims = set(c.SPATIAL_DIMS).intersection(array.dims)
            if not spatial_dims:
                continue
            if spatial_dims != set(c.SPATIAL_DIMS):
                violations.append(
                    f"{field_name}.{variable_name} uses only part of the spatial "
                    f"dimensions; found {tuple(array.dims)}"
                )
                continue
            violations += check_coords_match(
                reference,
                array,
                c.SPATIAL_DIMS,
                reference_name="scene reflectance",
                candidate_name=f"{field_name}.{variable_name}",
            )

    raise_if_violations("spatial_alignment", violations)
