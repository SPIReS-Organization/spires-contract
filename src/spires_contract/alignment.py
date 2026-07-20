"""Coordinate-alignment checks shared by SPIReS boundary validators."""

from spires_contract._validate import check_coords_match, raise_if_violations

__all__ = ["validate_coordinate_alignment"]


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
