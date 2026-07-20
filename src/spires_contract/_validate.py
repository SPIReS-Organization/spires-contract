"""Reusable validation primitives shared by all boundary contracts.

Each `check_*` function returns a list of human-readable violation strings
(empty list = conforms). Boundary modules compose these and call
`raise_if_violations` so a producer gets ONE error listing EVERY problem.
"""

import numpy as np


class ContractError(ValueError):
    """Raised when a data array violates a SPIReS boundary contract."""


def check_dims_present(da, required_dims):
    """Return a violation per required dim missing from `da.dims`."""
    return [
        f"missing required dimension {dim!r} (found dims: {tuple(da.dims)})"
        for dim in required_dims
        if dim not in da.dims
    ]


def check_dtype(da, accepted_dtypes):
    """Return a violation if `da`'s dtype is not among the accepted dtype(s).

    `accepted_dtypes` may be a single dtype-like or a collection of them. The
    array conforms if its dtype matches any accepted dtype. Passing a single
    dtype preserves the original single-dtype behavior.
    """
    if isinstance(accepted_dtypes, (list, tuple, set, frozenset)):
        accepted = tuple(np.dtype(dt) for dt in accepted_dtypes)
    else:
        accepted = (np.dtype(accepted_dtypes),)
    if np.dtype(da.dtype) not in accepted:
        expected = " or ".join(str(dt) for dt in accepted)
        return [f"dtype is {np.dtype(da.dtype)}, expected {expected}"]
    return []


def check_data_vars_present(dataset, required_vars):
    """Return a violation per required data variable missing from a dataset."""
    return [
        f"missing required data variable {name!r} "
        f"(found data variables: {tuple(dataset.data_vars)})"
        for name in required_vars
        if name not in dataset.data_vars
    ]


def check_coords_present(da, required_coords):
    """Return a violation per required coordinate missing from `da.coords`."""
    return [
        f"missing required coordinate {coord!r} (found coords: {tuple(da.coords)})"
        for coord in required_coords
        if coord not in da.coords
    ]


def check_no_extra_dims(da, allowed_dims):
    """Return a violation if `da` has any dimension not in `allowed_dims`."""
    extra = [d for d in da.dims if d not in allowed_dims]
    if extra:
        return [f"unexpected dimension(s) {extra!r} (allowed: {tuple(allowed_dims)})"]
    return []


def check_coords_match(
    reference,
    candidate,
    required_coords,
    *,
    reference_name="reference",
    candidate_name="candidate",
):
    """Return violations when required coordinates do not match exactly.

    Coordinate attributes are not part of alignment, but dimension structure,
    dtype, shape, and values are. Missing coordinates are reported here so the
    helper is independently actionable.
    """
    violations = []
    for coord in required_coords:
        reference_has_coord = coord in reference.coords
        candidate_has_coord = coord in candidate.coords
        if not reference_has_coord:
            violations.append(f"{reference_name} is missing coordinate {coord!r}")
        if not candidate_has_coord:
            violations.append(f"{candidate_name} is missing coordinate {coord!r}")
        if not (reference_has_coord and candidate_has_coord):
            continue

        reference_coord = reference.coords[coord]
        candidate_coord = candidate.coords[coord]
        matches = (
            tuple(reference_coord.dims) == tuple(candidate_coord.dims)
            and reference_coord.shape == candidate_coord.shape
            and np.dtype(reference_coord.dtype) == np.dtype(candidate_coord.dtype)
            and reference_coord.equals(candidate_coord)
        )
        if not matches:
            violations.append(
                f"coordinate {coord!r} on {candidate_name} does not exactly match "
                f"{reference_name}"
            )
    return violations


def check_binary_mask(da):
    """Return violations unless an array is Boolean or integer-valued 0/1."""
    dtype = np.dtype(da.dtype)
    if np.issubdtype(dtype, np.bool_):
        return []
    if not np.issubdtype(dtype, np.integer):
        return [f"mask dtype is {dtype}, expected bool or an integer dtype"]

    values = np.asarray(da.values)
    invalid = values[(values != 0) & (values != 1)]
    if invalid.size:
        found = np.unique(invalid)
        return [f"integer mask contains value(s) outside 0/1: {found.tolist()}"]
    return []


def check_cluster_labels(da):
    """Return violations for invalid cluster-label dtype or label values."""
    dtype = np.dtype(da.dtype)
    if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return [f"cluster-label dtype is {dtype}, expected an integer dtype"]

    values = np.asarray(da.values)
    violations = []
    below_sentinel = values[values < -1]
    if below_sentinel.size:
        found = np.unique(below_sentinel)
        violations.append(
            f"cluster labels contain value(s) below the -1 sentinel: {found.tolist()}"
        )

    labels = np.unique(values[values >= 0])
    expected = np.arange(labels.size, dtype=labels.dtype)
    if not np.array_equal(labels, expected):
        violations.append(
            "nonnegative cluster labels must be contiguous from 0; "
            f"found {labels.tolist()}"
        )
    return violations


def check_dims_order(da, required_dims):
    """Return a violation if `da.dims` is not exactly `required_dims`, in order.

    Order is part of the contract: the C++ inversion kernel indexes arrays
    positionally, so `(band, y, x)` is as wrong as a missing dimension. This is
    a cheap tuple comparison. Only reports an ordering violation when the same
    set of dims is present — missing/extra dims are left to the dedicated
    checks so errors don't double up.
    """
    if set(da.dims) == set(required_dims) and tuple(da.dims) != tuple(required_dims):
        return [
            f"dimensions {tuple(da.dims)} are not in canonical order "
            f"{tuple(required_dims)}"
        ]
    return []


def raise_if_violations(contract_name, violations):
    """Raise a single ContractError listing all violations, if any."""
    if violations:
        bullets = "\n".join(f"  - {v}" for v in violations)
        raise ContractError(
            f"{contract_name} contract violated:\n{bullets}"
        )
