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


def check_dtype(da, required_dtype):
    """Return a violation if `da` is not the required dtype."""
    if np.dtype(da.dtype) != np.dtype(required_dtype):
        return [
            f"dtype is {np.dtype(da.dtype)}, expected {np.dtype(required_dtype)}"
        ]
    return []


def check_coords_present(da, required_coords):
    """Return a violation per required coordinate missing from `da.coords`."""
    return [
        f"missing required coordinate {coord!r} (found coords: {tuple(da.coords)})"
        for coord in required_coords
        if coord not in da.coords
    ]


def raise_if_violations(contract_name, violations):
    """Raise a single ContractError listing all violations, if any."""
    if violations:
        bullets = "\n".join(f"  - {v}" for v in violations)
        raise ContractError(
            f"{contract_name} contract violated:\n{bullets}"
        )
