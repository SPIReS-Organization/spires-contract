"""QA mask contract: a fixed vocabulary of per-pixel boolean quality flags.

``qa_mask`` is a ``(flag, y, x)`` boolean DataArray whose ``flag`` coordinate
is exactly :data:`QA_FLAGS`. Producers map sensor-specific QA products
(Landsat QA_PIXEL, HLS Fmask, Sentinel-2 SCL, ...) onto this vocabulary; a
flag the source cannot express is all-False and omitted from the optional
``qa_flags_available`` attribute (whitespace-separated subset of QA_FLAGS).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "QA_FLAGS",
    "QA_INVALID_FLAGS",
    "CLOUD_FLAGS",
    "validate_qa_mask",
]

from spires_contract import conventions as c
from spires_contract._validate import check_dtype, raise_if_violations

QA_FLAGS = (
    "fill",
    "saturated",
    "cloud",
    "cirrus",
    "cloud_buffer",
    "cloud_shadow",
    "snow",
    "water",
)
"""Canonical flag vocabulary and the required order of the ``flag`` coordinate."""

QA_INVALID_FLAGS = (
    "fill",
    "saturated",
    "cloud",
    "cirrus",
    "cloud_buffer",
    "cloud_shadow",
)
"""Flags that mark a pixel as not invertible. ``snow`` and ``water`` are descriptive."""

CLOUD_FLAGS = ("cloud", "cirrus", "cloud_buffer", "cloud_shadow")
"""Flags that together form a cloud mask (excludes fill/saturation)."""


def validate_qa_mask(da: xr.DataArray) -> None:
    """Raise :class:`ContractError` unless ``da`` is a conforming ``qa_mask``."""
    violations = []
    if tuple(da.dims) != c.QA_MASK_DIMS:
        violations.append(f"dims are {tuple(da.dims)}, expected {c.QA_MASK_DIMS}")
    violations += check_dtype(da, np.bool_)
    for dim in c.QA_MASK_DIMS:
        if dim in da.dims and dim not in da.coords:
            violations.append(f"missing coordinate for dim {dim!r}")
    # Compare the vocabulary only when the flag coordinate exists; otherwise
    # da["flag"] falls back to positional indices and the message misleads.
    if "flag" in da.dims and "flag" in da.coords:
        flags = tuple(str(f) for f in da["flag"].values)
        if flags != QA_FLAGS:
            violations.append(
                f"flag coordinate is {flags}, expected exactly {QA_FLAGS}"
            )
    available = da.attrs.get("qa_flags_available")
    if available is not None:
        unknown = set(str(available).split()) - set(QA_FLAGS)
        if unknown:
            violations.append(
                f"qa_flags_available names unknown flag(s) {sorted(unknown)}"
            )
    raise_if_violations("qa_mask", violations)
