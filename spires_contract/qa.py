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

QA_INVALID_FLAGS = ("fill", "saturated", "cloud", "cirrus", "cloud_buffer", "cloud_shadow")
"""Flags that mark a pixel as not invertible. ``snow`` and ``water`` are descriptive."""

CLOUD_FLAGS = ("cloud", "cirrus", "cloud_buffer", "cloud_shadow")
"""Flags that together form a cloud mask (excludes fill/saturation)."""

QA_MASK_DIMS = ("flag", "y", "x")


def validate_qa_mask(da: xr.DataArray) -> None:
    """Raise :class:`ContractError` unless ``da`` is a conforming ``qa_mask``."""
    violations = []
    if tuple(da.dims) != QA_MASK_DIMS:
        violations.append(f"dims are {tuple(da.dims)}, expected {QA_MASK_DIMS}")
    violations += check_dtype(da, np.bool_)
    if "flag" in da.dims:
        flags = tuple(str(f) for f in da["flag"].values)
        if flags != QA_FLAGS:
            violations.append(f"flag coordinate is {flags}, expected exactly {QA_FLAGS}")
    for dim in ("y", "x"):
        if dim in da.dims and dim not in da.coords:
            violations.append(f"missing coordinate for dim {dim!r}")
    available = da.attrs.get("qa_flags_available")
    if available is not None:
        unknown = set(str(available).split()) - set(QA_FLAGS)
        if unknown:
            violations.append(f"qa_flags_available names unknown flag(s) {sorted(unknown)}")
    raise_if_violations("qa_mask", violations)
