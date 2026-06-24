import numpy as np
import pytest
import xarray as xr

from spires_contract._validate import (
    ContractError,
    check_dims_present,
    check_dtype,
    check_coords_present,
    raise_if_violations,
)


def _da(dims, coords=None, dtype=np.float64):
    shape = tuple(2 for _ in dims)
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords or {})


def test_check_dims_present_no_violation():
    da = _da(("y", "x", "band"))
    assert check_dims_present(da, ("y", "x", "band")) == []


def test_check_dims_present_reports_missing():
    da = _da(("y", "x"))
    msgs = check_dims_present(da, ("y", "x", "band"))
    assert len(msgs) == 1
    assert "band" in msgs[0]


def test_check_dtype_no_violation():
    da = _da(("y", "x"), dtype=np.float64)
    assert check_dtype(da, np.float64) == []


def test_check_dtype_reports_mismatch():
    da = _da(("y", "x"), dtype=np.float32)
    msgs = check_dtype(da, np.float64)
    assert len(msgs) == 1
    assert "float64" in msgs[0] and "float32" in msgs[0]


def test_check_coords_present_reports_missing():
    da = _da(("y", "x", "band"))  # no coords assigned
    msgs = check_coords_present(da, ("band",))
    assert len(msgs) == 1
    assert "band" in msgs[0]


def test_check_coords_present_no_violation():
    da = _da(("band",), coords={"band": [1, 2]})
    assert check_coords_present(da, ("band",)) == []


def test_raise_if_violations_raises_with_all_messages():
    with pytest.raises(ContractError) as exc:
        raise_if_violations("target_spectra", ["missing dim 'band'", "dtype is float32"])
    text = str(exc.value)
    assert "target_spectra" in text
    assert "missing dim 'band'" in text
    assert "dtype is float32" in text


def test_raise_if_violations_silent_when_empty():
    raise_if_violations("target_spectra", [])  # must not raise
