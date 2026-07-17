import numpy as np
import pytest
import xarray as xr

from spires_contract import lut
from spires_contract._validate import ContractError

# Canonical LUT dims (band, solar_angle, dust_concentration, grain_size).
_DIMS = ("band", "solar_angle", "dust_concentration", "grain_size")
_SIZES = {"band": 9, "solar_angle": 4, "dust_concentration": 3, "grain_size": 5}


def make_lut(dims=_DIMS, dtype=np.float32, coords=_DIMS):
    """A tiny LUT DataArray in canonical form. `coords` lists which dimension
    coordinates to attach (default: all)."""
    shape = tuple(_SIZES[d] for d in dims)
    coord_map = {d: np.arange(_SIZES[d], dtype=np.float64) for d in coords if d in dims}
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coord_map)


def test_validate_lut_accepts_valid():
    lut.validate_lut(make_lut())  # must not raise


def test_validate_lut_accepts_float32():
    lut.validate_lut(make_lut(dtype=np.float32))  # canonical dtype; must not raise


def test_validate_lut_rejects_float64():
    # float64 is no longer accepted: the boundary is float32-only.
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(make_lut(dtype=np.float64))
    assert "float64" in str(exc.value)


def test_validate_lut_rejects_non_float_dtype():
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(make_lut(dtype=np.int32))
    text = str(exc.value)
    assert "int32" in text and "float32" in text  # actual + expected dtype named


def test_validate_lut_rejects_wrong_dim_order():
    # order is part of the contract: a transposed LUT must be rejected
    da = make_lut(dims=("solar_angle", "band", "dust_concentration", "grain_size"))
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "canonical order" in str(exc.value)


def test_validate_lut_rejects_missing_dim():
    da = make_lut(dims=("band", "solar_angle", "dust_concentration"))
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "grain_size" in str(exc.value)


def test_validate_lut_rejects_extra_dim():
    da = make_lut().expand_dims("time")  # (time, band, solar_angle, ...)
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "time" in str(exc.value)


def test_validate_lut_rejects_missing_coordinate():
    # interpolator reads coord values to locate query points; a missing one is a violation
    da = make_lut(coords=("band", "solar_angle", "dust_concentration"))  # no grain_size coord
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "grain_size" in str(exc.value)


def test_validate_lut_collects_multiple_violations():
    # wrong dtype AND missing coordinate -> both reported in one error
    da = make_lut(dtype=np.int32, coords=("band", "solar_angle", "dust_concentration"))
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    text = str(exc.value)
    assert "int32" in text
    assert "grain_size" in text
