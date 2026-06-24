import numpy as np
import pytest
import xarray as xr

from spires_contract import spectra
from spires_contract._validate import ContractError


def make_target(dims=("y", "x", "band"), dtype=np.float64, with_band_coord=True):
    shape = tuple({"y": 3, "x": 4, "band": 5}[d] for d in dims)
    coords = {"band": np.arange(shape[dims.index("band")])} if with_band_coord and "band" in dims else {}
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords)


def make_solar(dims=("y", "x"), dtype=np.float64):
    shape = tuple({"y": 3, "x": 4}[d] for d in dims)
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims)


def test_validate_target_spectra_accepts_valid():
    spectra.validate_target_spectra(make_target())  # must not raise


def test_validate_target_spectra_rejects_missing_band_dim():
    da = make_target(dims=("y", "x"), with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_rejects_wrong_dtype():
    da = make_target(dtype=np.float32)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "float64" in str(exc.value)


def test_validate_target_spectra_rejects_missing_band_coord():
    da = make_target(with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "band" in str(exc.value)


def test_validate_target_spectra_collects_multiple_violations():
    # wrong dtype AND missing band coordinate -> both reported in one error
    da = make_target(dtype=np.float32, with_band_coord=False)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    text = str(exc.value)
    assert "float64" in text
    assert "band" in text


def test_validate_target_spectra_accepts_any_dim_order():
    # transposed order is still valid (validator checks presence, not order)
    da = make_target(dims=("band", "y", "x"))
    spectra.validate_target_spectra(da)  # must not raise


def test_validate_background_spectra_accepts_valid():
    spectra.validate_background_spectra(make_target())  # same shape rules as target


def test_validate_solar_angles_accepts_valid():
    spectra.validate_solar_angles(make_solar())


def test_validate_solar_angles_rejects_band_dim():
    da = xr.DataArray(np.zeros((3, 4, 5), dtype=np.float64), dims=("y", "x", "band"))
    with pytest.raises(ContractError):
        spectra.validate_solar_angles(da)


def test_conform_target_spectra_transposes_to_canonical_order():
    da = make_target(dims=("band", "y", "x"))
    out = spectra.conform_target_spectra(da)
    assert out.dims == ("y", "x", "band")


def test_conform_target_spectra_casts_dtype():
    da = make_target(dtype=np.float32)
    out = spectra.conform_target_spectra(da)
    assert out.dtype == np.float64


def test_conform_target_spectra_output_passes_validation():
    da = make_target(dims=("band", "y", "x"), dtype=np.float32)
    out = spectra.conform_target_spectra(da)
    spectra.validate_target_spectra(out)  # must not raise


def test_conform_solar_angles_transposes_and_casts():
    da = xr.DataArray(np.zeros((4, 3), dtype=np.float32), dims=("x", "y"))
    out = spectra.conform_solar_angles(da)
    assert out.dims == ("y", "x")
    assert out.dtype == np.float64


def test_conform_target_spectra_raises_when_dim_absent():
    # conform cannot fix a genuinely missing dimension; it should surface that
    da = make_target(dims=("y", "x"), with_band_coord=False)
    with pytest.raises(ContractError):
        spectra.conform_target_spectra(da)


def test_conform_target_spectra_raises_when_band_coord_absent():
    # all dims present but no band coordinate -> conform must refuse, not return invalid
    da = make_target(with_band_coord=False)
    with pytest.raises(ContractError):
        spectra.conform_target_spectra(da)


def test_conform_background_spectra_returns_canonical():
    # smoke test the background delegation: callable, transposes, casts
    da = make_target(dims=("band", "y", "x"), dtype=np.float32)
    out = spectra.conform_background_spectra(da)
    assert out.dims == ("y", "x", "band")
    assert out.dtype == np.float64
    spectra.validate_background_spectra(out)  # must not raise


def test_validate_target_spectra_rejects_extra_dim():
    da = make_target(dims=("y", "x", "band"))
    da = da.expand_dims("time")  # now (time, y, x, band)
    with pytest.raises(ContractError) as exc:
        spectra.validate_target_spectra(da)
    assert "time" in str(exc.value)


def test_validate_solar_angles_rejects_extra_dim():
    da = make_solar().expand_dims("time")  # (time, y, x)
    with pytest.raises(ContractError) as exc:
        spectra.validate_solar_angles(da)
    assert "time" in str(exc.value)


def test_conform_target_spectra_rejects_extra_dim_cleanly():
    # the bug: extra dim used to crash conform with a raw ValueError instead of ContractError
    da = make_target(dims=("y", "x", "band")).expand_dims("time")
    with pytest.raises(ContractError):
        spectra.conform_target_spectra(da)
