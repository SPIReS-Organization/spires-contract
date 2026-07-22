import numpy as np
import pytest
import xarray as xr

from spires_contract import lut
from spires_contract import conventions as c
from spires_contract._validate import ContractError

# Legacy normalized MATLAB LUT dims retained for validate_lut().
_DIMS = ("band", "solar_angle", "lap_concentration", "grain_size")
_SIZES = {"band": 9, "solar_angle": 4, "lap_concentration": 3, "grain_size": 5}


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
    da = make_lut(dims=("solar_angle", "band", "lap_concentration", "grain_size"))
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "canonical order" in str(exc.value)


def test_validate_lut_rejects_missing_dim():
    da = make_lut(dims=("band", "solar_angle", "lap_concentration"))
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
    da = make_lut(coords=("band", "solar_angle", "lap_concentration"))  # no grain_size coord
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    assert "grain_size" in str(exc.value)


def test_validate_lut_collects_multiple_violations():
    # wrong dtype AND missing coordinate -> both reported in one error
    da = make_lut(dtype=np.int32, coords=("band", "solar_angle", "lap_concentration"))
    with pytest.raises(ContractError) as exc:
        lut.validate_lut(da)
    text = str(exc.value)
    assert "int32" in text
    assert "grain_size" in text


def make_reflectance_dataset(
    *,
    dims=c.REFLECTANCE_LUT_DIMS,
    dtype=np.float32,
    sqrt_units="um^0.5",
    lap_type="dust",
):
    sizes = {
        "band": 2,
        "solar_angle": 3,
        "lap_concentration": 4,
        "sqrt_grain_radius": 5,
        "grain_size": 5,
    }
    coords = {
        "band": ["b1", "b2"],
        "solar_angle": xr.DataArray(
            [0.0, 30.0, 60.0], dims="solar_angle", attrs={"units": "degrees"}
        ),
        "lap_concentration": xr.DataArray(
            np.linspace(0, 1000, 4),
            dims="lap_concentration",
            attrs={"units": "ppm", "lap_type": lap_type},
        ),
        "sqrt_grain_radius": xr.DataArray(
            np.linspace(5, 40, 5),
            dims="sqrt_grain_radius",
            attrs={"units": sqrt_units},
        ),
        "grain_size": xr.DataArray(
            np.linspace(25, 1600, 5),
            dims="grain_size",
            attrs={"units": "um"},
        ),
    }
    shape = tuple(sizes[dim] for dim in dims)
    return xr.Dataset(
        {
            c.REFLECTANCE_LUT_VARIABLE: xr.DataArray(
                np.zeros(shape, dtype=dtype),
                dims=dims,
                coords={dim: coords[dim] for dim in dims},
            )
        }
    )


def make_albedo_dataset(
    *,
    optional_dims=(),
    dtype=np.float32,
    sqrt_units="sqrt(um)",
    altitude_units="km",
):
    dims = c.ALBEDO_LUT_REQUIRED_DIMS + tuple(optional_dims)
    coords = {
        "solar_zenith": xr.DataArray(
            [0.0, 45.0], dims="solar_zenith", attrs={"units": "deg"}
        ),
        "illumination_angle": xr.DataArray(
            [0.0, 30.0], dims="illumination_angle", attrs={"units": "degree"}
        ),
        "lap_concentration": xr.DataArray(
            [0.0, 1000.0],
            dims="lap_concentration",
            attrs={"units": "parts per million", "lap_type": "dust"},
        ),
        "sqrt_grain_radius": xr.DataArray(
            [5.0, 40.0],
            dims="sqrt_grain_radius",
            attrs={"units": sqrt_units},
        ),
        "skyview": xr.DataArray(
            [0.0, 1.0], dims="skyview", attrs={"units": "fraction"}
        ),
        "altitude": xr.DataArray(
            [0.0, 4.0], dims="altitude", attrs={"units": altitude_units}
        ),
    }
    shape = tuple(coords[dim].size for dim in dims)
    return xr.Dataset(
        {
            c.ALBEDO_LUT_VARIABLE: xr.DataArray(
                np.zeros(shape, dtype=dtype),
                dims=dims,
                coords={dim: coords[dim] for dim in dims},
            )
        }
    )


def test_validate_reflectance_lut_accepts_canonical_dataset():
    lut.validate_reflectance_lut(make_reflectance_dataset())


@pytest.mark.parametrize("units", ["um^0.5", "um^(1/2)", "sqrt(um)", "µm^0.5"])
def test_validate_reflectance_lut_accepts_sqrt_grain_unit_aliases(units):
    lut.validate_reflectance_lut(make_reflectance_dataset(sqrt_units=units))


def test_validate_reflectance_lut_rejects_unsquared_grain_units():
    with pytest.raises(ContractError) as exc:
        lut.validate_reflectance_lut(make_reflectance_dataset(sqrt_units="um"))
    assert "um^0.5" in str(exc.value)


def test_validate_reflectance_lut_requires_dataset():
    with pytest.raises(ContractError) as exc:
        lut.validate_reflectance_lut(make_reflectance_dataset()["reflectance"])
    assert "xarray.Dataset" in str(exc.value)


def test_validate_reflectance_lut_rejects_legacy_grain_axis():
    dataset = make_reflectance_dataset(
        dims=("band", "solar_angle", "lap_concentration", "grain_size")
    )
    with pytest.raises(ContractError) as exc:
        lut.validate_reflectance_lut(dataset)
    text = str(exc.value)
    assert "sqrt_grain_radius" in text and "grain_size" in text


def test_validate_reflectance_lut_requires_lap_type():
    dataset = make_reflectance_dataset()
    del dataset.coords["lap_concentration"].attrs["lap_type"]
    with pytest.raises(ContractError) as exc:
        lut.validate_reflectance_lut(dataset)
    assert "lap_type" in str(exc.value)


def test_validate_reflectance_lut_checks_expected_lap_type():
    dataset = make_reflectance_dataset(lap_type="soot")
    lut.validate_reflectance_lut(dataset, expected_lap_type="Soot")
    with pytest.raises(ContractError) as exc:
        lut.validate_reflectance_lut(dataset, expected_lap_type="dust")
    assert "expected 'dust'" in str(exc.value)


def test_validate_albedo_lut_accepts_optional_axes_and_unit_aliases():
    dataset = make_albedo_dataset(optional_dims=("skyview", "altitude"))
    lut.validate_albedo_lut(dataset, expected_lap_type="dust")


def test_validate_albedo_lut_rejects_altitude_in_metres():
    dataset = make_albedo_dataset(
        optional_dims=("altitude",), altitude_units="m"
    )
    with pytest.raises(ContractError) as exc:
        lut.validate_albedo_lut(dataset)
    assert "'km'" in str(exc.value)


def test_validate_albedo_lut_rejects_optional_axis_order():
    dataset = make_albedo_dataset(optional_dims=("altitude", "skyview"))
    with pytest.raises(ContractError) as exc:
        lut.validate_albedo_lut(dataset)
    assert "canonical order" in str(exc.value)


def test_validate_albedo_lut_rejects_out_of_range_skyview():
    dataset = make_albedo_dataset(optional_dims=("skyview",))
    dataset = dataset.assign_coords(skyview=[0.0, 1.1])
    dataset.coords["skyview"].attrs["units"] = "1"
    with pytest.raises(ContractError) as exc:
        lut.validate_albedo_lut(dataset)
    assert "within [0, 1]" in str(exc.value)


def test_validate_albedo_lut_rejects_float64_values():
    with pytest.raises(ContractError) as exc:
        lut.validate_albedo_lut(make_albedo_dataset(dtype=np.float64))
    assert "float64" in str(exc.value)
