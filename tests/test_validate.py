import numpy as np
import pytest
import xarray as xr

from spires_contract import (
    SpiresData,
    inversion_exclusion_metadata,
    validate_spires_data,
)
from spires_contract import conventions as c
from spires_contract._validate import (
    ContractError,
    check_dims_present,
    check_dtype,
    check_coords_present,
    check_no_extra_dims,
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


def test_check_dims_present_reports_multiple_missing():
    da = _da(("y",))
    msgs = check_dims_present(da, ("y", "x", "band"))
    assert len(msgs) == 2
    assert any("x" in m for m in msgs)
    assert any("band" in m for m in msgs)


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


def test_check_coords_present_reports_multiple_missing():
    da = _da(("y", "x", "band"))  # no coords assigned
    msgs = check_coords_present(da, ("band", "time"))
    assert len(msgs) == 2
    assert any("band" in m for m in msgs)
    assert any("time" in m for m in msgs)


def test_raise_if_violations_raises_with_all_messages():
    with pytest.raises(ContractError) as exc:
        raise_if_violations("target_spectra", ["missing dim 'band'", "dtype is float32"])
    text = str(exc.value)
    assert "target_spectra" in text
    assert "missing dim 'band'" in text
    assert "dtype is float32" in text


def test_raise_if_violations_silent_when_empty():
    raise_if_violations("target_spectra", [])  # must not raise


def test_check_no_extra_dims_no_violation():
    da = _da(("y", "x", "band"))
    assert check_no_extra_dims(da, ("y", "x", "band")) == []


def test_check_no_extra_dims_reports_extra():
    da = _da(("y", "x", "band", "time"))
    msgs = check_no_extra_dims(da, ("y", "x", "band"))
    assert len(msgs) == 1
    assert "time" in msgs[0]


def _packed_exclusion_array(name, values):
    return xr.DataArray(
        np.asarray(values, dtype=np.uint16),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        attrs=inversion_exclusion_metadata(name),
        name=name,
    )


def _exclusion_scene(flags, assessed):
    flags_array = _packed_exclusion_array(
        c.INVERSION_EXCLUSION_FLAGS_VARIABLE, flags
    )
    assessed_array = _packed_exclusion_array(
        c.INVERSION_EXCLUSION_ASSESSED_VARIABLE, assessed
    )
    valid_mask = xr.DataArray(
        np.asarray(flags, dtype=np.uint16) == 0,
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        name=c.VALID_INVERSION_MASK_VARIABLE,
    )
    return xr.Dataset(
        {
            c.INVERSION_EXCLUSION_FLAGS_VARIABLE: flags_array,
            c.INVERSION_EXCLUSION_ASSESSED_VARIABLE: assessed_array,
            c.VALID_INVERSION_MASK_VARIABLE: valid_mask,
        }
    )


def test_validate_spires_data_accepts_absent_exclusion_set():
    validate_spires_data(SpiresData(scene=xr.Dataset()))


def test_validate_spires_data_accepts_valid_exclusion_set():
    cloud = c.INVERSION_EXCLUSION_BITS["cloud"]
    water = c.INVERSION_EXCLUSION_BITS["water"]
    scene = _exclusion_scene(
        [[cloud, cloud | water], [0, 0]],
        [[cloud | water, cloud | water], [water, 0]],
    )
    validate_spires_data(SpiresData(scene=scene))


def test_validate_spires_data_rejects_partial_exclusion_set():
    scene = xr.Dataset(
        {
            c.VALID_INVERSION_MASK_VARIABLE: xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [10, 11]},
            )
        }
    )
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "atomic" in str(exc.value)


def test_validate_spires_data_rejects_flag_without_assessment():
    cloud = c.INVERSION_EXCLUSION_BITS["cloud"]
    scene = _exclusion_scene([[cloud, 0], [0, 0]], [[0, 0], [0, 0]])
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "matching inversion_exclusion_assessed bit is not set" in str(exc.value)


def test_validate_spires_data_rejects_mask_inconsistent_with_flags():
    cloud = c.INVERSION_EXCLUSION_BITS["cloud"]
    scene = _exclusion_scene([[cloud, 0], [0, 0]], [[cloud, 0], [0, 0]])
    scene[c.VALID_INVERSION_MASK_VARIABLE][:] = True
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "must be true exactly where" in str(exc.value)


def test_validate_spires_data_rejects_reserved_bits():
    reserved = 1 << c.INVERSION_EXCLUSION_RESERVED_BITS[0]
    scene = _exclusion_scene([[reserved, 0], [0, 0]], [[reserved, 0], [0, 0]])
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "reserved or unknown" in str(exc.value)


def test_validate_spires_data_rejects_incorrect_exclusion_metadata():
    scene = _exclusion_scene([[0, 0], [0, 0]], [[0, 0], [0, 0]])
    del scene[c.INVERSION_EXCLUSION_FLAGS_VARIABLE].attrs["flag_meanings"]
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "flag_meanings" in str(exc.value)


def test_validate_spires_data_rejects_wrong_exclusion_dtype():
    scene = _exclusion_scene([[0, 0], [0, 0]], [[0, 0], [0, 0]])
    scene[c.INVERSION_EXCLUSION_FLAGS_VARIABLE] = scene[
        c.INVERSION_EXCLUSION_FLAGS_VARIABLE
    ].astype(np.uint32, keep_attrs=True)
    with pytest.raises(ContractError) as exc:
        validate_spires_data(SpiresData(scene=scene))
    assert "uint32" in str(exc.value) and "uint16" in str(exc.value)
