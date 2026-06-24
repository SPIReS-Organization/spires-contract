import numpy as np
from spires_contract import conventions as c


def test_canonical_spectra_dims():
    assert c.SPECTRA_DIMS == ("y", "x", "band")


def test_canonical_solar_angle_dims():
    assert c.SOLAR_ANGLE_DIMS == ("y", "x")


def test_canonical_lut_dims():
    assert c.LUT_DIMS == ("band", "solar_angle", "dust_concentration", "grain_size")


def test_result_variables_order():
    assert c.RESULT_VARIABLES == ("fsca", "fshade", "dust_concentration", "grain_size")


def test_required_dtype_is_float64():
    assert c.REQUIRED_DTYPE == np.float64
