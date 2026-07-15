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


def test_accepted_dtypes_are_float32_and_float64():
    # Permissive boundary during the float32 migration (bd spires-h4e); to be
    # re-tightened to float32-only later (bd spires-cfp).
    assert set(np.dtype(dt) for dt in c.ACCEPTED_DTYPES) == {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }


def test_required_dtype_alias_points_at_accepted_set():
    # REQUIRED_DTYPE kept as a back-compat alias for the accepted set.
    assert c.REQUIRED_DTYPE == c.ACCEPTED_DTYPES
