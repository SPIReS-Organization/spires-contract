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


def test_accepted_dtypes_are_float32_only():
    # Canonical boundary dtype is float32 (see conventions.ACCEPTED_DTYPES): the
    # inversion kernel stores float32 and promotes to double at read time, so a
    # single dtype keeps the batch path deterministic and rejects float64.
    assert set(np.dtype(dt) for dt in c.ACCEPTED_DTYPES) == {np.dtype(np.float32)}


def test_required_dtype_alias_points_at_accepted_set():
    # REQUIRED_DTYPE kept as a back-compat alias for the accepted set.
    assert c.REQUIRED_DTYPE == c.ACCEPTED_DTYPES
