"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, lut, r0, results) speaks
the same dimension-name and dtype vocabulary.
"""

import numpy as np

# Spatial + spectral spectra arrays (target / background reflectance).
SPECTRA_DIMS = ("y", "x", "band")

# Per-pixel solar zenith angle (degrees).
SOLAR_ANGLE_DIMS = ("y", "x")

# Reflectance lookup table produced from Mie theory.
LUT_DIMS = ("band", "solar_angle", "dust_concentration", "grain_size")

# Inversion output vector, in this order, along the trailing result axis.
RESULT_VARIABLES = ("fsca", "fshade", "dust_concentration", "grain_size")

# Accepted floating dtypes at the I/O -> inversion boundary. The C++/SWIG
# inversion layer stores these large arrays as-is and promotes each value to
# double at read time for the interpolation/cost math (NLopt stays double), so
# both float32 (half the memory) and float64 are valid. float32 is preferred by
# producers (spires-io loads reflectance as float32); float64 remains accepted
# for back-compatibility.
ACCEPTED_DTYPES = (np.float32, np.float64)

# Back-compat alias: some callers referenced REQUIRED_DTYPE. Kept pointing at the
# accepted set so a single source of truth drives validation.
REQUIRED_DTYPE = ACCEPTED_DTYPES
