"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, r0, results) speaks
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

# The C++/SWIG inversion layer requires double-precision arrays.
REQUIRED_DTYPE = np.float64
