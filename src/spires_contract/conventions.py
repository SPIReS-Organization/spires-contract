"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, lut, r0, results) speaks
the same dimension-name and dtype vocabulary.
"""

import numpy as np

# Spatial + spectral spectra arrays (target / background reflectance).
SPECTRA_DIMS = ("y", "x", "band")

# Per-pixel solar zenith angle (degrees).
SOLAR_ANGLE_DIMS = ("y", "x")

# Reflectance lookup table produced from Mie theory. The light-absorbing-particle
# axis is named generically ('lap') rather than after one species (dust), since
# the LAP can be dust, black carbon, etc.
LUT_DIMS = ("band", "solar_angle", "lap_concentration", "grain_size")

# Inversion output vector, in this order, along the trailing result axis.
# 'fsnow' is the RAW inversion snow fraction (before fshade / canopy correction);
# 'fsca' is reserved for the post-processed product (see spires-postprocess).
# 'lap_concentration' is the light-absorbing-particle concentration.
RESULT_VARIABLES = ("fsnow", "fshade", "lap_concentration", "grain_size")

# Canonical floating dtype at the I/O -> inversion boundary: float32 only. The
# C++/SWIG inversion layer stores these large arrays as float32 (half the memory)
# and promotes each value to double at read time, so the interpolation/cost math
# and NLopt run in full double precision regardless. Enforcing a single dtype here
# keeps the batch inversion path deterministic (one kernel, no runtime dtype
# branch) and rejects float64 inputs at the boundary rather than silently
# round-tripping them float64 -> float32 -> double.
ACCEPTED_DTYPES = (np.float32,)

# Back-compat alias: some callers referenced REQUIRED_DTYPE. Kept pointing at the
# accepted set so a single source of truth drives validation.
REQUIRED_DTYPE = ACCEPTED_DTYPES
