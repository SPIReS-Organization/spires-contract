"""Canonical naming and dtype conventions shared across all SPIReS contracts.

Single source of truth so every boundary module (spectra, lut, r0, results) speaks
the same dimension-name and dtype vocabulary.
"""

import numpy as np

# Canonical single-scene spatial dimensions.
SPATIAL_DIMS = ("y", "x")

# Spatial + spectral spectra arrays (target / background reflectance).
SPECTRA_DIMS = SPATIAL_DIMS + ("band",)

# Per-pixel solar zenith angle (degrees).
SOLAR_ANGLE_DIMS = SPATIAL_DIMS

# Cluster-level arrays and the excluded-pixel sentinel used by cluster labels.
CLUSTER_DIM = "cluster"
CLUSTER_DIMS = (CLUSTER_DIM,)
CLUSTER_LABEL_DIMS = SPATIAL_DIMS
CLUSTER_LABEL_SENTINEL = -1

# Reflectance lookup table produced from Mie theory. The light-absorbing-particle
# axis is generic because LUTs may represent dust, black carbon, or another LAP.
LUT_DIMS = ("band", "solar_angle", "lap_concentration", "grain_size")

# Inversion output vector, in this order, along the trailing result axis.
RESULT_VARIABLES = ("fsnow", "fshade", "lap_concentration", "grain_size")
RESULT_DIMS = SPATIAL_DIMS

# Required scene variables at the inversion boundary.
REQUIRED_SCENE_VARIABLES = (
    "reflectance",
    "solar_zenith",
)

# Optional atomic inversion-exclusion provenance set. These variables are all
# present or all absent in a validated SpiresData scene.
VALID_INVERSION_MASK_VARIABLE = "valid_inversion_mask"
INVERSION_EXCLUSION_FLAGS_VARIABLE = "inversion_exclusion_flags"
INVERSION_EXCLUSION_ASSESSED_VARIABLE = "inversion_exclusion_assessed"
INVERSION_EXCLUSION_VARIABLES = (
    INVERSION_EXCLUSION_FLAGS_VARIABLE,
    INVERSION_EXCLUSION_ASSESSED_VARIABLE,
    VALID_INVERSION_MASK_VARIABLE,
)
OPTIONAL_SCENE_VARIABLES = INVERSION_EXCLUSION_VARIABLES

# Stable schema-v1 bit assignments. Published assignments must never be
# reordered; new meanings may use only currently reserved bits.
INVERSION_EXCLUSION_SCHEMA_VERSION = 1
INVERSION_EXCLUSION_DTYPE = np.dtype(np.uint16)
INVERSION_EXCLUSION_REASONS = (
    "invalid_reflectance",
    "invalid_geometry",
    "insufficient_observations",
    "poor_surface_reflectance_quality",
    "cloud",
    "cloud_shadow",
    "water",
    "ice",
    "playa",
    "low_reflectance",
    "user_exclusion",
)
INVERSION_EXCLUSION_BITS = {
    reason: 1 << bit_index
    for bit_index, reason in enumerate(INVERSION_EXCLUSION_REASONS)
}
INVERSION_EXCLUSION_KNOWN_MASK = sum(INVERSION_EXCLUSION_BITS.values())
INVERSION_EXCLUSION_RESERVED_BITS = tuple(range(11, 16))
INVERSION_EXCLUSION_SCHEMA_ATTR = "inversion_exclusion_schema_version"

# Required variables for a complete clustered scene.
CLUSTER_LABEL_VARIABLE = "cluster_label"
CLUSTER_COUNT_VARIABLE = "cluster_count"
CLUSTER_REPRESENTATIVE_VARIABLES = (
    "cluster_representative_reflectance",
    "cluster_representative_background",
    "cluster_representative_solar_zenith",
)
REQUIRED_CLUSTER_VARIABLES = (
    CLUSTER_LABEL_VARIABLE,
    CLUSTER_COUNT_VARIABLE,
) + CLUSTER_REPRESENTATIVE_VARIABLES
OPTIONAL_CLUSTER_VARIABLES = ("cluster_representative_cosine_illumination",)
CLUSTER_MASK_POLICY_ATTR = "valid_inversion_mask_applied"

# Canonical result metadata. Additional attributes are allowed.
RESULT_LONG_NAMES = {
    "fsnow": "Fractional Snow-Covered Area",
    "fshade": "Fractional Shaded Area",
    "lap_concentration": "Light-Absorbing Particle Concentration in Snow",
    "grain_size": "Effective Snow Grain Radius",
}
RESULT_UNITS = {
    "fsnow": "1",
    "fshade": "1",
    "lap_concentration": "ppm",
    "grain_size": "um",
}
GRAIN_SIZE_UNIT_ALIASES = ("um", "µm", "μm")
INITIAL_LAP_TYPE = "dust"

# Canonical floating dtype for scientific data at SPIReS package boundaries.
# Internal numerical kernels may promote values for computation, but public
# producers emit float32 and validators never silently cast float64 inputs.
ACCEPTED_DTYPES = (np.float32,)

# Back-compat alias: some callers referenced REQUIRED_DTYPE. Kept pointing at the
# accepted set so a single source of truth drives validation.
REQUIRED_DTYPE = ACCEPTED_DTYPES
