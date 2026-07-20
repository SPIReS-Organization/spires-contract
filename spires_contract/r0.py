"""R_0 -> inversion boundary contract (STUB — not yet implemented).

Planned spec: background (snow-free) reflectance R_0 with dims (y, x, band),
float32, matching the spatial grid and band coordinate of the target spectra it
will be paired with. To be implemented when the spires-r0 package is built.
"""

def validate_r0(da):
    """Validate an R_0 background reflectance array. Not yet implemented."""
    raise NotImplementedError(
        "The r0 -> inversion contract is not implemented yet; "
        "it will be added when the spires-r0 package is built."
    )
