"""LUT -> inversion boundary contract (STUB — not yet implemented).

Planned spec: a Mie-theory reflectance lookup table with dims
(band, solar_angle, dust_concentration, grain_size) — see conventions.LUT_DIMS —
float64, with a coordinate present for each of the four dimensions (the
interpolator reads coordinate values to locate query points). To be implemented
when the spires-lut package is built.
"""

def validate_lut(da):
    """Validate a reflectance lookup table DataArray. Not yet implemented."""
    raise NotImplementedError(
        "The lut -> inversion contract is not implemented yet; "
        "it will be added when the spires-lut package is built."
    )
