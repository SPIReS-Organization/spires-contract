"""inversion -> postprocess boundary contract (STUB — not yet implemented).

Planned spec: inversion output with dims (y, x) per variable, one variable each
for fsca, fshade, dust_concentration (ppm), grain_size (μm) — see
conventions.RESULT_VARIABLES — likely as an xarray.Dataset. To be implemented
when the spires-postprocess package is built.
"""

def validate_results(ds):
    """Validate an inversion results Dataset. Not yet implemented."""
    raise NotImplementedError(
        "The inversion -> postprocess contract is not implemented yet; "
        "it will be added when the spires-postprocess package is built."
    )
