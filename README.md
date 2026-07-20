# spires-contract

Shared data structures and executable boundary contracts for the
[SPIReS](https://github.com/SPIReS-Organization) package family.

`spires-contract` owns the neutral `SpiresData` container, canonical names,
and validation rules used by `spires-io`, `spires-inversion`, and
`spires-postprocess`. It depends only on NumPy and xarray and never imports a
scientific stage package.

The initial object contract is intentionally limited to one in-memory scene.
Existing time-aware or Dask-backed lower-level stage APIs can remain available
outside this object workflow.

## Install

The package is not yet published to PyPI. Install it from a checkout:

```bash
pip install ./spires-contract
```

For local development:

```bash
pip install -e ./spires-contract
```

## Shared data object

`SpiresData` is a frozen envelope that is enriched as a scene moves through
the scientific stages:

```python
import xarray as xr

from spires_contract import SpiresData

data = SpiresData(scene=scene)
data = data.with_background(background)
data = data.with_ancillary(ancillary)
data = data.with_results(results)
```

Only `scene` is required at construction. `background`, `ancillary`, and
`results` may be added later. Replacement methods make shallow xarray copies,
preserving lazy backing arrays while preventing accidental replacement of a
field on the frozen object.

The object provides neutral accessors for target reflectance, solar zenith,
and the inversion-validity mask. Loading, clustering, inversion,
postprocessing, and writing remain package-level behavior in the packages that
own those operations.

## Canonical single-scene boundary

Scientific numeric data variables crossing package boundaries are `float32`.
Validators reject float64 producer output rather than silently casting it.
Coordinates retain the native dtype appropriate to the grid, time, band, or
cluster identifier.

Required inputs before inversion are:

| Location | Variable | Dimensions | Dtype |
| --- | --- | --- | --- |
| `data.scene` | `reflectance` | `(y, x, band)` | `float32` |
| `data.scene` | `solar_zenith` | `(y, x)` | `float32` |
| `data.scene` | `valid_inversion_mask` | `(y, x)` | Boolean or integer `0/1` |
| `data.background` | background reflectance | `(y, x, band)` | `float32` |

Spatial and band coordinates must match exactly. Validation inspects inputs but
does not transpose, cast, resample, clip, or otherwise mutate them.

```python
from spires_contract import validate_for_inversion

validate_for_inversion(data)
```

## Clustered inputs

A scene is either unclustered or contains the complete canonical cluster
schema:

| Variable | Dimensions | Dtype |
| --- | --- | --- |
| `cluster_label` | `(y, x)` | integer |
| `cluster_count` | `(cluster,)` | integer |
| `cluster_representative_reflectance` | `(cluster, band)` | `float32` |
| `cluster_representative_background` | `(cluster, band)` | `float32` |
| `cluster_representative_solar_zenith` | `(cluster,)` | `float32` |
| `cluster_representative_cosine_illumination` | `(cluster,)` | `float32`, optional |

`cluster_label == -1` marks excluded pixels. Nonnegative labels are contiguous
from zero, counts must exactly match label membership, and representative
coordinates must match the scene. Empty clustering is represented by all `-1`
labels and zero-length cluster arrays. Cluster-label metadata records
`valid_inversion_mask_applied` as Boolean or integer `0/1`.

```python
from spires_contract import validate_clusters

validate_clusters(data)
```

## Canonical inversion results

The four base variables share exact `(y, x)` coordinates:

| Variable | Dtype | Units | Finite-value rule |
| --- | --- | --- | --- |
| `fsnow` | `float32` | `1` | `[0, 1]` |
| `fshade` | `float32` | `1` | `[0, 1]` |
| `lap_concentration` | `float32` | `ppm` | nonnegative |
| `grain_size` | `float32` | `um` | nonnegative |

`lap_concentration` additionally carries `lap_type="dust"`. NaNs are allowed,
but infinities and invalid finite values are rejected. When an effective
eligibility mask is supplied, every base result must be NaN outside that mask.
Extra result variables are allowed so postprocessing can enrich the same
dataset. The initial shared contract does not enforce
`fsnow + fshade <= 1`.

```python
from spires_contract import validate_results

validate_results(
    data.results,
    scene=data.scene,
    eligibility_mask=effective_eligibility,
)
```

Validation never clips values into physical ranges.

## Validation behavior

Validators raise `ContractError` with related violations collected into one
actionable message. They validate data at package boundaries, not scientific
correctness, and never normalize accepted inputs.

The main public validators are:

- `validate_spires_data(data)` for the neutral container type.
- `validate_for_inversion(data)` for complete full-resolution or clustered
  inversion input.
- `validate_spatial_alignment(data)` for exact field alignment.
- `validate_clusters(data)` for complete-or-absent cluster state.
- `validate_results(results, scene=..., eligibility_mask=...)` for canonical
  inversion output.
- `validate_lut(lut)` for the float32 LUT boundary.

Standalone `validate_r0()` remains deferred until the `spires-r0` package is
implemented; background reflectance used by inversion is already validated by
`validate_for_inversion()`.

## Package boundaries

| Boundary | Contract | Status |
| --- | --- | --- |
| shared lifecycle | `SpiresData` | implemented |
| I/O → inversion | scene, background, masks, alignment | implemented |
| clustered I/O → inversion | cluster labels, counts, representatives | implemented |
| LUT → inversion | lookup table | implemented |
| inversion → postprocess | canonical result dataset | implemented |
| standalone R₀ product | `validate_r0` | deferred stub |

## Development and release

Run verification in the workspace environment:

```bash
module load miniforge
mamba run -n spipy14 python -m pytest
```

Versions are derived from Git tags with `setuptools-scm`. The planned release
for this expanded shared contract is `v0.4.0`; creating and publishing that tag
is a separate release action after review.
