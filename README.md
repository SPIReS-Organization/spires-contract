# spires-contract

Data-interface contracts for the [SPIReS](https://github.com/SPIReS-Organization)
package family. Defines, as executable `xarray` validators, the array shapes,
dimension names, dtypes, and coordinate conventions that flow between SPIReS
packages (I/O, LUT generation, R_0 production, inversion, postprocessing).

Depends only on `numpy` and `xarray`.

## The idea

A contract is a handshake the two sides agree on before they ever connect: both
commit to the same data shape in advance, so neither has to guess.

Here the "shape" is literal — target spectra are `(y, x, band)`, float64, with a
`band` coordinate. One package produces that array, another consumes it. Rather
than each privately assuming the shape and hitting the mismatch only once wired
together (the classic "x and y are swapped" bug), both point at **one definition,
used from two sides**: the producer checks *"does what I emit match?"*, the
consumer checks *"can I handle anything that matches?"*

This agreement used to live informally in docstrings — exactly where
dimension-order and dtype bugs creep in. `spires-contract` makes it executable
and shared: each boundary is defined once, here, and packages on both sides check
themselves against the same source of truth. The contract validates the data at
each seam, not the science, so packages stay independently buildable and
releasable without drifting apart.

## Install

Not yet published to PyPI. Install from source:

```bash
git clone https://github.com/SPIReS-Organization/spires-contract.git
pip install ./spires-contract
```

Or, for local development (editable install from a checkout):

```bash
pip install -e .
```

## Usage

Each boundary module exposes `validate_*(da)` functions that raise
`ContractError` listing **every** violation at once — wrong dimension order,
missing or extra dimension, missing coordinate, wrong dtype — so a producer
gets one actionable error rather than fixing problems one at a time.

```python
import spires_contract.spectra as spectra

spectra.validate_target_spectra(da)   # raises ContractError listing all violations
```

The contract **validates**; it does not mutate. There are deliberately no
`conform`/normalize helpers: a contract that silently transposed or cast arrays
would hide a per-call performance cost (a large-array copy) inside what looks
like a check. Producers are expected to hand over data already in canonical
form — including **dimension order**, which is part of the contract because the
inversion kernel indexes arrays positionally.

## How to use it from both sides of a boundary

The contract is defined once and used to test **both directions** of each
boundary.

**Producer side** (e.g. `spires-io` emitting target spectra) — assert that what
it produces conforms:

```python
from spires_contract import spectra

def test_io_output_conforms():
    da = load_sentinel2_target(...)        # the producer's real output
    spectra.validate_target_spectra(da)    # must not raise
```

**Consumer side** (e.g. `spires-inversion` accepting target spectra) — two
obligations:

1. Accept *anything the contract permits*. Build a conforming input, certify
   it with the validator, and assert the consumer handles it:

   ```python
   from spires_contract import spectra

   def test_inversion_accepts_contract_valid_spectra():
       da = make_conforming_target(dims=("y", "x", "band"))  # canonical
       spectra.validate_target_spectra(da)                   # precondition: it IS valid
       result = invert(da, ...)                              # consumer must not choke
   ```

   Certifying the fixture with the validator guarantees you are testing against a
   genuinely conforming input, not your assumption of one.

2. As a producer of downstream data (its `results`), the consumer gets a
   producer-side test against *that* boundary's validator.

### What a contract does and does not guarantee

A contract validates **data, not behavior**. It guarantees the inversion
*receives* a well-formed `(y, x, band)` float64 array with a `band` coordinate;
it does **not** guarantee the inversion interprets the band axis correctly or
returns physically sensible output. Those remain each package's own numerical
and correctness tests. The contract simply removes the entire class of
shape/dtype/dimension-naming mismatches at the seams.

This package deliberately stays minimal: `validate_*` only. It does not ship
shared example-builder fixtures — each package builds its own test fixtures and
certifies them with the validators above.

## Boundaries

| Module                    | Boundary                   | Status      |
|---------------------------|----------------------------|-------------|
| `spires_contract.spectra` | I/O → inversion            | implemented |
| `spires_contract.lut`     | LUT → inversion            | stub        |
| `spires_contract.r0`      | R_0 → inversion            | stub        |
| `spires_contract.results` | inversion → postprocess    | stub        |

### Canonical forms

| Data                  | dims                                                  | dtype   | notes                          |
|-----------------------|-------------------------------------------------------|---------|--------------------------------|
| target spectra        | `(y, x, band)`                                        | float32 | mixed reflectance, `band` coord |
| background spectra    | `(y, x, band)`                                        | float32 | R_0 reflectance, `band` coord   |
| solar angles          | `(y, x)`                                              | float32 | solar zenith, degrees          |
| LUT                   | `(band, solar_angle, lap_concentration, grain_size)`  | float32 | Mie-theory reflectance table   |
| results               | `(y, x)` per variable                                 | float64 | `fsnow`, `fshade`, `lap`, grain |

The canonical dimension names, dtype, and result-variable ordering are the
single source of truth in `spires_contract.conventions`.
```
