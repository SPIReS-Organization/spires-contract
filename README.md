# spires-contract

Data-interface contracts for the [SPIReS](https://github.com/SPIReS-Organization)
package family. Defines, as executable `xarray` validators, the array shapes,
dimension names, dtypes, and coordinate conventions that flow between SPIReS
packages (I/O, LUT generation, R_0 production, inversion, postprocessing).

Depends only on `numpy` and `xarray`.

## The idea

Think of a contract the way two people agree on a handshake before they ever
shake hands: both sides commit to the same shape in advance, so neither has to
guess what the other will do.

Here, the "shape" is literal — what does an array of target spectra look like?
`(y, x, band)`, float64, with a `band` coordinate. One package *produces* that
array; another *consumes* it. Instead of each package privately assuming the
shape and discovering the mismatch only when they're wired together (the classic
"my x and y are swapped" bug), they both point at **one definition, used from
two sides**:

- the producer asks *"does what I hand over match the agreed shape?"*
- the consumer asks *"can I handle anything that matches the agreed shape?"*

The contract is the referee they both trust. It doesn't do the science — it just
makes sure the data showing up at each boundary is the shape everyone agreed on,
so the packages can be built, tested, and released independently without drifting
apart.

## Why a contract package

The SPIReS packages are developed and released independently, but they have to
agree on the exact form of the data passed between them — dimension names and
order, dtype, required coordinates, units. Historically that agreement lived
informally in docstrings, which is where dimension-ordering and dtype bugs creep
in at the seams.

`spires-contract` makes that agreement **executable and shared**. The interface
for each boundary is defined exactly once, here, and every package on either
side of a boundary depends on this package to check itself against the same
source of truth.

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

Each boundary module exposes two kinds of function:

- `validate_*(da)` — raises `ContractError` listing **every** violation at once
  (wrong dims, missing coordinate, wrong dtype), so a producer gets one
  actionable error rather than fixing problems one at a time.
- `conform_*(da)` — returns the array normalized to canonical form (transposed
  to the canonical dimension order and cast to the required dtype). It repairs
  order and dtype, but a genuinely missing dimension or coordinate cannot be
  invented, so `conform_*` raises `ContractError` in that case.

```python
import spires_contract.spectra as spectra

spectra.validate_target_spectra(da)        # raises ContractError listing all violations
da = spectra.conform_target_spectra(da)    # -> canonical (y, x, band), float64
```

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

1. Accept *anything the contract permits*. Build conforming inputs, certify
   them with the validator, and assert the consumer handles them — including
   legal-but-awkward cases such as a transposed dimension order:

   ```python
   from spires_contract import spectra

   def test_inversion_accepts_any_contract_valid_spectra():
       da = make_conforming_target(dims=("band", "y", "x"))  # legal, transposed
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

This package deliberately stays minimal: `validate_*` + `conform_*` only. It
does not ship shared example-builder fixtures — each package builds its own test
fixtures and certifies them with the validators above.

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
| target spectra        | `(y, x, band)`                                        | float64 | mixed reflectance, `band` coord |
| background spectra    | `(y, x, band)`                                        | float64 | R_0 reflectance, `band` coord   |
| solar angles          | `(y, x)`                                              | float64 | solar zenith, degrees          |
| LUT                   | `(band, solar_angle, dust_concentration, grain_size)` | float64 | Mie-theory reflectance table   |
| results               | `(y, x)` per variable                                 | float64 | `fsca`, `fshade`, dust, grain  |

The canonical dimension names, dtype, and result-variable ordering are the
single source of truth in `spires_contract.conventions`.
```
