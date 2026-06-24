# spires-contract

Data-interface contracts for the [SPIReS](https://github.com/SPIReS-Organization)
package family. Defines, as executable xarray validators, the array shapes,
dimension names, dtypes, and coordinate conventions that flow between SPIReS
packages (I/O, R_0 production, inversion, postprocessing).

Depends only on `numpy` and `xarray`.

## Install

```bash
pip install spires-contract
```

## Usage

```python
import spires_contract.spectra as spectra

spectra.validate_target_spectra(da)   # raises ContractError listing all violations
da = spectra.conform_target_spectra(da)  # transpose/cast to canonical (y, x, band) float64
```

## Boundaries

| Module                    | Boundary                   | Status      |
|---------------------------|----------------------------|-------------|
| `spires_contract.spectra` | I/O → inversion            | implemented |
| `spires_contract.r0`      | R_0 → inversion            | stub        |
| `spires_contract.results` | inversion → postprocess    | stub        |
