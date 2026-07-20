"""spires-contract: data-interface contracts for the SPIReS package family."""

# Version from setuptools_scm (git tags)
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spires_contract")
except PackageNotFoundError:
    __version__ = "unknown"

from spires_contract._validate import ContractError
from spires_contract.data import SpiresData
from spires_contract.lut import validate_lut

__all__ = ["ContractError", "SpiresData", "validate_lut", "__version__"]
