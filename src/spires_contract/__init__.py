"""spires-contract: data-interface contracts for the SPIReS package family."""

# Version from setuptools_scm (git tags)
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spires_contract")
except PackageNotFoundError:
    __version__ = "unknown"

from spires_contract._validate import ContractError
from spires_contract.alignment import (
    validate_coordinate_alignment,
    validate_spatial_alignment,
)
from spires_contract.clusters import clusters_present, validate_clusters
from spires_contract.data import (
    SpiresData,
    validate_for_inversion,
    validate_spires_data,
)
from spires_contract.lut import validate_lut
from spires_contract.results import validate_results

__all__ = [
    "ContractError",
    "SpiresData",
    "clusters_present",
    "validate_coordinate_alignment",
    "validate_clusters",
    "validate_for_inversion",
    "validate_lut",
    "validate_results",
    "validate_spatial_alignment",
    "validate_spires_data",
    "__version__",
]
