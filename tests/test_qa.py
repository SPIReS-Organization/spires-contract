import numpy as np
import pytest
import xarray as xr

from spires_contract import qa
from spires_contract._validate import ContractError


def make_qa_mask(flags=qa.QA_FLAGS, dtype=bool, dims=("flag", "y", "x")):
    data = np.zeros((len(flags), 3, 4), dtype=dtype)
    return xr.DataArray(
        data,
        dims=dims,
        coords={"flag": list(flags), "y": np.arange(3), "x": np.arange(4)},
        name="qa_mask",
    )


def test_vocabulary_is_fixed():
    assert qa.QA_FLAGS == (
        "fill", "saturated", "cloud", "cirrus", "cloud_buffer",
        "cloud_shadow", "snow", "water",
    )
    assert set(qa.QA_INVALID_FLAGS) == set(qa.QA_FLAGS) - {"snow", "water"}
    assert qa.CLOUD_FLAGS == ("cloud", "cirrus", "cloud_buffer", "cloud_shadow")
    assert set(qa.CLOUD_FLAGS) <= set(qa.QA_INVALID_FLAGS)


def test_validate_qa_mask_accepts_valid():
    qa.validate_qa_mask(make_qa_mask())  # must not raise


def test_validate_qa_mask_rejects_non_bool():
    with pytest.raises(ContractError, match="dtype"):
        qa.validate_qa_mask(make_qa_mask(dtype=np.uint8))


def test_validate_qa_mask_rejects_wrong_dim_order():
    da = make_qa_mask().transpose("y", "x", "flag")
    with pytest.raises(ContractError, match="dims"):
        qa.validate_qa_mask(da)


def test_validate_qa_mask_rejects_missing_flag():
    with pytest.raises(ContractError, match="flag"):
        qa.validate_qa_mask(make_qa_mask(flags=qa.QA_FLAGS[:-1]))


def test_validate_qa_mask_rejects_misordered_flags():
    with pytest.raises(ContractError, match="flag"):
        qa.validate_qa_mask(make_qa_mask(flags=tuple(reversed(qa.QA_FLAGS))))


def test_validate_qa_mask_rejects_unknown_available_attr():
    da = make_qa_mask()
    da.attrs["qa_flags_available"] = "cloud haze"
    with pytest.raises(ContractError, match="qa_flags_available"):
        qa.validate_qa_mask(da)


def test_validate_qa_mask_accepts_available_attr_subset():
    da = make_qa_mask()
    da.attrs["qa_flags_available"] = "fill cloud cirrus"
    qa.validate_qa_mask(da)
