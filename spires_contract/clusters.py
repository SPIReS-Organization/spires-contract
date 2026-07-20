"""Validation for clustered single-scene SPIReS inversion inputs."""

import numpy as np

from spires_contract import conventions as c
from spires_contract._validate import (
    check_cluster_labels,
    check_coords_match,
    check_coords_present,
    check_data_vars_present,
    check_dims_order,
    check_dims_present,
    check_dtype,
    check_no_extra_dims,
    raise_if_violations,
)
from spires_contract.data import SpiresData

__all__ = ["clusters_present", "validate_clusters"]


def clusters_present(data):
    """Return whether any canonical cluster variable is present in ``scene``."""
    if not isinstance(data, SpiresData):
        return False
    known = set(c.REQUIRED_CLUSTER_VARIABLES + c.OPTIONAL_CLUSTER_VARIABLES)
    return bool(known.intersection(data.scene.data_vars))


def validate_clusters(data):
    """Validate a complete clustered-scene schema, or accept its absence."""
    if not isinstance(data, SpiresData):
        raise_if_violations(
            "clusters",
            [f"data must be a SpiresData instance, got {type(data).__name__}"],
        )

    scene = data.scene
    if not clusters_present(data):
        return

    violations = check_data_vars_present(scene, c.REQUIRED_CLUSTER_VARIABLES)
    if "reflectance" not in scene.data_vars:
        violations.append(
            "scene is missing 'reflectance', required to validate cluster alignment"
        )

    label = scene.get(c.CLUSTER_LABEL_VARIABLE)
    count = scene.get(c.CLUSTER_COUNT_VARIABLE)
    reflectance = scene.get("reflectance")

    if label is not None:
        violations += _check_array_structure(
            label,
            c.CLUSTER_LABEL_DIMS,
            c.CLUSTER_LABEL_DIMS,
        )
        violations += check_cluster_labels(label)
        if reflectance is not None:
            violations += check_coords_match(
                reflectance,
                label,
                c.SPATIAL_DIMS,
                reference_name="scene reflectance",
                candidate_name=c.CLUSTER_LABEL_VARIABLE,
            )
        violations += _check_mask_policy(label)

    n_clusters = None
    count_values = None
    if count is not None:
        violations += _check_array_structure(count, c.CLUSTER_DIMS, c.CLUSTER_DIMS)
        count_dtype = np.dtype(count.dtype)
        if not np.issubdtype(count_dtype, np.integer) or np.issubdtype(
            count_dtype, np.bool_
        ):
            violations.append(
                f"{c.CLUSTER_COUNT_VARIABLE} dtype is {count_dtype}, "
                "expected an integer dtype"
            )
        else:
            count_values = np.asarray(count.values)
            if np.any(count_values < 0):
                violations.append(f"{c.CLUSTER_COUNT_VARIABLE} must be nonnegative")
        if c.CLUSTER_DIM in count.dims:
            n_clusters = int(count.sizes[c.CLUSTER_DIM])
            violations += _check_cluster_coordinate(count, n_clusters)

    representative_specs = {
        "cluster_representative_reflectance": (c.CLUSTER_DIMS + ("band",), True),
        "cluster_representative_background": (c.CLUSTER_DIMS + ("band",), True),
        "cluster_representative_solar_zenith": (c.CLUSTER_DIMS, False),
        "cluster_representative_cosine_illumination": (c.CLUSTER_DIMS, False),
    }
    for name, (dims, spectral) in representative_specs.items():
        representative = scene.get(name)
        if representative is None:
            continue
        violations += _check_array_structure(representative, dims, dims)
        violations += check_dtype(representative, c.ACCEPTED_DTYPES)
        values = np.asarray(representative.values)
        if np.any(~np.isfinite(values)):
            violations.append(f"{name} contains nonfinite value(s)")

        if n_clusters is not None and c.CLUSTER_DIM in representative.dims:
            if representative.sizes[c.CLUSTER_DIM] != n_clusters:
                violations.append(
                    f"{name} has {representative.sizes[c.CLUSTER_DIM]} clusters, "
                    f"expected {n_clusters}"
                )
        if count is not None:
            violations += check_coords_match(
                count,
                representative,
                c.CLUSTER_DIMS,
                reference_name=c.CLUSTER_COUNT_VARIABLE,
                candidate_name=name,
            )
        if spectral and reflectance is not None:
            violations += check_coords_match(
                reflectance,
                representative,
                ("band",),
                reference_name="scene reflectance",
                candidate_name=name,
            )

    if label is not None and count is not None and n_clusters is not None:
        violations += _check_label_counts(label, count_values, n_clusters)

    raise_if_violations("clusters", violations)


def _check_array_structure(array, required_dims, required_coords):
    violations = []
    violations += check_dims_present(array, required_dims)
    violations += check_no_extra_dims(array, required_dims)
    violations += check_dims_order(array, required_dims)
    violations += check_coords_present(array, required_coords)
    return violations


def _check_cluster_coordinate(array, n_clusters):
    if c.CLUSTER_DIM not in array.coords:
        return []
    coord = array.coords[c.CLUSTER_DIM]
    dtype = np.dtype(coord.dtype)
    violations = []
    if not np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        violations.append(f"cluster coordinate dtype is {dtype}, expected integer")
        return violations
    expected = np.arange(n_clusters, dtype=dtype)
    if not np.array_equal(np.asarray(coord.values), expected):
        violations.append(
            "cluster coordinate must contain contiguous identifiers from 0; "
            f"found {np.asarray(coord.values).tolist()}"
        )
    return violations


def _check_label_counts(label, count_values, n_clusters):
    dtype = np.dtype(label.dtype)
    if (
        not np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.bool_)
        or count_values is None
    ):
        return []

    labels = np.asarray(label.values)
    nonnegative = labels[labels >= 0]
    referenced_clusters = int(nonnegative.max()) + 1 if nonnegative.size else 0
    violations = []
    if referenced_clusters != n_clusters:
        violations.append(
            f"cluster labels reference {referenced_clusters} clusters, "
            f"but {c.CLUSTER_COUNT_VARIABLE} has length {n_clusters}"
        )
        return violations

    expected_counts = np.bincount(nonnegative, minlength=n_clusters)
    if not np.array_equal(count_values, expected_counts):
        violations.append(
            f"{c.CLUSTER_COUNT_VARIABLE} does not match cluster_label membership; "
            f"found {count_values.tolist()}, expected {expected_counts.tolist()}"
        )
    return violations


def _check_mask_policy(label):
    name = c.CLUSTER_MASK_POLICY_ATTR
    if name not in label.attrs:
        return [f"{c.CLUSTER_LABEL_VARIABLE} is missing required attribute {name!r}"]
    value = label.attrs[name]
    if isinstance(value, (bool, np.bool_)):
        return []
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return []
    return [f"{c.CLUSTER_LABEL_VARIABLE} attribute {name!r} must be Boolean or 0/1"]
