"""Durable publication and independent loading of Diffusion checkpoints.

Persists the accepted prepared Diffusion best/last states with an
integrity-checked manifest and reloads them into the existing prepared-state
representation. It does not train, sample, activate runtime training or modify
the accepted checkpoint-selection policy.

Bundle layout:

    <destination>/manifest.json
    <destination>/best/w000.npy ...
    <destination>/last/w000.npy ...

The manifest is written last; a ``.bundle.pending`` marker distinguishes an
incomplete publication from a complete bundle.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .diffusion_checkpoint_state import (
    _approved_configuration,
    _digest,
    _weights,
)
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS


SCHEMA = "trustforge.paper01.diffusion-checkpoint-bundle.v1"
MANIFEST_FILENAME = "manifest.json"
PENDING_FILENAME = ".bundle.pending"
ROLES = ("best", "last")

STATE_FIELDS = frozenset({
    "dataset_id", "experimental_seed", "class_mapping", "class_count",
    "scientific_configuration", "checkpoint_role", "selected_epoch",
    "completed_updates", "validation_mse", "weights", "state_sha256",
    "publication_status",
})
MANIFEST_FIELDS = frozenset({
    "schema", "dataset_id", "experimental_seed", "class_mapping",
    "class_count", "scientific_configuration", "source_commit",
    "dataset_provenance", "states",
})
STATE_MANIFEST_FIELDS = frozenset({
    "checkpoint_role", "selected_epoch", "completed_updates",
    "validation_mse", "state_sha256", "weight_count", "tensors",
})
TENSOR_FIELDS = frozenset({
    "filename", "dtype", "shape", "sha256", "bytes",
})

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TENSOR_NAME = re.compile(r"^(best|last)/w[0-9]{3}\.npy$")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_TENSOR_BYTES = 512 * 1024 * 1024


class DiffusionBundleError(ValueError):
    """Raised when a Diffusion checkpoint bundle is invalid or incomplete."""


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest_value(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise DiffusionBundleError(name + " requires a lowercase SHA-256")
    return value


def _write_exclusive(path, raw, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_regular(path, *, name, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise DiffusionBundleError(name + " must be a regular file")
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise DiffusionBundleError(name + " has an invalid bounded size")
    return path.read_bytes(), size


def _validate_mapping(mapping):
    if (
        not isinstance(mapping, (list, tuple))
        or not mapping
        or any(
            type(value) not in (str, int) or type(value) is bool
            for value in mapping
        )
        or len(set(mapping)) != len(mapping)
    ):
        raise DiffusionBundleError("invalid class mapping")
    return tuple(mapping)


def _validate_state(state, role):
    if type(state) is not dict or set(state) != STATE_FIELDS:
        raise DiffusionBundleError("prepared state fields are invalid")
    if state["checkpoint_role"] != role:
        raise DiffusionBundleError("prepared state role mismatch")
    if state["publication_status"] != "NOT_PUBLISHED":
        raise DiffusionBundleError(
            "prepared state must retain the in-memory publication marker",
        )
    if (
        type(state["selected_epoch"]) is not int
        or state["selected_epoch"] < 0
    ):
        raise DiffusionBundleError("invalid selected epoch")
    if (
        type(state["completed_updates"]) is not int
        or state["completed_updates"] < 1
    ):
        raise DiffusionBundleError("invalid completed updates")
    value = state["validation_mse"]
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DiffusionBundleError("invalid validation MSE")
    weights = _weights(state["weights"])
    digest = _digest(weights)
    if digest != _digest_value(state["state_sha256"], name="state digest"):
        raise DiffusionBundleError("prepared state digest mismatch")
    return weights, digest


def _state_manifest(role, state, weights):
    tensors = []
    for index, array in enumerate(weights):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        raw = buffer.getvalue()
        tensors.append({
            "filename": f"{role}/w{index:03d}.npy",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha(raw),
            "bytes": len(raw),
            "_raw": raw,
        })
    return {
        "checkpoint_role": role,
        "selected_epoch": state["selected_epoch"],
        "completed_updates": state["completed_updates"],
        "validation_mse": float(state["validation_mse"]),
        "state_sha256": state["state_sha256"],
        "weight_count": len(weights),
        "tensors": tensors,
    }


def publish_diffusion_checkpoint_bundle(
    *, prepared, destination, source_commit, dataset_provenance,
):
    """Publish prepared best/last states exclusively and atomically."""
    if type(prepared) is not dict or set(prepared) != set(ROLES):
        raise DiffusionBundleError("exactly best and last states required")
    if (
        type(source_commit) is not str
        or HEX40.fullmatch(source_commit) is None
    ):
        raise DiffusionBundleError("source commit must be a full SHA-1")
    if type(dataset_provenance) is not dict or not dataset_provenance:
        raise DiffusionBundleError("dataset provenance mapping required")

    best = prepared["best"]
    last = prepared["last"]
    best_weights, _ = _validate_state(best, "best")
    last_weights, _ = _validate_state(last, "last")

    dataset = best["dataset_id"]
    seed = best["experimental_seed"]
    mapping = _validate_mapping(best["class_mapping"])
    if dataset not in DATASET_CLASS_COUNTS:
        raise DiffusionBundleError("unsupported dataset identity")
    if len(mapping) != DATASET_CLASS_COUNTS[dataset]:
        raise DiffusionBundleError("class mapping length mismatch")
    if (
        best["class_count"] != len(mapping)
        or last["class_count"] != len(mapping)
    ):
        raise DiffusionBundleError("class count mismatch")
    configuration = _approved_configuration()
    if (
        best["scientific_configuration"] != configuration
        or last["scientific_configuration"] != configuration
    ):
        raise DiffusionBundleError("unapproved scientific configuration")
    for role, state in (("best", best), ("last", last)):
        if (
            state["dataset_id"] != dataset
            or state["experimental_seed"] != seed
            or _validate_mapping(state["class_mapping"]) != mapping
            or state["scientific_configuration"] != configuration
        ):
            raise DiffusionBundleError(
                role + " state identity differs from the bundle identity",
            )

    manifest = {
        "schema": SCHEMA,
        "dataset_id": dataset,
        "experimental_seed": seed,
        "class_mapping": list(mapping),
        "class_count": len(mapping),
        "scientific_configuration": configuration,
        "source_commit": source_commit,
        "dataset_provenance": dataset_provenance,
        "states": {
            "best": _state_manifest("best", best, best_weights),
            "last": _state_manifest("last", last, last_weights),
        },
    }

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("checkpoint bundle destination already exists")
    if not destination.parent.is_dir():
        raise FileNotFoundError("checkpoint bundle parent is unavailable")
    os.mkdir(destination, 0o700)
    _write_exclusive(destination / PENDING_FILENAME, b"")

    for role in ROLES:
        role_directory = destination / role
        os.mkdir(role_directory, 0o700)
        for tensor in manifest["states"][role]["tensors"]:
            _write_exclusive(
                destination / tensor["filename"], tensor.pop("_raw"),
            )

    manifest_raw = _canonical(manifest) + b"\n"
    _write_exclusive(destination / MANIFEST_FILENAME, manifest_raw)
    os.unlink(destination / PENDING_FILENAME)
    return {
        "path": str(destination.resolve()),
        "manifest_sha256": _sha(manifest_raw),
        "bytes": len(manifest_raw),
    }


def _reject_duplicates(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise DiffusionBundleError("duplicate JSON field: " + key)
        record[key] = value
    return record


def _reject_constant(value):
    raise DiffusionBundleError("non-finite JSON constant: " + str(value))


def _load_manifest(directory, expected_manifest_sha256):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise DiffusionBundleError("checkpoint bundle directory is unavailable")
    if (directory / PENDING_FILENAME).exists():
        raise DiffusionBundleError("checkpoint bundle publication is incomplete")
    if set(os.listdir(directory)) != {MANIFEST_FILENAME, "best", "last"}:
        raise DiffusionBundleError("checkpoint bundle entries are unexpected")

    raw, _ = _read_regular(
        directory / MANIFEST_FILENAME, name="bundle manifest",
        limit=MAX_MANIFEST_BYTES,
    )
    if _sha(raw) != _digest_value(
        expected_manifest_sha256, name="expected manifest",
    ):
        raise DiffusionBundleError("bundle manifest digest mismatch")
    try:
        manifest = json.loads(
            raw, parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeError, ValueError) as error:
        raise DiffusionBundleError("bundle manifest is unreadable: " + str(error))
    if type(manifest) is not dict or set(manifest) != MANIFEST_FIELDS:
        raise DiffusionBundleError("bundle manifest fields are invalid")
    return directory, manifest


def _load_state(directory, role, record, expected):
    if type(record) is not dict or set(record) != STATE_MANIFEST_FIELDS:
        raise DiffusionBundleError(role + " state manifest is invalid")
    if record["checkpoint_role"] != role:
        raise DiffusionBundleError(role + " role mismatch")
    if (
        type(record["selected_epoch"]) is not int
        or record["selected_epoch"] < 0
        or type(record["completed_updates"]) is not int
        or record["completed_updates"] < 1
        or type(record["validation_mse"]) not in (int, float)
        or not math.isfinite(float(record["validation_mse"]))
    ):
        raise DiffusionBundleError(role + " selection metadata is invalid")
    _digest_value(record["state_sha256"], name=role + " state digest")
    tensors = record["tensors"]
    if (
        type(tensors) is not list
        or not tensors
        or record["weight_count"] != len(tensors)
    ):
        raise DiffusionBundleError(role + " tensor inventory is invalid")

    arrays = []
    filenames = set()
    for index, tensor in enumerate(tensors):
        if type(tensor) is not dict or set(tensor) != TENSOR_FIELDS:
            raise DiffusionBundleError(role + " tensor fields are invalid")
        filename = tensor["filename"]
        expected_name = f"{role}/w{index:03d}.npy"
        if (
            type(filename) is not str
            or TENSOR_NAME.fullmatch(filename) is None
            or filename != expected_name
            or filename in filenames
        ):
            raise DiffusionBundleError(role + " tensor filename is invalid")
        filenames.add(filename)
        raw, size = _read_regular(
            directory / filename, name=role + " tensor",
            limit=MAX_TENSOR_BYTES,
        )
        if tensor["bytes"] != size or tensor["sha256"] != _sha(raw):
            raise DiffusionBundleError(role + " tensor integrity mismatch")
        try:
            array = np.load(io.BytesIO(raw), allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise DiffusionBundleError(
                role + " tensor is unreadable: " + str(error),
            )
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.str != tensor["dtype"]
            or list(array.shape) != tensor["shape"]
        ):
            raise DiffusionBundleError(role + " tensor metadata mismatch")
        validated = _weights([array])[0]
        arrays.append(validated)

    weights = tuple(arrays)
    if _digest(weights) != record["state_sha256"]:
        raise DiffusionBundleError(role + " reconstructed digest mismatch")
    return {
        **{key: expected[key] for key in (
            "dataset_id", "experimental_seed", "class_mapping",
            "class_count", "scientific_configuration",
        )},
        "checkpoint_role": role,
        "selected_epoch": record["selected_epoch"],
        "completed_updates": record["completed_updates"],
        "validation_mse": record["validation_mse"],
        "weights": weights,
        "state_sha256": record["state_sha256"],
        "publication_status": "NOT_PUBLISHED",
    }


def load_diffusion_checkpoint_bundle(
    directory,
    *,
    expected_manifest_sha256,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
    requested_role=None,
    expected_source_commit=None,
):
    """Load and verify a published bundle; return prepared state(s)."""
    if requested_role is not None and requested_role not in ROLES:
        raise DiffusionBundleError("invalid requested checkpoint role")
    directory, manifest = _load_manifest(
        directory, expected_manifest_sha256,
    )
    dataset = manifest["dataset_id"]
    mapping = _validate_mapping(manifest["class_mapping"])
    if (
        dataset != requested_dataset_id
        or dataset not in DATASET_CLASS_COUNTS
        or manifest["experimental_seed"] != requested_experimental_seed
        or mapping != _validate_mapping(requested_class_mapping)
        or manifest["class_count"] != len(mapping)
        or len(mapping) != DATASET_CLASS_COUNTS[dataset]
        or manifest["scientific_configuration"] != _approved_configuration()
        or type(manifest["dataset_provenance"]) is not dict
        or not manifest["dataset_provenance"]
        or type(manifest["source_commit"]) is not str
        or HEX40.fullmatch(manifest["source_commit"]) is None
    ):
        raise DiffusionBundleError("bundle experiment identity mismatch")
    if (
        expected_source_commit is not None
        and manifest["source_commit"] != expected_source_commit
    ):
        raise DiffusionBundleError("bundle source commit mismatch")

    states = manifest["states"]
    if type(states) is not dict or set(states) != set(ROLES):
        raise DiffusionBundleError("bundle state inventory is invalid")
    expected = {
        "dataset_id": dataset,
        "experimental_seed": manifest["experimental_seed"],
        "class_mapping": mapping,
        "class_count": len(mapping),
        "scientific_configuration": manifest["scientific_configuration"],
    }
    loaded = {
        role: _load_state(directory, role, states[role], expected)
        for role in ROLES
    }
    if requested_role is None:
        return loaded
    return {requested_role: loaded[requested_role]}


__all__ = [
    "DiffusionBundleError",
    "load_diffusion_checkpoint_bundle",
    "publish_diffusion_checkpoint_bundle",
]
