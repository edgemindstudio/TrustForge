"""Exact CIC StandardScaler evidence codec.

Captures supplied fitted state and declared context. Does not authenticate
sources, authorize execution, publish artifacts, or modify checkpoint schemas.
"""

import base64
from copy import deepcopy
import hashlib
import json

import numpy as np
from sklearn.preprocessing import StandardScaler


SCHEMA = "trustforge.paper01.cic-fitted-state.v1"
ARRAY_FIELDS = ("mean_", "var_", "scale_", "n_samples_seen_")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _pack(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("state must contain finite numeric arrays")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _unpack(record):
    if type(record) is not dict or set(record) != {"dtype", "shape", "base64"}:
        raise ValueError("invalid array record")
    dtype = np.dtype(record["dtype"])
    shape = record["shape"]
    if dtype.kind not in "fiu":
        raise ValueError("unsupported state dtype")
    if (
        type(shape) is not list
        or any(type(n) is not int or n < 0 for n in shape)
        or len(shape) > 1
    ):
        raise ValueError("invalid state shape")
    raw = base64.b64decode(record["base64"], validate=True)
    expected = (shape[0] if shape else 1) * dtype.itemsize
    if len(raw) != expected:
        raise ValueError("array byte count mismatch")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError("nonfinite state")
    return array


def _validate_scaler(scaler, training_count):
    if type(scaler) is not StandardScaler:
        raise TypeError("expected StandardScaler")
    if scaler.with_mean is not True or scaler.with_std is not True:
        raise ValueError("CIC requires centering and scaling")
    if type(scaler.copy) is not bool:
        raise ValueError("invalid copy setting")
    if (
        isinstance(scaler.n_features_in_, (bool, np.bool_))
        or not isinstance(scaler.n_features_in_, (int, np.integer))
        or int(scaler.n_features_in_) != 139
    ):
        raise ValueError("expected 139 fitted features")
    for name in ("mean_", "var_", "scale_"):
        array = np.asarray(getattr(scaler, name))
        if (
            array.shape != (139,)
            or array.dtype != np.dtype("float64")
            or not np.isfinite(array).all()
        ):
            raise ValueError("invalid fitted parameter: " + name)
    if np.any(scaler.var_ < 0) or np.any(scaler.scale_ <= 0):
        raise ValueError("invalid variance or scale")
    seen = np.asarray(scaler.n_samples_seen_)
    if (
        seen.shape != ()
        or seen.dtype.kind not in "iu"
        or int(seen) != training_count
    ):
        raise ValueError("fitted sample count differs from training membership")


def capture_fitted_state(*, scaler, training_indices, context):
    """Capture state and declarations without fitting or reading source files."""
    indices = np.asarray(training_indices)
    if (
        indices.dtype != np.dtype("int64")
        or indices.ndim != 1
        or indices.size != 8117
        or np.any(indices < 0)
        or np.any(indices >= 11598)
        or np.unique(indices).size != indices.size
        or np.any(indices[1:] <= indices[:-1])
    ):
        raise ValueError("invalid ordered CIC training membership")
    if type(context) is not dict or not context:
        raise ValueError("nonempty declared provenance context required")
    _validate_scaler(scaler, len(indices))

    payload = {
        "schema": SCHEMA,
        "dataset_id": "cicmaldroid2020",
        "training_indices": _pack(indices),
        "scaler": {
            "parameters": {
                "copy": scaler.copy,
                "with_mean": scaler.with_mean,
                "with_std": scaler.with_std,
            },
            "n_features_in": int(scaler.n_features_in_),
            "arrays": {
                name: _pack(getattr(scaler, name)) for name in ARRAY_FIELDS
            },
        },
        "declared_context": deepcopy(context),
    }
    raw = _canonical(payload)
    return {
        "payload": payload,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def restore_fitted_state(record):
    """Validate a record and reconstruct an independently owned scaler."""
    if type(record) is not dict or set(record) != {"payload", "sha256"}:
        raise ValueError("invalid fitted-state envelope")
    payload = record["payload"]
    if hashlib.sha256(_canonical(payload)).hexdigest() != record["sha256"]:
        raise ValueError("fitted-state digest mismatch")
    if type(payload) is not dict or set(payload) != {
        "schema", "dataset_id", "training_indices", "scaler", "declared_context",
    }:
        raise ValueError("invalid fitted-state payload")
    if payload["schema"] != SCHEMA or payload["dataset_id"] != "cicmaldroid2020":
        raise ValueError("unsupported fitted-state identity")

    state = payload["scaler"]
    if type(state) is not dict or set(state) != {
        "parameters", "n_features_in", "arrays",
    }:
        raise ValueError("invalid scaler field inventory")
    parameters = state["parameters"]
    if type(parameters) is not dict or set(parameters) != {
        "copy", "with_mean", "with_std",
    } or any(type(v) is not bool for v in parameters.values()):
        raise ValueError("invalid scaler parameters")
    if type(state["n_features_in"]) is not int:
        raise ValueError("invalid feature count")
    if type(state["arrays"]) is not dict or set(state["arrays"]) != set(ARRAY_FIELDS):
        raise ValueError("invalid scaler array inventory")

    scaler = StandardScaler(**parameters)
    scaler.n_features_in_ = state["n_features_in"]
    for name in ARRAY_FIELDS:
        setattr(scaler, name, _unpack(state["arrays"][name]))
    indices = _unpack(payload["training_indices"])

    rebuilt = capture_fitted_state(
        scaler=scaler,
        training_indices=indices,
        context=payload["declared_context"],
    )
    if _canonical(rebuilt) != _canonical(record):
        raise ValueError("noncanonical fitted-state record")
    return scaler
