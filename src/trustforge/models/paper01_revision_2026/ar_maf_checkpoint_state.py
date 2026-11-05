"""Ordered AR/MAF model-state capture and assignment.

No filesystem access, model construction, training or publication.
Tensor compatibility alone does not establish architecture or provenance.
The caller must supply an isolated, built destination model.
"""

import numpy as np

from .ar_maf_checkpoint_npz import (
    encode_network_state,
    decode_network_state,
)


def capture_model_state(model):
    """Return independent arrays and their complete ordered specification."""
    variables = tuple(model.weights)
    if not variables:
        raise ValueError("model must be built with nonempty weight state")

    specification = []
    state = {}
    for index, variable in enumerate(variables):
        key = f"model_{index:04d}"
        value = np.asarray(variable.numpy())
        if value.dtype != np.dtype("float32"):
            raise TypeError("checkpoint requires float32 model weights")
        if not value.shape or any(d < 1 for d in value.shape):
            raise ValueError("unsupported empty or scalar model weight")
        if not np.isfinite(value).all():
            raise ValueError("nonfinite model weight")
        if type(variable.trainable) is not bool:
            raise TypeError("weight trainability must be boolean")
        specification.append({
            "key": key,
            "shape": list(value.shape),
            "dtype": "float32",
            "trainable": variable.trainable,
        })
        state[key] = np.array(value, copy=True)

    # Apply the codec's inventory and resource limits during capture.
    encode_network_state(
        role="model", state=state, specification=specification,
    )
    return specification, state


def encode_model_snapshot(*, model, weights):
    """Encode a best/last snapshot against the constructed model inventory."""
    specification, _ = capture_model_state(model)
    if not isinstance(weights, (list, tuple)):
        raise TypeError("snapshot weights must be a sequence")
    if len(weights) != len(specification):
        raise ValueError("snapshot weight count mismatch")
    state = {
        entry["key"]: np.array(value, copy=True)
        for entry, value in zip(specification, weights)
    }
    raw = encode_network_state(
        role="model", state=state, specification=specification,
    )
    return specification, raw


def restore_model_snapshot(*, model, specification, raw):
    """Validate all tensors before assignment into an isolated built model.

    Assignment failure invalidates the destination model; callers must discard
    it. This function does not promise rollback or authorize checkpoint use.
    """
    expected, _ = capture_model_state(model)
    if specification != expected:
        raise ValueError("destination model tensor specification mismatch")

    state = decode_network_state(
        role="model", raw=raw, specification=expected,
    )
    ordered = [state[entry["key"]] for entry in expected]
    model.set_weights(ordered)

    observed_specification, observed = capture_model_state(model)
    if observed_specification != expected or any(
        not np.array_equal(observed[key], value)
        for key, value in state.items()
    ):
        raise RuntimeError("assigned checkpoint state verification failed")
