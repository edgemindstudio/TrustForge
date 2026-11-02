"""Strict in-memory VAE checkpoint state integrity.

Paper 1 revision 2026.

This module captures and validates complete encoder/decoder
variable inventories. It performs no filesystem operations,
training, checkpoint publication, or model mutation.

Checkpoint persistence and restoration remain separately
governed engineering responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import tensorflow as tf


STATE_SCHEMA_VERSION = "1.0"

ROLE_ENCODER = "encoder"
ROLE_DECODER = "decoder"

ROLES = (
    ROLE_ENCODER,
    ROLE_DECODER,
)

EXPECTED_DTYPE = np.dtype("float32")


def _require_model(network, *, role):
    if role not in ROLES:
        raise ValueError("unsupported VAE checkpoint role")

    if not isinstance(network, tf.keras.Model):
        raise TypeError(
            f"{role} must be a TensorFlow Keras model"
        )

    variables = list(network.variables)

    if not variables:
        raise ValueError(
            f"{role} contains no checkpoint variables"
        )

    if len({id(v) for v in variables}) != len(variables):
        raise ValueError(
            f"{role} contains duplicate variable references"
        )

    return variables


def _tensor_key(role, index):
    return f"{role}_{index:04d}"


def variable_specification(network, *, role):
    """Return ordered state specifications for one network.

    The specification describes the actual model instance.
    A future artifact manifest must additionally bind the
    approved architecture version and class count.
    """
    variables = _require_model(network, role=role)

    trainable_ids = {
        id(variable)
        for variable in network.trainable_variables
    }

    specification = []

    for index, variable in enumerate(variables):
        shape = variable.shape

        if shape.rank is None or any(
            dimension is None for dimension in shape
        ):
            raise ValueError(
                f"unresolved variable shape at {role}:{index}"
            )

        dtype = np.dtype(variable.dtype)

        if dtype != EXPECTED_DTYPE:
            raise TypeError(
                f"unsupported variable dtype at {role}:{index}"
            )

        specification.append(
            {
                "key": _tensor_key(role, index),
                "shape": tuple(int(d) for d in shape),
                "dtype": EXPECTED_DTYPE.name,
                "trainable": id(variable) in trainable_ids,
            }
        )

    return tuple(specification)


def capture_network_state(network, *, role):
    """Capture every trainable and nontrainable variable."""
    specification = variable_specification(
        network,
        role=role,
    )

    variables = list(network.variables)
    result = {}

    for entry, variable in zip(
        specification,
        variables,
        strict=True,
    ):
        array = variable.numpy().copy()

        if array.dtype != EXPECTED_DTYPE:
            raise TypeError(
                f"unexpected tensor dtype: {entry['key']}"
            )

        if array.shape != entry["shape"]:
            raise ValueError(
                f"unexpected tensor shape: {entry['key']}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"nonfinite checkpoint tensor: {entry['key']}"
            )

        result[entry["key"]] = array

    return result


def validate_network_state(
    network,
    state,
    *,
    role,
):
    """Validate exact tensor inventory against one model.

    Returns detached copies suitable for later serialization
    or separately governed restoration.

    This function never modifies model variables.
    """
    specification = variable_specification(
        network,
        role=role,
    )

    if not isinstance(state, Mapping):
        raise TypeError(
            f"{role} state must be a mapping"
        )

    required = {
        entry["key"]
        for entry in specification
    }

    if set(state) != required:
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - required)

        raise ValueError(
            f"{role} checkpoint inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    validated = {}

    for entry in specification:
        key = entry["key"]
        array = state[key]

        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"checkpoint tensor must be ndarray: {key}"
            )

        if array.dtype != EXPECTED_DTYPE:
            raise TypeError(
                f"checkpoint tensor dtype mismatch: {key}"
            )

        if array.shape != entry["shape"]:
            raise ValueError(
                f"checkpoint tensor shape mismatch: {key}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"nonfinite checkpoint tensor: {key}"
            )

        validated[key] = array.copy()

    return validated


def capture_vae_state(*, encoder, decoder):
    """Capture a complete paired VAE state."""
    encoder_variables = _require_model(
        encoder,
        role=ROLE_ENCODER,
    )
    decoder_variables = _require_model(
        decoder,
        role=ROLE_DECODER,
    )

    if {
        id(variable) for variable in encoder_variables
    } & {
        id(variable) for variable in decoder_variables
    }:
        raise ValueError(
            "encoder and decoder share checkpoint variables"
        )

    return {
        ROLE_ENCODER: capture_network_state(
            encoder,
            role=ROLE_ENCODER,
        ),
        ROLE_DECODER: capture_network_state(
            decoder,
            role=ROLE_DECODER,
        ),
    }


def validate_vae_state(*, encoder, decoder, state):
    """Fail closed unless both complete network states match."""
    if not isinstance(state, Mapping):
        raise TypeError("VAE state must be a mapping")

    if set(state) != set(ROLES):
        raise ValueError(
            "VAE checkpoint must contain encoder and decoder"
        )

    encoder_variables = _require_model(
        encoder,
        role=ROLE_ENCODER,
    )
    decoder_variables = _require_model(
        decoder,
        role=ROLE_DECODER,
    )

    if {
        id(variable) for variable in encoder_variables
    } & {
        id(variable) for variable in decoder_variables
    }:
        raise ValueError(
            "encoder and decoder share checkpoint variables"
        )

    validated_encoder = validate_network_state(
        encoder,
        state[ROLE_ENCODER],
        role=ROLE_ENCODER,
    )

    validated_decoder = validate_network_state(
        decoder,
        state[ROLE_DECODER],
        role=ROLE_DECODER,
    )

    return {
        ROLE_ENCODER: validated_encoder,
        ROLE_DECODER: validated_decoder,
    }


__all__ = [
    "STATE_SCHEMA_VERSION",
    "variable_specification",
    "capture_network_state",
    "validate_network_state",
    "capture_vae_state",
    "validate_vae_state",
]
