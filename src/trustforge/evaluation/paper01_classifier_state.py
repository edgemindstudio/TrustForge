"""Recipe-derived, model-only EvalCNN weight serialization.

Reuses the bounded deterministic NPZ codec. No TensorFlow import, model
assignment, optimizer restoration, publication or provenance authentication.
"""

import re
import numpy as np

from .paper01_inputs import CLASS_COUNTS
from .paper01_classifier_control import selected_recipe, state_digest
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    encode_network_state, decode_network_state,
)


def classifier_specification(*, dataset_id, recipe_raw):
    recipe = selected_recipe(recipe_raw)
    if dataset_id not in CLASS_COUNTS:
        raise ValueError("authoritative dataset required")
    architecture = recipe["architecture"]
    shapes = []
    channels = recipe["canonical_input"]["shape"][-1]
    kernel = architecture["convolution_kernel"]
    for width in architecture["convolution_widths"]:
        shapes.extend(((kernel[0], kernel[1], channels, width), (width,)))
        channels = width
    classes = CLASS_COUNTS[dataset_id]
    shapes.extend(((channels, classes), (classes,)))
    return [
        {
            "key": f"model_{index:04d}",
            "shape": list(shape),
            "dtype": "float32",
            "trainable": True,
        }
        for index, shape in enumerate(shapes)
    ]


def _expected_digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("explicit selected-weight SHA-256 required")


def encode_classifier_weights(
    *, weights, dataset_id, recipe_raw, expected_selected_weights_sha256,
):
    specification = classifier_specification(
        dataset_id=dataset_id, recipe_raw=recipe_raw,
    )
    _expected_digest(expected_selected_weights_sha256)
    if type(weights) not in (list, tuple) or len(weights) != len(specification):
        raise ValueError("exact ordered classifier weights required")
    state = {}
    for entry, value in zip(specification, weights):
        if type(value) is not np.ndarray:
            raise TypeError("classifier weight must be an ndarray")
        state[entry["key"]] = value.copy()
    ordered = [state[entry["key"]] for entry in specification]
    if state_digest(ordered) != expected_selected_weights_sha256:
        raise ValueError("weights differ from selected expectation")
    return encode_network_state(
        role="model", state=state, specification=specification,
    )


def decode_classifier_weights(
    *, raw, dataset_id, recipe_raw, expected_selected_weights_sha256,
):
    specification = classifier_specification(
        dataset_id=dataset_id, recipe_raw=recipe_raw,
    )
    _expected_digest(expected_selected_weights_sha256)
    state = decode_network_state(
        role="model", raw=raw, specification=specification,
    )
    weights = tuple(state[entry["key"]] for entry in specification)
    if state_digest(weights) != expected_selected_weights_sha256:
        raise ValueError("decoded weights differ from selected expectation")
    return weights
