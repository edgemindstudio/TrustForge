"""Ordered held-out prediction collection with selected-state checks.

Callbacks provide inference and state access. Provenance authentication and
linkage of the expected weights to accepted training remain caller-owned.
Stores O(N*K) probabilities and O(N) labels, never a full image dataset.
"""

from copy import deepcopy
import re
import numpy as np

from .paper01_inputs import RealSplitView, validate_batch
from .paper01_classifier_control import state_digest
from .paper01_downstream_metrics import (
    downstream_metrics, selected_metric_recipe,
)


def collect_test_predictions(
    *, test_view, expected_test_identity, expected_selected_weights_sha256,
    predict_batch, get_weights, get_execution_state, metric_recipe_raw,
):
    selected_metric_recipe(metric_recipe_raw)
    if not isinstance(test_view, RealSplitView):
        raise TypeError("real split view required")
    identity = deepcopy(test_view.identity)
    if identity["split"] != "test" or identity != expected_test_identity:
        raise ValueError("expected held-out test identity required")
    if (
        type(expected_selected_weights_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_selected_weights_sha256) is None
    ):
        raise ValueError("explicit selected-weight SHA-256 required")
    if not all(callable(value) for value in (
        predict_batch, get_weights, get_execution_state,
    )):
        raise TypeError("inference and state callbacks required")
    if state_digest(get_weights()) != expected_selected_weights_sha256:
        raise ValueError("loaded weights differ from selected expectation")

    initial_state = state_digest(get_execution_state())
    size, classes = identity["size"], identity["class_count"]
    labels = np.empty(size, dtype=np.int64)
    probabilities = np.empty((size, classes), dtype=np.float32)
    offset = 0
    maximum_batch = 0

    def unchanged():
        if test_view.identity != identity or test_view.size != size:
            raise RuntimeError("test view identity changed")
        if state_digest(get_weights()) != expected_selected_weights_sha256:
            raise RuntimeError("selected weights changed during inference")
        if state_digest(get_execution_state()) != initial_state:
            raise RuntimeError("execution state changed during inference")

    for positions, images, one_hot in test_view.iter_ordered(batch_size=128):
        unchanged()
        count = len(positions)
        expected_positions = np.arange(offset, offset + count, dtype=np.int64)
        if (
            not 1 <= count <= 128
            or offset + count > size
            or not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or not np.array_equal(positions, expected_positions)
        ):
            raise RuntimeError("incomplete or misordered test traversal")
        validate_batch(images, one_hot, count=count, class_count=classes)
        expected_labels = one_hot.argmax(axis=1).astype(np.int64)
        observed = predict_batch(images)
        if (
            not isinstance(observed, np.ndarray)
            or observed.shape != (count, classes)
            or observed.dtype != np.float32
            or not np.isfinite(observed).all()
            or np.any(observed < 0) or np.any(observed > 1)
            or np.any(np.abs(
                observed.astype(np.float64).sum(axis=1) - 1.0
            ) > 1e-6)
        ):
            raise ValueError("invalid classifier probability batch")
        labels[offset:offset + count] = expected_labels
        probabilities[offset:offset + count] = observed
        offset += count
        maximum_batch = max(maximum_batch, count)
        unchanged()

    if offset != size:
        raise RuntimeError("incomplete held-out coverage")
    unchanged()
    metrics = downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=classes, recipe_raw=metric_recipe_raw,
    )
    return {
        "labels": labels,
        "probabilities": probabilities,
        "report": {
            "schema": "trustforge.paper01.test-predictions.v1",
            "test_identity": identity,
            "selected_weights_sha256": expected_selected_weights_sha256,
            "execution_state_sha256": initial_state,
            "ordered_prediction_state_sha256": state_digest([
                labels, probabilities,
            ]),
            "position_order": "complete_logical_positions_zero_through_size_minus_one",
            "sample_count": offset,
            "maximum_image_batch": maximum_batch,
            "inference_state_unchanged": True,
            "metrics": metrics,
            "complete_evaluation": False,
        },
    }
