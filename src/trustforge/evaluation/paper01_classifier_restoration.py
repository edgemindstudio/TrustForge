"""Verified selected-classifier weight assignment on allocated compute nodes.

Restores inference weights only, not optimizer state or training continuation.
Source/environment compatibility and provenance require outer verification.
"""

from .paper01_classifier_bundle import read_classifier_bundle
from .paper01_classifier_state import decode_classifier_weights
from .paper01_classifier_control import state_digest
from .paper01_classifier_tensorflow import (
    require_compute_node, build_classifier_binding,
)


def restore_selected_classifier(
    directory, *, recipe_raw, expected_manifest_sha256,
    expected_context, expected_selected_weights_sha256,
):
    require_compute_node()
    loaded = read_classifier_bundle(
        directory,
        recipe_raw=recipe_raw,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_context=expected_context,
        expected_selected_weights_sha256=expected_selected_weights_sha256,
    )
    context = loaded["manifest"]["context"]
    # Consume the verified returned bytes; never reopen the original state path.
    weights = decode_classifier_weights(
        raw=loaded["state_raw"],
        dataset_id=context["dataset_id"],
        recipe_raw=recipe_raw,
        expected_selected_weights_sha256=expected_selected_weights_sha256,
    )
    from .paper01_inputs import CLASS_COUNTS

    binding = build_classifier_binding(
        recipe_raw=recipe_raw,
        class_count=CLASS_COUNTS[context["dataset_id"]],
        evaluator_seed=context["evaluator_seed"],
    )
    model = binding["model"]
    current = model.get_weights()
    if (
        len(current) != len(weights)
        or any(
            original.shape != restored.shape
            or original.dtype != restored.dtype
            for original, restored in zip(current, weights)
        )
    ):
        raise ValueError("constructed classifier differs from verified state")
    model.set_weights([value.copy() for value in weights])
    if state_digest(model.get_weights()) != expected_selected_weights_sha256:
        raise RuntimeError("selected classifier assignment digest mismatch")
    if binding["controls"]["optimizer_updates"]() != 0:
        raise RuntimeError("restoration unexpectedly updated optimizer")

    return {
        "binding": binding,
        "manifest": loaded["manifest"],
        "report": {
            "schema": "trustforge.paper01.classifier-restoration.v1",
            "manifest_file_sha256": expected_manifest_sha256,
            "selected_weights_sha256": expected_selected_weights_sha256,
            "context": context,
            "selected_weight_assignment_verified": True,
            "optimizer_state_restored": False,
            "training_continuation_authorized": False,
            "complete_evaluation": False,
        },
    }
