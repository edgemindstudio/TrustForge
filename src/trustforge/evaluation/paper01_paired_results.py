"""Link two classifier arms and verify saved downstream metric replay.

This is result linkage, not provenance authentication or complete evaluation.
Source/environment compatibility, shared training/validation membership and
generator lineage must be established by the outer orchestration.
"""

from copy import deepcopy
import hashlib
import json
import re
import numpy as np

from .paper01_classifier_control import selected_recipe, state_digest
from .paper01_downstream_metrics import downstream_metrics, selected_metric_recipe


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("explicit SHA-256 identity required")
    return value


def _verify_arm_linkage(
    *, arm, evaluator_seed, expected_test_identity,
    expected_initial_weights_sha256, classifier_recipe_raw, metric_recipe_raw,
    shared_labels=None,
):
    """Verify one arm's training, prediction linkage and metric replay.

    When shared_labels is supplied, paired ordered-label agreement is
    enforced before the replay check. Provenance authentication, cross-arm
    pairing and completion remain outer requirements.
    """
    classifier = selected_recipe(classifier_recipe_raw)
    selected_metric_recipe(metric_recipe_raw)
    if (
        type(evaluator_seed) is not int
        or evaluator_seed not in classifier["experimental_seeds"]
    ):
        raise ValueError("selected evaluator seed required")
    initial = _digest(expected_initial_weights_sha256)
    test = deepcopy(expected_test_identity)
    if type(test) is not dict or test.get("split") != "test":
        raise ValueError("explicit held-out identity required")
    recipe_sha = hashlib.sha256(classifier_recipe_raw).hexdigest()

    training = arm["training"]
    collected = arm["predictions"]
    prediction = collected["report"]
    labels = collected["labels"]
    probabilities = collected["probabilities"]
    if (
        type(training["evaluator_seed"]) is not int
        or training["evaluator_seed"] != evaluator_seed
        or training["recipe_sha256"] != recipe_sha
        or training["initial_weights_sha256"] != initial
    ):
        raise ValueError("training seed, recipe or initialization mismatch")
    selected = _digest(training["selected_weights_sha256"])
    if (
        prediction["schema"] != "trustforge.paper01.test-predictions.v1"
        or prediction["test_identity"] != test
        or prediction["selected_weights_sha256"] != selected
        or prediction["inference_state_unchanged"] is not True
        or prediction["position_order"]
        != "complete_logical_positions_zero_through_size_minus_one"
        or prediction["sample_count"] != test["size"]
        or type(prediction["maximum_image_batch"]) is not int
        or not 1 <= prediction["maximum_image_batch"] <= 128
    ):
        raise ValueError("prediction linkage or coverage mismatch")
    if (
        not isinstance(labels, np.ndarray)
        or labels.dtype != np.int64
        or labels.shape != (test["size"],)
        or not isinstance(probabilities, np.ndarray)
        or probabilities.dtype != np.float32
        or probabilities.shape != (test["size"], test["class_count"])
    ):
        raise ValueError("ordered prediction array contract mismatch")
    if state_digest([labels, probabilities]) != _digest(
        prediction["ordered_prediction_state_sha256"]
    ):
        raise ValueError("ordered prediction identity mismatch")
    if shared_labels is not None and not np.array_equal(labels, shared_labels):
        raise ValueError("paired test labels differ")

    # Verification replay; retain the originally collected metric record.
    replayed = downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=test["class_count"], recipe_raw=metric_recipe_raw,
    )
    if replayed != prediction["metrics"]:
        raise ValueError("saved downstream metrics differ from replay")
    return {
        "training": deepcopy(training),
        "prediction": deepcopy(prediction),
        "labels": labels.copy(),
    }


def link_paired_results(
    *, arms, evaluator_seed, expected_test_identity,
    expected_initial_weights_sha256, classifier_recipe_raw, metric_recipe_raw,
):
    classifier = selected_recipe(classifier_recipe_raw)
    selected_metric_recipe(metric_recipe_raw)
    if (
        type(evaluator_seed) is not int
        or evaluator_seed not in classifier["experimental_seeds"]
    ):
        raise ValueError("selected evaluator seed required")
    initial = _digest(expected_initial_weights_sha256)
    if type(arms) is not dict or set(arms) != {
        "real_only", "real_plus_synthetic",
    }:
        raise ValueError("exactly two named comparison arms required")
    test = deepcopy(expected_test_identity)
    if type(test) is not dict or test.get("split") != "test":
        raise ValueError("explicit held-out identity required")

    recipe_sha = hashlib.sha256(classifier_recipe_raw).hexdigest()
    verified = {}
    shared_labels = None
    for name in ("real_only", "real_plus_synthetic"):
        result = _verify_arm_linkage(
            arm=arms[name],
            evaluator_seed=evaluator_seed,
            expected_test_identity=test,
            expected_initial_weights_sha256=initial,
            classifier_recipe_raw=classifier_recipe_raw,
            metric_recipe_raw=metric_recipe_raw,
            shared_labels=shared_labels,
        )
        shared_labels = result["labels"]
        verified[name] = {
            "training": result["training"],
            "prediction": result["prediction"],
        }

    real_f1 = verified["real_only"]["prediction"]["metrics"]["macro_f1"]
    augmented_f1 = verified["real_plus_synthetic"]["prediction"]["metrics"]["macro_f1"]
    report = {
        "schema": "trustforge.paper01.paired-result-linkage.v1",
        "status": "LINKED_NOT_COMPLETE_EVALUATION",
        "evaluator_seed": evaluator_seed,
        "classifier_recipe_sha256": recipe_sha,
        "metric_recipe_sha256": hashlib.sha256(metric_recipe_raw).hexdigest(),
        "initial_weights_sha256": initial,
        "test_identity": test,
        "ordered_labels_sha256": state_digest([shared_labels]),
        "arms": verified,
        "primary_endpoint": "augmented_minus_real_only_macro_f1",
        "macro_f1_difference": augmented_f1 - real_f1,
        "metric_replay_verified": True,
        "complete_evaluation": False,
    }
    json.dumps(report, allow_nan=False)
    return report
