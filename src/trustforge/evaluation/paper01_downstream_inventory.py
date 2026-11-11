"""Bind downstream inventory bytes to prediction and paired-result replay.

Coverage units: test examples for each arm; comparison arms for paired linkage.
A pending or failed arm without prediction arrays is declared None: it cannot
be verified or paired, but the other arm is still verified independently.
Other successful groups receive byte validation only. No completion authority.
"""

import hashlib
import json

from .paper01_inputs import CLASS_COUNTS
from .paper01_result_artifacts import bind_result_artifacts
from .paper01_downstream_result_validation import verify_downstream_result
from .paper01_paired_results import _verify_arm_linkage, link_paired_results


ARM_RESULTS = {
    "real_only": "real_only_downstream",
    "real_plus_synthetic": "real_plus_synthetic_downstream",
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def verify_downstream_inventory(
    *, dataset_id, experimental_seed, records, artifact_bytes,
    paired_arguments, optional_result_ids=(),
):
    bound = bind_result_artifacts(
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        records=records,
        artifact_bytes=artifact_bytes,
        optional_result_ids=optional_result_ids,
    )
    if type(paired_arguments) is not dict or set(paired_arguments) != {
        "arms", "evaluator_seed", "expected_test_identity",
        "expected_initial_weights_sha256",
        "classifier_recipe_raw", "metric_recipe_raw",
    }:
        raise ValueError("exact paired replay inputs required")
    test = paired_arguments["expected_test_identity"]
    if (
        type(test) is not dict
        or test.get("dataset_id") != dataset_id
        or test.get("split") != "test"
        or type(test.get("class_count")) is not int
        or test["class_count"] != CLASS_COUNTS[dataset_id]
        or type(test.get("size")) is not int
        or test["size"] <= 0
        or type(paired_arguments["evaluator_seed"]) is not int
        or paired_arguments["evaluator_seed"] != experimental_seed
    ):
        raise ValueError("inventory and paired evaluation identities differ")

    arms = paired_arguments["arms"]
    if type(arms) is not dict or set(arms) != set(ARM_RESULTS):
        raise ValueError("exactly two named comparison arms required")
    supplied = {name for name, arm in arms.items() if arm is not None}
    if not supplied:
        raise ValueError(
            "at least one comparison arm with prediction arrays required"
        )

    # Replay from supplied arrays and reports, never a verification receipt.
    verified = {}
    paired = None
    if supplied == set(ARM_RESULTS):
        paired = link_paired_results(**paired_arguments)
        verified = paired["arms"]
    else:
        name = next(iter(supplied))
        verified[name] = _verify_arm_linkage(
            arm=arms[name],
            evaluator_seed=paired_arguments["evaluator_seed"],
            expected_test_identity=test,
            expected_initial_weights_sha256=paired_arguments[
                "expected_initial_weights_sha256"
            ],
            classifier_recipe_raw=paired_arguments["classifier_recipe_raw"],
            metric_recipe_raw=paired_arguments["metric_recipe_raw"],
        )

    results = bound["bound_results"]
    receipts = {}
    metric_raw = paired_arguments["metric_recipe_raw"]
    metric_sha = hashlib.sha256(metric_raw).hexdigest()

    for arm, result_id in ARM_RESULTS.items():
        if arm not in supplied:
            if result_id in results:
                raise ValueError(
                    "successful arm result requires prediction arrays"
                )
            continue
        if result_id not in results:
            continue
        item = results[result_id]
        collected = arms[arm]["predictions"]
        if (
            item["configuration_sha256"] != metric_sha
            or item["declared_coverage"]
            != {"expected": test["size"], "computed": test["size"]}
        ):
            raise ValueError("downstream recipe or example coverage mismatch")
        receipt = verify_downstream_result(
            dataset_id=dataset_id,
            labels=collected["labels"],
            probabilities=collected["probabilities"],
            expected_prediction_state_sha256=collected["report"][
                "ordered_prediction_state_sha256"
            ],
            result_raw=artifact_bytes[result_id]["result_raw"],
            expected_result_file_sha256=item["result_file_sha256"],
            metric_recipe_raw=metric_raw,
        )
        if _canonical(item["result"]) != _canonical(
            verified[arm]["prediction"]["metrics"]
        ):
            raise ValueError("arm artifact differs from paired metric record")
        receipts[result_id] = receipt

    paired_id = "paired_downstream"
    if paired_id in results:
        if set(receipts) != set(ARM_RESULTS.values()):
            raise ValueError("successful paired result requires both arm artifacts")
        item = results[paired_id]
        configuration = {
            "classifier_recipe_sha256": paired["classifier_recipe_sha256"],
            "metric_recipe_sha256": paired["metric_recipe_sha256"],
        }
        if (
            _canonical(item["configuration"]) != _canonical(configuration)
            or item["declared_coverage"] != {"expected": 2, "computed": 2}
            or _canonical(item["result"]) != _canonical(paired)
        ):
            raise ValueError("paired artifact, configuration or arm coverage mismatch")
        receipts[paired_id] = {
            "result_file_sha256": item["result_file_sha256"],
            "paired_replay_verified": True,
            "comparison_arm_count": 2,
        }

    return {
        "schema": "trustforge.paper01.downstream-inventory-verification.v1",
        "artifact_binding": bound,
        "downstream_verifications": receipts,
        "semantic_validation_not_performed": sorted(set(results) - set(receipts)),
        "required_status_blockers": bound["status_inventory"][
            "required_status_blockers"
        ],
        "provenance_validation": "NOT_PERFORMED",
        "publication_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
