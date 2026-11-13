"""Compose one seed-level evaluation from status, bytes and replay inputs.

Delegates semantic validation to the downstream and fidelity/diversity
inventory verifiers, checks cross-input identities, reports declared
augmented-arm pool linkage, and validates an explicit consumption record
against independently established augmented training evidence. Missing or
unbindable records retain an explicit blocker; conflicting records are
rejected. Generator/source lineage, feature-extraction and image-score
authentication, durable publication and completion authorization remain outer
requirements.
"""

import re

from .paper01_inputs import CLASS_COUNTS
from .paper01_status_inventory import validate_status_inventory
from .paper01_downstream_inventory import verify_downstream_inventory
from .paper01_fidelity_inventory import verify_fidelity_inventory


HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARM_NAMES = ("real_only", "real_plus_synthetic")
DOWNSTREAM_GROUPS = (
    "real_only_downstream", "real_plus_synthetic_downstream", "paired_downstream",
)
FIDELITY_INPUT_FIELDS = {
    "expected_synthetic_manifest_sha256", "kid_inputs", "ms_ssim_inputs",
}


def _downstream_supply(arguments, experimental_seed):
    """Inspect paired replay inputs; defer malformed arm values to the verifier."""
    if arguments is None:
        return frozenset(), False
    if type(arguments) is not dict:
        raise ValueError("exact paired replay inputs required")
    seed = arguments.get("evaluator_seed")
    if type(seed) is not int or seed != experimental_seed:
        raise ValueError("inventory and paired evaluation identities differ")
    arms = arguments.get("arms")
    if type(arms) is not dict or set(arms) != set(ARM_NAMES):
        raise ValueError("exactly two named comparison arms required")
    supplied = set()
    for name in ARM_NAMES:
        value = arms[name]
        if type(value) is dict:
            supplied.add(name)
        elif value is not None:
            return frozenset(), True
    return frozenset(supplied), bool(supplied)


def _validate_pool_linkage(
    linkage, *, dataset_id, experimental_seed, manifest,
):
    """Check declared linkage fields against the shared evaluation identity.

    Agreement establishes a declared identity link only, never verified
    lineage or consumption evidence.
    """
    if type(linkage) is not dict or set(linkage) != {
        "arm", "dataset_id", "experimental_seed", "family",
        "manifest_sha256", "checkpoint_sha256", "selected_weights_sha256",
    }:
        raise ValueError("exact augmented pool linkage fields required")
    if linkage["arm"] != "real_plus_synthetic":
        raise ValueError("pool linkage must bind the augmented classifier arm")
    if (
        linkage["dataset_id"] != dataset_id
        or type(linkage["experimental_seed"]) is not int
        or linkage["experimental_seed"] != experimental_seed
        or linkage["manifest_sha256"] != manifest
    ):
        raise ValueError("pool linkage conflicts with the evaluation identity")
    for field in (
        "manifest_sha256", "checkpoint_sha256", "selected_weights_sha256",
    ):
        if type(linkage[field]) is not str or HEX64.fullmatch(linkage[field]) is None:
            raise ValueError("pool linkage requires explicit SHA-256 identities")
    if type(linkage["family"]) is not str or not linkage["family"].strip():
        raise ValueError("pool linkage requires a generator family identity")
    return {
        "arm": "real_plus_synthetic",
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "family": linkage["family"],
        "manifest_sha256": linkage["manifest_sha256"],
        "checkpoint_sha256": linkage["checkpoint_sha256"],
        "selected_weights_sha256": linkage["selected_weights_sha256"],
        "identity_agreement": "REPORTED",
        "generator_lineage_authentication": "NOT_PERFORMED",
    }


def _training_example_count(training_report):
    """Return one consistent per-epoch training count, or None if unavailable."""
    if type(training_report) is not dict:
        return None
    history = training_report.get("history")
    if (
        type(history) is not list or not history
        or not all(type(row) is dict for row in history)
    ):
        return None
    counts = {row.get("training_examples") for row in history}
    if len(counts) != 1:
        return None
    count = next(iter(counts))
    if type(count) is not int or count <= 0:
        return None
    return count


def _check_pool_consumption_claims(
    record, *, dataset_id, experimental_seed, manifest,
    training_report, real_training_report=None, linkage=None,
):
    """Reject consumption-record claims conflicting with known evidence.

    The record's own training identity is never accepted as an expected
    identity. Its fields are checked as claims against evaluation identity,
    declared linkage, verified training reports and independently available
    training sizes.
    """
    if type(record) is not dict:
        raise ValueError("exact augmented pool consumption record required")
    report = training_report if type(training_report) is dict else {}
    identity = record.get("training_identity")
    claimed_real_size = None
    if type(identity) is dict:
        real = identity.get("real")
        synthetic = identity.get("synthetic")
        if type(real) is dict and type(synthetic) is dict:
            for observed, expected in (
                (identity.get("condition"), "real_plus_synthetic"),
                (identity.get("position_order"), "real_then_synthetic"),
                (real.get("dataset_id"), dataset_id),
                (real.get("split"), "train"),
                (real.get("class_count"), CLASS_COUNTS[dataset_id]),
                (synthetic.get("dataset_id"), dataset_id),
                (synthetic.get("experimental_seed"), experimental_seed),
                (synthetic.get("manifest_sha256"), manifest),
            ):
                if observed is not None and observed != expected:
                    raise ValueError(
                        "consumption record conflicts with evaluation pool "
                        "identity"
                    )
            if linkage is not None:
                for observed, expected in (
                    (synthetic.get("family"), linkage["family"]),
                    (synthetic.get("checkpoint_sha256"),
                     linkage["checkpoint_sha256"]),
                ):
                    if observed is not None and observed != expected:
                        raise ValueError(
                            "consumption record conflicts with declared pool "
                            "linkage"
                        )
            if type(real.get("size")) is int and real["size"] > 0:
                claimed_real_size = real["size"]

    for record_key, report_key, message in (
        ("selected_weights_sha256", "selected_weights_sha256",
         "consumption record conflicts with selected weights"),
        ("classifier_recipe_sha256", "recipe_sha256",
         "consumption record conflicts with classifier recipe"),
        ("evaluator_seed", "evaluator_seed",
         "consumption record conflicts with training seed"),
        ("selected_epoch_index", "selected_epoch_index",
         "consumption record conflicts with selected epoch"),
        ("completed_epochs", "completed_epochs",
         "consumption record conflicts with completed epochs"),
    ):
        if report_key in report and record.get(record_key) != report[report_key]:
            raise ValueError(message)

    selected = report.get("selected_epoch_index")
    completed = report.get("completed_epochs")
    history = report.get("history")
    selected_prefix = record.get("selected_prefix")
    final_prefix = record.get("final_prefix")
    if (
        type(selected_prefix) is dict
        and type(selected) is int and type(history) is list
        and 0 <= selected < len(history)
        and type(history[selected]) is dict
    ):
        updates = history[selected].get("optimizer_updates")
        if (
            updates is not None
            and selected_prefix.get("optimizer_updates") != updates
        ):
            raise ValueError(
                "consumption record conflicts with selected optimizer updates"
            )
    if (
        type(final_prefix) is dict
        and "optimizer_updates" in report
        and final_prefix.get("optimizer_updates") != report["optimizer_updates"]
    ):
        raise ValueError(
            "consumption record conflicts with final optimizer updates"
        )

    training_size = _training_example_count(training_report)
    known_real_size = _training_example_count(real_training_report)
    if claimed_real_size is not None and known_real_size is not None:
        if claimed_real_size != known_real_size:
            raise ValueError(
                "consumption record conflicts with real training identity"
            )
    expected_per_epoch = None
    if training_size is not None and known_real_size is not None:
        expected_per_epoch = training_size - known_real_size
        if expected_per_epoch <= 0:
            raise ValueError(
                "consumption record conflicts with training counts"
            )
        if record.get("synthetic_examples_per_epoch") != expected_per_epoch:
            raise ValueError(
                "consumption record conflicts with training counts"
            )
    else:
        claimed = record.get("synthetic_examples_per_epoch")
        if type(claimed) is int and claimed > 0:
            expected_per_epoch = claimed
    if expected_per_epoch is not None:
        if (
            type(selected_prefix) is dict and type(selected) is int
            and selected >= 0
            and selected_prefix.get("synthetic_examples")
            != (selected + 1) * expected_per_epoch
        ):
            raise ValueError(
                "consumption record conflicts with selected-epoch counts"
            )
        if (
            type(final_prefix) is dict and type(completed) is int
            and completed >= 0
            and final_prefix.get("synthetic_examples")
            != completed * expected_per_epoch
        ):
            raise ValueError(
                "consumption record conflicts with final training counts"
            )


def verify_evaluation(
    *, dataset_id, experimental_seed, records, artifact_bytes,
    downstream_arguments, fidelity_arguments, augmented_pool_linkage=None,
    augmented_pool_consumption=None, optional_result_ids=(),
):
    inventory = validate_status_inventory(
        dataset_id=dataset_id, experimental_seed=experimental_seed,
        records=records, optional_result_ids=optional_result_ids,
    )
    statuses = {
        record["result_id"]: record["status"] for record in inventory["records"]
    }
    if type(fidelity_arguments) is not dict or set(fidelity_arguments) != set(
        FIDELITY_INPUT_FIELDS
    ):
        raise ValueError("exact fidelity replay inputs required")
    manifest = fidelity_arguments["expected_synthetic_manifest_sha256"]

    if downstream_arguments is not None and type(downstream_arguments) is not dict:
        raise ValueError("exact paired replay inputs required")
    test = (
        downstream_arguments.get("expected_test_identity")
        if type(downstream_arguments) is dict else None
    )
    if downstream_arguments is not None and (
        type(test) is not dict
        or test.get("dataset_id") != dataset_id
        or test.get("split") != "test"
        or type(test.get("class_count")) is not int
        or test["class_count"] != CLASS_COUNTS[dataset_id]
        or type(test.get("size")) is not int
        or test["size"] <= 0
    ):
        raise ValueError(
            "held-out test identity conflicts with evaluation identity"
        )
    supplied, delegate = _downstream_supply(
        downstream_arguments, experimental_seed,
    )

    verified = {}
    if delegate:
        verified.update(verify_downstream_inventory(
            dataset_id=dataset_id, experimental_seed=experimental_seed,
            records=records, artifact_bytes=artifact_bytes,
            paired_arguments=downstream_arguments,
            optional_result_ids=optional_result_ids,
        )["downstream_verifications"])
    elif any(statuses[name] == "SUCCEEDED" for name in DOWNSTREAM_GROUPS):
        raise ValueError(
            "successful downstream result requires prediction arrays"
        )
    verified.update(verify_fidelity_inventory(
        dataset_id=dataset_id, experimental_seed=experimental_seed,
        records=records, artifact_bytes=artifact_bytes,
        optional_result_ids=optional_result_ids,
        **fidelity_arguments,
    )["fidelity_diversity_verifications"])

    augmented_weights = None
    if "real_plus_synthetic_downstream" in verified:
        augmented_weights = downstream_arguments["arms"]["real_plus_synthetic"][
            "predictions"
        ]["report"]["selected_weights_sha256"]
    lineage_blockers = []
    declared_linkage = None
    if augmented_pool_linkage is not None:
        candidate = _validate_pool_linkage(
            augmented_pool_linkage, dataset_id=dataset_id,
            experimental_seed=experimental_seed, manifest=manifest,
        )
        if augmented_weights is not None and (
            candidate["selected_weights_sha256"] != augmented_weights
        ):
            raise ValueError(
                "pool linkage conflicts with the verified augmented classifier"
            )
        declared_linkage = candidate
    consumption_gap = None
    if augmented_weights is not None:
        if augmented_pool_consumption is None:
            consumption_gap = "augmented-arm consumption record unavailable"
            reason = (
                "augmented-arm consumption of the declared synthetic pool "
                "remains unverified"
            )
        else:
            augmented_training = downstream_arguments["arms"][
                "real_plus_synthetic"
            ]["training"]
            real_training = None
            if "real_only_downstream" in verified:
                real_arm = downstream_arguments["arms"]["real_only"]
                if type(real_arm) is dict:
                    real_training = real_arm.get("training")
            _check_pool_consumption_claims(
                augmented_pool_consumption, dataset_id=dataset_id,
                experimental_seed=experimental_seed, manifest=manifest,
                training_report=augmented_training,
                real_training_report=real_training,
                linkage=declared_linkage,
            )
            consumption_gap = (
                "independent augmented training identity unavailable from "
                "verified training/evaluation evidence"
            )
            reason = consumption_gap
        lineage_blockers.append({
            "evidence": "augmented_pool_consumption",
            "status": "BLOCKED",
            "reason": reason,
        })

    return {
        "schema": "trustforge.paper01.evaluation-verification.v1",
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "status_inventory": inventory,
        "synthetic_manifest_sha256": manifest,
        "verified_groups": verified,
        "verified_group_count": len(verified),
        "semantic_validation_not_performed": sorted(
            name for name, status in statuses.items()
            if status == "SUCCEEDED" and name not in verified
        ),
        "required_status_blockers": inventory["required_status_blockers"],
        "lineage_blockers": lineage_blockers,
        "declared_pool_linkage": declared_linkage,
        "pool_consumption_gap": consumption_gap,
        "feature_extraction_authentication": "NOT_PERFORMED",
        "image_score_recomputation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "publication_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
