"""Compose one seed-level evaluation from status, bytes and replay inputs.

Delegates semantic validation to the downstream and fidelity/diversity
inventory verifiers, checks cross-input identities, reports declared
augmented-arm pool linkage and retains an explicit consumption-evidence
blocker when that arm is verified. Generator/source lineage,
feature-extraction and image-score authentication, durable publication and
completion authorization remain outer requirements.
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


def verify_evaluation(
    *, dataset_id, experimental_seed, records, artifact_bytes,
    downstream_arguments, fidelity_arguments, augmented_pool_linkage=None,
    optional_result_ids=(),
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
    if augmented_weights is not None:
        lineage_blockers.append({
            "evidence": "augmented_pool_consumption",
            "status": "BLOCKED",
            "reason": (
                "augmented-arm consumption of the declared synthetic pool "
                "remains unverified"
            ),
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
        "feature_extraction_authentication": "NOT_PERFORMED",
        "image_score_recomputation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "publication_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
