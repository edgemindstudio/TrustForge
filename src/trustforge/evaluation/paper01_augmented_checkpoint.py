"""Bind an augmented training-view identity through published checkpoint bytes."""

import hashlib
import json
import re

from .paper01_classifier_bundle import read_classifier_bundle
from .paper01_result_artifacts import _verified_json


SCHEMA = "trustforge.paper01.augmented-checkpoint-identity.v1"
EXPECTATIONS_SCHEMA = (
    "trustforge.paper01.paired-restoration-expectations.v1"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("explicit SHA-256 identity required")
    return value


def _digest_or_none(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        return None
    return value


def verify_augmented_checkpoint_identity(
    *, dataset_id, experimental_seed, expected_synthetic_manifest_sha256,
    directory, recipe_raw, expectations_raw, expected_expectations_sha256,
    context_evidence_raw,
):
    """Verify the training-view identity through receipt, context and bundle.

    Returns an evidence summary, or an explicit gap when the supplied evidence
    is too old/incomplete to establish the binding. Conflicting evidence is
    rejected. Generator lineage remains unauthenticated.
    """
    if type(recipe_raw) is not bytes or not recipe_raw:
        raise ValueError("classifier recipe bytes required")
    receipt = _verified_json(
        expectations_raw, _digest(expected_expectations_sha256),
    )
    if receipt.get("schema") != EXPECTATIONS_SCHEMA:
        return None, "legacy or unsupported external expectation receipt"
    if (
        type(receipt.get("evaluator_seed")) is not int
        or receipt["evaluator_seed"] != experimental_seed
    ):
        raise ValueError("expectation receipt evaluator seed mismatch")
    if receipt.get("classifier_recipe_sha256") != _sha(recipe_raw):
        raise ValueError("expectation receipt classifier recipe mismatch")
    arms = receipt.get("arms")
    expected = arms.get("real_plus_synthetic") if type(arms) is dict else None
    if type(expected) is not dict or not all(
        key in expected for key in (
            "context", "manifest_sha256", "selected_weights_sha256",
        )
    ):
        return None, (
            "external expectation receipt lacks augmented checkpoint binding"
        )
    context = expected["context"]
    if (
        type(context) is not dict
        or context.get("arm") != "real_plus_synthetic"
        or context.get("dataset_id") != dataset_id
        or type(context.get("evaluator_seed")) is not int
        or context["evaluator_seed"] != experimental_seed
        or type(context.get("training_sample_count")) is not int
        or context["training_sample_count"] <= 0
    ):
        raise ValueError("augmented expectation context conflicts with evaluation")
    context_sha = receipt.get("context_evidence_sha256")
    if _digest_or_none(context_sha) is None:
        return None, "external expectation receipt lacks context evidence"
    context_document = _verified_json(context_evidence_raw, context_sha)
    identity = context_document.get("augmented_training_identity")
    if type(identity) is not dict:
        return None, (
            "external context evidence lacks augmented training identity"
        )
    real, synthetic = identity.get("real"), identity.get("synthetic")
    if type(real) is not dict or type(synthetic) is not dict:
        return None, (
            "external context evidence lacks complete augmented training identity"
        )
    if (
        identity.get("condition") != "real_plus_synthetic"
        or identity.get("position_order") != "real_then_synthetic"
        or real.get("dataset_id") != dataset_id
        or synthetic.get("dataset_id") != dataset_id
        or type(synthetic.get("experimental_seed")) is not int
        or synthetic["experimental_seed"] != experimental_seed
        or synthetic.get("manifest_sha256")
        != expected_synthetic_manifest_sha256
    ):
        raise ValueError("augmented training identity conflicts with evaluation")
    identity_sha = _sha(_canonical(identity))
    if identity_sha != context.get("training_identity_sha256"):
        raise ValueError("context evidence conflicts with expectation receipt")

    loaded = read_classifier_bundle(
        directory, recipe_raw=recipe_raw,
        expected_manifest_sha256=_digest(expected["manifest_sha256"]),
        expected_context=context,
        expected_selected_weights_sha256=_digest(
            expected["selected_weights_sha256"]
        ),
    )
    return {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "evaluator_seed": experimental_seed,
        "classifier_recipe_sha256": _sha(recipe_raw),
        "training_identity": identity,
        "training_identity_sha256": identity_sha,
        "training_size": context["training_sample_count"],
        "selected_weights_sha256": expected["selected_weights_sha256"],
        "manifest_sha256": expected["manifest_sha256"],
        "bundle_manifest_sha256": _sha(loaded["manifest_raw"]),
        "expectations_sha256": expected_expectations_sha256,
        "context_evidence_sha256": context_sha,
        "bundle_verified": True,
        "generator_lineage_authentication": "NOT_PERFORMED",
        "complete_evaluation": False,
    }, None
