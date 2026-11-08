"""Selected-classifier manifest preparation and byte verification.

Context digests are declared evidence links, not provenance authentication.
No filesystem access, TensorFlow import, model assignment or publication.
"""

import hashlib
import json
import math
import re

from .paper01_inputs import CLASS_COUNTS
from .paper01_classifier_control import selected_recipe
from .paper01_classifier_state import (
    classifier_specification, decode_classifier_weights,
)
from .paper01_result_artifacts import _verified_json


def canonical_manifest(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _digest(value, length=64):
    if type(value) is not str or re.fullmatch(
        r"[0-9a-f]{" + str(length) + "}", value,
    ) is None:
        raise ValueError("invalid explicit digest")


def _context(context, recipe):
    fields = {
        "dataset_id", "evaluator_seed", "arm",
        "training_sample_count", "validation_sample_count",
        "training_identity_sha256", "validation_identity_sha256",
        "source_commit", "source_files_sha256", "environment_sha256",
    }
    if type(context) is not dict or set(context) != fields:
        raise ValueError("exact classifier context required")
    if context["dataset_id"] not in CLASS_COUNTS:
        raise ValueError("unsupported dataset")
    if (
        type(context["evaluator_seed"]) is not int
        or context["evaluator_seed"] not in recipe["experimental_seeds"]
        or context["arm"] not in ("real_only", "real_plus_synthetic")
    ):
        raise ValueError("invalid seed or comparison arm")
    for key in ("training_sample_count", "validation_sample_count"):
        if type(context[key]) is not int or context[key] <= 0:
            raise ValueError("positive sample counts required")
    for key in (
        "training_identity_sha256", "validation_identity_sha256",
        "source_files_sha256", "environment_sha256",
    ):
        _digest(context[key])
    _digest(context["source_commit"], 40)


def _training(report, context, recipe, recipe_sha):
    fields = {
        "recipe_sha256", "evaluator_seed", "initial_weights_sha256",
        "selected_weights_sha256", "selected_epoch_index",
        "selected_validation_loss", "completed_epochs", "optimizer_updates",
        "stopping_outcome", "history",
    }
    if type(report) is not dict or set(report) != fields:
        raise ValueError("exact training report required")
    if (
        report["recipe_sha256"] != recipe_sha
        or type(report["evaluator_seed"]) is not int
        or report["evaluator_seed"] != context["evaluator_seed"]
    ):
        raise ValueError("training recipe or seed mismatch")
    for key in ("initial_weights_sha256", "selected_weights_sha256"):
        _digest(report[key])
    controls = recipe["implementation_selection"]["validation_and_selection"]
    epochs = report["completed_epochs"]
    history = report["history"]
    if (
        type(epochs) is not int or not 1 <= epochs <= controls["maximum_epochs"]
        or type(history) is not list or len(history) != epochs
    ):
        raise ValueError("invalid completed history")
    updates_per_epoch = (
        context["training_sample_count"] + recipe["training"]["batch_size"] - 1
    ) // recipe["training"]["batch_size"]
    best, selected, wait = math.inf, None, 0
    for index, row in enumerate(history):
        if wait >= controls["patience"]:
            raise ValueError("history continues after patience stopping")
        expected = {
            "epoch_index": index,
            "training_examples": context["training_sample_count"],
            "validation_examples": context["validation_sample_count"],
            "optimizer_updates": (index + 1) * updates_per_epoch,
        }
        if type(row) is not dict or set(row) != set(expected) | {
            "training_loss", "validation_loss",
        }:
            raise ValueError("invalid history row")
        if any(type(row[k]) is not int or row[k] != v for k, v in expected.items()):
            raise ValueError("epoch coverage or update mismatch")
        for key in ("training_loss", "validation_loss"):
            if type(row[key]) is not float or not math.isfinite(row[key]):
                raise ValueError("finite recorded float losses required")
        if row["validation_loss"] < best:
            best, selected, wait = row["validation_loss"], index, 0
        else:
            wait += 1
    stopping = "patience" if wait >= controls["patience"] else "maximum_epochs"
    if stopping == "maximum_epochs" and epochs != controls["maximum_epochs"]:
        raise ValueError("training stopped before a selected stopping condition")
    if (
        type(report["selected_epoch_index"]) is not int
        or report["selected_epoch_index"] != selected
        or type(report["selected_validation_loss"]) is not float
        or report["selected_validation_loss"] != best
        or type(report["optimizer_updates"]) is not int
        or report["optimizer_updates"] != epochs * updates_per_epoch
        or report["stopping_outcome"] != stopping
    ):
        raise ValueError("selected state or stopping report mismatch")


def prepare_classifier_manifest(*, context, training_report, state_raw, recipe_raw):
    recipe = selected_recipe(recipe_raw)
    recipe_sha = hashlib.sha256(recipe_raw).hexdigest()
    _context(context, recipe)
    _training(training_report, context, recipe, recipe_sha)
    decode_classifier_weights(
        raw=state_raw, dataset_id=context["dataset_id"], recipe_raw=recipe_raw,
        expected_selected_weights_sha256=training_report["selected_weights_sha256"],
    )
    manifest = {
        "schema": "trustforge.paper01.selected-classifier-manifest.v1",
        "context": context,
        "classifier_recipe_sha256": recipe_sha,
        "training_report": training_report,
        "tensor_specification": classifier_specification(
            dataset_id=context["dataset_id"], recipe_raw=recipe_raw,
        ),
        "state_file_sha256": hashlib.sha256(state_raw).hexdigest(),
        "state_file_bytes": len(state_raw),
        "optimizer_state_included": False,
    }
    # Return an independent JSON representation.
    return json.loads(canonical_manifest(manifest))


def verify_classifier_manifest(
    *, manifest_raw, state_raw, recipe_raw, expected_manifest_sha256,
    expected_context, expected_selected_weights_sha256,
):
    _digest(expected_manifest_sha256)
    _digest(expected_selected_weights_sha256)
    manifest = _verified_json(manifest_raw, expected_manifest_sha256)
    if manifest_raw != canonical_manifest(manifest):
        raise ValueError("noncanonical classifier manifest")
    if canonical_manifest(manifest.get("context")) != canonical_manifest(expected_context):
        raise ValueError("requested classifier context mismatch")
    if manifest["training_report"]["selected_weights_sha256"] != expected_selected_weights_sha256:
        raise ValueError("requested selected-weight identity mismatch")
    rebuilt = prepare_classifier_manifest(
        context=expected_context, training_report=manifest["training_report"],
        state_raw=state_raw, recipe_raw=recipe_raw,
    )
    if canonical_manifest(manifest) != canonical_manifest(rebuilt):
        raise ValueError("manifest differs from validated classifier evidence")
    return rebuilt
