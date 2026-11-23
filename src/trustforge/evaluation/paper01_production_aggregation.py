"""Authenticated production linkage into the accepted seven-seed reporter.

This module connects persisted paired-evaluation artifacts to
``summarize_declared_paired_seeds``. It does not change the accepted
arithmetic. For each declared successful attempt it:

- authenticates the persisted unified evaluation record;
- binds dataset, family, seed and synthetic-pool manifest identity;
- replays paired metric linkage from saved test predictions with the accepted
  classifier and metric recipes;
- selects the lowest-numbered attempt that passes, never the best score;
- preserves failed, incomplete and pending seed identities.

Only persisted artifacts are read. No model inference, feature extraction,
training or experiment execution occurs here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from .paper01_augmented_checkpoint import (
    verify_augmented_checkpoint_identity,
)
from .paper01_classifier_control import state_digest
from .paper01_downstream_metrics import downstream_metrics
from .paper01_paired_results import link_paired_results
from .paper01_pool_consumption import validate_pool_consumption
from .paper01_seed_reporting import (
    EXPECTED_SEEDS,
    STATUSES,
    summarize_declared_paired_seeds,
)


SCHEMA = "trustforge.paper01.production-aggregation.v1"
INVENTORY_SCHEMA = "trustforge.paper01.production-aggregation-inventory.v1"

EXPECTED_DATASETS = (
    "ustc_tfc2016", "cicmaldroid2020", "malnet_image", "malex",
)
EXPECTED_FAMILIES = ("gan", "vae", "diffusion", "ar", "maf", "rbm", "gmm")

CLASSIFIER_RECIPE_SHA256 = (
    "1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982"
)
METRIC_RECIPE_SHA256 = (
    "d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85"
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_NPZ_BYTES = 64 * 1024 * 1024
ARM_NAMES = ("real_only", "real_plus_synthetic")


class AggregationError(ValueError):
    """Raised when production aggregation evidence is invalid or conflicting."""


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise AggregationError(name + " requires a lowercase SHA-256")
    return value


def _require_text(value, *, name):
    if type(value) is not str or not value.strip():
        raise AggregationError(name + " requires nonempty text")
    return value


def _reject_duplicate_keys(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise AggregationError("duplicate JSON field: " + key)
        record[key] = value
    return record


def _reject_constant(value):
    raise AggregationError("non-finite JSON constant: " + str(value))


def _read_regular(path, *, name, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AggregationError(name + " must be a regular file")
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise AggregationError(name + " has an invalid bounded size")
    raw = path.read_bytes()
    return raw, _sha(raw), size


def _read_json(path, *, name):
    raw, digest, size = _read_regular(path, name=name, limit=MAX_JSON_BYTES)
    try:
        value = json.loads(
            raw, parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, ValueError) as error:
        raise AggregationError(name + " is unreadable: " + str(error))
    return value, {"path": str(Path(path).resolve()), "sha256": digest,
                   "bytes": size}


def expected_cells():
    """Enumerate the accepted dataset/family/seed matrix."""
    return [
        (dataset, family, seed)
        for dataset in EXPECTED_DATASETS
        for family in EXPECTED_FAMILIES
        for seed in EXPECTED_SEEDS
    ]


# ---------------------------------------------------------------------------
# Inventory validation
# ---------------------------------------------------------------------------


def _validate_attempt(raw, *, name):
    if type(raw) is not dict or set(raw) != {
        "attempt", "run_dir", "declared_status", "reason",
    }:
        raise AggregationError(name + " has invalid fields")
    attempt = raw["attempt"]
    if type(attempt) is not int or attempt < 1:
        raise AggregationError(name + " attempt must be a positive integer")
    run_dir = raw["run_dir"]
    if type(run_dir) is not str or not Path(run_dir).is_absolute():
        raise AggregationError(name + " run_dir must be an absolute path")
    status = raw["declared_status"]
    if type(status) is not str or status not in STATUSES:
        raise AggregationError(name + " declared_status is invalid")
    reason = raw["reason"]
    if status == "SUCCEEDED":
        if reason is not None:
            raise AggregationError(name + " success requires a null reason")
    elif type(reason) is not str or not reason.strip():
        raise AggregationError(name + " non-success requires a reason")
    return {
        "attempt": attempt, "run_dir": run_dir,
        "declared_status": status, "reason": reason,
    }


def _validate_seed(raw, *, name):
    if type(raw) is not dict or set(raw) != {
        "experimental_seed", "pool_manifest_sha256", "attempts",
    }:
        raise AggregationError(name + " has invalid fields")
    seed = raw["experimental_seed"]
    if type(seed) is not int or seed not in EXPECTED_SEEDS:
        raise AggregationError(name + " seed must be one of 42-48")
    pool_sha = raw["pool_manifest_sha256"]
    if pool_sha is not None:
        pool_sha = _require_digest(pool_sha, name=name + " pool manifest")
    attempts = raw["attempts"]
    if type(attempts) is not list:
        raise AggregationError(name + " attempts must be a list")
    validated = []
    seen = set()
    for index, attempt in enumerate(attempts):
        item = _validate_attempt(
            attempt, name=f"{name}.attempts[{index}]",
        )
        if item["attempt"] in seen:
            raise AggregationError(name + " repeats an attempt number")
        seen.add(item["attempt"])
        validated.append(item)
    validated.sort(key=lambda item: item["attempt"])
    if pool_sha is None and any(
        item["declared_status"] == "SUCCEEDED" for item in validated
    ):
        raise AggregationError(
            name + " successful attempt requires a seed pool identity",
        )
    return {
        "experimental_seed": seed,
        "pool_manifest_sha256": pool_sha,
        "attempts": validated,
    }


def _validate_cell(raw, *, name):
    if type(raw) is not dict or set(raw) != {
        "dataset_id", "family", "seeds",
    }:
        raise AggregationError(name + " has invalid fields")
    dataset = raw["dataset_id"]
    family = raw["family"]
    if dataset not in EXPECTED_DATASETS or family not in EXPECTED_FAMILIES:
        raise AggregationError(name + " dataset or family is unsupported")
    seeds = raw["seeds"]
    if type(seeds) is not list:
        raise AggregationError(name + " seeds must be a list")
    validated = {}
    for index, item in enumerate(seeds):
        entry = _validate_seed(item, name=f"{name}.seeds[{index}]")
        seed = entry["experimental_seed"]
        if seed in validated:
            raise AggregationError(name + " repeats a seed")
        validated[seed] = {
            "pool_manifest_sha256": entry["pool_manifest_sha256"],
            "attempts": entry["attempts"],
        }
    if set(validated) != set(EXPECTED_SEEDS):
        raise AggregationError(
            name + " must explicitly represent all seven expected seeds",
        )
    return {
        "dataset_id": dataset, "family": family, "seeds": validated,
    }


def validate_inventory(raw):
    if type(raw) is not dict or set(raw) != {"schema", "cells"}:
        raise AggregationError("inventory has invalid fields")
    if raw["schema"] != INVENTORY_SCHEMA:
        raise AggregationError("unsupported aggregation inventory schema")
    cells = raw["cells"]
    if type(cells) is not list or not cells:
        raise AggregationError("inventory requires at least one cell")
    validated = {}
    for index, cell in enumerate(cells):
        entry = _validate_cell(cell, name=f"cells[{index}]")
        key = (entry["dataset_id"], entry["family"])
        if key in validated:
            raise AggregationError("inventory repeats a dataset/family cell")
        validated[key] = entry
    return validated


# ---------------------------------------------------------------------------
# Evidence authentication
# ---------------------------------------------------------------------------


def _load_predictions(run, arm):
    raw, digest, size = _read_regular(
        run / (arm + "_predictions.npz"),
        name=arm + " predictions", limit=MAX_NPZ_BYTES,
    )
    import io

    try:
        data = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise AggregationError(
            arm + " predictions are unreadable: " + str(error)
        )
    if set(data.files) != {"labels", "probabilities"}:
        raise AggregationError(arm + " prediction arrays are incomplete")
    labels = data["labels"]
    probabilities = data["probabilities"]
    return {
        "report": None,
        "labels": labels,
        "probabilities": probabilities,
        "_artifact": {
            "path": str((run / (arm + "_predictions.npz")).resolve()),
            "sha256": digest, "bytes": size,
        },
    }


def _authenticate_attempt(cell, attempt, classifier_raw, metric_raw):
    run = Path(attempt["run_dir"])
    if not run.is_dir() or run.is_symlink():
        raise AggregationError("run directory is unavailable")

    evaluation, evaluation_art = _read_json(
        run / "evaluation.json", name="evaluation record",
    )
    restore, restore_art = _read_json(
        run / "restore_report.json", name="restore report",
    )
    train, train_art = _read_json(
        run / "train_report.json", name="train report",
    )
    paired, paired_art = _read_json(
        run / "paired_downstream.json", name="paired downstream record",
    )

    seed = cell["seed"]
    dataset = cell["dataset_id"]
    family = cell["family"]
    pool_sha = cell["pool_manifest_sha256"]
    if pool_sha is None:
        raise AggregationError("seed pool identity is unavailable")

    if (
        type(evaluation) is not dict
        or evaluation.get("dataset_id") != dataset
        or type(evaluation.get("experimental_seed")) is not int
        or evaluation["experimental_seed"] != seed
        or evaluation.get("synthetic_manifest_sha256") != pool_sha
        or evaluation.get("verified_group_count") != 7
        or evaluation.get("required_status_blockers") != []
        or evaluation.get("lineage_blockers") != []
        or type(evaluation.get("verified_pool_consumption")) is not dict
        or evaluation.get("verified_pool_consumption") is None
        or evaluation.get("complete_evaluation") is not False
    ):
        raise AggregationError("unified evaluation identity or completeness")

    lineage = restore.get("generator_lineage") if isinstance(restore, dict) else None
    if (
        type(lineage) is not dict
        or lineage.get("dataset_id") != dataset
        or lineage.get("family") != family
        or lineage.get("experimental_seed") != seed
        or lineage.get("pool_manifest_sha256") != pool_sha
        or restore.get("complete_evaluation") is not False
        or restore.get("full_benchmark_go") is not False
    ):
        raise AggregationError("generator lineage or restoration identity")

    test_identity = restore.get("test_identity")
    if (
        type(test_identity) is not dict
        or test_identity.get("dataset_id") != dataset
        or test_identity.get("split") != "test"
        or type(test_identity.get("size")) is not int
        or test_identity["size"] <= 0
        or type(test_identity.get("class_count")) is not int
    ):
        raise AggregationError("held-out test identity")
    if (
        type(train) is not dict
        or train.get("test_accessed") is not False
        or train.get("test_identity") != test_identity
        or train.get("complete_evaluation") is not False
        or type(train.get("initial_weights_sha256")) is not str
    ):
        raise AggregationError("paired training identity or test isolation")

    arms = {}
    artifacts = {
        "evaluation": evaluation_art,
        "restore_report": restore_art,
        "train_report": train_art,
        "paired_downstream": paired_art,
    }
    for arm in ARM_NAMES:
        stored = paired.get("arms", {}).get(arm) if isinstance(paired, dict) else None
        if (
            type(stored) is not dict
            or type(stored.get("prediction")) is not dict
            or type(train.get("arms", {}).get(arm)) is not dict
        ):
            raise AggregationError("paired arm evidence is incomplete")
        predictions = _load_predictions(run, arm)
        predictions["report"] = stored["prediction"]
        arms[arm] = {
            "training": train["arms"][arm],
            "predictions": predictions,
        }
        artifacts[arm + "_predictions"] = predictions["_artifact"]

    expectations_raw, expectations_digest, expectations_bytes = _read_regular(
        run / "expectations.json", name="training receipt",
        limit=MAX_JSON_BYTES,
    )
    context_raw, _, _ = _read_regular(
        run / "context_evidence.json", name="context evidence",
        limit=MAX_JSON_BYTES,
    )
    consumption, consumption_art = _read_json(
        run / "pool_consumption.json", name="pool consumption record",
    )
    bound, gap = verify_augmented_checkpoint_identity(
        dataset_id=dataset,
        experimental_seed=seed,
        expected_synthetic_manifest_sha256=pool_sha,
        directory=run / "checkpoints" / "real_plus_synthetic",
        recipe_raw=classifier_raw,
        expectations_raw=expectations_raw,
        expected_expectations_sha256=expectations_digest,
        context_evidence_raw=context_raw,
    )
    if bound is None or gap is not None:
        raise AggregationError(
            "accepted checkpoint/consumption identity unavailable",
        )
    if evaluation.get("verified_pool_consumption") != bound:
        raise AggregationError(
            "unified evaluation consumption evidence differs from replay",
        )
    if restore.get("verified_pool_consumption") != bound:
        raise AggregationError(
            "restoration consumption evidence differs from replay",
        )
    validate_pool_consumption(
        consumption,
        training_identity=bound["training_identity"],
        training_size=bound["training_size"],
        training_report=train["arms"]["real_plus_synthetic"],
    )

    linked = link_paired_results(
        arms=arms,
        evaluator_seed=seed,
        expected_test_identity=test_identity,
        expected_initial_weights_sha256=train["initial_weights_sha256"],
        classifier_recipe_raw=classifier_raw,
        metric_recipe_raw=metric_raw,
    )
    if _canonical(linked) != _canonical(paired):
        raise AggregationError("persisted paired record differs from replay")
    if (
        linked.get("metric_replay_verified") is not True
        or linked.get("status") != "LINKED_NOT_COMPLETE_EVALUATION"
        or linked.get("complete_evaluation") is not False
    ):
        raise AggregationError("paired replay did not authenticate")
    augmented_selected = linked["arms"]["real_plus_synthetic"][
        "prediction"
    ]["selected_weights_sha256"]
    if bound["selected_weights_sha256"] != augmented_selected:
        raise AggregationError(
            "checkpoint selected weights differ from paired evidence",
        )
    artifacts.update({
        "expectations": {
            "path": str((run / "expectations.json").resolve()),
            "sha256": expectations_digest,
            "bytes": expectations_bytes,
        },
        "context_evidence": {
            "path": str((run / "context_evidence.json").resolve()),
            "sha256": _sha(context_raw),
            "bytes": len(context_raw),
        },
        "pool_consumption": consumption_art,
    })

    real_f1 = linked["arms"]["real_only"]["prediction"]["metrics"]["macro_f1"]
    augmented_f1 = (
        linked["arms"]["real_plus_synthetic"]["prediction"]["metrics"]["macro_f1"]
    )
    return {
        "real_only_macro_f1": real_f1,
        "real_plus_synthetic_macro_f1": augmented_f1,
        "macro_f1_difference": linked["macro_f1_difference"],
        "test_identity": test_identity,
        "artifacts": artifacts,
        "evidence": {
            "authentication": "PAIRED_RESULT_REPLAY_VERIFIED",
            "checkpoint_consumption_authentication": "BUNDLE_VERIFIED",
            "verified_group_count": 7,
            "pool_consumption_verified": True,
            "expectations_sha256": bound["expectations_sha256"],
            "checkpoint_bundle_manifest_sha256": bound[
                "bundle_manifest_sha256"
            ],
            "training_identity_sha256": bound["training_identity_sha256"],
            "selected_weights_sha256": {
                arm: linked["arms"][arm]["prediction"]["selected_weights_sha256"]
                for arm in ARM_NAMES
            },
            "fidelity_semantics": "RECORDED_UNIFIED_VERIFICATION",
            "feature_replay": "NOT_PERFORMED",
        },
    }


def _aggregate_seed(cell, attempts, classifier_raw, metric_raw):
    evidence = []
    selected = None
    for attempt in attempts:
        item = {
            "attempt": attempt["attempt"],
            "run_dir": attempt["run_dir"],
            "declared_status": attempt["declared_status"],
            "reason": attempt["reason"],
        }
        if attempt["declared_status"] == "SUCCEEDED":
            try:
                result = _authenticate_attempt(
                    {**cell, "seed": cell["seed"]}, attempt,
                    classifier_raw, metric_raw,
                )
            except (
                AggregationError, ValueError, FileNotFoundError,
                OSError, EOFError, KeyError,
            ) as error:
                item.update({
                    "authentication": "FAILED",
                    "failure_reason": str(error),
                })
            else:
                item.update({"authentication": "PASS", **result})
                if selected is None:
                    selected = item
        else:
            item.update({"authentication": "NOT_ATTEMPTED"})
        evidence.append(item)

    if selected is not None:
        record = {
            "experimental_seed": cell["seed"],
            "status": "SUCCEEDED",
            "reason": None,
            "real_only_macro_f1": selected["real_only_macro_f1"],
            "real_plus_synthetic_macro_f1":
                selected["real_plus_synthetic_macro_f1"],
        }
        return record, evidence, selected

    if not attempts:
        status, reason = "PENDING", "no execution attempt declared"
    else:
        failed = [item for item in evidence
                  if item["declared_status"] == "FAILED"]
        incomplete = [item for item in evidence
                      if item["declared_status"] == "INCOMPLETE"]
        rejected = [item for item in evidence
                    if item.get("authentication") == "FAILED"]
        if failed:
            status, reason = "FAILED", failed[0]["reason"]
        elif incomplete:
            status, reason = "INCOMPLETE", incomplete[0]["reason"]
        elif rejected:
            status = "FAILED"
            reason = "declared success failed authentication: " + (
                rejected[0]["failure_reason"]
            )
        else:
            status, reason = "PENDING", "no authenticated successful attempt"
    return (
        {
            "experimental_seed": cell["seed"], "status": status,
            "reason": reason, "real_only_macro_f1": None,
            "real_plus_synthetic_macro_f1": None,
        },
        evidence,
        None,
    )


def build_aggregate(*, inventory, classifier_recipe_raw, metric_recipe_raw):
    """Authenticate persisted evidence and invoke the accepted reporter."""
    if not isinstance(classifier_recipe_raw, (bytes, bytearray)) or not (
        classifier_recipe_raw
    ):
        raise AggregationError("classifier recipe bytes required")
    if not isinstance(metric_recipe_raw, (bytes, bytearray)) or not (
        metric_recipe_raw
    ):
        raise AggregationError("metric recipe bytes required")
    classifier_recipe_raw = bytes(classifier_recipe_raw)
    metric_recipe_raw = bytes(metric_recipe_raw)
    if _sha(classifier_recipe_raw) != CLASSIFIER_RECIPE_SHA256:
        raise AggregationError("classifier recipe differs from the accepted one")
    if _sha(metric_recipe_raw) != METRIC_RECIPE_SHA256:
        raise AggregationError("metric recipe differs from the accepted one")

    cells = validate_inventory(inventory)
    results = []
    for (dataset, family), cell in sorted(cells.items()):
        records, evidence, selected = [], [], {}
        for seed in EXPECTED_SEEDS:
            seed_cell = {
                **cell, "seed": seed,
                "pool_manifest_sha256": cell["seeds"][seed][
                    "pool_manifest_sha256"
                ],
            }
            record, attempts, chosen = _aggregate_seed(
                seed_cell, cell["seeds"][seed]["attempts"],
                classifier_recipe_raw, metric_recipe_raw,
            )
            records.append(record)
            evidence.append({
                "experimental_seed": seed,
                "pool_manifest_sha256": cell["seeds"][seed][
                    "pool_manifest_sha256"
                ],
                "selected_attempt": (
                    chosen["attempt"] if chosen is not None else None
                ),
                "attempts": attempts,
            })
            if chosen is not None:
                selected[seed] = chosen["evidence"]
        summary = summarize_declared_paired_seeds(records=records)
        results.append({
            "dataset_id": dataset,
            "family": family,
            "authenticated_success_count": summary[
                "declared_successful_pair_count"
            ],
            "records": records,
            "summary": summary,
            "attempt_evidence": evidence,
            "selected_evidence": {
                str(seed): value for seed, value in selected.items()
            },
            "aggregation_authentication": "PAIRED_RESULT_REPLAY_VERIFIED",
            "benchmark_complete": False,
            "full_benchmark_go": False,
            "primary_inference_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "classes": sorted(cells),
        "cell_count": len(results),
        "cells": results,
        "recipe_sha256": {
            "classifier": CLASSIFIER_RECIPE_SHA256,
            "metric": METRIC_RECIPE_SHA256,
        },
        "scope": (
            "authenticated bounded paired evidence; not full benchmark "
            "completion or primary inference authorization"
        ),
        "benchmark_complete": False,
        "full_benchmark_go": False,
        "primary_inference_authorized": False,
    }


def publish_aggregate(path, document):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("aggregation output already exists")
    raw = _canonical(document) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Authenticated Paper 1 production aggregation linkage",
    )
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classifier-recipe", required=True)
    parser.add_argument("--metric-recipe", required=True)
    args = parser.parse_args(argv)

    inventory, _ = _read_json(args.inventory, name="aggregation inventory")
    classifier_raw, _, _ = _read_regular(
        args.classifier_recipe, name="classifier recipe", limit=MAX_JSON_BYTES,
    )
    metric_raw, _, _ = _read_regular(
        args.metric_recipe, name="metric recipe", limit=MAX_JSON_BYTES,
    )
    document = build_aggregate(
        inventory=inventory,
        classifier_recipe_raw=classifier_raw,
        metric_recipe_raw=metric_raw,
    )
    published = publish_aggregate(args.output, document)
    print("STAGE5_AGGREGATION_OUTPUT=" + published["sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
