"""Write paired-restoration context evidence and external expectations."""

import hashlib
import json
from pathlib import Path


RECEIPT_SCHEMA = "trustforge.paper01.paired-restoration-expectations.v1"


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    with path.open("xb") as handle:
        handle.write(_canonical(value))


def write_paired_restoration_evidence(
    *, run_dir, shared, job_id, train_pid, evaluator_seed,
    initial_weights_sha256, classifier_recipe_sha256, metric_recipe_sha256,
    test_identity, fixture_pool_manifest_sha256,
    real_identity, augmented_identity, validation_identity,
    sources, environment, arms,
):
    """Persist the training-view evidence and receipt consumed by restoration."""
    run = Path(run_dir)
    context_evidence = {
        "source_files": sources,
        "environment": environment,
        "real_training_identity": real_identity,
        "augmented_training_identity": augmented_identity,
        "validation_identity": validation_identity,
    }
    context_raw = _canonical(context_evidence)
    with (run / "context_evidence.json").open("xb") as handle:
        handle.write(context_raw)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        **shared,
        "job_id": job_id,
        "train_pid": train_pid,
        "evaluator_seed": evaluator_seed,
        "initial_weights_sha256": initial_weights_sha256,
        "classifier_recipe_sha256": classifier_recipe_sha256,
        "metric_recipe_sha256": metric_recipe_sha256,
        "test_identity": test_identity,
        "fixture_pool_manifest_sha256": fixture_pool_manifest_sha256,
        "context_evidence_sha256": _sha(context_raw),
        "arms": arms,
    }
    receipt_raw = _canonical(receipt)
    with (run / "expectations.json").open("xb") as handle:
        handle.write(receipt_raw)
    return {
        "context_evidence_raw": context_raw,
        "receipt": receipt,
        "receipt_raw": receipt_raw,
        "receipt_sha256": _sha(receipt_raw),
    }
