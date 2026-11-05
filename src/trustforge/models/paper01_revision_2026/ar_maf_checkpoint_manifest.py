"""Pure AR/MAF best/last checkpoint manifest preparation.

Checks declared evidence and supplied archive bytes. Does not access files,
authenticate training history, publish checkpoints or authorize execution.
"""

from copy import deepcopy
import hashlib
import math
import re

from .ar_maf_checkpoint_identity import (
    build_checkpoint_identity, canonical_json_bytes,
)
from .ar_maf_checkpoint_npz import decode_network_state

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HISTORY_FIELDS = {
    "epoch_index", "train_nll", "validation_nll",
    "training_examples", "validation_examples",
    "training_batches", "validation_batches", "completed_training_steps",
}


def _positive_integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("expected positive integer")
    return value


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("invalid SHA-256 identity")
    return value


def prepare_checkpoint_manifest(
    *, identity, recipe_raw, history, specification,
    best_raw, last_raw, source_commit, source_files,
):
    """Build a validated manifest from complete two-epoch evidence.

    Source-file hashes are declarations here; verify them at execution.
    Historical input authenticity and adapter agreement remain outer checks.
    """
    if type(identity) is not dict:
        raise TypeError("identity must be a dictionary")
    rebuilt = build_checkpoint_identity(
        family=identity["family"],
        dataset_id=identity["dataset_id"],
        experimental_seed=identity["experimental_seed"],
        class_mapping=identity["class_mapping"],
        recipe_raw=recipe_raw,
        dataset_provenance=identity["dataset_provenance"],
    )
    if canonical_json_bytes(identity) != canonical_json_bytes(rebuilt):
        raise ValueError("checkpoint identity differs from rebuilt contract")

    if type(source_commit) is not str or HEX40.fullmatch(source_commit) is None:
        raise ValueError("invalid source commit")
    if type(source_files) is not dict or not source_files:
        raise ValueError("source file inventory required")
    for name, digest in source_files.items():
        if (
            type(name) is not str or not name or "\x00" in name
            or "\\" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError("source path must be repository-relative")
        _digest(digest)

    if not isinstance(history, (list, tuple)) or len(history) != 2:
        raise ValueError("exactly two completed epochs required")
    rows = []
    previous_steps = 0
    split_counts = None
    for epoch, row in enumerate(history):
        if type(row) is not dict or set(row) != HISTORY_FIELDS:
            raise ValueError("training history field inventory mismatch")
        if type(row["epoch_index"]) is not int or row["epoch_index"] != epoch:
            raise ValueError("invalid epoch order")
        for key in ("train_nll", "validation_nll"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("history losses must be finite numbers")
        train_count = _positive_integer(row["training_examples"])
        val_count = _positive_integer(row["validation_examples"])
        train_batches = _positive_integer(row["training_batches"])
        val_batches = _positive_integer(row["validation_batches"])
        steps = _positive_integer(row["completed_training_steps"])
        if (
            train_batches != (train_count + 63) // 64
            or val_batches != (val_count + 63) // 64
            or steps != previous_steps + train_batches
        ):
            raise ValueError("inconsistent traversal or update accounting")
        counts = (train_count, val_count)
        if split_counts is not None and counts != split_counts:
            raise ValueError("split counts changed between epochs")
        split_counts = counts
        previous_steps = steps
        rows.append(deepcopy(row))

    selected = min(range(2), key=lambda i: rows[i]["validation_nll"])
    artifacts = {}
    for role, raw in (("best", best_raw), ("last", last_raw)):
        decode_network_state(
            role="model", raw=raw, specification=specification,
        )
        artifacts[role] = {
            "filename": role + "_state.npz",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "epoch_index": selected if role == "best" else 1,
        }
    if selected == 1 and best_raw != last_raw:
        raise ValueError("best and last disagree for the same selected epoch")

    return {
        "schema": "trustforge.paper01.ar-maf.checkpoint-manifest.v1",
        "identity": deepcopy(rebuilt),
        "source_commit": source_commit,
        "source_files": deepcopy(source_files),
        "history": rows,
        "completed_epochs": 2,
        "completed_training_steps": previous_steps,
        "best_epoch_index": selected,
        "best_validation_nll": rows[selected]["validation_nll"],
        "tensor_specification": deepcopy(specification),
        "artifacts": artifacts,
        "training_resume_supported": False,
    }
