#!/usr/bin/env python3
"""Diagnostic-only CIC RBM seed-42 gradient-history replay.

This runner never persists a model checkpoint and never changes
the governed training budget or optimizer implementation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np

from trustforge.data.paper01_revision_2026.cicmaldroid import (
    CICMalDroidRevision2026,
)
from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_training as training,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    load_rbm_checkpoint,
)
from trustforge.provenance import canonical_json_bytes


DATASET_ID = "cicmaldroid2020"
SEED = 42
CLASS_COUNT = 5
MAX_UPDATES = 64
VALIDATION_STEPS = (0, 32, 64)
EXPECTED_STATE_SHA256 = (
    "10f64703515290ea1493b731bb8e941f7b200c6ad18a61acfc8a1be8bd17cd72"
)
OUTPUT_NAME = "rbm_gradient_diagnostic_cicmaldroid2020_seed42.json"


def require_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("diagnostic replay requires a Slurm allocation")
    if "login" in socket.gethostname().lower():
        raise RuntimeError("diagnostic replay cannot run on a login node")


def validated_output(repo: Path) -> Path:
    audit_root = (
        repo / "studies" / "paper01_benchmark"
        / "audits" / "revision_2026"
    ).resolve(strict=True)
    if not audit_root.is_relative_to(repo):
        raise ValueError("audit directory escapes repository")
    target = audit_root / OUTPUT_NAME
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"diagnostic evidence exists: {target}")
    return target


def exclusive_json(path: Path, payload: dict) -> None:
    encoded = canonical_json_bytes(payload) + b"\n"
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    created_stat = os.fstat(fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            current_stat = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if (
                current_stat.st_dev == created_stat.st_dev
                and current_stat.st_ino == created_stat.st_ino
            ):
                path.unlink()
        raise



def verify_replay_equivalence(*, replay, reference):
    """Fail closed unless replay exactly matches authoritative state."""

    replay_state = replay.model.state_dict()
    reference_state = reference.model.state_dict()

    differences = {}
    exact = {}

    for key in ("W", "a", "b", "U"):
        left, right = replay_state[key], reference_state[key]

        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(
                f"checkpoint state structure mismatch: {key}"
            )

        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise RuntimeError(
                f"checkpoint state contains nonfinite values: {key}"
            )

        differences[key] = float(np.max(np.abs(left - right)))
        exact[key] = bool(np.array_equal(left, right))

    history = []
    for item in replay.validation_history:
        mse = float(item.diagnostics.reconstruction_mse)
        if not np.isfinite(mse):
            raise RuntimeError(
                "RBM validation history contains non-finite MSE"
            )
        history.append(
            {
                "step": int(item.step),
                "reconstruction_mse": mse,
                "finite_state": True,
            }
        )

    validation_exact = (
        history == reference.metadata["validation_history"]
    )

    if not all(exact.values()) or not validation_exact:
        raise RuntimeError(
            "diagnostic replay differs from authoritative checkpoint; "
            f"max_absolute_differences={differences}; "
            f"validation_exact={validation_exact}; "
            "no diagnostic evidence written"
        )

    return differences, exact, history, validation_exact

def main() -> None:
    require_compute_node()

    repo = Path.cwd().resolve()
    if not (repo / ".git").is_dir():
        raise RuntimeError("run from TrustForge repository root")

    target = validated_output(repo)

    checkpoint_dir = (
        Path.home()
        / "trustforge_revision_2026_artifacts"
        / "paper01" / "rbm" / DATASET_ID / "seed42"
    )
    reference = load_rbm_checkpoint(
        checkpoint_dir,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=tuple(str(c) for c in range(CLASS_COUNT)),
    )
    metadata = reference.metadata

    expected = {
        "maximum_updates": MAX_UPDATES,
        "selected_step": MAX_UPDATES,
        "class_count": CLASS_COUNT,
        "batch_size": 256,
        "cd_depth": 1,
        "learning_rate": 0.0005,
        "class_balance_exponent": 0.85,
        "parameter_dtype": "float64",
        "optimizer": "manual_sgd",
        "checkpoint_selection_rule": "final_governed_update",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint configuration mismatch: {key}")

    if metadata.get("state_sha256") != EXPECTED_STATE_SHA256:
        raise RuntimeError("unexpected authoritative checkpoint digest")
    if tuple(metadata["validation_schedule"]) != VALIDATION_STEPS:
        raise RuntimeError("unexpected authoritative validation schedule")

    home = Path.home()
    adapter = CICMalDroidRevision2026(
        home / "gencys" / "data" / "CICMalDroid2020_raw"
        / "feature_vectors_syscalls_frequency_5_Cat.csv",
        home / "gencys" / "data" / "CICMalDroid2020_stage3"
        / "feature_columns.json",
    )
    labels = adapter.labels("train")
    if labels.shape != (8117,):
        raise RuntimeError("unexpected CIC training split")

    positions = {
        c: np.flatnonzero(labels == c).astype(np.int64)
        for c in range(CLASS_COUNT)
    }
    if sum(len(p) for p in positions.values()) != len(labels):
        raise RuntimeError("class inventory mismatch")

    original_optimizer = training.apply_manual_sgd_update
    records = []
    cumulative_gradient = np.zeros((CLASS_COUNT, 128), np.float64)
    cumulative_class_norms = np.zeros(CLASS_COUNT, np.float64)

    def observing_optimizer(*, model, cd_result):
        gradient = np.array(cd_result.delta_U, copy=True)
        if gradient.dtype != np.float64 or gradient.shape != (5, 128):
            raise RuntimeError("invalid class gradient structure")
        if not np.isfinite(gradient).all():
            raise RuntimeError("nonfinite class gradient")

        gradient.setflags(write=False)
        result = original_optimizer(model=model, cd_result=cd_result)

        class_norms = np.linalg.norm(gradient, axis=1)
        cumulative_gradient[:] += gradient
        cumulative_class_norms[:] += class_norms
        U = np.array(model.U.numpy(), copy=True)

        if not np.isfinite(U).all():
            raise RuntimeError("nonfinite class parameters")
        if not np.isclose(
            np.linalg.norm(gradient),
            result.delta_U_norm,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("observer optimizer norm mismatch")

        pairs = {
            f"{a}-{b}": float(np.linalg.norm(U[a] - U[b]))
            for a in range(CLASS_COUNT)
            for b in range(a + 1, CLASS_COUNT)
        }
        records.append({
            "step": len(records) + 1,
            "delta_U_norm": result.delta_U_norm,
            "class_gradient_norms": class_norms.tolist(),
            "class_U_norms": np.linalg.norm(U, axis=1).tolist(),
            "pairwise_U_distances": pairs,
        })
        return result

    print("[P1 RBM GRADIENT] replay=BEGIN", flush=True)

    training.apply_manual_sgd_update = observing_optimizer
    try:
        replay = training.train_shared_conditional_rbm(
            adapter=adapter,
            dataset_id=DATASET_ID,
            experimental_seed=SEED,
            train_class_positions=positions,
        )
    finally:
        training.apply_manual_sgd_update = original_optimizer

    if replay.optimizer_steps_completed != MAX_UPDATES:
        raise RuntimeError("replay update-count mismatch")
    if len(records) != MAX_UPDATES:
        raise RuntimeError("gradient observation-count mismatch")

    differences, exact, history, validation_exact = (
        verify_replay_equivalence(
            replay=replay,
            reference=reference,
        )
    )

    print(
        "[P1 RBM GRADIENT] parameter_exact=" + str(exact),
        flush=True,
    )
    print(
        "[P1 RBM GRADIENT] validation_exact=" + str(validation_exact),
        flush=True,
    )


    cancellation = []
    for c in range(CLASS_COUNT):
        denominator = float(cumulative_class_norms[c])
        numerator = float(np.linalg.norm(cumulative_gradient[c]))
        cancellation.append({
            "class_id": c,
            "signed_gradient_sum_norm": numerator,
            "sum_gradient_norms": denominator,
            "directional_retention_ratio": (
                numerator / denominator if denominator > 0 else None
            ),
        })

    repository_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    payload = {
        "schema_version": "1.0",
        "study_id": "paper01_revision_2026",
        "audit_type": "rbm_class_gradient_diagnostic_replay",
        "dataset_id": DATASET_ID,
        "experimental_seed": SEED,
        "repository_head": repository_head,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "node": socket.gethostname(),
        "checkpoint_state_sha256": EXPECTED_STATE_SHA256,
        "authoritative_checkpoint_replaced": False,
        "training_budget_modified": False,
        "production_chain_depth_selected": False,
        "optimizer_updates": replay.optimizer_steps_completed,
        "parameter_exact_equality": exact,
        "max_absolute_parameter_differences": differences,
        "validation_exact_equality": validation_exact,
        "validation_history": history,
        "class_cumulative_gradient": cancellation,
        "per_update_diagnostics": records,
        "replay_equivalence": "EXACT_PASS",
    }
    exclusive_json(target, payload)
    print(f"[P1 RBM GRADIENT] output={target}", flush=True)
    print("[P1 RBM GRADIENT] status=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
