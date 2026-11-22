"""Bounded independent VAE checkpoint verification and restoration.

Reads the published job-316796 generations, verifies both committed
generations, restores BEST into isolated models, and records evidence.
Does not train, publish, synthesize, evaluate or modify checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_committed_generation import (
    verify_committed_vae_generation,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_loader import (
    load_governed_vae_checkpoint,
)

DATASET_ID = "cicmaldroid2020"
SEED = 42
ATTEMPT = 2
RUN_IDENTITY = "commit1aec01c_attempt02"
SOURCE_COMMIT = "1aec01c26fab51b87362a47e2cfa797bd428c359"
CLASS_MAPPING = ("0", "1", "2", "3", "4")
BEST_ID = "35ca3cd756a88d6b93dcc291f02c4747"
LAST_ID = "0d808f873b163a2d054f87aac34f3043"
BEST_MANIFEST = "f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e"
BEST_COMMIT_RECORD = "f1fcd7cdc8ac7ee99edf1e794eab726a0219213b68007b3a882e5dbebf282118"
LAST_MANIFEST = "ba2a5231db0ef90ac9d724525e61cec9fa4bda6c0838740b52312a13cd8816ac"
LAST_COMMIT_RECORD = "a98b11cfbe7119944ee13c42eadf95660ab278d713636faeb0e2741575d95f15"
ENCODER_SHA = "c308765d65090a97ca872133ed833d2d6f975c52bb8a031aceb2005a5faa0141"
DECODER_SHA = "c7b4f3ff56a13c416f0997e4dd375812cd1db3b6bdadd60d60770a97d06557b0"
PROVENANCE_SHA = "e6736e56b7bd8f2e81505d3f8b1811dfc95a0e0514c629d524025a88a03138ae"
EXPECTED_VALIDATION_LOSS = 4508.749372571244
WT = Path("/home/bruno.fonkeng/TrustForge-paper01-vae-stage4-attempt02")
PUBLICATION_ROOT = Path(
    "/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/"
    "vae/cicmaldroid2020/seed42/checkpoints/production/"
    "commit1aec01c_attempt02"
)
STDOUT_LOG = WT / "p1r1-vae-prod.316796.out"
STDERR_LOG = WT / "p1r1-vae-prod.316796.err"


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    root = Path(root)
    record = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        info = path.stat()
        record[str(path.relative_to(root))] = {
            "sha256": sha256_file(path),
            "size": info.st_size,
            "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        }
    return record


def require_equal(actual, expected, name):
    if actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def verify_role(role, generation_id, expected_manifest, expected_commit):
    verified = verify_committed_vae_generation(
        PUBLICATION_ROOT,
        generation_id=generation_id,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )
    metadata = verified["metadata"]
    record = verified["commit_record"]
    provenance = verified["dataset_provenance"]

    require_equal(verified["generation_id"], generation_id, "generation_id")
    require_equal(record["generation_id"], generation_id, "commit generation_id")
    require_equal(record["manifest_sha256"], expected_manifest, "manifest digest")
    require_equal(metadata["checkpoint_role"], role, "checkpoint_role")
    require_equal(metadata["dataset_id"], DATASET_ID, "dataset_id")
    require_equal(metadata["experimental_seed"], SEED, "experimental_seed")
    require_equal(metadata["class_mapping"], list(CLASS_MAPPING), "class_mapping")
    require_equal(metadata["class_count"], 5, "class_count")
    require_equal(metadata["completed_epochs"], 2, "completed_epochs")
    require_equal(metadata["completed_updates"], 254, "completed_updates")
    require_equal(metadata["latent_dim"], 100, "latent_dim")
    require_equal(metadata["image_shape"], [40, 40, 3], "image_shape")
    require_equal(
        metadata["architecture_version"],
        "paper01_revision2026_conditional_vae_arch_v1",
        "architecture_version",
    )
    require_equal(
        metadata["observation_version"],
        "gaussian_fixed_variance_1p0_v1",
        "observation_version",
    )
    require_equal(
        metadata["selection_policy_id"],
        "min_validation_total_loss_earliest_epoch_v1",
        "selection_policy_id",
    )
    require_equal(
        metadata["validation_seed_policy_id"],
        "fixed_per_batch_across_epochs_v1",
        "validation_seed_policy_id",
    )
    require_equal(
        metadata["validation_criterion"],
        {
            "metric": "validation_total_loss",
            "direction": "minimize",
            "value": EXPECTED_VALIDATION_LOSS,
        },
        "validation_criterion",
    )
    config = metadata["effective_configuration"]
    require_equal(config["source_commit"], SOURCE_COMMIT, "source_commit")
    require_equal(config["attempt"], ATTEMPT, "attempt")
    require_equal(config["run_identity"], RUN_IDENTITY, "run_identity")
    require_equal(
        metadata["state_files"],
        {
            "vae_encoder_state.npz": ENCODER_SHA,
            "vae_decoder_state.npz": DECODER_SHA,
        },
        "state_files",
    )

    state_bytes = verified["verified_state_bytes"]
    require_equal(
        set(state_bytes),
        {"vae_encoder_state.npz", "vae_decoder_state.npz"},
        "verified state inventory",
    )
    state_hashes = {
        name: sha256_bytes(raw)
        for name, raw in state_bytes.items()
    }
    require_equal(state_hashes["vae_encoder_state.npz"], ENCODER_SHA, "encoder bytes")
    require_equal(state_hashes["vae_decoder_state.npz"], DECODER_SHA, "decoder bytes")

    provenance_bytes = canonical_json_bytes(provenance)
    provenance_sha = sha256_bytes(provenance_bytes)
    require_equal(metadata["dataset_provenance_sha256"], PROVENANCE_SHA, "metadata provenance digest")
    require_equal(provenance_sha, PROVENANCE_SHA, "provenance bytes")
    require_equal(record["manifest_filename"], "vae_checkpoint_manifest.json", "manifest filename")

    commit_path = PUBLICATION_ROOT / "commits" / (generation_id + ".commit.json")
    manifest_path = PUBLICATION_ROOT / "generations" / generation_id / "vae_checkpoint_manifest.json"
    commit_raw_sha = sha256_file(commit_path)
    manifest_raw_sha = sha256_file(manifest_path)
    require_equal(commit_raw_sha, expected_commit, "commit-record raw digest")
    require_equal(manifest_raw_sha, expected_manifest, "manifest raw digest")

    return {
        "role": role,
        "generation_id": generation_id,
        "commit_filename": generation_id + ".commit.json",
        "commit_record": record,
        "commit_record_raw_sha256": commit_raw_sha,
        "manifest_raw_sha256": manifest_raw_sha,
        "metadata": metadata,
        "dataset_provenance": provenance,
        "dataset_provenance_sha256": provenance_sha,
        "state_hashes": state_hashes,
    }


def model_summary(model):
    weights = model.get_weights()
    return {
        "variable_count": len(weights),
        "parameter_count": int(sum(array.size for array in weights)),
        "shapes": [list(array.shape) for array in weights],
        "dtypes": [str(array.dtype) for array in weights],
        "all_finite": bool(all(np.isfinite(array).all() for array in weights)),
    }


def parse_training_log():
    raw = STDOUT_LOG.read_bytes()
    text = raw.decode("utf-8")
    lines = [
        line for line in text.splitlines()
        if line.startswith("VAE_REVISION_PRODUCTION_RESULT ")
    ]
    if not lines:
        raise AssertionError("production result line missing")
    summary = json.loads(lines[-1].split(" ", 1)[1])
    return summary, raw


def main():
    before = inventory(PUBLICATION_ROOT)

    best = verify_role("best", BEST_ID, BEST_MANIFEST, BEST_COMMIT_RECORD)
    last = verify_role("last", LAST_ID, LAST_MANIFEST, LAST_COMMIT_RECORD)

    restored = load_governed_vae_checkpoint(
        PUBLICATION_ROOT / "generations" / BEST_ID,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )
    require_equal(restored["metadata"], best["metadata"], "restored metadata")
    require_equal(restored["dataset_provenance"], best["dataset_provenance"], "restored provenance")
    if restored.get("encoder") is None or restored.get("decoder") is None:
        raise AssertionError("restored models are missing")

    encoder = model_summary(restored["encoder"])
    decoder = model_summary(restored["decoder"])
    if encoder["variable_count"] == 0 or decoder["variable_count"] == 0:
        raise AssertionError("restored model variable inventory is empty")
    if not encoder["all_finite"] or not decoder["all_finite"]:
        raise AssertionError("restored model contains non-finite values")

    after = inventory(PUBLICATION_ROOT)
    unchanged = before == after
    if not unchanged:
        raise AssertionError("checkpoint artifact inventory changed")

    production, stdout_raw = parse_training_log()
    stderr_raw = STDERR_LOG.read_bytes()
    nested = production.get("checkpoint_publication", {})

    result = {
        "task": "VAE Stage 4 independent committed-generation verification and BEST restoration",
        "status": "PASS",
        "dataset_id": DATASET_ID,
        "generator": "vae",
        "experimental_seed": SEED,
        "attempt": ATTEMPT,
        "run_identity": RUN_IDENTITY,
        "source_commit": SOURCE_COMMIT,
        "publication_root": str(PUBLICATION_ROOT),
        "process": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "python_version": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "worktree": {
            "path": str(WT),
            "head": subprocess.run(
                ["git", "-C", str(WT), "rev-parse", "HEAD"],
                check=True, stdout=subprocess.PIPE, text=True,
            ).stdout.strip(),
            "status": subprocess.run(
                ["git", "-C", str(WT), "status", "--porcelain", "--untracked-files=normal"],
                check=True, stdout=subprocess.PIPE, text=True,
            ).stdout,
        },
        "training_logs": {
            "stdout_path": str(STDOUT_LOG),
            "stdout_sha256": sha256_bytes(stdout_raw),
            "stderr_path": str(STDERR_LOG),
            "stderr_sha256": sha256_bytes(stderr_raw),
            "training_result": production.get("training_result"),
            "top_level_checkpoint_status": production.get("checkpoint_status"),
            "top_level_publication_status": production.get("publication_status"),
            "nested_publication_status": nested.get("publication_status"),
            "nested_publication_authorization": production.get("publication_authorization"),
        },
        "best": best,
        "last": last,
        "restoration": {
            "best_generation_id": BEST_ID,
            "encoder": encoder,
            "decoder": decoder,
        },
        "checkpoint_inventory": {
            "before": before,
            "after": after,
            "unchanged": unchanged,
            "CHECKPOINT_ARTIFACTS_UNCHANGED": "YES" if unchanged else "NO",
        },
        "reporting_discrepancy": {
            "top_level_checkpoint_status": production.get("checkpoint_status"),
            "top_level_publication_status": production.get("publication_status"),
            "nested_publication_status": nested.get("publication_status"),
            "explanation": "The top-level training result remained NOT_PREPARED/NOT_PUBLISHED while the nested checkpoint-publication result reported PUBLISHED; commit records, manifests and state files corroborate the nested result.",
        },
        "limitations": [
            "GPFS crash durability remains unverified.",
            "crash_durability_verified=False",
            "Same-account operational custody does not authenticate against a malicious same-UID actor.",
            "Checkpoint loading does not independently authenticate checkpoint origin or establish scientific utility.",
            "Dataset provenance does not independently prove partition execution or continuous source immutability.",
        ],
    }

    result_raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
    result_path = Path(os.environ["RESTORATION_RESULT_PATH"])
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(result_path, flags, 0o600)
    try:
        view = memoryview(result_raw)
        position = 0
        while position < len(view):
            written = os.write(fd, view[position:])
            if written <= 0:
                raise OSError("restoration result write made no progress")
            position += written
        os.fsync(fd)
    finally:
        os.close(fd)

    print("RESTORATION_RESULT " + json.dumps(result, sort_keys=True, default=str))
    print("RESTORATION_RESULT_SHA256 " + sha256_bytes(result_raw))


if __name__ == "__main__":
    main()
