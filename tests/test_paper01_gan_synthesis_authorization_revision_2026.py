"""Focused tests for the synthesis-only CIC GAN authorization boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_synthesis_authorization as authorization,
)
from trustforge.provenance import canonical_json_bytes


SOURCE = "abc1234" + "0" * 33
CHECKPOINT_NAME = "commit42c40fc_attempt01"
RUN_IDENTITY = "commitabc1234_attempt01"
LAUNCHER_SHA256 = "b" * 64


def metadata():
    return {
        "schema_version": "3.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "class_mapping": ["0", "1", "2", "3", "4"],
        "class_count": 5,
        "generator_iteration": 500,
        "discriminator_updates": 1000,
        "planned_generator_iterations": 500,
        "checkpoint_selection": "terminal_governed_iteration",
        "state_files": {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        "dataset_provenance_file": "gan_dataset_provenance.json",
        "dataset_provenance_sha256": "c" * 64,
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "d" * 64,
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
    }


def grant_record(
    *,
    checkpoint_root,
    pool_root,
    checkpoint_sha256,
    source=SOURCE,
    **overrides,
):
    run_identity = f"commit{source[:7]}_attempt01"
    record = {
        "schema": authorization.SCHEMA,
        "paper": authorization.PAPER,
        "revision": authorization.REVISION,
        "family": authorization.FAMILY,
        "dataset_id": authorization.DATASET_ID,
        "experimental_seed": authorization.EXPERIMENTAL_SEED,
        "attempt": authorization.ATTEMPT,
        "source_commit": source,
        "launcher_sha256": LAUNCHER_SHA256,
        "checkpoint_directory": str(
            checkpoint_root / CHECKPOINT_NAME
        ),
        "checkpoint_run_identity": CHECKPOINT_NAME,
        "checkpoint_sha256": checkpoint_sha256,
        "destination": str(pool_root / run_identity),
        "run_identity": run_identity,
        "recipe_sha256": authorization.RECIPE_SHA256,
        "class_mapping": list(authorization.CLASS_MAPPING),
        "quota_per_class": authorization.QUOTA_PER_CLASS,
        "total_samples": authorization.TOTAL_SAMPLES,
        "batch_size": authorization.BATCH_SIZE,
        "batches_per_class": authorization.BATCHES_PER_CLASS,
        "total_batches": authorization.TOTAL_BATCHES,
        "resources": dict(authorization.EXPECTED_RESOURCES),
        "authorization_id": "gan-pool-seed42-approved",
    }
    record.update(overrides)
    return record


@pytest.fixture
def context(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints" / "production"
    pool_root = tmp_path / "pools" / "production"
    checkpoint = checkpoint_root / CHECKPOINT_NAME
    checkpoint.mkdir(parents=True)
    meta = metadata()
    raw = canonical_json_bytes(meta) + b"\n"
    (checkpoint / "model_metadata.json").write_bytes(raw)
    checkpoint_sha256 = hashlib.sha256(
        canonical_json_bytes(meta)
    ).hexdigest()

    path = tmp_path / "authorization.json"

    monkeypatch.setattr(
        authorization, "CHECKPOINT_ROOT", checkpoint_root
    )
    monkeypatch.setattr(authorization, "POOL_ROOT", pool_root)
    monkeypatch.setattr(
        authorization, "AUTHORIZATION_PATH", path
    )
    monkeypatch.setattr(
        authorization, "ACCOUNT_HOME", Path.home().resolve()
    )
    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=SOURCE, dirty=False),
    )

    record = grant_record(
        checkpoint_root=checkpoint_root,
        pool_root=pool_root,
        checkpoint_sha256=checkpoint_sha256,
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)

    return SimpleNamespace(
        tmp_path=tmp_path,
        checkpoint_root=checkpoint_root,
        pool_root=pool_root,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        path=path,
        record=record,
        meta=meta,
    )


def require(context, **overrides):
    values = {
        "destination": context.pool_root / RUN_IDENTITY,
        "checkpoint_directory": context.checkpoint,
        "dataset_id": authorization.DATASET_ID,
        "experimental_seed": authorization.EXPERIMENTAL_SEED,
        "repository_root": context.tmp_path,
    }
    values.update(overrides)
    return authorization.require_gan_synthesis_authorization(**values)


def write_grant(context, record):
    context.path.write_text(json.dumps(record), encoding="utf-8")
    context.path.chmod(0o600)


def test_valid_grant_is_accepted(context):
    grant = require(context)

    assert grant.dataset_id == "cicmaldroid2020"
    assert grant.experimental_seed == 42
    assert grant.attempt == 1
    assert grant.source_commit == SOURCE
    assert grant.checkpoint_directory == context.checkpoint
    assert grant.checkpoint_run_identity == CHECKPOINT_NAME
    assert grant.checkpoint_sha256 == context.checkpoint_sha256
    assert grant.destination == context.pool_root / RUN_IDENTITY
    assert grant.run_identity == RUN_IDENTITY
    assert not grant.destination.exists()


def test_missing_authorization_denies(context):
    context.path.unlink()

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="absent",
    ):
        require(context)


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "unsupported"},
        {"paper": "paper02"},
        {"revision": "revision_2025"},
        {"family": "vae"},
        {"dataset_id": "malex"},
        {"experimental_seed": 43},
        {"attempt": 2},
        {"source_commit": "abc"},
        {"launcher_sha256": "short"},
        {"checkpoint_directory": "/tmp/not-the-checkpoint"},
        {"checkpoint_run_identity": "commitdeadbee_attempt01"},
        {"checkpoint_sha256": "short"},
        {"destination": "/tmp/not-the-pool"},
        {"run_identity": "commitabc1234_attempt02"},
        {"run_identity": "commitdeadbee_attempt01"},
        {"recipe_sha256": "0" * 64},
        {"class_mapping": ["0", "1", "2", "3"]},
        {"quota_per_class": 1999},
        {"total_samples": 9999},
        {"batch_size": 8},
        {"batches_per_class": 250},
        {"total_batches": 1250},
        {
            "resources": {
                **authorization.EXPECTED_RESOURCES,
                "cpus_per_task": 4,
            }
        },
        {"authorization_id": "RESEARCHER_DECISION_PENDING"},
        {"authorization_id": "bad id"},
    ],
)
def test_invalid_grant_fields_deny(context, overrides):
    record = grant_record(
        checkpoint_root=context.checkpoint_root,
        pool_root=context.pool_root,
        checkpoint_sha256=context.checkpoint_sha256,
    )
    record.update(overrides)
    write_grant(context, record)

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        require(context)


def test_training_authorization_schema_is_rejected(context):
    record = dict(context.record)
    record["schema"] = (
        "trustforge.paper01.gan.production-authorization.v1"
    )
    write_grant(context, record)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="schema",
    ):
        require(context)


def test_missing_and_extra_fields_deny(context):
    record = dict(context.record)
    del record["launcher_sha256"]
    write_grant(context, record)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="fields",
    ):
        require(context)

    record = dict(context.record)
    record["unexpected"] = True
    write_grant(context, record)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="fields",
    ):
        require(context)


def test_symlinked_and_writable_authorization_deny(context):
    target = context.path.with_name("real_authorization.json")
    target.write_bytes(context.path.read_bytes())
    context.path.unlink()
    context.path.symlink_to(target)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="absent",
    ):
        require(context)

    context.path.unlink()
    write_grant(context, context.record)
    context.path.chmod(0o666)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="writable",
    ):
        require(context)


def test_dirty_or_wrong_source_denies(context, monkeypatch):
    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=SOURCE, dirty=True),
    )
    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="dirty",
    ):
        require(context)

    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda root: SimpleNamespace(
            commit="b" * 40, dirty=False,
        ),
    )
    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="source",
    ):
        require(context)


def test_existing_destination_denies(context):
    (context.pool_root / RUN_IDENTITY).mkdir(parents=True)

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="exists",
    ):
        require(context)


def test_wrong_requested_identity_denies(context):
    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        require(context, dataset_id="malex")

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        require(context, experimental_seed=43)

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        require(context, checkpoint_directory=context.tmp_path)

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        require(context, destination=context.tmp_path / "other")


def test_checkpoint_metadata_digest_preflight(context):
    digest = authorization.verify_checkpoint_metadata_digest(
        context.checkpoint, context.checkpoint_sha256,
    )
    assert digest == context.checkpoint_sha256

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="digest",
    ):
        authorization.verify_checkpoint_metadata_digest(
            context.checkpoint, "0" * 64,
        )


def test_noncanonical_or_missing_metadata_denies(context):
    path = context.checkpoint / "model_metadata.json"
    path.write_bytes(json.dumps(context.meta).encode())

    with pytest.raises(
        authorization.GANSynthesisAuthorizationError,
        match="canonical",
    ):
        authorization.verify_checkpoint_metadata_digest(
            context.checkpoint, context.checkpoint_sha256,
        )

    path.unlink()

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        authorization.verify_checkpoint_metadata_digest(
            context.checkpoint, context.checkpoint_sha256,
        )
