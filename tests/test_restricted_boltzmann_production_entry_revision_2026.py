"""Focused tests for the RBM governed production-sampling entry point.

These tests use synthetic requests, mocked checkpoint loading and mocked
sampling only. No real dataset, production checkpoint, training or synthesis
is accessed, and no Slurm job is submitted.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_production_entry as entry,
    restricted_boltzmann_production_sampling as sampler,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "run_paper01_rbm_production_sampling.py"
LAUNCHER_PATH = (
    REPO_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "rbm_production_sampling.slurm"
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
STATE_SHA = "c" * 64
OTHER_STATE_SHA = "d" * 64

ALLOCATION = {
    "JobId": "12345",
    "JobState": "RUNNING",
    "Partition": "talon",
    "NumNodes": "1",
    "NumTasks": "1",
    "NumCPUs": "8",
    "TRES": "cpu=8,mem=32G,node=1,billing=8",
    "TimeLimit": "12:00:00",
    "Requeue": "0",
    "Gres": "(null)",
}

# Observed Talon 25.05.6 controller representation for job 317176. It has
# AllocTRES and ReqTRES but no standalone TRES field.
TALON_ALLOCATION = {
    "JobId": "317176",
    "JobName": "p1r1-rbm-sample",
    "JobState": "FAILED",
    "Reason": "NonZeroExitCode",
    "Requeue": "0",
    "Restarts": "0",
    "ExitCode": "1:0",
    "RunTime": "00:00:40",
    "TimeLimit": "00:10:00",
    "Partition": "talon",
    "NodeList": "talon16",
    "BatchHost": "talon16",
    "NumNodes": "1",
    "NumCPUs": "8",
    "NumTasks": "1",
    "CPUs/Task": "8",
    "ReqTRES": "cpu=8,mem=32G,node=1,billing=8",
    "AllocTRES": "cpu=8,mem=32G,node=1,billing=8",
    "MinCPUsNode": "8",
    "MinMemoryNode": "32G",
    "TresPerTask": "cpu=8",
}


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_rbm_production_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load RBM production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_request(tmp_path, **overrides):
    values = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "class_id": 0,
        "sample_start": 0,
        "sample_stop": 2,
        "expected_source_commit": COMMIT,
        "repository_root": REPO_ROOT,
        "checkpoint_directory": tmp_path / "checkpoint",
        "expected_checkpoint_state_sha256": STATE_SHA,
    }
    values.update(overrides)
    return entry.RBMProductionSamplingRequest(**values)


def make_record(request, **overrides):
    mapping = entry.authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )
    policy = entry.policy_for_request(request, mapping)

    record = {
        "schema": entry.SCHEMA,
        "paper": entry.PAPER,
        "revision": entry.REVISION,
        "family": entry.FAMILY,
        "dataset_id": request.dataset_id,
        "experimental_seed": request.experimental_seed,
        "attempt": request.attempt,
        "run_identity": request.run_identity,
        "class_id": request.class_id,
        "sample_start": request.sample_start,
        "sample_stop": request.sample_stop,
        "source_commit": request.expected_source_commit,
        "checkpoint_directory": str(request.checkpoint_directory),
        "checkpoint_state_sha256": request.expected_checkpoint_state_sha256,
        "policy_digest": policy.digest(),
        "authorization_id": "p1-rbm-cic-s42-a01",
        "resources": dict(entry.DEFAULT_RESOURCES),
    }
    record.update(overrides)
    return record


def install_authorization(
    request,
    tmp_path,
    monkeypatch,
    **overrides,
):
    root = tmp_path / "authorizations"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(entry, "AUTHORIZATION_ROOT", root)

    path = entry.authorization_path(request)
    path.write_text(
        json.dumps(make_record(request, **overrides), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return path


def fake_source_binding(request):
    return {
        "requested_source_commit": request.expected_source_commit,
        "resolved_source_commit": request.expected_source_commit,
        "repository_root": str(REPO_ROOT),
        "module_paths": {"restricted_boltzmann_production_entry": __file__},
    }


# Execution identity validation


@pytest.mark.parametrize(
    "dataset_id",
    ["unknown", "", "CICMalDroid2020"],
)
def test_unsupported_dataset_rejected(tmp_path, dataset_id):
    with pytest.raises((TypeError, ValueError)):
        make_request(tmp_path, dataset_id=dataset_id)


@pytest.mark.parametrize("seed", [41, 49, 0, -1])
def test_ungoverned_seed_rejected(tmp_path, seed):
    with pytest.raises(ValueError):
        make_request(tmp_path, experimental_seed=seed)


@pytest.mark.parametrize("attempt", [0, -1])
def test_invalid_attempt_rejected(tmp_path, attempt):
    with pytest.raises(ValueError):
        make_request(tmp_path, attempt=attempt)


def test_requested_class_exceeds_governed_universe(tmp_path):
    request = make_request(tmp_path, class_id=5)

    with pytest.raises(ValueError, match="class identity"):
        entry.validate_request_identity(request)


@pytest.mark.parametrize(
    "start,stop",
    [(2, 2), (3, 2), (-1, 2), (0, 2001)],
)
def test_invalid_sample_scope_rejected(tmp_path, start, stop):
    with pytest.raises(ValueError):
        make_request(tmp_path, sample_start=start, sample_stop=stop)


@pytest.mark.parametrize(
    "commit",
    ["abc", "A" * 40, "a" * 39, "a" * 41],
)
def test_invalid_source_commit_rejected(tmp_path, commit):
    with pytest.raises(ValueError):
        make_request(tmp_path, expected_source_commit=commit)


def test_invalid_checkpoint_state_hash_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_request(
            tmp_path,
            expected_checkpoint_state_sha256="not-a-hash",
        )


# Authoritative class mappings


def test_cic_authoritative_mapping():
    mapping = entry.authoritative_class_mapping(
        dataset_id="cicmaldroid2020",
        repository_root=REPO_ROOT,
    )
    assert mapping == ("0", "1", "2", "3", "4")


def test_malex_authoritative_mapping():
    mapping = entry.authoritative_class_mapping(
        dataset_id="malex",
        repository_root=REPO_ROOT,
    )
    assert mapping == ("benign", "malware")


def test_malnet_authoritative_mapping_uses_committed_record():
    mapping = entry.authoritative_class_mapping(
        dataset_id="malnet_image",
        repository_root=REPO_ROOT,
    )
    assert len(mapping) == 47
    assert mapping[0] == "addisplay"
    assert len(set(mapping)) == 47


def test_missing_malnet_mapping_rejected(tmp_path):
    with pytest.raises(ValueError, match="MalNet class mapping"):
        entry.authoritative_class_mapping(
            dataset_id="malnet_image",
            repository_root=tmp_path,
        )


# Authorization and activation


def test_activation_defaults_to_disabled(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    monkeypatch.setattr(entry, "RBM_PRODUCTION_SAMPLING_ENABLED", False)
    monkeypatch.setattr(
        entry,
        "bind_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded"),
    )
    monkeypatch.setattr(
        entry,
        "sample_production_images",
        lambda *args, **kwargs: pytest.fail("sampler invoked"),
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.execute_rbm_production_sampling(
            request,
        )


def test_missing_authorization_denies_before_checkpoint(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    root = tmp_path / "authorizations"
    root.mkdir()
    monkeypatch.setattr(entry, "AUTHORIZATION_ROOT", root)
    monkeypatch.setattr(entry, "RBM_PRODUCTION_SAMPLING_ENABLED", True)
    monkeypatch.setattr(entry, "require_source_binding", fake_source_binding)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(
        entry,
        "capture_slurm_allocation",
        lambda: ALLOCATION,
    )
    monkeypatch.setattr(
        entry,
        "bind_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded"),
    )
    monkeypatch.setattr(
        entry,
        "sample_production_images",
        lambda *args, **kwargs: pytest.fail("sampler invoked"),
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.execute_rbm_production_sampling(
            request,
        )


def test_valid_authorization_accepted(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)

    grant = entry.validate_authorization(
        request,
        allocation=ALLOCATION,
    )

    assert grant.authorization_id == "p1-rbm-cic-s42-a01"
    assert grant.dataset_id == request.dataset_id
    assert grant.policy_digest == entry.policy_for_request(
        request,
        entry.authoritative_class_mapping(
            dataset_id=request.dataset_id,
            repository_root=request.repository_root,
        ),
    ).digest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_id", "malex"),
        ("experimental_seed", 43),
        ("attempt", 2),
        ("run_identity", "commitbbbbbbb_attempt01"),
        ("class_id", 1),
        ("sample_start", 1),
        ("sample_stop", 3),
        ("source_commit", OTHER_COMMIT),
        ("checkpoint_state_sha256", OTHER_STATE_SHA),
        ("policy_digest", "e" * 64),
        ("schema", "wrong"),
    ],
)
def test_mismatched_authorization_field_denies(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch, **{field: value})

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=ALLOCATION,
        )


def test_mismatched_checkpoint_directory_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        checkpoint_directory=str(tmp_path / "other-checkpoint"),
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=ALLOCATION,
        )


def test_duplicate_authorization_field_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    root = tmp_path / "authorizations"
    root.mkdir()
    monkeypatch.setattr(entry, "AUTHORIZATION_ROOT", root)
    path = entry.authorization_path(request)
    path.write_text(
        '{"schema": "duplicate", "schema": "duplicate"}\n',
        encoding="utf-8",
    )
    os.chmod(path, 0o600)

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.load_authorization_record(request)


def test_group_writable_authorization_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    path = install_authorization(request, tmp_path, monkeypatch)
    os.chmod(path, 0o660)

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.load_authorization_record(request)


# Resource envelope and Slurm allocation


def test_allocation_cpu_mismatch_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)
    allocation = dict(ALLOCATION, NumCPUs="16")

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_allocation_partition_mismatch_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)
    allocation = dict(ALLOCATION, Partition="other")

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_allocation_requeue_denies(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)
    allocation = dict(ALLOCATION, Requeue="1")

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_core_hour_ceiling_enforced(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["cpus_per_task"] = 64
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )

    with pytest.raises(entry.RBMProductionAuthorizationError, match="650"):
        entry.validate_authorization(
            request,
            allocation=ALLOCATION,
        )


def test_job_core_hours_include_authorized_task_count(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["tasks"] = 2
    resources["cpus_per_task"] = 4
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        ALLOCATION,
        NumTasks="2",
        NumCPUs="8",
        TRES="cpu=8,mem=32G,node=1",
    )

    grant = entry.validate_authorization(
        request,
        allocation=allocation,
    )

    assert grant.resources["tasks"] == 2
    assert grant.resources["cpus_per_task"] == 4
    assert entry._individual_job_cpu_core_hours(grant.resources) == 96.0
    assert entry._authorized_total_cpus(grant.resources) == 8


def test_total_cpu_allocation_must_match_authorized_tasks(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["tasks"] = 2
    resources["cpus_per_task"] = 4
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        ALLOCATION,
        NumTasks="2",
        NumCPUs="4",
        TRES="cpu=4,mem=32G,node=1",
    )

    with pytest.raises(entry.RBMProductionAuthorizationError, match="total CPU"):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_core_hour_ceiling_accounts_for_authorized_tasks(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["tasks"] = 2
    resources["cpus_per_task"] = 32
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        ALLOCATION,
        NumTasks="2",
        NumCPUs="64",
        TRES="cpu=64,mem=32G,node=1",
    )

    with pytest.raises(entry.RBMProductionAuthorizationError, match="650"):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_gpu_envelope_rejected(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["gpu_enabled"] = True
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=ALLOCATION,
        )


def test_talon_alloctres_allocation_accepted(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )

    grant = entry.validate_authorization(
        request,
        allocation=TALON_ALLOCATION,
    )

    assert grant.resources["walltime_seconds"] == 600
    assert grant.resources["cpus_per_task"] == 8
    assert grant.resources["memory_gb"] == 32


def test_allocated_memory_mismatch_rejected(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        TALON_ALLOCATION,
        AllocTRES="cpu=8,mem=16G,node=1,billing=8",
    )

    with pytest.raises(entry.RBMProductionAuthorizationError, match="memory"):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_requested_memory_does_not_conceal_allocated_mismatch(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        TALON_ALLOCATION,
        ReqTRES="cpu=8,mem=32G,node=1,billing=8",
        AllocTRES="cpu=8,mem=16G,node=1,billing=8",
    )

    with pytest.raises(entry.RBMProductionAuthorizationError, match="memory"):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_requested_tres_alone_cannot_authorize(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = {
        key: value
        for key, value in TALON_ALLOCATION.items()
        if key != "AllocTRES"
    }

    with pytest.raises(
        entry.RBMProductionAuthorizationError,
        match="allocated-resource evidence",
    ):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


@pytest.mark.parametrize(
    "allocated",
    [
        None,
        "",
        "cpu=8,node=1",
        "mem=32G,cpu",
        "mem=32G,cpu=8,cpu=8",
    ],
)
def test_missing_or_malformed_allocated_evidence_rejected(
    tmp_path,
    monkeypatch,
    allocated,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(TALON_ALLOCATION)

    if allocated is None:
        allocation.pop("AllocTRES")
    else:
        allocation["AllocTRES"] = allocated

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


@pytest.mark.parametrize(
    "legacy",
    [
        "cpu=8,mem=16G,node=1",
        "cpu=4,mem=32G,node=1",
        "cpu=8,mem=32G,node",
    ],
)
def test_contradictory_allocated_representations_rejected(
    tmp_path,
    monkeypatch,
    legacy,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(TALON_ALLOCATION, TRES=legacy)

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


@pytest.mark.parametrize(
    "allocated,legacy",
    [
        ("mem=32G", "cpu=4,mem=32G,node=1"),
        ("cpu=8,mem=32G", "cpu=8,mem=32G,node=2"),
    ],
)
def test_legacy_supplied_value_contradicts_envelope_when_alloctres_omits_field(
    tmp_path,
    monkeypatch,
    allocated,
    legacy,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        TALON_ALLOCATION,
        AllocTRES=allocated,
        TRES=legacy,
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_compatible_dual_allocated_representations_accepted(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(
        TALON_ALLOCATION,
        TRES="cpu=8,mem=32G,node=1",
    )

    grant = entry.validate_authorization(
        request,
        allocation=allocation,
    )

    assert grant.resources["memory_gb"] == 32
    assert grant.resources["walltime_seconds"] == 600


@pytest.mark.parametrize(
    "allocation_update",
    [
        {
            "AllocTRES": "cpu=8,mem=32G,node=1,billing=8,gres/gpu=1",
        },
        {
            "TRES": "cpu=8,mem=32G,node=1,gres/gpu=1",
        },
        {"Gres": "gpu:1"},
    ],
)
def test_gpu_allocation_rejected(
    tmp_path,
    monkeypatch,
    allocation_update,
):
    request = make_request(tmp_path)
    resources = dict(entry.DEFAULT_RESOURCES)
    resources["walltime_seconds"] = 600
    install_authorization(
        request,
        tmp_path,
        monkeypatch,
        resources=resources,
    )
    allocation = dict(TALON_ALLOCATION, **allocation_update)

    with pytest.raises(entry.RBMProductionAuthorizationError, match="GPU"):
        entry.validate_authorization(
            request,
            allocation=allocation,
        )


def test_legacy_tres_allocation_supported(tmp_path, monkeypatch):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)

    grant = entry.validate_authorization(
        request,
        allocation=ALLOCATION,
    )

    assert grant.resources["memory_gb"] == 32
    assert grant.resources["walltime_seconds"] == 12 * 60 * 60


def test_checkpoint_state_mismatch_denied_before_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_state.npz").write_bytes(b"not the governed state")
    (checkpoint / "model_metadata.json").write_bytes(b"{}\n")
    request = make_request(tmp_path)

    monkeypatch.setattr(
        entry,
        "load_rbm_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint loaded"),
    )

    with pytest.raises(entry.RBMProductionSamplingError, match="state file"):
        entry.bind_checkpoint(
            request,
            class_mapping=("0", "1", "2", "3", "4"),
        )


# Source and module binding


def test_source_binding_matches_expected_commit(monkeypatch):
    monkeypatch.setattr(
        entry,
        "capture_git_state",
        lambda root: SimpleNamespace(
            dirty=False,
            commit=COMMIT,
            repo_root=REPO_ROOT,
        ),
    )
    request = make_request(Path("/tmp"))

    binding = entry.require_source_binding(request)

    assert binding["requested_source_commit"] == COMMIT
    assert binding["resolved_source_commit"] == COMMIT


def test_source_binding_rejects_wrong_commit(monkeypatch):
    monkeypatch.setattr(
        entry,
        "capture_git_state",
        lambda root: SimpleNamespace(
            dirty=False,
            commit=OTHER_COMMIT,
            repo_root=REPO_ROOT,
        ),
    )

    with pytest.raises(entry.RBMProductionSamplingError, match="source"):
        entry.require_source_binding(make_request(Path("/tmp")))


def test_source_binding_rejects_dirty_tree(monkeypatch):
    monkeypatch.setattr(
        entry,
        "capture_git_state",
        lambda root: SimpleNamespace(
            dirty=True,
            commit=COMMIT,
            repo_root=REPO_ROOT,
        ),
    )

    with pytest.raises(entry.RBMProductionSamplingError, match="clean"):
        entry.require_source_binding(make_request(Path("/tmp")))


def test_module_binding_rejects_foreign_module():
    fake = SimpleNamespace(__file__="/tmp/foreign_rbm_module.py")

    with pytest.raises(entry.RBMProductionSamplingError, match="module path"):
        entry.require_module_binding(
            REPO_ROOT,
            modules={"foreign": fake},
        )


def test_module_binding_accepts_revision_modules():
    paths = entry.require_module_binding(REPO_ROOT)

    assert "restricted_boltzmann_production_entry" in paths
    assert "restricted_boltzmann_production_sampling" in paths


# Delegation and policy preservation


def test_execution_delegates_to_accepted_sampler(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)
    monkeypatch.setattr(entry, "RBM_PRODUCTION_SAMPLING_ENABLED", True)
    monkeypatch.setattr(entry, "require_source_binding", fake_source_binding)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(
        entry,
        "capture_slurm_allocation",
        lambda: ALLOCATION,
    )

    mapping = entry.authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )
    policy = entry.policy_for_request(request, mapping)
    checkpoint = object()
    captured = {}

    monkeypatch.setattr(
        entry,
        "bind_checkpoint",
        lambda request, *, class_mapping: (
            checkpoint,
            policy,
            "f" * 64,
        ),
    )

    def fake_sample(*, checkpoint, policy, class_id, sample_indices):
        captured["checkpoint"] = checkpoint
        captured["policy"] = policy
        captured["class_id"] = class_id
        captured["sample_indices"] = tuple(sample_indices)

        provenance = sampler.RBMProductionSamplingProvenance(
            dataset_id=policy.dataset_id,
            experimental_seed=policy.experimental_seed,
            class_id=class_id,
            class_label=policy.class_mapping[class_id],
            sample_indices=tuple(sample_indices),
            sample_count=len(sample_indices),
            transition_count=policy.transition_count,
            burn_in=policy.burn_in,
            initialization=policy.initialization,
            class_conditioning=policy.class_conditioning,
            chain_semantics=policy.chain_semantics,
            internal_dtype=policy.internal_dtype,
            canonical_shape=policy.canonical_shape,
            canonical_dtype=policy.canonical_dtype,
            checkpoint_state_sha256=policy.checkpoint_state_sha256,
            policy_digest=policy.digest(),
        )
        images = np.zeros(
            (len(sample_indices), *policy.canonical_shape),
            dtype=np.float32,
        )
        return sampler.RBMProductionSamplingResult(
            images=images,
            provenance=provenance,
        )

    monkeypatch.setattr(entry, "sample_production_images", fake_sample)

    execution = entry.execute_rbm_production_sampling(request)
    record = execution.to_mapping()

    assert captured["checkpoint"] is checkpoint
    assert captured["policy"] is policy
    assert captured["class_id"] == 0
    assert captured["sample_indices"] == (0, 1)
    assert captured["policy"].transition_count == 10
    assert captured["policy"].burn_in == 0
    assert captured["policy"].primary_samples_per_class == 2000
    assert captured["policy"].thinning == "none"
    assert captured["policy"].continuation_chains is False
    assert captured["policy"].adaptive_stopping is False

    assert record["status"] == "SAMPLING_COMPLETE"
    assert record["transition_count"] == 10
    assert record["burn_in"] == 0
    assert record["sample_count"] == 2
    assert record["policy_digest"] == policy.digest()
    assert record["individual_job_cpu_core_hours"] == 96.0
    assert record["cumulative_rbm_planning_core_hours"] == 650
    assert record["cumulative_workload_enforcement"] == "NOT_PERFORMED"
    assert record["measured_resource_consumption"] == "NOT_MEASURED"
    assert record["pool_publication"] == "NOT_PERFORMED"
    assert record["dataset_source_provenance"] == (
        "NOT_ESTABLISHED_BY_RBM_CHECKPOINT_METADATA"
    )
    assert record["requested_source_commit"] == COMMIT
    assert record["resolved_source_commit"] == COMMIT


def test_returned_provenance_mismatch_rejected(
    tmp_path,
    monkeypatch,
):
    request = make_request(tmp_path)
    install_authorization(request, tmp_path, monkeypatch)
    monkeypatch.setattr(entry, "RBM_PRODUCTION_SAMPLING_ENABLED", True)
    monkeypatch.setattr(entry, "require_source_binding", fake_source_binding)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(
        entry,
        "capture_slurm_allocation",
        lambda: ALLOCATION,
    )

    mapping = entry.authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )
    policy = entry.policy_for_request(request, mapping)

    monkeypatch.setattr(
        entry,
        "bind_checkpoint",
        lambda request, *, class_mapping: (object(), policy, "f" * 64),
    )

    def wrong_provenance(**kwargs):
        provenance = sampler.RBMProductionSamplingProvenance(
            dataset_id=policy.dataset_id,
            experimental_seed=policy.experimental_seed,
            class_id=request.class_id,
            class_label=policy.class_mapping[request.class_id],
            sample_indices=request.sample_indices,
            sample_count=len(request.sample_indices),
            transition_count=1,
            burn_in=0,
            initialization=policy.initialization,
            class_conditioning=policy.class_conditioning,
            chain_semantics=policy.chain_semantics,
            internal_dtype=policy.internal_dtype,
            canonical_shape=policy.canonical_shape,
            canonical_dtype=policy.canonical_dtype,
            checkpoint_state_sha256=policy.checkpoint_state_sha256,
            policy_digest=policy.digest(),
        )
        return sampler.RBMProductionSamplingResult(
            images=np.zeros((2, 40, 40, 3), dtype=np.float32),
            provenance=provenance,
        )

    monkeypatch.setattr(entry, "sample_production_images", wrong_provenance)

    with pytest.raises(entry.RBMProductionSamplingError, match="provenance"):
        entry.execute_rbm_production_sampling(
            request,
        )


# CLI and launcher static contract


def test_cli_parser_builds_explicit_identity():
    cli = load_cli()
    args = cli.make_parser().parse_args(
        [
            "--dataset-id",
            "cicmaldroid2020",
            "--experimental-seed",
            "42",
            "--attempt",
            "1",
            "--class-id",
            "0",
            "--sample-start",
            "0",
            "--sample-stop",
            "2000",
            "--expected-source-commit",
            COMMIT,
            "--repository-root",
            str(REPO_ROOT),
            "--checkpoint-dir",
            "/fixture/checkpoint",
            "--expected-checkpoint-state-sha256",
            STATE_SHA,
        ]
    )

    assert args.dataset_id == "cicmaldroid2020"
    assert args.sample_stop == 2000
    assert args.expected_checkpoint_state_sha256 == STATE_SHA


def test_activation_boundary_remains_disabled_in_source():
    source = Path(entry.__file__).read_text(encoding="utf-8")

    assert "RBM_PRODUCTION_SAMPLING_ENABLED = False" in source
    assert "os.environ.get(\"SLURM_JOB_ID\"" in source
    assert "automatic_retry=disabled" in Path(
        LAUNCHER_PATH
    ).read_text(encoding="utf-8").lower()


def test_launcher_is_inert_and_retry_free():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --requeue" not in source
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "RBM_EXPECTED_SOURCE_COMMIT" in source
    assert "run_paper01_rbm_production_sampling.py" in source
    assert "RESOURCE_ENVELOPE=PROVISIONAL" in source
    assert "POOL_PUBLICATION=DISABLED" in source
    assert "AUTOMATIC_RETRY=DISABLED" in source


def test_launcher_does_not_inspect_checkpoint_artifacts():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "model_state.npz" not in source
    assert "model_metadata.json" not in source
    assert '[[ -f "$RBM_CHECKPOINT_DIR' not in source
    assert '[[ -d "$RBM_CHECKPOINT_DIR' not in source
    assert "--checkpoint-dir \"$RBM_CHECKPOINT_DIR\"" in source
    assert '^[0-9a-f]{64}$' in source
