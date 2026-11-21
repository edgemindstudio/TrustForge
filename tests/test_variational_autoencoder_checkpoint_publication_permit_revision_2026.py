"""Focused tests for the VAE checkpoint-publication permit validator.

Synthetic fixtures only. No real dataset, production checkpoint, Slurm job,
synthesis, evaluation or publication is used.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_permit as permit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "train_paper01_vae_revision_production.py"
)
SOURCE_COMMIT = "a" * 40
APPROVAL_COMMIT = "b" * 40
AUTHORIZATION_ID = (
    "P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION"
)
RUN_IDENTITY = "commit" + SOURCE_COMMIT[:7] + "_attempt02"


def _record(artifact_root):
    publication_root = (
        Path(artifact_root)
        / "vae"
        / "cicmaldroid2020"
        / "seed42"
        / "checkpoints"
        / "production"
        / RUN_IDENTITY
    )

    return {
        "schema": permit.SCHEMA,
        "paper": permit.PAPER,
        "revision": permit.REVISION,
        "family": permit.FAMILY,
        "dataset_id": permit.DATASET_ID,
        "experimental_seed": permit.EXPERIMENTAL_SEED,
        "attempt": permit.ATTEMPT,
        "run_identity": RUN_IDENTITY,
        "execution_source_commit": SOURCE_COMMIT,
        "publication_root": str(publication_root),
        "class_mapping": list(permit.CLASS_MAPPING),
        "checkpoint_roles": list(permit.CHECKPOINT_ROLES),
        "observation_version": permit.OBSERVATION_VERSION,
        "checkpoint_contract": {
            "schema_type": permit.CHECKPOINT_SCHEMA_TYPE,
            "schema_version": permit.CHECKPOINT_SCHEMA_VERSION,
            "architecture_version": permit.ARCHITECTURE_VERSION,
            "selection_policy_id": permit.SELECTION_POLICY_ID,
        },
        "authorization_id": AUTHORIZATION_ID,
        "approval_record": permit.APPROVAL_RECORD,
        "approval_commit": APPROVAL_COMMIT,
        "scope": {
            "checkpoint_publication": True,
            "committed_generation_verification": True,
            "synthesis": False,
            "evaluation": False,
        },
    }


def _write_record(path, record, *, mode=0o600, raw=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if raw is None:
        raw = canonical_json_bytes(record) + b"\n"

    path.write_bytes(raw)
    path.chmod(mode)
    return path


def _approval_entry(
    record,
    *,
    authorization_id=None,
    scope_sha256=None,
):
    return {
        "authorization_id":
            authorization_id or record["authorization_id"],
        "authorization_scope_sha256":
            scope_sha256
            or permit.vae_authorization_scope_sha256(record),
    }


def _approval_entry_line(record, **overrides):
    entry = _approval_entry(record, **overrides)
    return (
        permit.APPROVAL_ENTRY_PREFIX
        + canonical_json_bytes(entry).decode("utf-8")
        + permit.APPROVAL_ENTRY_SUFFIX
    )


def _approval_text(record, *, entry_lines=None, extra=()):
    lines = ["# Synthetic VAE approval ledger", ""]

    if entry_lines is None:
        lines.append(_approval_entry_line(record))
    else:
        lines.extend(entry_lines)

    lines.extend(extra)

    return "\n".join(lines) + "\n"


@pytest.fixture
def permit_context(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    permit_path = (
        artifact_root
        / "vae"
        / "authorizations"
        / "vae_checkpoint_publication_permit.json"
    )
    permit_path.parent.mkdir(parents=True)

    record = _record(artifact_root)
    approval_text = _approval_text(record)

    monkeypatch.setattr(permit, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(permit, "PERMIT_PATH", permit_path)
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: approval_text,
    )

    _write_record(permit_path, record)

    return SimpleNamespace(
        artifact_root=artifact_root,
        permit_path=permit_path,
        record=record,
        approval_text=approval_text,
    )


def _require(context, **overrides):
    values = {
        "dataset_id": permit.DATASET_ID,
        "experimental_seed": permit.EXPERIMENTAL_SEED,
        "attempt": permit.ATTEMPT,
        "execution_source_commit": SOURCE_COMMIT,
        "publication_root": context.record["publication_root"],
        "class_mapping": permit.CLASS_MAPPING,
        "repository_root": REPO_ROOT,
    }
    values.update(overrides)
    return permit.require_vae_checkpoint_publication_permit(**values)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_vae_permit_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load VAE production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cli_args(publication_root=None):
    values = [
        "--dataset-id",
        permit.DATASET_ID,
        "--experimental-seed",
        str(permit.EXPERIMENTAL_SEED),
        "--attempt",
        str(permit.ATTEMPT),
        "--expected-source-commit",
        SOURCE_COMMIT,
        "--repository-root",
        str(REPO_ROOT),
        "--cic-source-csv",
        "source.csv",
        "--cic-feature-columns",
        "features.json",
    ]

    if publication_root is not None:
        values.extend(
            [
                "--publish-checkpoints",
                "--publication-root",
                str(publication_root),
            ]
        )

    return values


def test_valid_permit_loads_and_returns_canonical_identity(
    permit_context,
):
    loaded = permit.load_vae_checkpoint_publication_permit()
    assert loaded == permit_context.record

    result = _require(permit_context)

    expected_digest = hashlib.sha256(
        canonical_json_bytes(permit_context.record)
    ).hexdigest()

    assert isinstance(result, permit.VAECheckpointPublicationPermit)
    assert result.permit_sha256 == expected_digest
    assert result.authorization_scope_sha256 == (
        permit.vae_authorization_scope_sha256(
            permit_context.record
        )
    )
    assert result.authorization_id == AUTHORIZATION_ID
    assert result.run_identity == RUN_IDENTITY
    assert result.execution_source_commit == SOURCE_COMMIT
    assert result.publication_root == Path(
        permit_context.record["publication_root"]
    )
    assert result.class_mapping == permit.CLASS_MAPPING
    assert result.checkpoint_roles == permit.CHECKPOINT_ROLES
    assert result.observation_version == permit.OBSERVATION_VERSION
    assert result.scope["synthesis"] is False
    assert result.scope["evaluation"] is False


def test_missing_permit_is_rejected(permit_context):
    permit_context.permit_path.unlink()

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="absent",
    ):
        _require(permit_context)


@pytest.mark.parametrize(
    "raw",
    [
        b"{not-json}\n",
        b'{"schema": "one", "schema": "two"}\n',
        b'{"schema": NaN}\n',
        b"   " * (permit.MAX_PERMIT_BYTES // 3 + 2),
    ],
)
def test_malformed_permit_is_rejected(permit_context, raw):
    _write_record(permit_context.permit_path, permit_context.record, raw=raw)

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        _require(permit_context)


def test_noncanonical_permit_is_rejected(permit_context):
    raw = json.dumps(
        permit_context.record,
        indent=2,
    ).encode("utf-8") + b"\n"

    _write_record(permit_context.permit_path, permit_context.record, raw=raw)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="canonically encoded",
    ):
        _require(permit_context)


def test_unknown_and_missing_fields_are_rejected(permit_context):
    record = dict(permit_context.record)
    record["extra"] = 1
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="field inventory",
    ):
        _require(permit_context)

    record = dict(permit_context.record)
    del record["authorization_id"]
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="field inventory",
    ):
        _require(permit_context)


@pytest.mark.parametrize(
    "case",
    [
        "dataset",
        "seed",
        "attempt",
        "source",
        "root",
        "class_mapping",
        "role",
        "observation",
        "contract",
        "requested_class_mapping",
    ],
)
def test_identity_mismatch_is_rejected(permit_context, case):
    record = dict(permit_context.record)
    overrides = {}

    if case == "dataset":
        record["dataset_id"] = "ustc_tfc2016"
    elif case == "seed":
        record["experimental_seed"] = 43
    elif case == "attempt":
        record["attempt"] = 1
    elif case == "source":
        record["execution_source_commit"] = "c" * 40
    elif case == "root":
        record["publication_root"] = str(
            permit_context.artifact_root / "wrong-root"
        )
    elif case == "class_mapping":
        record["class_mapping"] = list(permit.CLASS_MAPPING[:-1])
    elif case == "role":
        record["checkpoint_roles"] = ["best"]
    elif case == "observation":
        record["observation_version"] = (
            "gaussian_fixed_variance_1p0_v2"
        )
    elif case == "contract":
        contract = dict(record["checkpoint_contract"])
        contract["selection_policy_id"] = "different_policy"
        record["checkpoint_contract"] = contract
    elif case == "requested_class_mapping":
        overrides["class_mapping"] = permit.CLASS_MAPPING[:-1]

    _write_record(permit_context.permit_path, record)

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        _require(permit_context, **overrides)


@pytest.mark.parametrize(
    "case",
    [
        "seed_float",
        "attempt_bool",
        "source_short",
        "mapping_string",
    ],
)
def test_requested_identity_types_are_strict(permit_context, case):
    overrides = {}

    if case == "seed_float":
        overrides["experimental_seed"] = 42.0
    elif case == "attempt_bool":
        overrides["attempt"] = True
    elif case == "source_short":
        overrides["execution_source_commit"] = "a" * 39
    elif case == "mapping_string":
        overrides["class_mapping"] = "01234"

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        _require(permit_context, **overrides)


def test_wrong_approval_record_and_commit_are_rejected(permit_context):
    record = dict(permit_context.record)
    record["approval_record"] = "other.md"
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval record",
    ):
        _require(permit_context)

    record = dict(permit_context.record)
    record["approval_commit"] = "not-a-commit"
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval_commit",
    ):
        _require(permit_context)


def test_wrong_approval_scope_digest_is_rejected(
    permit_context,
    monkeypatch,
):
    line = _approval_entry_line(
        permit_context.record,
        scope_sha256="0" * 64,
    )
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=[line],
        ),
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="scope digest mismatch",
    ):
        _require(permit_context)


def test_approval_entry_for_other_authorization_is_rejected(
    permit_context,
    monkeypatch,
):
    line = _approval_entry_line(
        permit_context.record,
        authorization_id="P1R2026-VAE-OTHER-AUTHORIZATION",
    )
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=[line],
        ),
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval entry is missing",
    ):
        _require(permit_context)


def test_digest_only_in_unrelated_ledger_text_is_insufficient(
    permit_context,
    monkeypatch,
):
    digest = permit.vae_authorization_scope_sha256(
        permit_context.record
    )
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=[],
            extra=[
                "Unrelated ledger prose mentions "
                + digest
                + " without an approval entry."
            ],
        ),
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval entry is missing",
    ):
        _require(permit_context)


@pytest.mark.parametrize(
    "case, expected",
    [
        ("duplicated", "duplicated or ambiguous"),
        ("ambiguous", "duplicated or ambiguous"),
        ("malformed", "malformed"),
    ],
)
def test_bad_approval_entry_sets_are_rejected(
    permit_context,
    monkeypatch,
    case,
    expected,
):
    correct = _approval_entry_line(permit_context.record)
    wrong = _approval_entry_line(
        permit_context.record,
        scope_sha256="0" * 64,
    )

    if case == "duplicated":
        lines = [correct, correct]
    elif case == "ambiguous":
        lines = [correct, wrong]
    else:
        lines = [
            permit.APPROVAL_ENTRY_PREFIX
            + canonical_json_bytes(
                _approval_entry(permit_context.record)
            ).decode("utf-8")
        ]

    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=lines,
        ),
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match=expected,
    ):
        _require(permit_context)


def test_destination_change_requires_new_scope_digest(
    permit_context,
    monkeypatch,
    tmp_path,
):
    record_a = permit_context.record
    digest_a = permit.vae_authorization_scope_sha256(record_a)

    moved_root = tmp_path / "moved-artifacts"
    monkeypatch.setattr(permit, "ARTIFACT_ROOT", moved_root)
    record_b = _record(moved_root)
    digest_b = permit.vae_authorization_scope_sha256(record_b)

    assert digest_a != digest_b
    _write_record(permit_context.permit_path, record_b)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="scope digest mismatch",
    ):
        _require(
            permit_context,
            publication_root=record_b["publication_root"],
        )

    approved_b = _approval_text(record_b)
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: approved_b,
    )

    result = _require(
        permit_context,
        publication_root=record_b["publication_root"],
    )
    assert result.authorization_scope_sha256 == digest_b
    assert result.publication_root == Path(
        record_b["publication_root"]
    )


def test_scope_digest_binds_complete_execution_identity(
    permit_context,
):
    base = permit_context.record
    base_digest = permit.vae_authorization_scope_sha256(base)
    mutations = {}

    for name, value in {
        "schema": "different-schema",
        "paper": "paper02",
        "revision": "revision_2027",
        "family": "gan",
        "dataset_id": "ustc_tfc2016",
        "experimental_seed": 43,
        "attempt": 1,
        "run_identity": "commitbbbbbbb_attempt02",
        "execution_source_commit": "c" * 40,
        "publication_root": str(Path(base["publication_root"]).parent / "other"),
        "class_mapping": list(permit.CLASS_MAPPING[:-1]),
        "checkpoint_roles": ["best"],
        "observation_version": "different-observation",
        "authorization_id": "P1R2026-VAE-CHANGED",
        "approval_record": "other.md",
    }.items():
        record = copy.deepcopy(base)
        record[name] = value
        mutations[name] = record

    for name in (
        "schema_type",
        "schema_version",
        "architecture_version",
        "selection_policy_id",
    ):
        record = copy.deepcopy(base)
        record["checkpoint_contract"][name] = "different"
        mutations["checkpoint_contract." + name] = record

    for name in (
        "checkpoint_publication",
        "committed_generation_verification",
        "synthesis",
        "evaluation",
    ):
        record = copy.deepcopy(base)
        record["scope"][name] = not record["scope"][name]
        mutations["scope." + name] = record

    for name, record in mutations.items():
        assert (
            permit.vae_authorization_scope_sha256(record)
            != base_digest
        ), name


def test_approval_commit_is_outside_scope_but_required(
    permit_context,
    monkeypatch,
):
    record_a = permit_context.record
    record_b = copy.deepcopy(record_a)
    record_b["approval_commit"] = "c" * 40

    assert (
        permit.vae_authorization_scope_sha256(record_a)
        == permit.vae_authorization_scope_sha256(record_b)
    )
    assert (
        hashlib.sha256(canonical_json_bytes(record_a)).hexdigest()
        != hashlib.sha256(canonical_json_bytes(record_b)).hexdigest()
    )

    _write_record(permit_context.permit_path, record_b)

    def read_approval(**kwargs):
        if kwargs["approval_commit"] != APPROVAL_COMMIT:
            raise permit.VAECheckpointPublicationPermitError(
                "permit approval record is unavailable"
            )
        return _approval_text(record_a)

    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        read_approval,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="unavailable",
    ):
        _require(permit_context)


def test_approval_commit_change_is_rejected_on_revalidation(
    permit_context,
):
    validated = _require(permit_context)

    record = copy.deepcopy(permit_context.record)
    record["approval_commit"] = "c" * 40
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="changed after validation",
    ):
        permit.verify_vae_publication_permit_unchanged(validated)


def test_missing_approval_entry_is_rejected(
    permit_context,
    monkeypatch,
):
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=[],
        ),
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval entry is missing",
    ):
        _require(permit_context)


def test_unavailable_approval_record_is_rejected(
    permit_context,
    monkeypatch,
):
    def unavailable(**kwargs):
        raise permit.VAECheckpointPublicationPermitError(
            "permit approval record is unavailable"
        )

    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        unavailable,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="unavailable",
    ):
        _require(permit_context)


def test_approval_record_is_read_from_git_history(
    tmp_path,
    monkeypatch,
):
    calls = []
    record = _record(tmp_path / "artifacts")
    approval_text = _approval_text(record)

    def available(command, **kwargs):
        calls.append(command)

        if "cat-file" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=b"",
                stderr=b"",
            )

        return SimpleNamespace(
            returncode=0,
            stdout=approval_text.encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(permit.subprocess, "run", available)

    text = permit._read_approval_record(
        repository_root=tmp_path,
        approval_commit=APPROVAL_COMMIT,
        approval_record=permit.APPROVAL_RECORD,
    )

    assert AUTHORIZATION_ID in text
    assert permit.vae_authorization_scope_sha256(record) in text
    assert calls[0][0] == "git"
    assert calls[0][1:3] == ["-C", str(tmp_path)]
    assert "cat-file" in calls[0]
    assert "show" in calls[1]

    def unavailable(command, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(permit.subprocess, "run", unavailable)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="unavailable",
    ):
        permit._read_approval_record(
            repository_root=tmp_path,
            approval_commit="0" * 40,
            approval_record=permit.APPROVAL_RECORD,
        )


def test_symlinked_permit_is_rejected(permit_context, tmp_path):
    target = tmp_path / "target-permit.json"
    _write_record(target, permit_context.record)
    permit_context.permit_path.unlink()
    permit_context.permit_path.symlink_to(target)

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        _require(permit_context)


def test_group_writable_permit_is_rejected(permit_context):
    _write_record(
        permit_context.permit_path,
        permit_context.record,
        mode=0o664,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="group/world writable",
    ):
        _require(permit_context)


def test_wrong_owner_is_rejected(permit_context, monkeypatch):
    real_uid = os.geteuid()
    monkeypatch.setattr(
        permit.os,
        "geteuid",
        lambda: real_uid + 1,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="different owner",
    ):
        _require(permit_context)


def test_changed_permit_is_rejected_on_revalidation(permit_context):
    validated = _require(permit_context)

    record = dict(permit_context.record)
    record["authorization_id"] = (
        "P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION-CHANGED"
    )
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="changed after validation",
    ):
        permit.verify_vae_publication_permit_unchanged(validated)


def test_removed_permit_is_rejected_on_revalidation(permit_context):
    validated = _require(permit_context)
    permit_context.permit_path.unlink()

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        permit.verify_vae_publication_permit_unchanged(validated)


def test_existing_destination_is_rejected(permit_context):
    Path(permit_context.record["publication_root"]).mkdir(
        parents=True
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="already exists",
    ):
        _require(permit_context)


def test_symlinked_destination_is_rejected(permit_context, tmp_path):
    target = tmp_path / "destination-target"
    target.mkdir()
    requested = Path(permit_context.record["publication_root"])
    requested.parent.mkdir(parents=True)
    requested.symlink_to(target)

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        _require(permit_context)


def test_repository_destination_is_rejected(
    permit_context,
    monkeypatch,
):
    monkeypatch.setattr(
        permit,
        "ARTIFACT_ROOT",
        REPO_ROOT / "vae-permit-test-artifacts",
    )
    record = _record(permit.ARTIFACT_ROOT)
    approval_text = _approval_text(record)
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: approval_text,
    )
    _write_record(permit_context.permit_path, record)

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="outside the source checkout",
    ):
        _require(
            permit_context,
            publication_root=record["publication_root"],
        )


def test_training_only_cli_does_not_require_permit(
    permit_context,
    monkeypatch,
):
    cli = _load_cli()
    args = cli.make_parser().parse_args(_cli_args())
    calls = []

    def unexpected_permit_load():
        calls.append("permit")
        raise AssertionError("permit must not load")

    monkeypatch.setattr(
        permit,
        "load_vae_checkpoint_publication_permit",
        unexpected_permit_load,
    )

    def stop_at_environment(args):
        raise RuntimeError("environment reached")

    monkeypatch.setattr(
        cli,
        "require_environment",
        stop_at_environment,
    )

    with pytest.raises(RuntimeError, match="environment reached"):
        cli.run(args)

    assert calls == []


def test_publication_activation_with_valid_test_permit_reaches_environment(
    permit_context,
    monkeypatch,
):
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )

    assert cli.VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED is True
    assert permit.load_vae_checkpoint_publication_permit() == (
        permit_context.record
    )

    def stop_at_environment(args):
        raise RuntimeError("environment reached")

    monkeypatch.setattr(
        cli,
        "require_environment",
        stop_at_environment,
    )

    with pytest.raises(RuntimeError, match="environment reached"):
        cli.run(args)

    assert not Path(
        permit_context.record["publication_root"]
    ).exists()


def test_publication_activation_without_permit_denied_before_environment(
    permit_context,
    monkeypatch,
):
    permit_context.permit_path.unlink()
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )

    def unexpected_environment(args):
        raise AssertionError("environment must not run")

    monkeypatch.setattr(
        cli,
        "require_environment",
        unexpected_environment,
    )

    with pytest.raises(
        PermissionError,
        match="permit is absent",
    ):
        cli.run(args)

    assert not Path(
        permit_context.record["publication_root"]
    ).exists()


def test_publication_activation_invalid_approval_denied_before_environment(
    permit_context,
    monkeypatch,
):
    monkeypatch.setattr(
        permit,
        "_read_approval_record",
        lambda **kwargs: _approval_text(
            permit_context.record,
            entry_lines=[],
        ),
    )
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )

    def unexpected_environment(args):
        raise AssertionError("environment must not run")

    monkeypatch.setattr(
        cli,
        "require_environment",
        unexpected_environment,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="approval entry is missing",
    ):
        cli.run(args)


def test_publication_activation_wrong_source_denied_before_environment(
    permit_context,
    monkeypatch,
):
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )
    args.expected_source_commit = "c" * 40

    def unexpected_environment(args):
        raise AssertionError("environment must not run")

    monkeypatch.setattr(
        cli,
        "require_environment",
        unexpected_environment,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="permit source does not match",
    ):
        cli.run(args)


def test_publication_activation_destination_conflict_denied_before_environment(
    permit_context,
    monkeypatch,
):
    Path(permit_context.record["publication_root"]).mkdir(
        parents=True
    )
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )

    def unexpected_environment(args):
        raise AssertionError("environment must not run")

    monkeypatch.setattr(
        cli,
        "require_environment",
        unexpected_environment,
    )

    with pytest.raises(
        permit.VAECheckpointPublicationPermitError,
        match="already exists",
    ):
        cli.run(args)


def test_publication_activation_governed_training_gate_enforced(
    permit_context,
    monkeypatch,
):
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )

    class Governed:
        @staticmethod
        def require_vae_production_authorization():
            raise PermissionError("training not authorized")

        @staticmethod
        def run_governed_vae_training(**kwargs):
            raise AssertionError("training must not run")

    monkeypatch.setattr(
        cli,
        "require_environment",
        lambda args: SimpleNamespace(
            actual_source_commit=SOURCE_COMMIT,
            repository_root=REPO_ROOT,
            module_paths={},
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_governed_module",
        lambda: Governed,
    )
    monkeypatch.setattr(
        cli,
        "build_adapter",
        lambda args: (_ for _ in ()).throw(
            AssertionError("adapter must not run")
        ),
    )

    with pytest.raises(
        PermissionError,
        match="training not authorized",
    ):
        cli.run(args)


def test_publication_activation_revalidates_before_publish(
    permit_context,
    monkeypatch,
):
    pytest.importorskip("tensorflow")

    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_checkpoint_publication_coordinator as coordinator,
    )

    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )
    calls = []

    class Governed:
        @staticmethod
        def require_vae_production_authorization():
            return None

        @staticmethod
        def run_governed_vae_training(**kwargs):
            return {
                "training_result": {
                    "epochs_completed": 2,
                    "completed_updates": 254,
                },
                "checkpoint_status": "NOT_PREPARED",
                "publication_status": "NOT_PUBLISHED",
                "dataset_provenance": {},
            }

    def verify_unchanged(value):
        calls.append("permit")
        return True

    def require_destination(value, **kwargs):
        calls.append("destination")
        return value.publication_root

    def publish(**kwargs):
        calls.append("publish")
        return {"publication_status": "PUBLISHED"}

    monkeypatch.setattr(
        cli,
        "require_environment",
        lambda args: SimpleNamespace(
            actual_source_commit=SOURCE_COMMIT,
            repository_root=REPO_ROOT,
            module_paths={},
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_governed_module",
        lambda: Governed,
    )
    monkeypatch.setattr(cli, "build_adapter", lambda args: object())
    monkeypatch.setattr(
        cli,
        "effective_configuration",
        lambda: {"epochs": 2},
    )
    monkeypatch.setattr(
        permit,
        "verify_vae_publication_permit_unchanged",
        verify_unchanged,
    )
    monkeypatch.setattr(
        permit,
        "require_unused_vae_publication_destination",
        require_destination,
    )
    monkeypatch.setattr(
        coordinator,
        "publish_governed_vae_checkpoints",
        publish,
    )

    summary = cli.run(args)

    assert calls[-3:] == ["permit", "destination", "publish"]
    assert calls.count("destination") == 2
    assert summary["publication_authorization"]["authorization_id"] == (
        AUTHORIZATION_ID
    )


def test_cli_permit_validator_accepts_test_permit(permit_context):
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.record["publication_root"])
    )
    request = cli.validate_request(args)

    result = cli.validate_publication_permit(args, request)

    assert result.authorization_id == AUTHORIZATION_ID
    assert result.publication_root == Path(
        permit_context.record["publication_root"]
    )


def test_cli_permit_validator_rejects_wrong_root(permit_context):
    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_args(permit_context.artifact_root / "wrong-root")
    )
    request = cli.validate_request(args)

    with pytest.raises(permit.VAECheckpointPublicationPermitError):
        cli.validate_publication_permit(args, request)
