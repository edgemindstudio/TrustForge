"""Identity-bound VAE checkpoint-publication permit validation.

Default deny. The permit record must bind one approved dataset, seed,
attempt, execution source, publication destination, class ordering,
checkpoint roles and observation contract. This module only validates the
record and its destination; it does not train, prepare checkpoints, publish
artifacts, activate publication or create approval records.

The cooperative same-account custody model provides accidental-modification
and identity-confusion protection. It does not authenticate approval against
a malicious same-UID actor, and a SHA-256 digest is not proof of researcher
authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trustforge.provenance import canonical_json_bytes


SCHEMA = "trustforge.paper01.vae.checkpoint-publication-permit.v1"
PAPER = "paper01"
REVISION = "revision_2026"
FAMILY = "vae"
DATASET_ID = "cicmaldroid2020"
EXPERIMENTAL_SEED = 42
ATTEMPT = 2
CLASS_MAPPING = ("0", "1", "2", "3", "4")
CHECKPOINT_ROLES = ("best", "last")
OBSERVATION_VERSION = "gaussian_fixed_variance_1p0_v1"
CHECKPOINT_SCHEMA_TYPE = (
    "paper01_revision2026_conditional_vae_checkpoint"
)
CHECKPOINT_SCHEMA_VERSION = "1.0"
ARCHITECTURE_VERSION = (
    "paper01_revision2026_conditional_vae_arch_v1"
)
SELECTION_POLICY_ID = (
    "min_validation_total_loss_earliest_epoch_v1"
)
APPROVAL_RECORD = (
    "studies/paper01_benchmark/revision_2026/"
    "production_readiness_ledger.md"
)

ARTIFACT_ROOT = (
    Path.home() / "trustforge_revision_2026_artifacts" / "paper01"
)
PERMIT_PATH = (
    ARTIFACT_ROOT
    / "vae"
    / "authorizations"
    / "vae_checkpoint_publication_permit.json"
)

MAX_PERMIT_BYTES = 65536
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)
PLACEHOLDER_ID = "RESEARCHER_DECISION_PENDING"

# Canonical approval entry line:
# <!-- VAE_CHECKPOINT_PUBLICATION_APPROVAL {"authorization_id":"...","authorization_scope_sha256":"..."} -->
APPROVAL_ENTRY_PREFIX = (
    "<!-- VAE_CHECKPOINT_PUBLICATION_APPROVAL "
)
APPROVAL_ENTRY_SUFFIX = " -->"
APPROVAL_ENTRY_FIELDS = frozenset({
    "authorization_id",
    "authorization_scope_sha256",
})

TOP_LEVEL_FIELDS = frozenset({
    "schema",
    "paper",
    "revision",
    "family",
    "dataset_id",
    "experimental_seed",
    "attempt",
    "run_identity",
    "execution_source_commit",
    "publication_root",
    "class_mapping",
    "checkpoint_roles",
    "observation_version",
    "checkpoint_contract",
    "authorization_id",
    "approval_record",
    "approval_commit",
    "scope",
})

CHECKPOINT_CONTRACT_FIELDS = frozenset({
    "schema_type",
    "schema_version",
    "architecture_version",
    "selection_policy_id",
})

SCOPE_FIELDS = frozenset({
    "checkpoint_publication",
    "committed_generation_verification",
    "synthesis",
    "evaluation",
})


class VAECheckpointPublicationPermitError(PermissionError):
    """Raised when publication authorization is absent or invalid."""


@dataclass(frozen=True)
class VAECheckpointPublicationPermit:
    """Validated immutable publication scope for one VAE execution."""

    schema: str
    paper: str
    revision: str
    family: str
    dataset_id: str
    experimental_seed: int
    attempt: int
    run_identity: str
    execution_source_commit: str
    publication_root: Path
    class_mapping: tuple[str, ...]
    checkpoint_roles: tuple[str, ...]
    observation_version: str
    checkpoint_contract: dict
    authorization_id: str
    approval_record: str
    approval_commit: str
    scope: dict
    authorization_scope_sha256: str
    permit_sha256: str
    permit_path: Path


def _fail(message):
    raise VAECheckpointPublicationPermitError(message)


def _reject_constant(value):
    _fail("JSON contains a non-finite constant")


def _reject_duplicate_keys(pairs):
    record = {}

    for key, value in pairs:
        if key in record:
            _fail("JSON contains a duplicate field")
        record[key] = value

    return record


def _require_text(value, name):
    if type(value) is not str or not value or value != value.strip():
        _fail(f"invalid permit field: {name}")
    return value


def _require_exact_int(value, name, expected):
    if type(value) is not int or value != expected:
        _fail(f"invalid permit field: {name}")
    return value


def _require_commit(value, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        _fail(f"invalid permit commit: {name}")
    return value


def _require_digest(value, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        _fail(f"invalid approval digest: {name}")
    return value


def _require_authorization_id(value):
    if (
        type(value) is not str
        or value == PLACEHOLDER_ID
        or AUTHORIZATION_ID.fullmatch(value) is None
    ):
        _fail("permit authorization ID is not approved")
    return value


def _require_mapping(value, name, expected_fields):
    if type(value) is not dict:
        _fail(f"invalid permit field: {name}")

    if set(value) != set(expected_fields):
        _fail(f"invalid permit field inventory: {name}")

    return value


def _require_class_mapping(value):
    if type(value) is not list or list(value) != list(CLASS_MAPPING):
        _fail("permit class mapping is not the governed CIC ordering")
    return tuple(value)


def _require_checkpoint_roles(value):
    if type(value) is not list or list(value) != list(CHECKPOINT_ROLES):
        _fail("permit checkpoint roles must be exactly best and last")
    return tuple(value)


def _expected_run_identity(source_commit, attempt):
    return f"commit{source_commit[:7]}_attempt{attempt:02d}"


def _expected_publication_root(run_identity):
    return (
        ARTIFACT_ROOT
        / "vae"
        / DATASET_ID
        / f"seed{EXPERIMENTAL_SEED}"
        / "checkpoints"
        / "production"
        / run_identity
    )


def vae_authorization_scope_payload(record):
    """Return every approved field except the later approval commit."""
    if type(record) is not dict:
        raise TypeError("permit record must be a dictionary")

    return {
        name: record[name]
        for name in sorted(TOP_LEVEL_FIELDS - {"approval_commit"})
    }


def vae_authorization_scope_sha256(record):
    """Return the stable digest of the complete approved execution scope."""
    return hashlib.sha256(
        canonical_json_bytes(
            vae_authorization_scope_payload(record)
        )
    ).hexdigest()


def _read_permit_bytes(path):
    path = Path(path).expanduser()

    if path.parent.is_symlink() or not path.parent.is_dir():
        _fail("permit parent directory is unsafe or unavailable")

    required = ("O_NOFOLLOW", "O_CLOEXEC")

    if any(not hasattr(os, flag) for flag in required):
        raise RuntimeError("required permit file flags are unavailable")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC

    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VAECheckpointPublicationPermitError(
            "permit is absent or is not a regular file"
        ) from exc

    try:
        info = os.fstat(descriptor)

        if not stat.S_ISREG(info.st_mode):
            _fail("permit must be a regular file")

        if info.st_uid != os.geteuid():
            _fail("permit has a different owner")

        if stat.S_IMODE(info.st_mode) & 0o022:
            _fail("permit is group/world writable")

        if info.st_size > MAX_PERMIT_BYTES:
            _fail("permit is too large")

        chunks = []
        total = 0

        while True:
            chunk = os.read(descriptor, min(8192, MAX_PERMIT_BYTES + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_PERMIT_BYTES:
                _fail("permit is too large")

        return b"".join(chunks)

    finally:
        os.close(descriptor)


def _parse_permit(raw):
    try:
        text = raw.decode("utf-8")
        record = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except VAECheckpointPublicationPermitError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise VAECheckpointPublicationPermitError(
            "permit JSON is unreadable"
        ) from exc

    if type(record) is not dict:
        _fail("permit JSON must be an object")

    if raw != canonical_json_bytes(record) + b"\n":
        _fail("permit is not canonically encoded")

    return record


def load_vae_checkpoint_publication_permit():
    """Load and parse the one canonical permit record, or fail closed."""
    path = Path(PERMIT_PATH).expanduser()
    raw = _read_permit_bytes(path)
    return _parse_permit(raw)


def _read_approval_record(
    *,
    repository_root,
    approval_commit,
    approval_record,
):
    """Read the referenced approval evidence from Git history.

    Tests may replace this boundary with fixture text. Production reads the
    exact approval record blob from the stated commit, without modifying the
    repository.
    """
    if approval_record != APPROVAL_RECORD:
        _fail("permit approval record is not the accepted ledger")

    _require_commit(approval_commit, "approval_commit")

    root = Path(repository_root).expanduser()

    commands = (
        ["cat-file", "-e", approval_commit + "^{commit}"],
        ["show", f"{approval_commit}:{approval_record}"],
    )

    outputs = []

    for arguments in commands:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VAECheckpointPublicationPermitError(
                "permit approval record cannot be read"
            ) from exc

        if completed.returncode != 0:
            _fail("permit approval record is unavailable")

        outputs.append(completed.stdout)

    try:
        return outputs[-1].decode("utf-8")
    except UnicodeError as exc:
        raise VAECheckpointPublicationPermitError(
            "permit approval record is not text"
        ) from exc


def _parse_approval_entries(text):
    entries = []

    for line in text.splitlines():
        if APPROVAL_ENTRY_PREFIX not in line:
            continue

        if (
            not line.startswith(APPROVAL_ENTRY_PREFIX)
            or not line.endswith(APPROVAL_ENTRY_SUFFIX)
            or line.count(APPROVAL_ENTRY_PREFIX) != 1
            or line.count(APPROVAL_ENTRY_SUFFIX) != 1
        ):
            _fail("permit approval entry is malformed")

        body = line[
            len(APPROVAL_ENTRY_PREFIX):-len(APPROVAL_ENTRY_SUFFIX)
        ]

        try:
            entry = json.loads(
                body,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except VAECheckpointPublicationPermitError:
            raise
        except (UnicodeError, ValueError) as exc:
            raise VAECheckpointPublicationPermitError(
                "permit approval entry is unreadable"
            ) from exc

        if type(entry) is not dict:
            _fail("permit approval entry must be an object")

        if set(entry) != APPROVAL_ENTRY_FIELDS:
            _fail("permit approval entry field inventory is invalid")

        if body != canonical_json_bytes(entry).decode("utf-8"):
            _fail("permit approval entry is not canonically encoded")

        _require_authorization_id(entry["authorization_id"])
        _require_digest(
            entry["authorization_scope_sha256"],
            "authorization_scope_sha256",
        )
        entries.append(entry)

    return entries


def _validate_approval_evidence(
    record,
    *,
    authorization_scope_sha256,
    repository_root,
):
    if record["approval_record"] != APPROVAL_RECORD:
        _fail("permit approval record is not the accepted ledger")

    text = _read_approval_record(
        repository_root=repository_root,
        approval_commit=record["approval_commit"],
        approval_record=record["approval_record"],
    )

    entries = _parse_approval_entries(text)
    matching = [
        entry
        for entry in entries
        if entry["authorization_id"] == record["authorization_id"]
    ]

    if not matching:
        _fail("permit approval entry is missing")

    if len(matching) != 1:
        _fail("permit approval entry is duplicated or ambiguous")

    if (
        matching[0]["authorization_scope_sha256"]
        != authorization_scope_sha256
    ):
        _fail("permit approval scope digest mismatch")


def validate_vae_checkpoint_publication_permit(
    record,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_attempt,
    requested_source_commit,
    requested_publication_root,
    requested_class_mapping=CLASS_MAPPING,
    repository_root,
):
    """Validate one permit record against one requested execution."""
    if type(record) is not dict:
        _fail("permit record must be a dictionary")

    if type(requested_dataset_id) is not str:
        _fail("requested dataset identity is invalid")

    if type(requested_experimental_seed) is not int:
        _fail("requested experimental seed is invalid")

    if type(requested_attempt) is not int:
        _fail("requested attempt identity is invalid")

    if (
        type(requested_source_commit) is not str
        or HEX40.fullmatch(requested_source_commit) is None
    ):
        _fail("requested execution source is invalid")

    if type(requested_class_mapping) not in (tuple, list):
        _fail("requested class mapping is invalid")

    if set(record) != TOP_LEVEL_FIELDS:
        _fail("permit field inventory is invalid")

    expected_text = {
        "schema": SCHEMA,
        "paper": PAPER,
        "revision": REVISION,
        "family": FAMILY,
        "dataset_id": DATASET_ID,
        "observation_version": OBSERVATION_VERSION,
    }

    for name, expected in expected_text.items():
        if _require_text(record[name], name) != expected:
            _fail(f"permit field differs: {name}")

    if _require_text(
        record["dataset_id"], "dataset_id"
    ) != requested_dataset_id:
        _fail("permit dataset does not match the execution")

    if _require_exact_int(
        record["experimental_seed"],
        "experimental_seed",
        EXPERIMENTAL_SEED,
    ) != requested_experimental_seed:
        _fail("permit seed does not match the execution")

    if _require_exact_int(
        record["attempt"], "attempt", ATTEMPT
    ) != requested_attempt:
        _fail("permit attempt does not match the execution")

    source = _require_commit(
        record["execution_source_commit"],
        "execution_source_commit",
    )

    if source != requested_source_commit:
        _fail("permit source does not match the execution")

    identity = _require_text(
        record["run_identity"], "run_identity"
    )

    if identity != _expected_run_identity(
        source, requested_attempt
    ):
        _fail("permit run identity is inconsistent")

    mapping = _require_class_mapping(record["class_mapping"])

    if mapping != tuple(requested_class_mapping):
        _fail("permit class mapping does not match the execution")

    roles = _require_checkpoint_roles(record["checkpoint_roles"])

    if roles != CHECKPOINT_ROLES:
        _fail("permit checkpoint roles are unsupported")

    contract = _require_mapping(
        record["checkpoint_contract"],
        "checkpoint_contract",
        CHECKPOINT_CONTRACT_FIELDS,
    )

    expected_contract = {
        "schema_type": CHECKPOINT_SCHEMA_TYPE,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "selection_policy_id": SELECTION_POLICY_ID,
    }

    for name, expected in expected_contract.items():
        if _require_text(
            contract[name], "checkpoint_contract." + name
        ) != expected:
            _fail(f"permit checkpoint contract differs: {name}")

    authorization_id = _require_authorization_id(
        record["authorization_id"]
    )

    approval_record = _require_text(
        record["approval_record"], "approval_record"
    )

    if approval_record != APPROVAL_RECORD:
        _fail("permit approval record is not the accepted ledger")

    approval_commit = _require_commit(
        record["approval_commit"], "approval_commit"
    )

    scope = _require_mapping(
        record["scope"], "scope", SCOPE_FIELDS
    )

    expected_scope = {
        "checkpoint_publication": True,
        "committed_generation_verification": True,
        "synthesis": False,
        "evaluation": False,
    }

    for name, expected in expected_scope.items():
        if type(scope[name]) is not bool or scope[name] is not expected:
            _fail(f"permit scope differs: {name}")

    requested_root = Path(
        requested_publication_root
    ).expanduser()

    if not requested_root.is_absolute():
        _fail("publication root must be absolute")

    if ".." in requested_root.parts:
        _fail("publication root must not contain parent traversal")

    granted_root = Path(
        record["publication_root"]
    ).expanduser()

    if granted_root != requested_root:
        _fail("permit publication root does not match the execution")

    expected_root = _expected_publication_root(identity)

    if granted_root != expected_root:
        _fail("permit publication root is not the canonical destination")

    authorization_scope_sha256 = vae_authorization_scope_sha256(
        record
    )

    _validate_approval_evidence(
        record,
        authorization_scope_sha256=authorization_scope_sha256,
        repository_root=repository_root,
    )

    return VAECheckpointPublicationPermit(
        schema=record["schema"],
        paper=record["paper"],
        revision=record["revision"],
        family=record["family"],
        dataset_id=record["dataset_id"],
        experimental_seed=record["experimental_seed"],
        attempt=record["attempt"],
        run_identity=identity,
        execution_source_commit=source,
        publication_root=granted_root,
        class_mapping=mapping,
        checkpoint_roles=roles,
        observation_version=record["observation_version"],
        checkpoint_contract=dict(contract),
        authorization_id=authorization_id,
        approval_record=approval_record,
        approval_commit=approval_commit,
        scope=dict(scope),
        authorization_scope_sha256=authorization_scope_sha256,
        permit_sha256=hashlib.sha256(
            canonical_json_bytes(record)
        ).hexdigest(),
        permit_path=Path(PERMIT_PATH).expanduser(),
    )


def _validate_publication_destination(
    publication_root,
    *,
    repository_root,
):
    requested = Path(publication_root).expanduser()

    if not requested.is_absolute():
        _fail("publication destination must be absolute")

    if ".." in requested.parts:
        _fail("publication destination must not contain traversal")

    if requested.is_symlink():
        _fail("publication destination must not be a symlink")

    repository = Path(repository_root).expanduser().resolve()
    resolved = requested.parent.resolve() / requested.name

    if resolved == repository or repository in resolved.parents:
        _fail("publication destination must be outside the source checkout")

    if requested.exists():
        _fail("publication destination already exists")

    ancestor = requested.parent

    while not ancestor.exists():
        if ancestor.is_symlink():
            _fail("publication destination contains a symlink")
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent

    if ancestor.is_symlink() or not ancestor.is_dir():
        _fail("publication destination parent is unsafe or unavailable")

    return requested


def require_unused_vae_publication_destination(
    permit,
    *,
    repository_root,
):
    """Require the permit destination to be fresh and outside the checkout."""
    if not isinstance(permit, VAECheckpointPublicationPermit):
        raise TypeError("permit must be a validated permit")

    return _validate_publication_destination(
        permit.publication_root,
        repository_root=repository_root,
    )


def verify_vae_publication_permit_unchanged(permit):
    """Re-read the canonical permit and require identical canonical bytes."""
    if not isinstance(permit, VAECheckpointPublicationPermit):
        raise TypeError("permit must be a validated permit")

    record = load_vae_checkpoint_publication_permit()
    digest = hashlib.sha256(
        canonical_json_bytes(record)
    ).hexdigest()

    if digest != permit.permit_sha256:
        _fail("permit changed after validation")

    return True


def vae_publication_permit_sha256(permit):
    """Return the canonical permit digest used for audit evidence."""
    if not isinstance(permit, VAECheckpointPublicationPermit):
        raise TypeError("permit must be a validated permit")
    return permit.permit_sha256


def require_vae_checkpoint_publication_permit(
    *,
    dataset_id,
    experimental_seed,
    attempt,
    execution_source_commit,
    publication_root,
    class_mapping=CLASS_MAPPING,
    repository_root,
):
    """Load, validate and bind one permit to one requested execution."""
    record = load_vae_checkpoint_publication_permit()

    permit = validate_vae_checkpoint_publication_permit(
        record,
        requested_dataset_id=dataset_id,
        requested_experimental_seed=experimental_seed,
        requested_attempt=attempt,
        requested_source_commit=execution_source_commit,
        requested_publication_root=publication_root,
        requested_class_mapping=class_mapping,
        repository_root=repository_root,
    )

    require_unused_vae_publication_destination(
        permit,
        repository_root=repository_root,
    )

    return permit


__all__ = [
    "APPROVAL_ENTRY_FIELDS",
    "APPROVAL_ENTRY_PREFIX",
    "APPROVAL_ENTRY_SUFFIX",
    "APPROVAL_RECORD",
    "ARCHITECTURE_VERSION",
    "ARTIFACT_ROOT",
    "ATTEMPT",
    "AUTHORIZATION_ID",
    "CHECKPOINT_ROLES",
    "CHECKPOINT_SCHEMA_TYPE",
    "CHECKPOINT_SCHEMA_VERSION",
    "CLASS_MAPPING",
    "DATASET_ID",
    "EXPERIMENTAL_SEED",
    "OBSERVATION_VERSION",
    "PERMIT_PATH",
    "SCHEMA",
    "SELECTION_POLICY_ID",
    "VAECheckpointPublicationPermit",
    "VAECheckpointPublicationPermitError",
    "load_vae_checkpoint_publication_permit",
    "require_unused_vae_publication_destination",
    "require_vae_checkpoint_publication_permit",
    "vae_authorization_scope_payload",
    "vae_authorization_scope_sha256",
    "vae_publication_permit_sha256",
    "validate_vae_checkpoint_publication_permit",
    "verify_vae_publication_permit_unchanged",
]
