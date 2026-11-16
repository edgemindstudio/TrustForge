"""Synthesis-only authorization for the CIC GAN seed-42 production pool.

Default deny. One external authorization record must exactly bind the approved
synthesis source, checkpoint, destination, recipe, budget and resource
envelope. This module does not create grants, launch jobs, load checkpoints or
generate samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from trustforge.provenance import canonical_json_bytes
from trustforge.provenance.git import GitProvenanceError, capture_git_state

from .gan_checkpoint_v3_manifest import (
    validate_gan_checkpoint_v3_manifest,
)
from .gan_checkpoint_v3_secure_io import read_regular_file_securely


SCHEMA = "trustforge.paper01.gan.cic-pool-authorization.v1"
PAPER = "paper01"
REVISION = "revision_2026"
DATASET_ID = "cicmaldroid2020"
FAMILY = "gan"
EXPERIMENTAL_SEED = 42
ATTEMPT = 1
CLASS_MAPPING = ("0", "1", "2", "3", "4")
QUOTA_PER_CLASS = 2000
TOTAL_SAMPLES = 10000
BATCH_SIZE = 16
BATCHES_PER_CLASS = 125
TOTAL_BATCHES = 625
RECIPE_SHA256 = (
    "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
)
CHECKPOINT_RUN_IDENTITY = "commit42c40fc_attempt01"

RESOURCE_FIELDS = frozenset({
    "partition",
    "nodes",
    "tasks",
    "cpus_per_task",
    "memory_gb",
    "walltime_seconds",
    "gpu_enabled",
    "max_concurrent_jobs",
    "automatic_retries",
})
EXPECTED_RESOURCES = {
    "partition": "talon",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory_gb": 32,
    "walltime_seconds": 12 * 60 * 60,
    "gpu_enabled": False,
    "max_concurrent_jobs": 1,
    "automatic_retries": 0,
}
TOP_LEVEL_FIELDS = frozenset({
    "schema",
    "paper",
    "revision",
    "family",
    "dataset_id",
    "experimental_seed",
    "attempt",
    "source_commit",
    "launcher_sha256",
    "checkpoint_directory",
    "checkpoint_run_identity",
    "checkpoint_sha256",
    "destination",
    "run_identity",
    "recipe_sha256",
    "class_mapping",
    "quota_per_class",
    "total_samples",
    "batch_size",
    "batches_per_class",
    "total_batches",
    "resources",
    "authorization_id",
})

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
RUN_IDENTITY = re.compile(r"^commit([0-9a-f]{7})_attempt01$")
PLACEHOLDER_ID = "RESEARCHER_DECISION_PENDING"
MAX_AUTHORIZATION_BYTES = 65536
MAX_METADATA_BYTES = 1024 * 1024


class GANSynthesisAuthorizationError(PermissionError):
    """Raised when synthesis authorization is absent or invalid."""


@dataclass(frozen=True)
class GANSynthesisGrant:
    dataset_id: str
    experimental_seed: int
    attempt: int
    source_commit: str
    launcher_sha256: str
    checkpoint_directory: Path
    checkpoint_run_identity: str
    checkpoint_sha256: str
    destination: Path
    run_identity: str
    recipe_sha256: str
    authorization_id: str
    resources: dict


def _account_home():
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (ImportError, KeyError):
        return Path.home().resolve()


ACCOUNT_HOME = _account_home()
CHECKPOINT_ROOT = (
    ACCOUNT_HOME
    / "trustforge_revision_2026_artifacts"
    / "paper01"
    / "gan"
    / DATASET_ID
    / "seed42"
    / "checkpoints"
    / "production"
)
POOL_ROOT = (
    ACCOUNT_HOME
    / "trustforge_revision_2026_artifacts"
    / "paper01"
    / "gan"
    / DATASET_ID
    / "seed42"
    / "pools"
    / "production"
)
AUTHORIZATION_PATH = (
    ACCOUNT_HOME
    / "trustforge_revision_2026_artifacts"
    / "paper01"
    / "gan"
    / "authorizations"
    / "cicmaldroid2020_seed42_synthesis.json"
)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    record = {}

    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON field: {key}")
        record[key] = value

    return record


def _require_text(value, name):
    if type(value) is not str or not value or value != value.strip():
        raise GANSynthesisAuthorizationError(
            f"invalid synthesis authorization field: {name}"
        )
    return value


def _require_exact_int(value, name, expected):
    if type(value) is not int or value != expected:
        raise GANSynthesisAuthorizationError(
            f"invalid synthesis authorization field: {name}"
        )
    return value


def _require_digest(value, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise GANSynthesisAuthorizationError(
            f"invalid synthesis authorization digest: {name}"
        )
    return value


def _validate_resources(resources):
    if type(resources) is not dict or set(resources) != RESOURCE_FIELDS:
        raise GANSynthesisAuthorizationError(
            "synthesis resource fields are invalid"
        )

    for name, expected in EXPECTED_RESOURCES.items():
        observed = resources[name]

        if type(observed) is not type(expected) or observed != expected:
            raise GANSynthesisAuthorizationError(
                f"synthesis resource differs: {name}"
            )


def _load_authorization():
    if Path.home().resolve() != ACCOUNT_HOME:
        raise GANSynthesisAuthorizationError(
            "HOME differs from the canonical account home"
        )

    path = AUTHORIZATION_PATH

    if path.is_symlink() or not path.is_file():
        raise GANSynthesisAuthorizationError(
            "synthesis authorization is absent"
        )

    metadata = path.stat()

    if metadata.st_uid != os.getuid():
        raise GANSynthesisAuthorizationError(
            "synthesis authorization has a different owner"
        )

    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise GANSynthesisAuthorizationError(
            "synthesis authorization is group/world writable"
        )

    if metadata.st_size > MAX_AUTHORIZATION_BYTES:
        raise GANSynthesisAuthorizationError(
            "synthesis authorization is too large"
        )

    try:
        record = json.loads(
            path.read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise GANSynthesisAuthorizationError(
            "synthesis authorization is unreadable"
        ) from exc

    if type(record) is not dict or set(record) != TOP_LEVEL_FIELDS:
        raise GANSynthesisAuthorizationError(
            "synthesis authorization fields are invalid"
        )

    return record


def load_gan_synthesis_grant():
    """Return one statically validated synthesis grant; default deny."""
    record = _load_authorization()

    if _require_text(record["schema"], "schema") != SCHEMA:
        raise GANSynthesisAuthorizationError(
            "unsupported synthesis authorization schema"
        )

    if _require_text(record["paper"], "paper") != PAPER:
        raise GANSynthesisAuthorizationError(
            "authorization is not Paper 1"
        )

    if _require_text(record["revision"], "revision") != REVISION:
        raise GANSynthesisAuthorizationError(
            "authorization is not revision_2026"
        )

    if _require_text(record["family"], "family") != FAMILY:
        raise GANSynthesisAuthorizationError("authorization is not GAN")

    if _require_text(record["dataset_id"], "dataset_id") != DATASET_ID:
        raise GANSynthesisAuthorizationError(
            "authorization is not CICMalDroid2020"
        )

    if _require_exact_int(
        record["experimental_seed"], "experimental_seed", EXPERIMENTAL_SEED
    ) != EXPERIMENTAL_SEED:
        raise GANSynthesisAuthorizationError(
            "authorization seed is not 42"
        )

    if _require_exact_int(
        record["attempt"], "attempt", ATTEMPT
    ) != ATTEMPT:
        raise GANSynthesisAuthorizationError(
            "authorization attempt is not 1"
        )

    source = _require_text(record["source_commit"], "source_commit")
    if HEX40.fullmatch(source) is None:
        raise GANSynthesisAuthorizationError(
            "authorization source commit is malformed"
        )

    launcher = _require_digest(
        record["launcher_sha256"], "launcher_sha256"
    )

    checkpoint_directory = (
        CHECKPOINT_ROOT / CHECKPOINT_RUN_IDENTITY
    )
    granted_checkpoint = _require_text(
        record["checkpoint_directory"], "checkpoint_directory"
    )

    if (
        Path(granted_checkpoint) != checkpoint_directory
        or Path(granted_checkpoint).is_symlink()
    ):
        raise GANSynthesisAuthorizationError(
            "authorized checkpoint directory is invalid"
        )

    if _require_text(
        record["checkpoint_run_identity"], "checkpoint_run_identity"
    ) != CHECKPOINT_RUN_IDENTITY:
        raise GANSynthesisAuthorizationError(
            "authorized checkpoint run identity is invalid"
        )

    checkpoint_sha256 = _require_digest(
        record["checkpoint_sha256"], "checkpoint_sha256"
    )

    run_identity = _require_text(
        record["run_identity"], "run_identity"
    )
    match = RUN_IDENTITY.fullmatch(run_identity)

    if match is None or match.group(1) != source[:7]:
        raise GANSynthesisAuthorizationError(
            "authorized synthesis run identity is invalid"
        )

    destination = POOL_ROOT / run_identity
    granted_destination = _require_text(
        record["destination"], "destination"
    )

    if Path(granted_destination) != destination:
        raise GANSynthesisAuthorizationError(
            "authorized synthesis destination is invalid"
        )

    if _require_text(
        record["recipe_sha256"], "recipe_sha256"
    ) != RECIPE_SHA256:
        raise GANSynthesisAuthorizationError(
            "authorized synthesis recipe differs"
        )

    if record["class_mapping"] != list(CLASS_MAPPING):
        raise GANSynthesisAuthorizationError(
            "authorized class mapping differs"
        )

    if _require_exact_int(
        record["quota_per_class"], "quota_per_class", QUOTA_PER_CLASS
    ) != QUOTA_PER_CLASS:
        raise GANSynthesisAuthorizationError("authorized quota differs")

    if _require_exact_int(
        record["total_samples"], "total_samples", TOTAL_SAMPLES
    ) != TOTAL_SAMPLES:
        raise GANSynthesisAuthorizationError(
            "authorized total samples differ"
        )

    if _require_exact_int(
        record["batch_size"], "batch_size", BATCH_SIZE
    ) != BATCH_SIZE:
        raise GANSynthesisAuthorizationError("authorized batch size differs")

    if _require_exact_int(
        record["batches_per_class"],
        "batches_per_class",
        BATCHES_PER_CLASS,
    ) != BATCHES_PER_CLASS:
        raise GANSynthesisAuthorizationError(
            "authorized batches per class differ"
        )

    if _require_exact_int(
        record["total_batches"], "total_batches", TOTAL_BATCHES
    ) != TOTAL_BATCHES:
        raise GANSynthesisAuthorizationError(
            "authorized total batches differ"
        )

    _validate_resources(record["resources"])

    authorization_id = _require_text(
        record["authorization_id"], "authorization_id"
    )

    if (
        authorization_id == PLACEHOLDER_ID
        or AUTHORIZATION_ID.fullmatch(authorization_id) is None
    ):
        raise GANSynthesisAuthorizationError(
            "synthesis authorization is not approved"
        )

    return GANSynthesisGrant(
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        attempt=ATTEMPT,
        source_commit=source,
        launcher_sha256=launcher,
        checkpoint_directory=checkpoint_directory,
        checkpoint_run_identity=CHECKPOINT_RUN_IDENTITY,
        checkpoint_sha256=checkpoint_sha256,
        destination=destination,
        run_identity=run_identity,
        recipe_sha256=RECIPE_SHA256,
        authorization_id=authorization_id,
        resources=dict(record["resources"]),
    )


def require_gan_synthesis_authorization(
    *,
    destination,
    checkpoint_directory,
    dataset_id,
    experimental_seed,
    repository_root,
):
    """Validate a grant and the exact runtime identity of one synthesis run."""
    grant = load_gan_synthesis_grant()

    if dataset_id != DATASET_ID:
        raise GANSynthesisAuthorizationError(
            "requested synthesis dataset is not authorized"
        )

    if experimental_seed != EXPERIMENTAL_SEED:
        raise GANSynthesisAuthorizationError(
            "requested synthesis seed is not authorized"
        )

    requested_checkpoint = Path(checkpoint_directory).expanduser()
    if requested_checkpoint != grant.checkpoint_directory:
        raise GANSynthesisAuthorizationError(
            "requested checkpoint is not authorized"
        )

    if (
        not requested_checkpoint.is_dir()
        or requested_checkpoint.is_symlink()
    ):
        raise GANSynthesisAuthorizationError(
            "authorized checkpoint directory is unavailable"
        )

    requested_destination = Path(destination).expanduser()
    if requested_destination != grant.destination:
        raise GANSynthesisAuthorizationError(
            "requested synthesis destination is not authorized"
        )

    if (
        requested_destination.is_symlink()
        or requested_destination.exists()
    ):
        raise GANSynthesisAuthorizationError(
            "synthesis destination already exists"
        )

    try:
        state = capture_git_state(repository_root)
    except (GitProvenanceError, OSError) as exc:
        raise GANSynthesisAuthorizationError(
            "synthesis source identity is unavailable"
        ) from exc

    if state.dirty:
        raise GANSynthesisAuthorizationError(
            "synthesis source checkout is dirty"
        )

    if state.commit != grant.source_commit:
        raise GANSynthesisAuthorizationError(
            "synthesis source does not match the authorization"
        )

    return grant


def verify_checkpoint_metadata_digest(
    checkpoint_directory,
    expected_sha256,
):
    """Read-only metadata preflight; no TensorFlow model is constructed."""
    expected = _require_digest(expected_sha256, "expected_sha256")
    root = Path(checkpoint_directory).expanduser()

    if not root.is_dir() or root.is_symlink():
        raise GANSynthesisAuthorizationError(
            "checkpoint directory is unavailable"
        )

    path = root / "model_metadata.json"

    if path.is_symlink() or not path.is_file():
        raise GANSynthesisAuthorizationError(
            "checkpoint metadata is unavailable"
        )

    raw = read_regular_file_securely(
        path,
        max_bytes=MAX_METADATA_BYTES,
    )

    try:
        loaded = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise GANSynthesisAuthorizationError(
            "checkpoint metadata is invalid"
        ) from exc

    try:
        metadata = validate_gan_checkpoint_v3_manifest(loaded)
    except (TypeError, ValueError) as exc:
        raise GANSynthesisAuthorizationError(
            "checkpoint metadata is not a valid v3 manifest"
        ) from exc

    if raw != canonical_json_bytes(metadata) + b"\n":
        raise GANSynthesisAuthorizationError(
            "checkpoint metadata is not canonical"
        )

    digest = hashlib.sha256(
        canonical_json_bytes(metadata)
    ).hexdigest()

    if digest != expected:
        raise GANSynthesisAuthorizationError(
            "checkpoint metadata digest does not match authorization"
        )

    return digest


__all__ = [
    "AUTHORIZATION_PATH",
    "BATCHES_PER_CLASS",
    "BATCH_SIZE",
    "CHECKPOINT_ROOT",
    "CHECKPOINT_RUN_IDENTITY",
    "CLASS_MAPPING",
    "DATASET_ID",
    "EXPECTED_RESOURCES",
    "EXPERIMENTAL_SEED",
    "FAMILY",
    "GANSynthesisAuthorizationError",
    "GANSynthesisGrant",
    "POOL_ROOT",
    "RECIPE_SHA256",
    "SCHEMA",
    "TOTAL_BATCHES",
    "TOTAL_SAMPLES",
    "load_gan_synthesis_grant",
    "require_gan_synthesis_authorization",
    "verify_checkpoint_metadata_digest",
]
