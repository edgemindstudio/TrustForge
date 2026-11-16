"""Scoped production authorization for the Paper 1 revision-2026 GAN.

Default deny. A single external authorization record must exactly match the
planned run, accepted scientific baseline, actual executable source revision,
checkpoint destination and verified Slurm allocation.

This module does not create authorization records or production directories,
enable execution flags, authorize training, publish checkpoints or submit
jobs. The production authorization flags in ``gan_governed_runner`` remain the
outer execution boundary.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trustforge.provenance.git import GitProvenanceError, capture_git_state
from trustforge.storage import validate_storage_identifier

from .experimental_seed_policy import require_governed_experimental_seed
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from .gan_production_binding import (
    RUN_IDENTITY_PATTERN,
    resolve_gan_production_checkpoint_directory,
)


SCHEMA = "trustforge.paper01.gan.production-authorization.v1"
SCIENTIFIC_BASELINE_COMMIT = (
    "14982b142887095c53e62387749f5e308ce42d5d"
)
AUTHORIZATION_FILENAME = "production_authorization.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SLURM_FIELD = re.compile(r"([A-Za-z][A-Za-z0-9_/]*)=([^\s]*)")

TOP_LEVEL_FIELDS = frozenset({
    "schema",
    "paper",
    "family",
    "mode",
    "dataset_id",
    "experimental_seed",
    "attempt",
    "run_identity",
    "scientific_baseline_commit",
    "executable_source_commit",
    "checkpoint_directory",
    "authorization_id",
    "resources",
})

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


class GANProductionAuthorizationError(PermissionError):
    """Raised when scoped production authorization is absent or invalid."""


@dataclass(frozen=True)
class GANProductionScope:
    authorization_id: str
    dataset_id: str
    experimental_seed: int
    attempt: int
    run_identity: str
    scientific_baseline_commit: str
    executable_source_commit: str
    checkpoint_directory: Path
    resources: dict


def _account_home():
    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (ImportError, KeyError):
        return Path.home().resolve()


ACCOUNT_HOME = _account_home()
AUTHORIZATION_PATH = (
    ACCOUNT_HOME
    / "trustforge_revision_2026_artifacts"
    / "paper01"
    / "gan"
    / AUTHORIZATION_FILENAME
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


def _load_authorization():
    """Read the one external authorization record or fail closed."""
    if Path.home().resolve() != ACCOUNT_HOME:
        raise GANProductionAuthorizationError(
            "HOME differs from the canonical account home"
        )

    path = AUTHORIZATION_PATH

    if path.is_symlink() or not path.is_file():
        raise GANProductionAuthorizationError(
            "GAN production authorization is absent"
        )

    metadata = path.stat()

    if metadata.st_uid != os.getuid():
        raise GANProductionAuthorizationError(
            "GAN production authorization has a different owner"
        )

    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise GANProductionAuthorizationError(
            "GAN production authorization is group/world writable"
        )

    if metadata.st_size > 65536:
        raise GANProductionAuthorizationError(
            "GAN production authorization is too large"
        )

    try:
        record = json.loads(
            path.read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise GANProductionAuthorizationError(
            "GAN production authorization is unreadable"
        ) from exc

    if type(record) is not dict or set(record) != TOP_LEVEL_FIELDS:
        raise GANProductionAuthorizationError(
            "GAN production authorization fields are invalid"
        )

    return record


def _require_text(value, name):
    if type(value) is not str or not value or value != value.strip():
        raise GANProductionAuthorizationError(
            f"invalid authorization field: {name}"
        )
    return value


def _require_exact_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise GANProductionAuthorizationError(
            f"invalid authorization field: {name}"
        )
    return value


def _validate_resources(resources):
    if type(resources) is not dict or set(resources) != RESOURCE_FIELDS:
        raise GANProductionAuthorizationError(
            "authorized resource fields are invalid"
        )

    for name, expected in EXPECTED_RESOURCES.items():
        observed = resources[name]

        if type(observed) is not type(expected) or observed != expected:
            raise GANProductionAuthorizationError(
                f"authorized resource differs: {name}"
            )


def _validate_authorization_identity(
    record,
    *,
    dataset_id,
    experimental_seed,
    checkpoint_directory,
):
    if _require_text(record["paper"], "paper") != "paper01":
        raise GANProductionAuthorizationError("authorization is not Paper 1")

    if _require_text(record["family"], "family") != "gan":
        raise GANProductionAuthorizationError("authorization is not GAN")

    if _require_text(record["mode"], "mode") != "production":
        raise GANProductionAuthorizationError(
            "authorization is not production mode"
        )

    if _require_text(record["schema"], "schema") != SCHEMA:
        raise GANProductionAuthorizationError(
            "authorization schema is unsupported"
        )

    authorized_dataset = _require_text(
        record["dataset_id"], "dataset_id"
    )

    if authorized_dataset not in DATASET_CLASS_COUNTS:
        raise GANProductionAuthorizationError(
            "authorization dataset is unsupported"
        )

    try:
        authorized_seed = require_governed_experimental_seed(
            record["experimental_seed"]
        )
    except (TypeError, ValueError) as exc:
        raise GANProductionAuthorizationError(
            "authorization seed is unsupported"
        ) from exc

    authorized_attempt = _require_exact_int(
        record["attempt"], "attempt"
    )

    baseline = _require_text(
        record["scientific_baseline_commit"],
        "scientific_baseline_commit",
    )

    if baseline != SCIENTIFIC_BASELINE_COMMIT:
        raise GANProductionAuthorizationError(
            "scientific baseline differs from the accepted baseline"
        )

    executable = _require_text(
        record["executable_source_commit"],
        "executable_source_commit",
    )

    if HEX40.fullmatch(executable) is None:
        raise GANProductionAuthorizationError(
            "executable source commit is malformed"
        )

    identity = _require_text(record["run_identity"], "run_identity")

    try:
        validate_storage_identifier(identity, field_name="run_identity")
    except ValueError as exc:
        raise GANProductionAuthorizationError(
            "run identity is not a safe path segment"
        ) from exc

    match = RUN_IDENTITY_PATTERN.fullmatch(identity)

    if match is None:
        raise GANProductionAuthorizationError(
            "run identity is malformed"
        )

    if int(match.group(1), 16) != int(executable[:7], 16):
        raise GANProductionAuthorizationError(
            "run identity does not match the executable source"
        )

    if int(match.group(2)) != authorized_attempt:
        raise GANProductionAuthorizationError(
            "run identity does not match the authorized attempt"
        )

    authorization_id = _require_text(
        record["authorization_id"], "authorization_id"
    )

    if AUTHORIZATION_ID.fullmatch(authorization_id) is None:
        raise GANProductionAuthorizationError(
            "authorization identifier is malformed"
        )

    _validate_resources(record["resources"])

    if authorized_dataset != dataset_id:
        raise GANProductionAuthorizationError(
            "authorization dataset does not match the run"
        )

    if authorized_seed != experimental_seed:
        raise GANProductionAuthorizationError(
            "authorization seed does not match the run"
        )

    granted_directory = _require_text(
        record["checkpoint_directory"], "checkpoint_directory"
    )
    granted_path = Path(granted_directory)

    if (
        not granted_path.is_absolute()
        or ".." in granted_path.parts
        or granted_path.name != identity
    ):
        raise GANProductionAuthorizationError(
            "authorized checkpoint destination is invalid"
        )

    expected_directory = resolve_gan_production_checkpoint_directory(
        dataset_id=authorized_dataset,
        experimental_seed=authorized_seed,
        run_identity=identity,
        expected_source_commit=executable,
    )

    if granted_path != expected_directory:
        raise GANProductionAuthorizationError(
            "authorized checkpoint destination differs from the binding"
        )

    actual_directory = Path(checkpoint_directory).expanduser()

    if actual_directory != expected_directory:
        raise GANProductionAuthorizationError(
            "run checkpoint destination does not match the authorization"
        )

    if actual_directory.is_symlink():
        raise GANProductionAuthorizationError(
            "run checkpoint destination is a symlink"
        )

    return {
        "dataset_id": authorized_dataset,
        "experimental_seed": authorized_seed,
        "attempt": authorized_attempt,
        "run_identity": identity,
        "scientific_baseline_commit": baseline,
        "executable_source_commit": executable,
        "checkpoint_directory": expected_directory,
        "authorization_id": authorization_id,
        "resources": dict(record["resources"]),
    }


def _require_executable_source(record, repository_root):
    executable = record["executable_source_commit"]

    try:
        state = capture_git_state(repository_root)
    except (GitProvenanceError, OSError) as exc:
        raise GANProductionAuthorizationError(
            "executable source identity is unavailable"
        ) from exc

    if state.dirty:
        raise GANProductionAuthorizationError(
            "executable source working tree is dirty"
        )

    if state.commit != executable:
        raise GANProductionAuthorizationError(
            "executable source differs from the authorization"
        )


def parse_gan_slurm_allocation(raw):
    """Parse one ``scontrol show job -o`` line into a field mapping."""
    if type(raw) is not str:
        raise GANProductionAuthorizationError(
            "Slurm allocation record is invalid"
        )

    lines = [line for line in raw.splitlines() if line.strip()]

    if len(lines) != 1:
        raise GANProductionAuthorizationError(
            "Slurm allocation record is not a single line"
        )

    fields = dict(SLURM_FIELD.findall(lines[0]))

    if not fields:
        raise GANProductionAuthorizationError(
            "Slurm allocation record is empty"
        )

    return fields


def capture_gan_slurm_allocation():
    """Read the actual allocation for the current Slurm job."""
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()

    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise GANProductionAuthorizationError(
            "a Slurm job identity is required"
        )

    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        raise GANProductionAuthorizationError(
            "Slurm allocation cannot be verified"
        ) from exc

    if completed.returncode != 0:
        raise GANProductionAuthorizationError(
            "Slurm allocation lookup failed"
        )

    allocation = parse_gan_slurm_allocation(completed.stdout)

    if allocation.get("JobId") != job_id:
        raise GANProductionAuthorizationError(
            "Slurm allocation identity does not match the job"
        )

    return allocation


def _integer_field(allocation, name):
    value = allocation.get(name)

    if type(value) is not str or re.fullmatch(r"[0-9]+", value) is None:
        raise GANProductionAuthorizationError(
            f"Slurm allocation field is missing: {name}"
        )

    return int(value)


def _memory_mib(value):
    if type(value) is not str:
        raise GANProductionAuthorizationError(
            "Slurm memory value is missing"
        )

    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value.strip().upper())

    if match is None:
        raise GANProductionAuthorizationError(
            "Slurm memory value is malformed"
        )

    amount = int(match.group(1))
    unit = match.group(2)
    factor = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
    }[unit]
    result = amount * factor

    if not float(result).is_integer():
        raise GANProductionAuthorizationError(
            "Slurm memory value is not integral"
        )

    return int(result)


def _walltime_seconds(value):
    if type(value) is not str:
        raise GANProductionAuthorizationError(
            "Slurm walltime is missing"
        )

    text = value.strip()

    if re.fullmatch(r"[0-9]+", text):
        return int(text) * 60

    days = 0

    if "-" in text:
        prefix, text = text.split("-", 1)
        days = int(prefix)

    parts = text.split(":")

    if len(parts) == 3:
        hours, minutes, seconds = (int(part) for part in parts)
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = (int(part) for part in parts)
    else:
        raise GANProductionAuthorizationError(
            "Slurm walltime is malformed"
        )

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _tres_fields(value):
    fields = {}

    if type(value) is not str:
        return fields

    for token in value.split(","):
        if "=" not in token:
            continue
        name, field = token.split("=", 1)
        fields[name.strip()] = field.strip()

    return fields


def _gpu_requested(*values):
    for value in values:
        if type(value) is not str or "gpu" not in value.lower():
            continue

        counts = [
            int(number)
            for number in re.findall(
                r"gpu[^0-9]*([0-9]+)", value.lower(),
            )
        ]

        if not counts or any(count > 0 for count in counts):
            return True

    return False


def validate_gan_slurm_allocation(allocation, resources):
    """Require the actual allocation to match the authorized ceilings."""
    if type(allocation) is not dict:
        raise GANProductionAuthorizationError(
            "Slurm allocation record is invalid"
        )

    if allocation.get("Partition") != resources["partition"]:
        raise GANProductionAuthorizationError(
            "Slurm partition differs from the authorization"
        )

    if allocation.get("JobState") not in ("RUNNING", "COMPLETING"):
        raise GANProductionAuthorizationError(
            "Slurm job is not running"
        )

    if _integer_field(allocation, "NumNodes") != resources["nodes"]:
        raise GANProductionAuthorizationError(
            "Slurm node count differs from the authorization"
        )

    if _integer_field(allocation, "NumTasks") != resources["tasks"]:
        raise GANProductionAuthorizationError(
            "Slurm task count differs from the authorization"
        )

    if _integer_field(allocation, "NumCPUs") != (
        resources["cpus_per_task"] * resources["tasks"]
    ):
        raise GANProductionAuthorizationError(
            "Slurm CPU count differs from the authorization"
        )

    tres = _tres_fields(allocation.get("TRES"))

    tres_cpu = tres.get("cpu")

    if tres_cpu is not None:
        if (
            re.fullmatch(r"[0-9]+", tres_cpu) is None
            or int(tres_cpu) != resources["cpus_per_task"]
        ):
            raise GANProductionAuthorizationError(
                "Slurm TRES CPU count differs from the authorization"
            )

    tres_node = tres.get("node")

    if tres_node is not None:
        if (
            re.fullmatch(r"[0-9]+", tres_node) is None
            or int(tres_node) != resources["nodes"]
        ):
            raise GANProductionAuthorizationError(
                "Slurm TRES node count differs from the authorization"
            )

    memory = tres.get("mem")

    if memory is None:
        memory = allocation.get("MinMemoryNode")

    if memory is None:
        memory = allocation.get("MinMemoryPerNode")

    if _memory_mib(memory) != resources["memory_gb"] * 1024:
        raise GANProductionAuthorizationError(
            "Slurm memory differs from the authorization"
        )

    if _walltime_seconds(
        allocation.get("TimeLimit")
    ) != resources["walltime_seconds"]:
        raise GANProductionAuthorizationError(
            "Slurm walltime differs from the authorization"
        )

    if resources["gpu_enabled"] is not False or _gpu_requested(
        allocation.get("TRES"), allocation.get("Gres"),
    ):
        raise GANProductionAuthorizationError(
            "Slurm allocation requests a GPU"
        )

    if _integer_field(allocation, "Requeue") != (
        resources["automatic_retries"]
    ):
        raise GANProductionAuthorizationError(
            "Slurm requeue policy differs from the authorization"
        )

    restarts = allocation.get("Restarts")

    if restarts is not None and restarts != "0":
        raise GANProductionAuthorizationError(
            "Slurm job has restarted unexpectedly"
        )

    if resources["max_concurrent_jobs"] != 1:
        raise GANProductionAuthorizationError(
            "authorized concurrency is unsupported"
        )


def require_gan_production_scope(
    *,
    dataset_id,
    experimental_seed,
    checkpoint_directory,
    repository_root,
):
    """Validate the single external authorization against one planned run."""
    if type(dataset_id) is not str:
        raise GANProductionAuthorizationError(
            "run dataset identity must be a string"
        )

    if type(experimental_seed) is not int:
        raise GANProductionAuthorizationError(
            "run experimental seed must be an integer"
        )

    record = _load_authorization()

    scope = _validate_authorization_identity(
        record,
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        checkpoint_directory=checkpoint_directory,
    )

    _require_executable_source(scope, repository_root)

    allocation = capture_gan_slurm_allocation()

    validate_gan_slurm_allocation(
        allocation,
        scope["resources"],
    )

    return GANProductionScope(**scope)
