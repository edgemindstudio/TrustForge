"""Governed Diffusion production synthesis and restoration entry.

One invocation authorizes one explicit dataset/seed/attempt/class/range,
restores the published validation-best checkpoint in a fresh process, and
(only in synthesis mode) generates and durably publishes exactly that range.

Activation is default-deny. The source gate is necessary but not sufficient:
an execution-specific external authorization record, a clean committed
checkout, the approved synthesis recipe, the expected checkpoint identity and
the actual Slurm allocation must all agree before the checkpoint is loaded or
a model is built.

This module never trains, retrains, evaluates, selects checkpoints or submits
jobs. It does not introduce DDIM, guidance, observation noise or alternative
generation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

import numpy as np

from trustforge.models.paper01_revision_2026.diffusion_pool_publication import (
    ACCEPTED_SAMPLING_POLICY,
    CANONICAL_SHAPE,
    DiffusionRangeIdentity,
    PRIMARY_SAMPLES_PER_CLASS,
    publish_diffusion_range,
)
from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_entry import (
    authoritative_class_mapping,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    SynthesisRequest,
    load_synthesis_recipe,
    restore_diffusion_model_for_synthesis,
    synthesize_diffusion_batch,
)
from trustforge.provenance.git import capture_git_state


SCHEMA = "trustforge.paper01.diffusion-production-synthesis-authorization.v1"
EXECUTION_SCHEMA = (
    "trustforge.paper01.diffusion-production-synthesis-execution.v1"
)
PAPER = "paper01"
REVISION = "revision_2026"
FAMILY = "diffusion"
RECIPE_RELATIVE_PATH = (
    "studies/paper01_benchmark/revision_2026/synthesis_linkage_recipe.json"
)
REVISION_PACKAGE = (
    Path("src") / "trustforge" / "models" / "paper01_revision_2026"
)
ACCEPTED_MODEL_BUILDER = Path("diffusion") / "models.py"
MODES = ("synthesis", "restoration")

# Activation boundary. This is not a command-line option. Enabling it is a
# separate researcher authorization decision.
DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED = True

SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)
SLURM_FIELD_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_/]*)=([^\s]*)")

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

AUTHORIZATION_FIELDS = frozenset({
    "schema",
    "paper",
    "revision",
    "family",
    "mode",
    "dataset_id",
    "experimental_seed",
    "attempt",
    "class_id",
    "sample_start",
    "sample_stop",
    "checkpoint_directory",
    "checkpoint_manifest_sha256",
    "checkpoint_state_sha256",
    "checkpoint_training_source_commit",
    "synthesis_source_commit",
    "recipe_sha256",
    "output_destination",
    "authorization_id",
    "resources",
})


def _account_home() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except (ImportError, KeyError):
        return Path.home().resolve()


ACCOUNT_HOME = _account_home()
AUTHORIZATION_ROOT = (
    ACCOUNT_HOME
    / "trustforge_revision_2026_artifacts"
    / "paper01"
    / "diffusion"
    / "authorizations"
)


class DiffusionProductionSynthesisError(RuntimeError):
    """Raised when a governed Diffusion synthesis request is invalid."""


class DiffusionProductionAuthorizationError(PermissionError):
    """Raised when production synthesis is absent or not authorized."""


def _exact_integer(value, *, name, minimum=0):
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(name + " must be an exact integer")
    if value < minimum:
        raise ValueError(name + " must be >= " + str(minimum))
    return value


def _require_sha256(value, *, name):
    if type(value) is not str or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _require_commit(value, *, name):
    if (
        type(value) is not str
        or SOURCE_COMMIT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(name + " must be a full lowercase SHA-1")
    return value


def require_diffusion_production_synthesis_authorization():
    """Fail closed unless the production synthesis boundary is enabled."""
    if DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED is not True:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production synthesis is not authorized"
        )


@dataclass(frozen=True)
class DiffusionProductionSynthesisRequest:
    """Explicit identity for one bounded production synthesis range."""

    dataset_id: str
    experimental_seed: int
    attempt: int
    expected_authorization_id: str
    class_id: int
    sample_start: int
    sample_stop: int
    expected_synthesis_source_commit: str
    repository_root: Path
    checkpoint_directory: Path
    expected_checkpoint_manifest_sha256: str
    expected_checkpoint_state_sha256: str
    expected_checkpoint_training_source_commit: str
    expected_recipe_sha256: str
    output_destination: Path

    def __post_init__(self):
        if self.dataset_id not in DATASET_CLASS_COUNTS:
            raise ValueError("unsupported governed dataset ID")
        seed = require_governed_experimental_seed(self.experimental_seed)
        attempt = _exact_integer(self.attempt, name="attempt", minimum=1)
        if (
            type(self.expected_authorization_id) is not str
            or AUTHORIZATION_ID_PATTERN.fullmatch(
                self.expected_authorization_id
            )
            is None
        ):
            raise ValueError("expected authorization identity is invalid")
        class_id = _exact_integer(self.class_id, name="class_id")
        if class_id >= DATASET_CLASS_COUNTS[self.dataset_id]:
            raise ValueError(
                "class_id exceeds the governed class universe"
            )
        start = _exact_integer(
            self.sample_start,
            name="sample_start",
            minimum=0,
        )
        stop = _exact_integer(
            self.sample_stop,
            name="sample_stop",
            minimum=1,
        )
        if start >= stop:
            raise ValueError("sample_start must be less than sample_stop")
        if stop > PRIMARY_SAMPLES_PER_CLASS:
            raise ValueError(
                "sample_stop exceeds the governed 2,000-sample class budget"
            )
        source_commit = _require_commit(
            self.expected_synthesis_source_commit,
            name="expected synthesis source commit",
        )
        repository_root = Path(self.repository_root).expanduser()
        checkpoint_directory = Path(
            self.checkpoint_directory
        ).expanduser()
        output_destination = Path(self.output_destination).expanduser()
        manifest = _require_sha256(
            self.expected_checkpoint_manifest_sha256,
            name="expected checkpoint manifest sha256",
        )
        state = _require_sha256(
            self.expected_checkpoint_state_sha256,
            name="expected checkpoint state sha256",
        )
        training_commit = _require_commit(
            self.expected_checkpoint_training_source_commit,
            name="expected checkpoint training source commit",
        )
        recipe = _require_sha256(
            self.expected_recipe_sha256,
            name="expected recipe sha256",
        )
        object.__setattr__(self, "experimental_seed", seed)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "sample_start", start)
        object.__setattr__(self, "sample_stop", stop)
        object.__setattr__(
            self, "expected_synthesis_source_commit", source_commit,
        )
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(
            self, "checkpoint_directory", checkpoint_directory,
        )
        object.__setattr__(self, "output_destination", output_destination)
        if not output_destination.parent.is_dir():
            raise ValueError(
                "output destination parent directory is unavailable"
            )
        if output_destination.exists() or output_destination.is_symlink():
            raise FileExistsError(
                "output destination already exists"
            )
        object.__setattr__(
            self, "expected_checkpoint_manifest_sha256", manifest,
        )
        object.__setattr__(
            self, "expected_checkpoint_state_sha256", state,
        )
        object.__setattr__(
            self, "expected_checkpoint_training_source_commit",
            training_commit,
        )
        object.__setattr__(self, "expected_recipe_sha256", recipe)

    @property
    def run_identity(self):
        return (
            f"commit{self.expected_synthesis_source_commit[:7]}"
            f"_attempt{self.attempt:02d}"
        )

    @property
    def sample_indices(self):
        return tuple(range(self.sample_start, self.sample_stop))


@dataclass(frozen=True)
class DiffusionRestorationVerificationRequest:
    """Explicit identity for one restoration-only verification."""

    dataset_id: str
    experimental_seed: int
    attempt: int
    expected_authorization_id: str
    expected_synthesis_source_commit: str
    repository_root: Path
    checkpoint_directory: Path
    expected_checkpoint_manifest_sha256: str
    expected_checkpoint_state_sha256: str
    expected_checkpoint_training_source_commit: str
    expected_recipe_sha256: str

    def __post_init__(self):
        if self.dataset_id not in DATASET_CLASS_COUNTS:
            raise ValueError("unsupported governed dataset ID")
        seed = require_governed_experimental_seed(self.experimental_seed)
        attempt = _exact_integer(self.attempt, name="attempt", minimum=1)
        if (
            type(self.expected_authorization_id) is not str
            or AUTHORIZATION_ID_PATTERN.fullmatch(
                self.expected_authorization_id
            )
            is None
        ):
            raise ValueError("expected authorization identity is invalid")
        source_commit = _require_commit(
            self.expected_synthesis_source_commit,
            name="expected synthesis source commit",
        )
        object.__setattr__(self, "experimental_seed", seed)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(
            self, "expected_synthesis_source_commit", source_commit,
        )
        object.__setattr__(
            self,
            "repository_root",
            Path(self.repository_root).expanduser(),
        )
        object.__setattr__(
            self,
            "checkpoint_directory",
            Path(self.checkpoint_directory).expanduser(),
        )
        object.__setattr__(
            self,
            "expected_checkpoint_manifest_sha256",
            _require_sha256(
                self.expected_checkpoint_manifest_sha256,
                name="expected checkpoint manifest sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_checkpoint_state_sha256",
            _require_sha256(
                self.expected_checkpoint_state_sha256,
                name="expected checkpoint state sha256",
            ),
        )
        object.__setattr__(
            self,
            "expected_checkpoint_training_source_commit",
            _require_commit(
                self.expected_checkpoint_training_source_commit,
                name="expected checkpoint training source commit",
            ),
        )
        object.__setattr__(
            self,
            "expected_recipe_sha256",
            _require_sha256(
                self.expected_recipe_sha256,
                name="expected recipe sha256",
            ),
        )

    @property
    def run_identity(self):
        return (
            f"commit{self.expected_synthesis_source_commit[:7]}"
            f"_attempt{self.attempt:02d}"
        )


@dataclass(frozen=True)
class DiffusionProductionSynthesisGrant:
    """Validated execution-specific authorization record."""

    authorization_id: str
    mode: str
    dataset_id: str
    experimental_seed: int
    attempt: int
    class_id: int | None
    sample_start: int | None
    sample_stop: int | None
    checkpoint_directory: Path
    checkpoint_manifest_sha256: str
    checkpoint_state_sha256: str
    checkpoint_training_source_commit: str
    synthesis_source_commit: str
    recipe_sha256: str
    output_destination: Path | None
    resources: dict


@dataclass(frozen=True)
class SynthesisSourceBinding:
    """Clean-checkout source identity for one execution."""

    requested_source_commit: str
    resolved_source_commit: str
    repository_root: Path


@dataclass(frozen=True)
class ModuleBinding:
    """Verified Revision-2026 and model-builder import identity."""

    module_paths: dict
    model_builder_path: str


def authorization_path(request, *, mode):
    if mode not in MODES:
        raise ValueError("unsupported Diffusion execution mode")
    return (
        Path(AUTHORIZATION_ROOT)
        / (
            f"{request.dataset_id}_seed{request.experimental_seed}"
            f"_attempt{request.attempt:02d}_{mode}.json"
        )
    )


def _reject_duplicate_keys(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(
                "duplicate Diffusion authorization JSON field: " + key
            )
        record[key] = value
    return record


def _reject_constant(value):
    raise ValueError(
        "non-finite Diffusion authorization JSON constant: " + str(value)
    )


def _require_regular_authorization(path):
    if path.is_symlink():
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization must not be a symlink"
        )
    if not path.is_file():
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization is absent"
        )
    metadata = path.stat()
    if metadata.st_uid != os.getuid():
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization has a different owner"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization is group/world writable"
        )
    return path


def load_authorization_record(request, *, mode):
    """Read one external authorization record or fail closed."""
    path = _require_regular_authorization(
        authorization_path(request, mode=mode)
    )
    try:
        record = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization is not valid JSON"
        ) from exc
    if type(record) is not dict:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization root must be an object"
        )
    if set(record) != AUTHORIZATION_FIELDS:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization fields are invalid"
        )
    return record


def _validate_resources(raw):
    if type(raw) is not dict or set(raw) != RESOURCE_FIELDS:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production resource envelope fields are invalid"
        )
    partition = raw["partition"]
    if type(partition) is not str or not partition.strip():
        raise DiffusionProductionAuthorizationError(
            "Diffusion production partition is invalid"
        )
    for name in (
        "nodes", "tasks", "cpus_per_task", "memory_gb",
        "walltime_seconds", "max_concurrent_jobs",
    ):
        _exact_integer(raw[name], name=name, minimum=1)
    if type(raw["gpu_enabled"]) is not bool:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production gpu_enabled must be boolean"
        )
    _exact_integer(
        raw["automatic_retries"], name="automatic_retries", minimum=0,
    )
    if raw["automatic_retries"] != 0:
        raise DiffusionProductionAuthorizationError(
            "Diffusion production automatic retries must be zero"
        )
    return dict(raw)


def parse_slurm_allocation(raw):
    """Parse one ``scontrol show job -o`` line into a field mapping."""
    if type(raw) is not str:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation record is invalid"
        )
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation record must contain exactly one job"
        )
    record = {}
    for key, value in SLURM_FIELD_PATTERN.findall(lines[0]):
        record[key] = value
    if record.get("JobId") is None:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation record lacks JobId"
        )
    return record


def capture_slurm_allocation():
    """Return the actual Slurm allocation for the current job."""
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise DiffusionProductionAuthorizationError(
            "a Slurm job identity is required"
        )
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DiffusionProductionAuthorizationError(
            "unable to read the Slurm allocation"
        ) from exc
    if completed.returncode != 0:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation is unavailable"
        )
    allocation = parse_slurm_allocation(completed.stdout)
    if allocation.get("JobId") != job_id:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation identity does not match the job"
        )
    return allocation


def _parse_tres(value, *, field):
    if type(value) is not str or not value.strip():
        raise DiffusionProductionAuthorizationError(
            field + " is missing or empty"
        )
    result = {}
    for item in value.split(","):
        token = item.strip()
        if not token or "=" not in token:
            raise DiffusionProductionAuthorizationError(
                field + " is malformed"
            )
        key, raw = token.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key or not raw:
            raise DiffusionProductionAuthorizationError(
                field + " is malformed"
            )
        if key in result:
            raise DiffusionProductionAuthorizationError(
                field + " contains a duplicate key: " + key
            )
        result[key] = raw
    if not result:
        raise DiffusionProductionAuthorizationError(
            field + " is empty"
        )
    return result


def _memory_gb(value):
    match = re.fullmatch(
        r"([0-9]+)([KMGTP]?)", (value or "").strip().upper(),
    )
    if match is None:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocated memory is invalid"
        )
    number = int(match.group(1))
    unit = match.group(2)
    factors = {
        "": 1.0 / (1024 ** 3),
        "K": 1.0 / (1024 ** 2),
        "M": 1.0 / 1024,
        "G": 1.0,
        "T": 1024.0,
        "P": 1024.0 * 1024.0,
    }
    return number * factors[unit]


def _tres_integer(fields, name, *, field):
    value = fields.get(name)
    if value is None:
        return None
    if re.fullmatch(r"[0-9]+", value) is None:
        raise DiffusionProductionAuthorizationError(
            field + " " + name + " value is malformed"
        )
    return int(value)


def _tres_memory_gb(fields, *, field):
    value = fields.get("mem")
    if value is None:
        raise DiffusionProductionAuthorizationError(
            field + " lacks allocated memory"
        )
    return _memory_gb(value)


def _tres_has_gpu(fields):
    return any("gpu" in key.lower() for key in fields)


def _require_tres_matches_envelope(
    fields, *, field, resources, require_gpu,
):
    allocated_cpu = _tres_integer(fields, "cpu", field=field)
    expected_cpus = resources["cpus_per_task"] * resources["tasks"]
    if allocated_cpu is not None and allocated_cpu != expected_cpus:
        raise DiffusionProductionAuthorizationError(
            field + " CPU count differs from the authorized envelope"
        )
    allocated_nodes = _tres_integer(fields, "node", field=field)
    if (
        allocated_nodes is not None
        and allocated_nodes != resources["nodes"]
    ):
        raise DiffusionProductionAuthorizationError(
            field + " node count differs from the authorized envelope"
        )
    if "mem" in fields:
        if abs(
            _tres_memory_gb(fields, field=field)
            - float(resources["memory_gb"])
        ) > 1e-9:
            raise DiffusionProductionAuthorizationError(
                "Slurm allocated memory differs from the authorized envelope"
            )
    has_gpu = _tres_has_gpu(fields)
    if require_gpu or has_gpu:
        if has_gpu != resources["gpu_enabled"]:
            raise DiffusionProductionAuthorizationError(
                "Slurm GPU allocation differs from the authorization"
            )


def _allocated_tres(allocation, resources):
    """Return authoritative allocated TRES, preferring AllocTRES.

    ``ReqTRES`` is requested resources, not allocation evidence, and is never
    substituted when allocated evidence is absent.
    """
    has_alloc = "AllocTRES" in allocation
    has_legacy = "TRES" in allocation

    if has_alloc:
        allocated = _parse_tres(
            allocation["AllocTRES"], field="AllocTRES",
        )
        _require_tres_matches_envelope(
            allocated,
            field="AllocTRES",
            resources=resources,
            require_gpu=True,
        )
        if has_legacy:
            legacy = _parse_tres(
                allocation["TRES"], field="TRES",
            )
            _require_tres_matches_envelope(
                legacy,
                field="TRES",
                resources=resources,
                require_gpu=False,
            )
        _tres_memory_gb(allocated, field="AllocTRES")
        return allocated

    if has_legacy:
        legacy = _parse_tres(allocation["TRES"], field="TRES")
        _require_tres_matches_envelope(
            legacy,
            field="TRES",
            resources=resources,
            require_gpu=True,
        )
        _tres_memory_gb(legacy, field="TRES")
        return legacy

    raise DiffusionProductionAuthorizationError(
        "Slurm allocation lacks allocated-resource evidence"
    )


def _time_limit_seconds(value):
    parts = (value or "").split("-")
    if len(parts) == 1:
        days = 0
        clock = parts[0]
    elif len(parts) == 2:
        days = int(parts[0])
        clock = parts[1]
    else:
        raise DiffusionProductionAuthorizationError(
            "Slurm time limit is invalid"
        )
    fields = clock.split(":")
    if len(fields) == 3:
        hours, minutes, seconds = (int(item) for item in fields)
    elif len(fields) == 2:
        hours = 0
        minutes, seconds = (int(item) for item in fields)
    else:
        raise DiffusionProductionAuthorizationError(
            "Slurm time limit is invalid"
        )
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _allocation_integer(allocation, name):
    raw = allocation.get(name)
    if type(raw) is not str or re.fullmatch(r"[0-9]+", raw) is None:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation " + name + " is invalid"
        )
    return int(raw)


def validate_allocation(*, resources, allocation):
    """Require the actual allocation to satisfy the authorized envelope."""
    if type(allocation) is not dict:
        raise DiffusionProductionAuthorizationError(
            "Slurm allocation mapping is invalid"
        )
    for name in ("Partition", "NumNodes", "NumTasks", "NumCPUs", "TimeLimit"):
        if name not in allocation:
            raise DiffusionProductionAuthorizationError(
                "Slurm allocation lacks " + name
            )
    if allocation["Partition"] != resources["partition"]:
        raise DiffusionProductionAuthorizationError(
            "Slurm partition differs from the authorization"
        )
    if _allocation_integer(allocation, "NumNodes") != resources["nodes"]:
        raise DiffusionProductionAuthorizationError(
            "Slurm node count differs from the authorization"
        )
    if _allocation_integer(allocation, "NumTasks") != resources["tasks"]:
        raise DiffusionProductionAuthorizationError(
            "Slurm task count differs from the authorization"
        )
    expected_cpus = resources["cpus_per_task"] * resources["tasks"]
    if _allocation_integer(allocation, "NumCPUs") != expected_cpus:
        raise DiffusionProductionAuthorizationError(
            "Slurm CPU allocation differs from the authorization"
        )
    if _time_limit_seconds(allocation["TimeLimit"]) > resources[
        "walltime_seconds"
    ]:
        raise DiffusionProductionAuthorizationError(
            "Slurm walltime exceeds the authorization"
        )
    _allocated_tres(allocation, resources)
    gres = allocation.get("Gres") or allocation.get("AllocGRES") or ""
    gpu_in_gres = (
        "gpu" in gres and gres not in ("(null)", "N/A", "")
    )
    if gpu_in_gres and not resources["gpu_enabled"]:
        raise DiffusionProductionAuthorizationError(
            "Slurm GPU allocation differs from the authorization"
        )
    if allocation.get("Requeue") not in (None, "0"):
        raise DiffusionProductionAuthorizationError(
            "Slurm requeue must be disabled"
        )
    if allocation.get("Restarts") not in (None, "0"):
        raise DiffusionProductionAuthorizationError(
            "Slurm restart count must be zero"
        )
    if _allocation_integer(allocation, "NumTasks") < 1:
        raise DiffusionProductionAuthorizationError(
            "Slurm task count is invalid"
        )
    return dict(allocation)


def _require_equal(name, value, expected):
    if value != expected:
        raise DiffusionProductionAuthorizationError(
            name + " differs from the request"
        )


def validate_authorization(request, *, allocation, mode):
    """Validate the external authorization before any checkpoint access."""
    record = load_authorization_record(request, mode=mode)
    if (
        record["schema"] != SCHEMA
        or record["paper"] != PAPER
        or record["revision"] != REVISION
        or record["family"] != FAMILY
        or record["mode"] != mode
    ):
        raise DiffusionProductionAuthorizationError(
            "Diffusion production authorization identity is invalid"
        )
    _require_equal(
        "authorized dataset", record["dataset_id"], request.dataset_id,
    )
    _require_equal(
        "authorized seed",
        record["experimental_seed"],
        request.experimental_seed,
    )
    _require_equal(
        "authorized attempt", record["attempt"], request.attempt,
    )
    _require_equal(
        "authorized authorization_id",
        record["authorization_id"],
        request.expected_authorization_id,
    )
    if type(record["checkpoint_directory"]) is not str:
        raise DiffusionProductionAuthorizationError(
            "authorized checkpoint directory is invalid"
        )
    if (
        Path(record["checkpoint_directory"]).expanduser().resolve()
        != request.checkpoint_directory.resolve()
    ):
        raise DiffusionProductionAuthorizationError(
            "authorized checkpoint directory differs from the request"
        )
    for field in (
        "checkpoint_manifest_sha256",
        "checkpoint_state_sha256",
        "checkpoint_training_source_commit",
        "synthesis_source_commit",
        "recipe_sha256",
    ):
        _require_equal(
            "authorized " + field,
            record[field],
            getattr(request, "expected_" + field),
        )
    if mode == "synthesis":
        _require_equal(
            "authorized class", record["class_id"], request.class_id,
        )
        _require_equal(
            "authorized sample_start",
            record["sample_start"],
            request.sample_start,
        )
        _require_equal(
            "authorized sample_stop",
            record["sample_stop"],
            request.sample_stop,
        )
        if type(record["output_destination"]) is not str:
            raise DiffusionProductionAuthorizationError(
                "authorized output destination is invalid"
            )
        if (
            Path(record["output_destination"]).expanduser().resolve()
            != request.output_destination.resolve()
        ):
            raise DiffusionProductionAuthorizationError(
                "authorized output destination differs from the request"
            )
    else:
        if (
            record["class_id"] is not None
            or record["sample_start"] is not None
            or record["sample_stop"] is not None
            or record["output_destination"] is not None
        ):
            raise DiffusionProductionAuthorizationError(
                "restoration authorization must not bind sampling scope"
            )
    if AUTHORIZATION_ID_PATTERN.fullmatch(
        record["authorization_id"]
    ) is None:
        raise DiffusionProductionAuthorizationError(
            "authorized authorization_id is invalid"
        )
    for name in (
        "checkpoint_manifest_sha256",
        "checkpoint_state_sha256",
        "recipe_sha256",
    ):
        _require_sha256(record[name], name=name)
    for name in (
        "checkpoint_training_source_commit",
        "synthesis_source_commit",
    ):
        _require_commit(record[name], name=name)
    resources = _validate_resources(record["resources"])
    validate_allocation(resources=resources, allocation=allocation)
    return DiffusionProductionSynthesisGrant(
        authorization_id=record["authorization_id"],
        mode=mode,
        dataset_id=request.dataset_id,
        experimental_seed=request.experimental_seed,
        attempt=request.attempt,
        class_id=record["class_id"],
        sample_start=record["sample_start"],
        sample_stop=record["sample_stop"],
        checkpoint_directory=request.checkpoint_directory,
        checkpoint_manifest_sha256=record["checkpoint_manifest_sha256"],
        checkpoint_state_sha256=record["checkpoint_state_sha256"],
        checkpoint_training_source_commit=record[
            "checkpoint_training_source_commit"
        ],
        synthesis_source_commit=record["synthesis_source_commit"],
        recipe_sha256=record["recipe_sha256"],
        output_destination=(
            request.output_destination if mode == "synthesis" else None
        ),
        resources=resources,
    )


def require_source_binding(request) -> SynthesisSourceBinding:
    """Require a clean checkout at the expected synthesis source commit."""
    state = capture_git_state(request.repository_root)
    if state.dirty:
        raise DiffusionProductionSynthesisError(
            "Diffusion production synthesis requires a clean working tree "
            "and index"
        )
    if state.commit != request.expected_synthesis_source_commit:
        raise DiffusionProductionSynthesisError(
            "source revision does not match the expected synthesis commit"
        )
    return SynthesisSourceBinding(
        requested_source_commit=request.expected_synthesis_source_commit,
        resolved_source_commit=state.commit,
        repository_root=Path(state.repo_root),
    )


def _load_revision_modules():
    from trustforge.models.paper01_revision_2026 import (
        diffusion_checkpoint_bundle,
        diffusion_checkpoint_state,
        diffusion_ddpm_math,
        diffusion_governed_sampling,
        experimental_seed_policy,
        gan_checkpoint_manifest,
        synthesis_linkage,
    )

    return {
        "diffusion_checkpoint_bundle": diffusion_checkpoint_bundle,
        "diffusion_checkpoint_state": diffusion_checkpoint_state,
        "diffusion_ddpm_math": diffusion_ddpm_math,
        "diffusion_governed_sampling": diffusion_governed_sampling,
        "experimental_seed_policy": experimental_seed_policy,
        "gan_checkpoint_manifest": gan_checkpoint_manifest,
        "synthesis_linkage": synthesis_linkage,
    }


def _load_model_builder_module():
    from diffusion import models as model_builder

    return model_builder


def require_module_binding(
    *, repository_root, modules=None, model_builder=None,
) -> ModuleBinding:
    """Require accepted Revision-2026 and Diffusion builder imports."""
    root = Path(repository_root).resolve()
    package = (root / REVISION_PACKAGE).resolve()
    accepted_builder = (root / ACCEPTED_MODEL_BUILDER).resolve()
    if modules is None:
        modules = _load_revision_modules()
    if model_builder is None:
        model_builder = _load_model_builder_module()
    if not modules:
        raise DiffusionProductionSynthesisError(
            "no Revision-2026 modules were bound"
        )
    if not accepted_builder.is_file():
        raise DiffusionProductionSynthesisError(
            "accepted Diffusion model builder is missing: "
            + str(accepted_builder)
        )
    paths = {}
    for name, module in modules.items():
        path = Path(getattr(module, "__file__", "")).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise DiffusionProductionSynthesisError(
                name + " is not bound to the Revision-2026 module path"
            )
        paths[name] = str(path)
    builder_path = Path(
        getattr(model_builder, "__file__", "")
    ).resolve()
    if builder_path != accepted_builder:
        raise DiffusionProductionSynthesisError(
            "Diffusion model builder is not the accepted source: "
            + str(builder_path)
        )
    if not callable(
        getattr(model_builder, "build_diffusion_model", None)
    ):
        raise DiffusionProductionSynthesisError(
            "Diffusion model builder lacks build_diffusion_model"
        )
    return ModuleBinding(
        module_paths=paths,
        model_builder_path=str(builder_path),
    )


def _build_model_factory(model_builder, num_classes):
    from trustforge.models.paper01_revision_2026.diffusion_governed_training import (
        APPROVED_BASE_FILTERS,
        APPROVED_DEPTH,
        APPROVED_IMAGE_SHAPE,
        APPROVED_TIME_EMBED_DIM,
    )

    if tuple(APPROVED_IMAGE_SHAPE) != CANONICAL_SHAPE:
        raise DiffusionProductionSynthesisError(
            "accepted Diffusion image shape changed"
        )

    def factory():
        return model_builder.build_diffusion_model(
            img_shape=APPROVED_IMAGE_SHAPE,
            num_classes=num_classes,
            base_filters=APPROVED_BASE_FILTERS,
            depth=APPROVED_DEPTH,
            time_emb_dim=APPROVED_TIME_EMBED_DIM,
        )

    return factory


def _authorize(request, *, mode):
    require_diffusion_production_synthesis_authorization()
    mapping = authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )
    if mode == "synthesis" and request.class_id >= len(mapping):
        raise DiffusionProductionSynthesisError(
            "requested class exceeds the governed class universe"
        )
    binding = require_source_binding(request)
    recipe = load_synthesis_recipe(binding.repository_root)
    if recipe.sha256 != request.expected_recipe_sha256:
        raise DiffusionProductionSynthesisError(
            "synthesis recipe identity differs from the request"
        )
    allocation = capture_slurm_allocation()
    grant = validate_authorization(request, allocation=allocation, mode=mode)
    module_binding = require_module_binding(
        repository_root=binding.repository_root,
    )
    return mapping, binding, recipe, allocation, grant, module_binding


def _linkage_request(request, mapping):
    if hasattr(request, "sample_start"):
        class_id = request.class_id
        indices = request.sample_indices
        scope = (
            "production_complete_class"
            if (
                request.sample_start == 0
                and request.sample_stop == PRIMARY_SAMPLES_PER_CLASS
            )
            else "production_batch"
        )
    else:
        class_id = 0
        indices = (0,)
        scope = "production_batch"
    return SynthesisRequest(
        dataset_id=request.dataset_id,
        family=FAMILY,
        experimental_seed=request.experimental_seed,
        class_id=class_id,
        sample_indices=indices,
        class_mapping=mapping,
        scope=scope,
    )


def _checkpoint_identity(request, mapping):
    return {
        "dataset_id": request.dataset_id,
        "experimental_seed": request.experimental_seed,
        "class_mapping": list(mapping),
        "checkpoint_role": "best",
        "checkpoint_sha256": request.expected_checkpoint_state_sha256,
        "source_commit": request.expected_checkpoint_training_source_commit,
    }


def _restoration_report(model):
    weights = model.get_weights()
    return {
        "restoration_status": "VERIFIED",
        "restored_parameter_count": len(weights),
        "restored_element_count": int(
            sum(np.asarray(weight).size for weight in weights)
        ),
        "sampling_performed": False,
        "images_generated": 0,
    }


def _runtime_versions():
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf

        versions["tensorflow"] = tf.__version__
    except Exception:
        versions["tensorflow"] = None
    return versions


def _restore(
    *,
    request,
    mapping,
    recipe,
    model_builder,
    model_binding,
):
    linkage = _linkage_request(request, mapping)
    model_factory = _build_model_factory(
        model_builder, len(mapping),
    )
    model = restore_diffusion_model_for_synthesis(
        prepared=None,
        checkpoint_directory=request.checkpoint_directory,
        expected_manifest_sha256=(
            request.expected_checkpoint_manifest_sha256
        ),
        model_factory=model_factory,
        request=linkage,
        recipe=recipe,
        checkpoint_identity=_checkpoint_identity(request, mapping),
    )
    report = _restoration_report(model)
    report.update({
        "dataset_id": request.dataset_id,
        "experimental_seed": request.experimental_seed,
        "attempt": request.attempt,
        "run_identity": request.run_identity,
        "class_mapping": list(mapping),
        "checkpoint_role": "best",
        "checkpoint_manifest_sha256":
            request.expected_checkpoint_manifest_sha256,
        "checkpoint_state_sha256":
            request.expected_checkpoint_state_sha256,
        "checkpoint_training_source_commit":
            request.expected_checkpoint_training_source_commit,
        "synthesis_source_commit":
            request.expected_synthesis_source_commit,
        "recipe_sha256": recipe.sha256,
        "model_builder_path": model_binding.model_builder_path,
        "module_paths": dict(model_binding.module_paths),
        "runtime_versions": _runtime_versions(),
    })
    return model, linkage, report


def verify_diffusion_restoration(request):
    """Authorize and restore the best checkpoint without generating samples."""
    if not isinstance(request, DiffusionRestorationVerificationRequest):
        raise TypeError(
            "request must be DiffusionRestorationVerificationRequest"
        )
    mapping, binding, recipe, allocation, grant, module_binding = (
        _authorize(request, mode="restoration")
    )
    model_builder = _load_model_builder_module()
    _, _, report = _restore(
        request=request,
        mapping=mapping,
        recipe=recipe,
        model_builder=model_builder,
        model_binding=module_binding,
    )
    report.update({
        "schema": EXECUTION_SCHEMA,
        "status": "RESTORATION_VERIFIED",
        "mode": "restoration",
        "authorization_id": grant.authorization_id,
        "slurm_job_id": allocation.get("JobId"),
        "resources": dict(grant.resources),
        "source_binding": {
            "requested_source_commit": binding.requested_source_commit,
            "resolved_source_commit": binding.resolved_source_commit,
            "repository_root": str(binding.repository_root),
        },
    })
    return report


def execute_diffusion_production_synthesis(request):
    """Authorize, restore, synthesize and publish one bounded range."""
    if not isinstance(request, DiffusionProductionSynthesisRequest):
        raise TypeError("request must be DiffusionProductionSynthesisRequest")
    started = time.perf_counter()
    mapping, binding, recipe, allocation, grant, module_binding = (
        _authorize(request, mode="synthesis")
    )
    if grant.class_id != request.class_id:
        raise DiffusionProductionSynthesisError(
            "authorization class does not match the request"
        )
    model_builder = _load_model_builder_module()
    model, linkage, restoration = _restore(
        request=request,
        mapping=mapping,
        recipe=recipe,
        model_builder=model_builder,
        model_binding=module_binding,
    )
    result = synthesize_diffusion_batch(
        model=model,
        request=linkage,
        recipe=recipe,
        checkpoint_identity=_checkpoint_identity(request, mapping),
    )
    images = np.asarray(result["images"])
    if images.shape != (
        len(linkage.sample_indices), *CANONICAL_SHAPE,
    ):
        raise DiffusionProductionSynthesisError(
            "synthesis output shape differs from the requested range"
        )

    identity = DiffusionRangeIdentity(
        dataset_id=request.dataset_id,
        experimental_seed=request.experimental_seed,
        attempt=request.attempt,
        class_id=request.class_id,
        class_label=mapping[request.class_id],
        sample_start=request.sample_start,
        sample_stop=request.sample_stop,
        checkpoint_manifest_sha256=(
            request.expected_checkpoint_manifest_sha256
        ),
        checkpoint_state_sha256=request.expected_checkpoint_state_sha256,
        checkpoint_training_source_commit=(
            request.expected_checkpoint_training_source_commit
        ),
        synthesis_source_commit=(
            request.expected_synthesis_source_commit
        ),
        recipe_sha256=recipe.sha256,
        sampling_policy=dict(ACCEPTED_SAMPLING_POLICY),
        authorization_id=grant.authorization_id,
        slurm_job_id=allocation.get("JobId"),
        resources=dict(grant.resources),
        scope="production",
        checkpoint_role="best",
    )
    publication = publish_diffusion_range(
        identity=identity,
        images=images,
        destination=request.output_destination,
        expected_checkpoint_manifest_sha256=(
            request.expected_checkpoint_manifest_sha256
        ),
        expected_checkpoint_state_sha256=(
            request.expected_checkpoint_state_sha256
        ),
        expected_recipe_sha256=request.expected_recipe_sha256,
        repository_root=binding.repository_root,
    )
    elapsed = time.perf_counter() - started

    return {
        "schema": EXECUTION_SCHEMA,
        "status": "SYNTHESIS_COMPLETE",
        "mode": "synthesis",
        "dataset_id": request.dataset_id,
        "family": FAMILY,
        "experimental_seed": request.experimental_seed,
        "attempt": request.attempt,
        "run_identity": request.run_identity,
        "class_id": request.class_id,
        "class_label": mapping[request.class_id],
        "sample_start": request.sample_start,
        "sample_stop": request.sample_stop,
        "sample_count": len(linkage.sample_indices),
        "checkpoint_role": "best",
        "checkpoint_manifest_sha256":
            request.expected_checkpoint_manifest_sha256,
        "checkpoint_state_sha256":
            request.expected_checkpoint_state_sha256,
        "checkpoint_training_source_commit":
            request.expected_checkpoint_training_source_commit,
        "synthesis_source_commit":
            request.expected_synthesis_source_commit,
        "recipe_sha256": recipe.sha256,
        "authorization_id": grant.authorization_id,
        "slurm_job_id": allocation.get("JobId"),
        "resources": dict(grant.resources),
        "output_destination": str(request.output_destination),
        "source_binding": {
            "requested_source_commit": binding.requested_source_commit,
            "resolved_source_commit": binding.resolved_source_commit,
            "repository_root": str(binding.repository_root),
        },
        "model_builder_path": module_binding.model_builder_path,
        "module_paths": dict(module_binding.module_paths),
        "runtime_versions": _runtime_versions(),
        "restoration": restoration,
        "publication": publication,
        "elapsed_seconds": elapsed,
        "publication_status": publication["publication_status"],
        "authoritative_artifact": publication["authoritative_artifact"],
    }


__all__ = [
    "ACCEPTED_MODEL_BUILDER",
    "AUTHORIZATION_FIELDS",
    "AUTHORIZATION_ROOT",
    "DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED",
    "DiffusionProductionAuthorizationError",
    "DiffusionProductionSynthesisError",
    "DiffusionProductionSynthesisGrant",
    "DiffusionProductionSynthesisRequest",
    "DiffusionRestorationVerificationRequest",
    "ModuleBinding",
    "RESOURCE_FIELDS",
    "SCHEMA",
    "EXECUTION_SCHEMA",
    "SynthesisSourceBinding",
    "authorization_path",
    "capture_slurm_allocation",
    "execute_diffusion_production_synthesis",
    "load_authorization_record",
    "parse_slurm_allocation",
    "require_diffusion_production_synthesis_authorization",
    "require_module_binding",
    "require_source_binding",
    "validate_allocation",
    "validate_authorization",
    "verify_diffusion_restoration",
]
