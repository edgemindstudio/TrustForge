"""Governed production entry for the Paper 1 revision-2026 RBM sampler.

This module connects the accepted fixed finite-step production sampler to one
explicitly governed execution request. It does not train models, publish
synthetic pools, select checkpoints, enable runtime execution or claim
convergence, mixing sufficiency or stationary-distribution equivalence.

Activation is default-deny. An external execution-specific authorization
record must match the exact dataset, seed, attempt, source commit, checkpoint,
sampling policy, requested sample scope and resource envelope before a
production checkpoint is loaded or sampling is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import operator
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping

from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    METADATA_FILENAME,
    STATE_FILENAME,
    RBMPersistedCheckpoint,
    load_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_sampling import (
    PRIMARY_SAMPLES_PER_CLASS,
    RBMProductionSamplingPolicy,
    RBMProductionSamplingResult,
    sample_production_images,
    validate_production_sampling_policy,
)
from trustforge.provenance import sha256_file
from trustforge.provenance.git import capture_git_state


SCHEMA = "trustforge.paper01.rbm.production-sampling-authorization.v1"
EXECUTION_SCHEMA = (
    "trustforge.paper01.rbm.production-sampling-execution.v1"
)
PAPER = "paper01"
REVISION = "revision_2026"
FAMILY = "restricted_boltzmann"

SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SLURM_FIELD_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_/]*)=([^\s]*)")

REVISION_PACKAGE = (
    Path("src") / "trustforge" / "models" / "paper01_revision_2026"
)
MALNET_MAPPING_PATH = (
    Path("studies")
    / "paper01_benchmark"
    / "audits"
    / "revision_2026"
    / "malnet_image_type_mapping.json"
)

# Approved cumulative RBM synthesis planning ceiling. A single job may not
# exceed this value, but satisfying the per-job bound is only a necessary
# condition; it does not enforce or prove cumulative workload compliance.
CUMULATIVE_RBM_SYNTHESIS_PLANNING_CORE_HOURS = 650

DEFAULT_RESOURCES = {
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

RESOURCE_FIELDS = frozenset(DEFAULT_RESOURCES)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "paper",
        "revision",
        "family",
        "dataset_id",
        "experimental_seed",
        "attempt",
        "run_identity",
        "class_id",
        "sample_start",
        "sample_stop",
        "source_commit",
        "checkpoint_directory",
        "checkpoint_state_sha256",
        "policy_digest",
        "authorization_id",
        "resources",
    }
)

# Activation boundary. This is not a command-line option and must remain
# false. Enabling it is a separate researcher authorization decision.
RBM_PRODUCTION_SAMPLING_ENABLED = False


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
    / "rbm"
    / "authorizations"
)


class RBMProductionSamplingError(RuntimeError):
    """Raised when a governed production-sampling request is invalid."""


class RBMProductionAuthorizationError(PermissionError):
    """Raised when production execution is absent or not authorized."""


def _exact_integer(value: int, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")

    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an exact integer") from exc

    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return result


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")

    return value


def _authorized_total_cpus(resources: Mapping[str, Any]) -> int:
    """Return total allocated CPUs from the authorized task and CPU counts.

    Slurm counts total tasks, so nodes partition the task count; they do not
    multiply it. ``tasks * cpus_per_task`` is the authorized total.
    """

    return resources["tasks"] * resources["cpus_per_task"]


def _individual_job_cpu_core_hours(
    resources: Mapping[str, Any],
) -> float:
    """Return one job's maximum allocated CPU core-hours."""

    return (
        _authorized_total_cpus(resources)
        * resources["walltime_seconds"]
        / 3600.0
    )


def require_rbm_production_sampling_authorization() -> None:
    """Fail closed unless the production activation boundary is enabled."""
    if RBM_PRODUCTION_SAMPLING_ENABLED is not True:
        raise RBMProductionAuthorizationError(
            "RBM production sampling is not authorized"
        )


@dataclass(frozen=True)
class RBMProductionSamplingRequest:
    """Explicit execution identity for one bounded production request."""

    dataset_id: str
    experimental_seed: int
    attempt: int
    class_id: int
    sample_start: int
    sample_stop: int
    expected_source_commit: str
    repository_root: Path
    checkpoint_directory: Path
    expected_checkpoint_state_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str):
            raise TypeError("dataset_id must be a string")

        if self.dataset_id not in DATASET_CLASS_COUNTS:
            raise ValueError("unsupported governed dataset ID")

        seed = require_governed_experimental_seed(self.experimental_seed)
        attempt = _exact_integer(self.attempt, name="attempt", minimum=1)
        class_id = _exact_integer(self.class_id, name="class_id")
        sample_start = _exact_integer(
            self.sample_start,
            name="sample_start",
        )
        sample_stop = _exact_integer(
            self.sample_stop,
            name="sample_stop",
        )

        if sample_start >= sample_stop:
            raise ValueError("sample_start must be less than sample_stop")

        if sample_stop > PRIMARY_SAMPLES_PER_CLASS:
            raise ValueError(
                "sample_stop exceeds the governed 2000-sample class budget"
            )

        if (
            not isinstance(self.expected_source_commit, str)
            or SOURCE_COMMIT_PATTERN.fullmatch(
                self.expected_source_commit
            )
            is None
        ):
            raise ValueError(
                "expected source commit must be a full lowercase SHA-1"
            )

        repository_root = Path(self.repository_root).expanduser()
        checkpoint_directory = Path(self.checkpoint_directory).expanduser()

        _require_sha256(
            self.expected_checkpoint_state_sha256,
            name="expected_checkpoint_state_sha256",
        )

        object.__setattr__(self, "experimental_seed", seed)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "sample_start", sample_start)
        object.__setattr__(self, "sample_stop", sample_stop)
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "checkpoint_directory", checkpoint_directory)

    @property
    def run_identity(self) -> str:
        return (
            f"commit{self.expected_source_commit[:7]}_attempt{self.attempt:02d}"
        )

    @property
    def sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))


@dataclass(frozen=True)
class RBMProductionSamplingGrant:
    """Validated execution-specific authorization record."""

    authorization_id: str
    dataset_id: str
    experimental_seed: int
    attempt: int
    class_id: int
    sample_start: int
    sample_stop: int
    source_commit: str
    checkpoint_directory: Path
    checkpoint_state_sha256: str
    policy_digest: str
    resources: dict


def _load_malnet_mapping(repository_root: Path) -> tuple[str, ...]:
    path = (Path(repository_root).resolve() / MALNET_MAPPING_PATH)

    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"authoritative MalNet class mapping is missing: {path}"
        )

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "authoritative MalNet class mapping is not valid JSON"
        ) from exc

    if not isinstance(record, dict):
        raise ValueError("authoritative MalNet mapping root must be an object")

    count = DATASET_CLASS_COUNTS["malnet_image"]

    if record.get("num_classes") != count:
        raise ValueError("authoritative MalNet class count mismatch")

    id_to_class = record.get("id_to_class")
    class_to_id = record.get("class_to_id")

    if (
        not isinstance(id_to_class, dict)
        or set(id_to_class) != {str(index) for index in range(count)}
    ):
        raise ValueError("authoritative MalNet id_to_class is incomplete")

    labels = tuple(id_to_class[str(index)] for index in range(count))

    if (
        any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            for label in labels
        )
        or len(set(labels)) != count
    ):
        raise ValueError("authoritative MalNet labels are invalid")

    if class_to_id != {
        label: index for index, label in enumerate(labels)
    }:
        raise ValueError(
            "authoritative MalNet forward/reverse mappings disagree"
        )

    return labels


def authoritative_class_mapping(
    *,
    dataset_id: str,
    repository_root: Path,
) -> tuple[str, ...]:
    """Return the accepted authoritative class ordering for one dataset."""

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset ID")

    count = DATASET_CLASS_COUNTS[dataset_id]

    if dataset_id in ("ustc_tfc2016", "cicmaldroid2020"):
        return tuple(str(index) for index in range(count))

    if dataset_id == "malex":
        return ("benign", "malware")

    if dataset_id == "malnet_image":
        return _load_malnet_mapping(repository_root)

    raise AssertionError(dataset_id)


def validate_request_identity(
    request: RBMProductionSamplingRequest,
) -> tuple[str, ...]:
    """Validate the dataset class universe and requested class identity."""

    if not isinstance(request, RBMProductionSamplingRequest):
        raise TypeError("request must be RBMProductionSamplingRequest")

    mapping = authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )

    if request.class_id >= len(mapping):
        raise ValueError(
            "requested class identity exceeds the governed class universe"
        )

    return mapping


def policy_for_request(
    request: RBMProductionSamplingRequest,
    class_mapping: tuple[str, ...],
) -> RBMProductionSamplingPolicy:
    """Build the accepted production policy for one request."""

    return RBMProductionSamplingPolicy(
        dataset_id=request.dataset_id,
        experimental_seed=request.experimental_seed,
        class_mapping=class_mapping,
        checkpoint_state_sha256=request.expected_checkpoint_state_sha256,
    )


def authorization_path(
    request: RBMProductionSamplingRequest,
) -> Path:
    """Return the fixed execution-specific authorization path."""

    return (
        Path(AUTHORIZATION_ROOT)
        / (
            f"{request.dataset_id}_seed{request.experimental_seed}"
            f"_attempt{request.attempt:02d}.json"
        )
    )


def _reject_duplicate_keys(pairs):
    record = {}

    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate authorization JSON field: {key}")

        record[key] = value

    return record


def _reject_constant(value):
    raise ValueError(f"non-finite authorization JSON constant: {value}")


def _require_regular_authorization(path: Path) -> Path:
    if path.is_symlink():
        raise RBMProductionAuthorizationError(
            "RBM production authorization must not be a symlink"
        )

    if not path.is_file():
        raise RBMProductionAuthorizationError(
            "RBM production authorization is absent"
        )

    metadata = path.stat()

    if metadata.st_uid != os.getuid():
        raise RBMProductionAuthorizationError(
            "RBM production authorization has a different owner"
        )

    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise RBMProductionAuthorizationError(
            "RBM production authorization is group/world writable"
        )

    return path


def load_authorization_record(
    request: RBMProductionSamplingRequest,
) -> dict[str, Any]:
    """Read one external authorization record or fail closed."""

    path = _require_regular_authorization(authorization_path(request))

    try:
        raw = path.read_bytes()
        record = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RBMProductionAuthorizationError(
            "RBM production authorization is not valid JSON"
        ) from exc

    if not isinstance(record, dict):
        raise RBMProductionAuthorizationError(
            "RBM production authorization root must be an object"
        )

    if set(record) != _TOP_LEVEL_FIELDS:
        raise RBMProductionAuthorizationError(
            "RBM production authorization fields are invalid"
        )

    return record


def _validate_resources(raw: Any) -> dict:
    if not isinstance(raw, dict) or set(raw) != RESOURCE_FIELDS:
        raise RBMProductionAuthorizationError(
            "RBM production resource envelope fields are invalid"
        )

    partition = raw["partition"]

    if not isinstance(partition, str) or not partition.strip():
        raise RBMProductionAuthorizationError(
            "RBM production partition is invalid"
        )

    for name in ("nodes", "tasks", "cpus_per_task", "memory_gb"):
        _exact_integer(raw[name], name=name, minimum=1)

    _exact_integer(
        raw["walltime_seconds"],
        name="walltime_seconds",
        minimum=1,
    )

    if raw["gpu_enabled"] is not False:
        raise RBMProductionAuthorizationError(
            "RBM production sampling must be CPU-only"
        )

    if raw["max_concurrent_jobs"] != 1:
        raise RBMProductionAuthorizationError(
            "RBM production concurrency must be one"
        )

    if raw["automatic_retries"] != 0:
        raise RBMProductionAuthorizationError(
            "RBM production automatic retries must be zero"
        )

    job_core_hours = _individual_job_cpu_core_hours(raw)

    if job_core_hours > CUMULATIVE_RBM_SYNTHESIS_PLANNING_CORE_HOURS:
        raise RBMProductionAuthorizationError(
            "individual-job allocated CPU core-hours exceed the approved "
            "650 core-hour RBM synthesis planning ceiling; this per-job "
            "bound is necessary but does not enforce cumulative workload "
            "compliance"
        )

    return dict(raw)


def parse_slurm_allocation(raw: str) -> dict:
    """Parse one ``scontrol show job -o`` line into a field mapping."""

    if not isinstance(raw, str):
        raise RBMProductionAuthorizationError(
            "Slurm allocation record is invalid"
        )

    lines = [line for line in raw.splitlines() if line.strip()]

    if len(lines) != 1:
        raise RBMProductionAuthorizationError(
            "Slurm allocation record is not a single line"
        )

    fields = dict(SLURM_FIELD_PATTERN.findall(lines[0]))

    if not fields:
        raise RBMProductionAuthorizationError(
            "Slurm allocation record is empty"
        )

    return fields


def capture_slurm_allocation() -> dict:
    """Read the actual allocation for the current Slurm job."""

    job_id = os.environ.get("SLURM_JOB_ID", "").strip()

    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise RBMProductionAuthorizationError(
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
        raise RBMProductionAuthorizationError(
            "Slurm allocation cannot be verified"
        ) from exc

    if completed.returncode != 0:
        raise RBMProductionAuthorizationError(
            "Slurm allocation lookup failed"
        )

    allocation = parse_slurm_allocation(completed.stdout)

    if allocation.get("JobId") != job_id:
        raise RBMProductionAuthorizationError(
            "Slurm allocation identity does not match the job"
        )

    return allocation


def _allocation_integer(allocation: Mapping[str, str], name: str) -> int:
    value = allocation.get(name)

    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise RBMProductionAuthorizationError(
            f"Slurm allocation field is missing: {name}"
        )

    return int(value)


def _memory_gb(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value.strip().upper())

    if match is None:
        raise RBMProductionAuthorizationError(
            "Slurm memory field is invalid"
        )

    amount = int(match.group(1))
    unit = match.group(2)

    factors = {"": 1.0 / (1024**3), "K": 1.0 / (1024**2), "M": 1.0 / 1024,
               "G": 1.0, "T": 1024.0}

    return int(round(amount * factors[unit]))


def _tres_memory_gb(tres: str) -> int:
    for item in tres.split(","):
        if item.startswith("mem="):
            return _memory_gb(item.split("=", 1)[1])

    raise RBMProductionAuthorizationError(
        "Slurm allocation lacks a memory TRES value"
    )


def _time_limit_seconds(value: str) -> int:
    match = re.fullmatch(
        r"(?:([0-9]+)-)?([0-9]{1,2}):([0-9]{2}):([0-9]{2})",
        value.strip(),
    )

    if match is None:
        raise RBMProductionAuthorizationError(
            "Slurm time-limit field is invalid"
        )

    days = int(match.group(1) or 0)
    hours = int(match.group(2))
    minutes = int(match.group(3))
    seconds = int(match.group(4))

    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def validate_allocation(
    *,
    resources: Mapping[str, Any],
    allocation: Mapping[str, str],
) -> None:
    """Require the Slurm allocation to match the authorized envelope."""

    if not isinstance(allocation, Mapping):
        raise RBMProductionAuthorizationError(
            "Slurm allocation record is invalid"
        )

    if allocation.get("Partition") != resources["partition"]:
        raise RBMProductionAuthorizationError(
            "Slurm partition differs from the authorized envelope"
        )

    if _allocation_integer(allocation, "NumNodes") != resources["nodes"]:
        raise RBMProductionAuthorizationError(
            "Slurm node count differs from the authorized envelope"
        )

    if _allocation_integer(allocation, "NumTasks") != resources["tasks"]:
        raise RBMProductionAuthorizationError(
            "Slurm task count differs from the authorized envelope"
        )

    if (
        _allocation_integer(allocation, "NumCPUs")
        != _authorized_total_cpus(resources)
    ):
        raise RBMProductionAuthorizationError(
            "Slurm total CPU allocation differs from the authorized task "
            "and CPU counts"
        )

    tres = allocation.get("TRES")

    if not isinstance(tres, str):
        raise RBMProductionAuthorizationError(
            "Slurm allocation lacks TRES"
        )

    if _tres_memory_gb(tres) != resources["memory_gb"]:
        raise RBMProductionAuthorizationError(
            "Slurm memory differs from the authorized envelope"
        )

    time_limit = allocation.get("TimeLimit")

    if not isinstance(time_limit, str):
        raise RBMProductionAuthorizationError(
            "Slurm allocation lacks a time limit"
        )

    if (
        _time_limit_seconds(time_limit)
        != resources["walltime_seconds"]
    ):
        raise RBMProductionAuthorizationError(
            "Slurm walltime differs from the authorized envelope"
        )

    if allocation.get("Requeue") != "0":
        raise RBMProductionAuthorizationError(
            "Slurm automatic requeue must be disabled"
        )

    gres = allocation.get("Gres", "")

    if isinstance(gres, str) and "gpu" in gres.lower():
        raise RBMProductionAuthorizationError(
            "RBM production sampling must not request a GPU"
        )


def validate_authorization(
    request: RBMProductionSamplingRequest,
    *,
    allocation: Mapping[str, str],
) -> RBMProductionSamplingGrant:
    """Validate the authorization record without loading a checkpoint."""

    class_mapping = validate_request_identity(request)
    expected_policy = policy_for_request(request, class_mapping)
    record = load_authorization_record(request)

    if (
        record["schema"] != SCHEMA
        or record["paper"] != PAPER
        or record["revision"] != REVISION
        or record["family"] != FAMILY
    ):
        raise RBMProductionAuthorizationError(
            "RBM production authorization identity is invalid"
        )

    if record["dataset_id"] != request.dataset_id:
        raise RBMProductionAuthorizationError(
            "authorized dataset identity differs from the request"
        )

    if record["experimental_seed"] != request.experimental_seed:
        raise RBMProductionAuthorizationError(
            "authorized experimental seed differs from the request"
        )

    if record["attempt"] != request.attempt:
        raise RBMProductionAuthorizationError(
            "authorized attempt differs from the request"
        )

    if record["run_identity"] != request.run_identity:
        raise RBMProductionAuthorizationError(
            "authorized run identity differs from the request"
        )

    if record["class_id"] != request.class_id:
        raise RBMProductionAuthorizationError(
            "authorized class identity differs from the request"
        )

    if (
        record["sample_start"] != request.sample_start
        or record["sample_stop"] != request.sample_stop
    ):
        raise RBMProductionAuthorizationError(
            "authorized sample scope differs from the request"
        )

    if record["source_commit"] != request.expected_source_commit:
        raise RBMProductionAuthorizationError(
            "authorized source commit differs from the request"
        )

    if (
        Path(record["checkpoint_directory"]).expanduser().resolve()
        != request.checkpoint_directory.resolve()
    ):
        raise RBMProductionAuthorizationError(
            "authorized checkpoint directory differs from the request"
        )

    if (
        record["checkpoint_state_sha256"]
        != request.expected_checkpoint_state_sha256
    ):
        raise RBMProductionAuthorizationError(
            "authorized checkpoint state differs from the request"
        )

    if record["policy_digest"] != expected_policy.digest():
        raise RBMProductionAuthorizationError(
            "authorized sampling policy differs from the request"
        )

    for value, name in (
        (record["authorization_id"], "authorization_id"),
        (record["checkpoint_state_sha256"], "checkpoint_state_sha256"),
        (record["policy_digest"], "policy_digest"),
    ):
        if not isinstance(value, str):
            raise RBMProductionAuthorizationError(
                f"authorized {name} is invalid"
            )

    if AUTHORIZATION_ID_PATTERN.fullmatch(
        record["authorization_id"]
    ) is None:
        raise RBMProductionAuthorizationError(
            "authorized authorization_id is invalid"
        )

    _require_sha256(
        record["checkpoint_state_sha256"],
        name="authorized checkpoint_state_sha256",
    )
    _require_sha256(record["policy_digest"], name="authorized policy_digest")

    resources = _validate_resources(record["resources"])
    validate_allocation(resources=resources, allocation=allocation)

    return RBMProductionSamplingGrant(
        authorization_id=record["authorization_id"],
        dataset_id=request.dataset_id,
        experimental_seed=request.experimental_seed,
        attempt=request.attempt,
        class_id=request.class_id,
        sample_start=request.sample_start,
        sample_stop=request.sample_stop,
        source_commit=request.expected_source_commit,
        checkpoint_directory=request.checkpoint_directory,
        checkpoint_state_sha256=request.expected_checkpoint_state_sha256,
        policy_digest=expected_policy.digest(),
        resources=resources,
    )


def _load_revision_modules() -> dict:
    from trustforge.models.paper01_revision_2026 import (
        experimental_seed_policy,
        restricted_boltzmann_checkpoint,
        restricted_boltzmann_model,
        restricted_boltzmann_production_sampling,
        restricted_boltzmann_sampling,
    )

    return {
        "experimental_seed_policy": experimental_seed_policy,
        "restricted_boltzmann_checkpoint": restricted_boltzmann_checkpoint,
        "restricted_boltzmann_model": restricted_boltzmann_model,
        "restricted_boltzmann_production_entry": sys.modules[__name__],
        "restricted_boltzmann_production_sampling":
            restricted_boltzmann_production_sampling,
        "restricted_boltzmann_sampling": restricted_boltzmann_sampling,
    }


def require_module_binding(
    repository_root: Path,
    *,
    modules: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Require every consumed module to resolve inside this checkout."""

    root = Path(repository_root).resolve()
    package = (root / REVISION_PACKAGE).resolve()

    if modules is None:
        modules = _load_revision_modules()

    if not modules:
        raise RBMProductionSamplingError(
            "no Revision-2026 modules were bound"
        )

    paths: dict[str, str] = {}

    for name, module in modules.items():
        path = Path(getattr(module, "__file__", "")).resolve()

        if not path.is_relative_to(package):
            raise RBMProductionSamplingError(
                f"{name} is not bound to the Revision-2026 module path: "
                f"{path}"
            )

        if not path.is_file():
            raise RBMProductionSamplingError(
                f"Revision-2026 module is missing: {path}"
            )

        paths[name] = str(path)

    return paths


def require_source_binding(
    request: RBMProductionSamplingRequest,
) -> dict:
    """Require Slurm, a clean checkout and the exact expected commit."""

    if not isinstance(request, RBMProductionSamplingRequest):
        raise TypeError("request must be RBMProductionSamplingRequest")

    state = capture_git_state(request.repository_root)

    if state.dirty:
        raise RBMProductionSamplingError(
            "RBM production sampling requires a clean working tree and index"
        )

    if state.commit != request.expected_source_commit:
        raise RBMProductionSamplingError(
            "source revision does not match the expected commit"
        )

    module_paths = require_module_binding(state.repo_root)

    return {
        "requested_source_commit": request.expected_source_commit,
        "resolved_source_commit": state.commit,
        "repository_root": str(Path(state.repo_root).resolve()),
        "module_paths": module_paths,
    }


def bind_checkpoint(
    request: RBMProductionSamplingRequest,
    *,
    class_mapping: tuple[str, ...],
) -> tuple[RBMPersistedCheckpoint, RBMProductionSamplingPolicy, str]:
    """Load and validate the authorized checkpoint pair."""

    directory = request.checkpoint_directory

    if directory.is_symlink():
        raise RBMProductionSamplingError(
            "RBM production checkpoint directory must not be a symlink"
        )

    if not directory.is_dir():
        raise RBMProductionSamplingError(
            f"RBM production checkpoint directory is missing: {directory}"
        )

    state_path = directory / STATE_FILENAME
    metadata_path = directory / METADATA_FILENAME

    for path in (state_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise RBMProductionSamplingError(
                f"RBM production checkpoint file is missing or symlinked: "
                f"{path}"
            )

    observed_state_sha256 = sha256_file(state_path)

    if (
        observed_state_sha256
        != request.expected_checkpoint_state_sha256
    ):
        raise RBMProductionSamplingError(
            "RBM production checkpoint state file does not match the "
            "expected identity"
        )

    checkpoint = load_rbm_checkpoint(
        directory,
        requested_dataset_id=request.dataset_id,
        requested_experimental_seed=request.experimental_seed,
        requested_class_mapping=class_mapping,
    )

    policy = policy_for_request(request, class_mapping)

    validate_production_sampling_policy(
        checkpoint=checkpoint,
        policy=policy,
    )

    return checkpoint, policy, sha256_file(metadata_path)


@dataclass(frozen=True)
class RBMProductionSamplingExecution:
    """Requested, resolved and consumed identity for one bounded execution."""

    request: RBMProductionSamplingRequest
    source_binding: dict
    grant: RBMProductionSamplingGrant
    checkpoint_metadata_sha256: str
    result: RBMProductionSamplingResult
    slurm_job_id: str

    def to_mapping(self) -> dict[str, Any]:
        provenance = self.result.provenance

        return {
            "schema": EXECUTION_SCHEMA,
            "status": "SAMPLING_COMPLETE",
            "paper": PAPER,
            "revision": REVISION,
            "family": FAMILY,
            "dataset_id": self.request.dataset_id,
            "experimental_seed": self.request.experimental_seed,
            "attempt": self.request.attempt,
            "run_identity": self.request.run_identity,
            "class_id": self.request.class_id,
            "class_label": provenance.class_label,
            "sample_start": self.request.sample_start,
            "sample_stop": self.request.sample_stop,
            "sample_count": provenance.sample_count,
            "requested_source_commit":
                self.source_binding["requested_source_commit"],
            "resolved_source_commit":
                self.source_binding["resolved_source_commit"],
            "repository_root": self.source_binding["repository_root"],
            "module_paths": self.source_binding["module_paths"],
            "authorization_id": self.grant.authorization_id,
            "checkpoint_directory": str(self.request.checkpoint_directory),
            "checkpoint_state_sha256":
                self.request.expected_checkpoint_state_sha256,
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "policy_digest": self.grant.policy_digest,
            "transition_count": provenance.transition_count,
            "burn_in": provenance.burn_in,
            "initialization": provenance.initialization,
            "class_conditioning": provenance.class_conditioning,
            "chain_semantics": provenance.chain_semantics,
            "internal_dtype": provenance.internal_dtype,
            "canonical_shape": list(provenance.canonical_shape),
            "canonical_dtype": provenance.canonical_dtype,
            "slurm_job_id": self.slurm_job_id,
            "resource_envelope": dict(self.grant.resources),
            "individual_job_cpu_core_hours": (
                _individual_job_cpu_core_hours(self.grant.resources)
            ),
            "cumulative_rbm_planning_core_hours": (
                CUMULATIVE_RBM_SYNTHESIS_PLANNING_CORE_HOURS
            ),
            "cumulative_workload_enforcement": "NOT_PERFORMED",
            "measured_resource_consumption": "NOT_MEASURED",
            "sampling_provenance": provenance.to_mapping(),
            "dataset_source_provenance":
                "NOT_ESTABLISHED_BY_RBM_CHECKPOINT_METADATA",
            "pool_publication": "NOT_PERFORMED",
        }


def execute_rbm_production_sampling(
    request: RBMProductionSamplingRequest,
) -> RBMProductionSamplingExecution:
    """Authorize, bind and delegate one bounded production-sampling request.

    Authorization is validated before any production checkpoint is loaded or
    any random sampling primitive is invoked.
    """

    require_rbm_production_sampling_authorization()

    class_mapping = validate_request_identity(request)
    source_binding = require_source_binding(request)

    job_id = os.environ.get("SLURM_JOB_ID", "").strip()

    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise RBMProductionAuthorizationError(
            "a Slurm job identity is required"
        )

    allocation = capture_slurm_allocation()

    if allocation.get("JobId") != job_id:
        raise RBMProductionAuthorizationError(
            "Slurm allocation identity does not match the job"
        )

    grant = validate_authorization(
        request,
        allocation=allocation,
    )

    checkpoint, policy, metadata_sha256 = bind_checkpoint(
        request,
        class_mapping=class_mapping,
    )

    result = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=request.class_id,
        sample_indices=request.sample_indices,
    )

    provenance = result.provenance

    if (
        provenance.policy_digest != policy.digest()
        or provenance.transition_count != policy.transition_count
        or provenance.burn_in != policy.burn_in
        or provenance.class_id != request.class_id
        or provenance.sample_indices != request.sample_indices
        or provenance.checkpoint_state_sha256
        != request.expected_checkpoint_state_sha256
    ):
        raise RBMProductionSamplingError(
            "returned sampling provenance does not match the authorized "
            "request"
        )

    job_id = str(allocation.get("JobId", ""))

    return RBMProductionSamplingExecution(
        request=request,
        source_binding=source_binding,
        grant=grant,
        checkpoint_metadata_sha256=metadata_sha256,
        result=result,
        slurm_job_id=job_id,
    )


__all__ = [
    "AUTHORIZATION_ROOT",
    "CUMULATIVE_RBM_SYNTHESIS_PLANNING_CORE_HOURS",
    "DEFAULT_RESOURCES",
    "EXECUTION_SCHEMA",
    "FAMILY",
    "PAPER",
    "RBM_PRODUCTION_SAMPLING_ENABLED",
    "RBMProductionAuthorizationError",
    "RBMProductionSamplingError",
    "RBMProductionSamplingExecution",
    "RBMProductionSamplingGrant",
    "RBMProductionSamplingRequest",
    "REVISION",
    "RESOURCE_FIELDS",
    "SCHEMA",
    "authoritative_class_mapping",
    "authorization_path",
    "bind_checkpoint",
    "capture_slurm_allocation",
    "execute_rbm_production_sampling",
    "load_authorization_record",
    "parse_slurm_allocation",
    "policy_for_request",
    "require_module_binding",
    "require_rbm_production_sampling_authorization",
    "require_source_binding",
    "validate_allocation",
    "validate_authorization",
    "validate_request_identity",
]
