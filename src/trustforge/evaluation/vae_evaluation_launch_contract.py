"""Fail-closed authorization contract for the VAE seed-42 evaluation launcher.

Validates one external authorization record, the deployed source checkout, the
accepted VAE production pool, the governed dataset configuration and Inception
weights, the fixed evaluation destination and the live Slurm envelope.

This module authorizes nothing by itself. It only checks that an existing,
researcher-approved record matches the execution it is about to permit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


SCHEMA = "trustforge.paper01.vae.cic-evaluation-authorization.v1"
PAPER = "paper01"
REVISION = "revision_2026"
FAMILY = "vae"
DATASET_ID = "cicmaldroid2020"
SEED = 42
ATTEMPT = 1
JOB_NAME = "p1r1-vae-eval-s42"
PARTITION = "talon"

PLACEHOLDER_ID = "RESEARCHER_DECISION_PENDING"
AUTH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_AUTH_BYTES = 65536

DRIVER_RELATIVE = "scripts/run_paper01_cic_vae_seed42_bounded_evaluation.py"
LAUNCHER_RELATIVE = (
    "slurm/paper01_revision_2026/vae_cic_seed42_bounded_evaluation.slurm"
)
AUTH_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/vae/evaluations/"
    "authorizations/cicmaldroid2020_seed42_evaluation.json"
)
EVAL_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/vae/evaluations/"
    "cicmaldroid2020/seed42"
)
POOL_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/"
    "seed42/pools/production/"
    "checkpoint35ca3cd756a88d6b93dcc291f02c4747_synthesis-d4031a9_attempt01/"
    "pool/manifest.json"
)
DATASET_CONFIG_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/diagnostics/"
    "cic-fitted-state-e813232-ogfyI95a/dataset_config.json"
)
WEIGHTS_RELATIVE = (
    ".keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

TOP_FIELDS = frozenset({
    "schema", "paper", "revision", "family", "dataset_id",
    "experimental_seed", "attempt", "authorization_status",
    "authorization_id", "source_commit", "launcher_sha256",
    "driver_relative_path", "pool_manifest_path", "pool_manifest_sha256",
    "checkpoint_generation_id", "checkpoint_manifest_sha256",
    "checkpoint_source_commit", "synthesis_source_commit",
    "synthesis_recipe_sha256", "classifier_recipe_sha256",
    "metric_recipe_sha256", "dataset_config_path", "dataset_config_sha256",
    "inception_weights_path", "inception_weights_sha256", "run_dir",
    "resources",
})
RESOURCE_FIELDS = frozenset({
    "partition", "nodes", "tasks", "cpus_per_task", "memory_gb",
    "walltime_seconds", "gpu_enabled", "max_concurrent_jobs",
    "automatic_retries",
})

ACCEPTED = {
    "pool_manifest_sha256":
        "08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6",
    "checkpoint_generation_id": "35ca3cd756a88d6b93dcc291f02c4747",
    "checkpoint_manifest_sha256":
        "f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e",
    "checkpoint_source_commit":
        "1aec01c26fab51b87362a47e2cfa797bd428c359",
    "synthesis_source_commit":
        "d4031a9b39565e5382d08e2b28236162498a08ca",
    "synthesis_recipe_sha256":
        "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9",
    "classifier_recipe_sha256":
        "1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982",
    "metric_recipe_sha256":
        "d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85",
    "dataset_config_sha256":
        "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b",
    "inception_weights_sha256":
        "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a",
    "resources": {
        "partition": PARTITION,
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 8,
        "memory_gb": 32,
        "walltime_seconds": 2 * 60 * 60,
        "gpu_enabled": False,
        "max_concurrent_jobs": 1,
        "automatic_retries": 0,
    },
}


class LaunchContractError(RuntimeError):
    """Raised when the frozen execution contract is not satisfied."""


def _fail(message):
    raise LaunchContractError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_regular(path, name, *, owner=False, bounded=False, uid=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail(name + " must be a regular file")
    info = path.stat()
    if owner and uid is not None and info.st_uid != uid:
        _fail(name + " has a different owner")
    if owner and stat.S_IMODE(info.st_mode) & 0o022:
        _fail(name + " is group/world writable")
    if bounded and info.st_size > MAX_AUTH_BYTES:
        _fail(name + " is too large")
    return path


def _reject_duplicates(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            _fail("duplicate JSON field: " + key)
        record[key] = value
    return record


def _reject_constant(value):
    _fail("non-finite JSON constant: " + str(value))


def expected_launcher_path(checkout):
    return Path(checkout) / LAUNCHER_RELATIVE


def expected_run_dir(home):
    return Path(home) / EVAL_RELATIVE / "runs" / ("attempt%02d" % ATTEMPT)


def _git(checkout, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        _fail("evaluation checkout git inspection failed: " + str(error))


def validate_checkout(checkout, source_commit):
    checkout = Path(checkout).resolve()
    if not checkout.is_dir() or checkout.is_symlink():
        _fail("evaluation checkout is unavailable")
    if Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
        _fail("execution checkout mismatch")
    if HEX40.fullmatch(source_commit or "") is None:
        _fail("final evaluation source commit is required")
    if source_commit == ACCEPTED["synthesis_source_commit"]:
        _fail("generator/synthesis source cannot be the evaluation source")
    head = _git(checkout, "rev-parse", "HEAD")
    if head != source_commit:
        _fail("evaluation checkout HEAD differs from the approved source")
    dirty = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        _fail("evaluation checkout must be clean")
    driver = checkout / DRIVER_RELATIVE
    _require_regular(driver, "evaluation driver")
    return checkout, driver


def _resource_error(key, value, expected):
    if type(value) is not type(expected) or value != expected:
        _fail("evaluation resource differs: " + key)


def _validate_resources(record, accepted):
    resources = record.get("resources")
    if type(resources) is not dict or set(resources) != RESOURCE_FIELDS:
        _fail("evaluation resource fields are invalid")
    for key, expected in accepted["resources"].items():
        _resource_error(key, resources[key], expected)


def _validate_identity(record, accepted):
    if record.get("schema") != SCHEMA:
        _fail("unsupported evaluation authorization schema")
    for key, expected in (
        ("paper", PAPER),
        ("revision", REVISION),
        ("family", FAMILY),
        ("dataset_id", DATASET_ID),
        ("authorization_status", "APPROVED"),
        ("driver_relative_path", DRIVER_RELATIVE),
    ):
        if record.get(key) != expected:
            _fail("evaluation authorization field differs: " + key)
    if type(record.get("experimental_seed")) is not int or (
        record["experimental_seed"] != SEED
    ):
        _fail("evaluation authorization seed is not 42")
    if type(record.get("attempt")) is not int or record["attempt"] != ATTEMPT:
        _fail("evaluation authorization attempt is not 1")
    authorization_id = record.get("authorization_id")
    if (
        type(authorization_id) is not str
        or authorization_id == PLACEHOLDER_ID
        or AUTH_ID.fullmatch(authorization_id) is None
    ):
        _fail("evaluation authorization is not approved")
    for key in (
        "pool_manifest_sha256", "checkpoint_generation_id",
        "checkpoint_manifest_sha256", "checkpoint_source_commit",
        "synthesis_source_commit", "synthesis_recipe_sha256",
        "classifier_recipe_sha256", "metric_recipe_sha256",
        "dataset_config_sha256", "inception_weights_sha256",
    ):
        if record.get(key) != accepted[key]:
            _fail("evaluation authorization digest differs: " + key)
    _validate_resources(record, accepted)


def validate_authorization(
    *,
    authorization_path,
    checkout,
    launcher_path,
    pool_manifest,
    run_dir,
    dataset_config,
    weights,
    source_commit,
    home=None,
    uid=None,
    accepted=None,
    allocation=None,
):
    """Validate one frozen authorization record and its execution binding."""
    accepted = dict(ACCEPTED if accepted is None else accepted)
    home = Path(home if home is not None else Path.home()).resolve()
    if uid is None:
        uid = os.getuid()

    checkout, _driver = validate_checkout(checkout, source_commit)

    authorization = _require_regular(
        Path(authorization_path), "evaluation authorization",
        owner=True, bounded=True, uid=uid,
    )
    if authorization != home / AUTH_RELATIVE:
        _fail("evaluation authorization path differs from the approved path")
    try:
        record = json.loads(
            authorization.read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeError, ValueError):
        _fail("evaluation authorization is unreadable")
    if type(record) is not dict or set(record) != TOP_FIELDS:
        _fail("evaluation authorization fields are invalid")

    _validate_identity(record, accepted)

    if record["source_commit"] != source_commit:
        _fail("evaluation authorization source differs from the checkout")
    if HEX40.fullmatch(record["source_commit"]) is None:
        _fail("evaluation authorization source is malformed")

    launcher = expected_launcher_path(checkout)
    if Path(launcher_path).resolve() != launcher.resolve():
        _fail("evaluation launcher path differs from the approved path")
    _require_regular(launcher, "evaluation launcher")
    if record["launcher_sha256"] != sha256_file(launcher):
        _fail("evaluation launcher digest mismatch")

    expected_pool = home / POOL_RELATIVE
    if record["pool_manifest_path"] != str(expected_pool):
        _fail("evaluation pool path differs from the accepted production pool")
    if Path(pool_manifest).resolve() != expected_pool.resolve():
        _fail("pool manifest argument differs from the accepted production pool")
    _require_regular(expected_pool, "accepted VAE pool manifest")

    dataset_config = Path(dataset_config)
    weights = Path(weights)
    if not dataset_config.is_absolute() or not weights.is_absolute():
        _fail("governed input paths must be absolute")
    if record["dataset_config_path"] != str(dataset_config):
        _fail("dataset configuration path differs from the authorization")
    if record["inception_weights_path"] != str(weights):
        _fail("Inception weights path differs from the authorization")
    _require_regular(dataset_config, "dataset configuration")
    _require_regular(weights, "Inception weights")

    expected_run = expected_run_dir(home)
    if record["run_dir"] != str(expected_run) or Path(run_dir) != expected_run:
        _fail("evaluation run directory differs from the approved identity")
    if expected_run.exists() or expected_run.is_symlink():
        _fail("evaluation run directory already exists")
    if not expected_run.parent.is_dir():
        _fail("evaluation run parent is unavailable")

    fields = allocation if allocation is not None else capture_allocation()
    validate_allocation(fields, accepted)
    return record


# ---------------------------------------------------------------------------
# Slurm envelope
# ---------------------------------------------------------------------------


def _memory_mib(value):
    if type(value) is not str:
        _fail("Slurm memory is missing")
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value.strip().upper())
    if match is None:
        _fail("Slurm memory is malformed")
    amount = int(match.group(1))
    factor = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[
        match.group(2)
    ]
    result = amount * factor
    if not float(result).is_integer():
        _fail("Slurm memory is not integral")
    return int(result)


def _walltime_seconds(value):
    if type(value) is not str:
        _fail("Slurm walltime is missing")
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
        _fail("Slurm walltime is malformed")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_tres(value):
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
            for number in re.findall(r"gpu[^0-9]*([0-9]+)", value.lower())
        ]
        if not counts or any(count > 0 for count in counts):
            return True
    return False


def capture_allocation():
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if re.fullmatch(r"[0-9]+", job_id) is None:
        _fail("a Slurm job identity is required")
    if not os.environ.get("SLURM_JOB_NODELIST", "").strip():
        _fail("a Slurm node list is required")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None and cpus != str(ACCEPTED["resources"]["cpus_per_task"]):
        _fail("SLURM_CPUS_PER_TASK differs from the approved envelope")
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        _fail("Slurm allocation cannot be verified")
    if completed.returncode != 0:
        _fail("Slurm allocation lookup failed")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        _fail("Slurm allocation record is not a single line")
    fields = dict(re.findall(r"(\w+)=([^ ]*)", lines[0]))
    if not fields:
        _fail("Slurm allocation record is empty")
    return fields


def validate_allocation(fields, accepted=None):
    resources = (ACCEPTED if accepted is None else accepted)["resources"]
    if fields.get("JobId") != os.environ.get("SLURM_JOB_ID"):
        _fail("Slurm allocation identity does not match the job")
    if fields.get("JobName") != JOB_NAME:
        _fail("Slurm job name differs from the approved launcher")
    if fields.get("Partition") != resources["partition"]:
        _fail("Slurm partition differs from the approved envelope")
    if fields.get("JobState") not in ("RUNNING", "COMPLETING"):
        _fail("Slurm job is not running")
    for name, expected in (
        ("NumNodes", resources["nodes"]),
        ("NumTasks", resources["tasks"]),
        ("NumCPUs", resources["cpus_per_task"]),
    ):
        value = fields.get(name)
        if type(value) is not str or re.fullmatch(r"[0-9]+", value) is None:
            _fail("Slurm allocation field is missing: " + name)
        if int(value) != expected:
            _fail("Slurm allocation field differs: " + name)
    tres = _parse_tres(fields.get("TRES"))
    cpu = tres.get("cpu")
    if cpu is not None and (
        re.fullmatch(r"[0-9]+", cpu) is None
        or int(cpu) != resources["cpus_per_task"]
    ):
        _fail("Slurm TRES CPU count differs from the approved envelope")
    node = tres.get("node")
    if node is not None and (
        re.fullmatch(r"[0-9]+", node) is None
        or int(node) != resources["nodes"]
    ):
        _fail("Slurm TRES node count differs from the approved envelope")
    memory = (
        tres.get("mem") or fields.get("MinMemoryNode")
        or fields.get("MinMemoryPerNode")
    )
    if _memory_mib(memory) != resources["memory_gb"] * 1024:
        _fail("Slurm memory differs from the approved envelope")
    if _walltime_seconds(fields.get("TimeLimit")) != resources["walltime_seconds"]:
        _fail("Slurm walltime differs from the approved envelope")
    if fields.get("Requeue") != str(resources["automatic_retries"]):
        _fail("Slurm requeue policy differs from the approved envelope")
    restarts = fields.get("Restarts")
    if restarts is not None and restarts != "0":
        _fail("Slurm job has restarted unexpectedly")
    if resources["gpu_enabled"] is not False or _gpu_requested(
        fields.get("TRES"), fields.get("Gres"),
    ):
        _fail("Slurm allocation requests a GPU")
    if resources["max_concurrent_jobs"] != 1:
        _fail("authorized concurrency is unsupported")
    return fields


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 8:
        print(
            "VAE_EVAL_LAUNCHER_BLOCKED: authorization checkout launcher "
            "pool run-dir dataset-config weights source-commit required",
            file=sys.stderr,
        )
        return 78
    (
        authorization_path, checkout, launcher_path, pool_manifest,
        run_dir, dataset_config, weights, source_commit,
    ) = argv
    try:
        validate_authorization(
            authorization_path=authorization_path,
            checkout=checkout,
            launcher_path=launcher_path,
            pool_manifest=pool_manifest,
            run_dir=run_dir,
            dataset_config=dataset_config,
            weights=weights,
            source_commit=source_commit.strip(),
        )
    except LaunchContractError as error:
        print("VAE_EVAL_LAUNCHER_BLOCKED: " + str(error), file=sys.stderr)
        return 78
    print("VAE_EVAL_LAUNCHER_BINDING=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
