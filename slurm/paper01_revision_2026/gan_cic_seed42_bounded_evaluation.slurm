#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gan-eval-s42
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/evaluations/cicmaldroid2020/seed42/logs/gan-eval-s42-%j.out
#SBATCH --error=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/evaluations/cicmaldroid2020/seed42/logs/gan-eval-s42-%j.err

set -euo pipefail
umask 077
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

# Paper 1 Revision-2026 CICMalDroid2020 GAN seed-42 paired-evaluation launcher.
#
# Approved resource envelope: talon, 1 node, 1 task, 8 CPUs, 32 GB, 02:00:00,
# CPU-only, no requeue, no automatic retries, one bounded attempt.
#
# Resource approval is NOT execution authorization. This launcher remains
# non-operational until a separate researcher-approved evaluation
# authorization record exists at the fixed path below and the final
# evaluation-source revision and deployment paths are frozen.
#
# Prerequisites prepared outside this task (not created here):
# - log parent:
#   /home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/
#   evaluations/cicmaldroid2020/seed42/logs
# - evaluation run parent:
#   /home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/
#   evaluations/cicmaldroid2020/seed42/runs
# - evaluation authorization record:
#   $HOME/trustforge_revision_2026_artifacts/paper01/gan/evaluations/
#   authorizations/cicmaldroid2020_seed42_evaluation.json
# - final evaluation source commit and clean checkout
# - governed CIC dataset configuration and Inception feature-extractor weights
#
# The generator/synthesis source 4324aeecb45a47a72ac0b479d8cd994fe79dc1e0
# is distinct from the final evaluation source. This launcher rejects the
# generator source for evaluation.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_JOB_NODELIST:?Slurm node list required}"
: "${GAN_EVAL_EXECUTION_CHECKOUT:?Explicit clean evaluation checkout required}"
: "${GAN_EVAL_SOURCE_COMMIT:?Final evaluation source commit required}"
: "${GAN_EVAL_DATASET_CONFIG:?Governed CIC dataset configuration required}"
: "${GAN_EVAL_WEIGHTS:?Governed Inception weights required}"

[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$GAN_EVAL_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$GAN_EVAL_EXECUTION_CHECKOUT" && ! -L "$GAN_EVAL_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$GAN_EVAL_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

home="$(
    cd "$HOME"
    pwd -P
)"

PYTHON="$home/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

AUTHORIZATION="$home/trustforge_revision_2026_artifacts/paper01/gan/evaluations/authorizations/cicmaldroid2020_seed42_evaluation.json"
POOL_MANIFEST="$home/trustforge_revision_2026_artifacts/paper01/gan/cicmaldroid2020/seed42/pools/production/commit4324aee_attempt01/manifest.json"
# Fixed attempt-1 destination, known before Slurm submission; must be absent.
RUN_DIR="$home/trustforge_revision_2026_artifacts/paper01/gan/evaluations/cicmaldroid2020/seed42/runs/attempt01"
DRIVER="$checkout/scripts/run_paper01_cic_gan_seed42_bounded_evaluation.py"

# ---------------------------------------------------------------------------
# Fail-closed authorization, source, allocation and deployment-binding guard.
# This runs before any driver invocation and before classifier training.
# ---------------------------------------------------------------------------
"$PYTHON" -B - "$checkout" "$AUTHORIZATION" "$POOL_MANIFEST" "$RUN_DIR" "$DRIVER" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

JOB_NAME = "p1r1-gan-eval-s42"
PARTITION = "talon"
DATASET_ID = "cicmaldroid2020"
FAMILY = "gan"
SEED = 42
ATTEMPT = 1
POOL_SHA = "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1"
CHECKPOINT_SHA = (
    "963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530"
)
RECIPE_SHA = (
    "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
)
DATASET_CONFIG_SHA = (
    "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
)
WEIGHTS_SHA = (
    "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
)
GENERATOR_SOURCE = "4324aeecb45a47a72ac0b479d8cd994fe79dc1e0"
DRIVER_RELATIVE = "scripts/run_paper01_cic_gan_seed42_bounded_evaluation.py"
LAUNCHER_RELATIVE = (
    "slurm/paper01_revision_2026/gan_cic_seed42_bounded_evaluation.slurm"
)
POOL_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/gan/cicmaldroid2020/"
    "seed42/pools/production/commit4324aee_attempt01/manifest.json"
)
EVAL_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/gan/evaluations/"
    "cicmaldroid2020/seed42"
)
AUTH_RELATIVE = (
    "trustforge_revision_2026_artifacts/paper01/gan/evaluations/"
    "authorizations/cicmaldroid2020_seed42_evaluation.json"
)
AUTH_SCHEMA = "trustforge.paper01.gan.cic-evaluation-authorization.v1"
RECEIPT_SCHEMA = "trustforge.paper01.paired-restoration-expectations.v1"
PLACEHOLDER_ID = "RESEARCHER_DECISION_PENDING"
AUTH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_AUTH_BYTES = 65536

TOP_FIELDS = frozenset({
    "schema", "paper", "revision", "family", "dataset_id",
    "experimental_seed", "attempt", "authorization_status",
    "authorization_id", "source_commit", "launcher_sha256",
    "driver_relative_path", "pool_manifest_path", "pool_manifest_sha256",
    "checkpoint_sha256", "recipe_sha256", "dataset_config_path",
    "dataset_config_sha256", "inception_weights_path",
    "inception_weights_sha256", "run_dir", "resources",
})
RESOURCE_FIELDS = frozenset({
    "partition", "nodes", "tasks", "cpus_per_task", "memory_gb",
    "walltime_seconds", "gpu_enabled", "max_concurrent_jobs",
    "automatic_retries",
})
EXPECTED_RESOURCES = {
    "partition": PARTITION,
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory_gb": 32,
    "walltime_seconds": 2 * 60 * 60,
    "gpu_enabled": False,
    "max_concurrent_jobs": 1,
    "automatic_retries": 0,
}


def fail(message, code=78):
    print("GAN_EVAL_LAUNCHER_BLOCKED: " + message, file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_regular(path, name, *, owner=False, bounded=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        fail(name + " must be a regular file")
    info = path.stat()
    if owner and info.st_uid != os.getuid():
        fail(name + " has a different owner")
    if owner and stat.S_IMODE(info.st_mode) & 0o022:
        fail(name + " is group/world writable")
    if bounded and info.st_size > MAX_AUTH_BYTES:
        fail(name + " is too large")
    return path


def reject_duplicates(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            fail("duplicate JSON field: " + key)
        record[key] = value
    return record


def reject_constant(value):
    fail("non-finite JSON constant: " + str(value))


def memory_mib(value):
    if type(value) is not str:
        fail("Slurm memory is missing")
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value.strip().upper())
    if match is None:
        fail("Slurm memory is malformed")
    amount = int(match.group(1))
    factor = {
        "": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024,
    }[match.group(2)]
    result = amount * factor
    if not float(result).is_integer():
        fail("Slurm memory is not integral")
    return int(result)


def walltime_seconds(value):
    if type(value) is not str:
        fail("Slurm walltime is missing")
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
        fail("Slurm walltime is malformed")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_tres(value):
    fields = {}
    if type(value) is not str:
        return fields
    for token in value.split(","):
        if "=" not in token:
            continue
        name, field = token.split("=", 1)
        fields[name.strip()] = field.strip()
    return fields


def gpu_requested(*values):
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
        fail("a Slurm job identity is required")
    if not os.environ.get("SLURM_JOB_NODELIST", "").strip():
        fail("a Slurm node list is required")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None and cpus != str(EXPECTED_RESOURCES["cpus_per_task"]):
        fail("SLURM_CPUS_PER_TASK differs from the approved envelope")
    try:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        fail("Slurm allocation cannot be verified")
    if completed.returncode != 0:
        fail("Slurm allocation lookup failed")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        fail("Slurm allocation record is not a single line")
    fields = dict(re.findall(r"(\w+)=([^ ]*)", lines[0]))
    if not fields:
        fail("Slurm allocation record is empty")
    return fields


def validate_allocation(fields):
    if fields.get("JobId") != os.environ["SLURM_JOB_ID"]:
        fail("Slurm allocation identity does not match the job")
    if fields.get("JobName") != JOB_NAME:
        fail("Slurm job name differs from the approved launcher")
    if fields.get("Partition") != PARTITION:
        fail("Slurm partition differs from the approved envelope")
    if fields.get("JobState") not in ("RUNNING", "COMPLETING"):
        fail("Slurm job is not running")
    for name, expected in (
        ("NumNodes", EXPECTED_RESOURCES["nodes"]),
        ("NumTasks", EXPECTED_RESOURCES["tasks"]),
        ("NumCPUs", EXPECTED_RESOURCES["cpus_per_task"]),
    ):
        value = fields.get(name)
        if type(value) is not str or re.fullmatch(r"[0-9]+", value) is None:
            fail("Slurm allocation field is missing: " + name)
        if int(value) != expected:
            fail("Slurm allocation field differs: " + name)
    tres = parse_tres(fields.get("TRES"))
    cpu = tres.get("cpu")
    if cpu is not None and (
        re.fullmatch(r"[0-9]+", cpu) is None
        or int(cpu) != EXPECTED_RESOURCES["cpus_per_task"]
    ):
        fail("Slurm TRES CPU count differs from the approved envelope")
    node = tres.get("node")
    if node is not None and (
        re.fullmatch(r"[0-9]+", node) is None
        or int(node) != EXPECTED_RESOURCES["nodes"]
    ):
        fail("Slurm TRES node count differs from the approved envelope")
    memory = (
        tres.get("mem")
        or fields.get("MinMemoryNode")
        or fields.get("MinMemoryPerNode")
    )
    if memory_mib(memory) != EXPECTED_RESOURCES["memory_gb"] * 1024:
        fail("Slurm memory differs from the approved envelope")
    if walltime_seconds(fields.get("TimeLimit")) != (
        EXPECTED_RESOURCES["walltime_seconds"]
    ):
        fail("Slurm walltime differs from the approved envelope")
    if fields.get("Requeue") != str(EXPECTED_RESOURCES["automatic_retries"]):
        fail("Slurm requeue policy differs from the approved envelope")
    restarts = fields.get("Restarts")
    if restarts is not None and restarts != "0":
        fail("Slurm job has restarted unexpectedly")
    if EXPECTED_RESOURCES["gpu_enabled"] is not False or gpu_requested(
        fields.get("TRES"), fields.get("Gres"),
    ):
        fail("Slurm allocation requests a GPU")
    if EXPECTED_RESOURCES["max_concurrent_jobs"] != 1:
        fail("authorized concurrency is unsupported")


def require_identity(record):
    if record.get("schema") != AUTH_SCHEMA:
        fail("unsupported evaluation authorization schema")
    expected_text = {
        "paper": "paper01",
        "revision": "revision_2026",
        "family": FAMILY,
        "dataset_id": DATASET_ID,
        "authorization_status": "APPROVED",
        "driver_relative_path": DRIVER_RELATIVE,
    }
    for key, value in expected_text.items():
        if record.get(key) != value:
            fail("evaluation authorization field differs: " + key)
    if type(record.get("experimental_seed")) is not int or (
        record["experimental_seed"] != SEED
    ):
        fail("evaluation authorization seed is not 42")
    if type(record.get("attempt")) is not int or record["attempt"] != ATTEMPT:
        fail("evaluation authorization attempt is not 1")
    authorization_id = record.get("authorization_id")
    if (
        type(authorization_id) is not str
        or authorization_id == PLACEHOLDER_ID
        or AUTH_ID.fullmatch(authorization_id) is None
    ):
        fail("evaluation authorization is not approved")
    for key, expected in (
        ("pool_manifest_sha256", POOL_SHA),
        ("checkpoint_sha256", CHECKPOINT_SHA),
        ("recipe_sha256", RECIPE_SHA),
        ("dataset_config_sha256", DATASET_CONFIG_SHA),
        ("inception_weights_sha256", WEIGHTS_SHA),
    ):
        if record.get(key) != expected:
            fail("evaluation authorization digest differs: " + key)
    resources = record.get("resources")
    if type(resources) is not dict or set(resources) != RESOURCE_FIELDS:
        fail("evaluation resource fields are invalid")
    for key, expected in EXPECTED_RESOURCES.items():
        if type(resources[key]) is not type(expected) or (
            resources[key] != expected
        ):
            fail("evaluation resource differs: " + key)


def main():
    checkout = Path(sys.argv[1]).resolve()
    authorization = Path(sys.argv[2])
    pool_manifest = Path(sys.argv[3])
    run_dir = Path(sys.argv[4])
    driver = Path(sys.argv[5])
    home = Path(os.environ["HOME"]).resolve()
    source = os.environ.get("GAN_EVAL_SOURCE_COMMIT", "").strip()

    if HEX40.fullmatch(source) is None:
        fail("final evaluation source commit is required")
    if source == GENERATOR_SOURCE:
        fail("generator/synthesis source cannot be the evaluation source")
    if not checkout.is_dir() or checkout.is_symlink():
        fail("evaluation checkout is unavailable")
    if Path(
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve() != checkout:
        fail("execution checkout mismatch")
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != source:
        fail("evaluation checkout HEAD differs from the approved source")
    dirty = subprocess.check_output(
        [
            "git", "-C", str(checkout), "status", "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    if dirty.strip():
        fail("evaluation checkout must be clean")
    if driver != checkout / DRIVER_RELATIVE:
        fail("evaluation driver path differs from the approved path")
    require_regular(driver, "evaluation driver")

    authorization = require_regular(
        authorization, "evaluation authorization", owner=True, bounded=True,
    )
    if authorization != home / AUTH_RELATIVE:
        fail("evaluation authorization path differs from the approved path")
    try:
        record = json.loads(
            authorization.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, ValueError):
        fail("evaluation authorization is unreadable")
    if type(record) is not dict or set(record) != TOP_FIELDS:
        fail("evaluation authorization fields are invalid")
    require_identity(record)

    if record["source_commit"] != source:
        fail("evaluation authorization source differs from the checkout")
    if HEX40.fullmatch(record["source_commit"]) is None:
        fail("evaluation authorization source is malformed")
    launcher = checkout / LAUNCHER_RELATIVE
    require_regular(launcher, "evaluation launcher")
    if record["launcher_sha256"] != sha256_file(launcher):
        fail("evaluation launcher digest mismatch")

    expected_pool = home / POOL_RELATIVE
    if record["pool_manifest_path"] != str(expected_pool):
        fail("evaluation pool path differs from the accepted production pool")
    if pool_manifest != expected_pool:
        fail("pool manifest argument differs from the accepted production pool")
    require_regular(pool_manifest, "accepted GAN pool manifest")

    dataset_config = Path(os.environ["GAN_EVAL_DATASET_CONFIG"])
    weights = Path(os.environ["GAN_EVAL_WEIGHTS"])
    if not dataset_config.is_absolute() or not weights.is_absolute():
        fail("governed input paths must be absolute")
    if record["dataset_config_path"] != str(dataset_config):
        fail("dataset configuration path differs from the authorization")
    if record["inception_weights_path"] != str(weights):
        fail("Inception weights path differs from the authorization")
    require_regular(dataset_config, "dataset configuration")
    require_regular(weights, "Inception weights")

    expected_run = home / EVAL_RELATIVE / "runs" / "attempt01"
    if record["run_dir"] != str(expected_run) or run_dir != expected_run:
        fail("evaluation run directory differs from the approved identity")
    if run_dir.exists() or run_dir.is_symlink():
        fail("evaluation run directory already exists")
    if not run_dir.parent.is_dir():
        fail("evaluation run parent is unavailable")

    validate_allocation(capture_allocation())
    print("GAN_EVAL_LAUNCHER_BINDING=PASS")


main()
PY

# ---------------------------------------------------------------------------
# Runtime environment: governed interpreter, checkout-bound source, CPU only.
# ---------------------------------------------------------------------------
export PYTHONPATH="$checkout/src:$checkout"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -B - "$checkout" <<'PY'
import importlib.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit("gcs_py310 interpreter required")
for name in ("numpy", "tensorflow", "keras", "trustforge"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit("required module unavailable: " + name)
PY

check_source() {
    [[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$GAN_EVAL_SOURCE_COMMIT" ]]
    [[ -z "$(git -C "$checkout" status --porcelain --untracked-files=all)" ]]
}

COMMON_ARGS=(
    --experimental-seed 42
    --repository-root "$checkout"
    --expected-head "$GAN_EVAL_SOURCE_COMMIT"
    --dataset-config "$GAN_EVAL_DATASET_CONFIG"
    --config-sha256 "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
    --pool-manifest "$POOL_MANIFEST"
    --pool-manifest-sha256 "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1"
    --weights "$GAN_EVAL_WEIGHTS"
)

# ---------------------------------------------------------------------------
# Preflight must pass before any classifier training can begin.
# ---------------------------------------------------------------------------
check_source
"$PYTHON" -B -u "$DRIVER" --preflight-only "${COMMON_ARGS[@]}"
check_source

# ---------------------------------------------------------------------------
# Paired classifier training. The driver creates exactly one fresh run dir.
# ---------------------------------------------------------------------------
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]]
"$PYTHON" -B -u "$DRIVER" --phase train "${COMMON_ARGS[@]}" \
    --run-dir "$RUN_DIR"
check_source

# ---------------------------------------------------------------------------
# Receipt: hash the actual published bytes, verify bounded identity, then
# restore in a separate Python process with the exact digest.
# ---------------------------------------------------------------------------
RECEIPT="$RUN_DIR/expectations.json"
[[ -f "$RECEIPT" && ! -L "$RECEIPT" ]]
RECEIPT_SHA="$(
    "$PYTHON" -B - "$RECEIPT" <<'PY'
import hashlib
import sys

print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
[[ "$RECEIPT_SHA" =~ ^[0-9a-f]{64}$ ]]

"$PYTHON" -B - "$RECEIPT" "$GAN_EVAL_SOURCE_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "schema": "trustforge.paper01.paired-restoration-expectations.v1",
    "evaluator_seed": 42,
    "synthetic_pool_manifest_sha256":
        "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1",
    "source_commit": sys.argv[2],
    "job_id": sys.argv[3],
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit("training receipt identity mismatch: " + key)
if type(receipt.get("train_pid")) is not int:
    raise SystemExit("training receipt lacks a training process identity")
PY
printf 'EVALUATION_RECEIPT_SHA256=%s\n' "$RECEIPT_SHA"

check_source
"$PYTHON" -B -u "$DRIVER" --phase restore "${COMMON_ARGS[@]}" \
    --run-dir "$RUN_DIR" --receipt-sha256 "$RECEIPT_SHA"
check_source

printf 'GAN_CIC_SEED42_EVALUATION_LAUNCHER=PASS\n'
printf 'COMPLETE_EVALUATION=False\n'
printf 'FULL_BENCHMARK_GO=NO\n'
