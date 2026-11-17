"""Focused tests for the CIC GAN seed-42 bounded evaluation launcher.

All execution uses a temporary checkout, a fixture authorization record and a
stand-in driver. The production pool, checkpoint and classifier are never
accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPOSITORY_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "gan_cic_seed42_bounded_evaluation.slurm"
)

POOL_SHA = (
    "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1"
)
CHECKPOINT_SHA = (
    "963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530"
)
RECIPE_SHA = (
    "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
)
CONFIG_SHA = (
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
JOB_ID = "12345"

APPROVED_LINE = (
    f"JobId={JOB_ID} JobName=p1r1-gan-eval-s42 Partition=talon "
    "JobState=RUNNING NumNodes=1 NumTasks=1 NumCPUs=8 "
    "TRES=cpu=8,node=1,mem=32768M,billing=8 MinMemoryNode=32768M "
    "TimeLimit=02:00:00 Requeue=0 Restarts=0 Gres=(null)"
)

FAKE_SCONTROL = (
    "#!/bin/sh\n"
    'if [ "${FAKE_SCONTROL_FAIL:-0}" = "1" ]; then\n'
    "    exit 1\n"
    "fi\n"
    'if [ -n "${FAKE_SCONTROL_LINE:-}" ]; then\n'
    '    printf \'%s\\n\' "$FAKE_SCONTROL_LINE"\n'
    "else\n"
    f"    printf '%s\\n' \"{APPROVED_LINE}\"\n"
    "fi\n"
)

FAKE_DRIVER = '''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--preflight-only", action="store_true")
parser.add_argument("--phase", choices=("train", "restore"))
parser.add_argument("--experimental-seed", type=int)
parser.add_argument("--repository-root")
parser.add_argument("--expected-head")
parser.add_argument("--dataset-config")
parser.add_argument("--config-sha256")
parser.add_argument("--pool-manifest")
parser.add_argument("--pool-manifest-sha256")
parser.add_argument("--weights")
parser.add_argument("--run-dir")
parser.add_argument("--receipt-sha256")
args = parser.parse_args()

phase = (
    "preflight" if args.preflight_only
    else args.phase
)
record = {
    "phase": phase,
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "hashseed": os.environ.get("PYTHONHASHSEED"),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "omp": os.environ.get("OMP_NUM_THREADS"),
}
with Path(os.environ["GAN_EVAL_TEST_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\\n")

if args.preflight_only:
    if os.environ.get("FAKE_PREFLIGHT_FAIL") == "1":
        print("FAKE_PREFLIGHT_FAIL", file=sys.stderr)
        raise SystemExit(11)
    print("CIC_GAN_SEED42_EVALUATION_PREFLIGHT=PASS")
    raise SystemExit(0)

run = Path(args.run_dir)
if args.phase == "train":
    if os.environ.get("FAKE_TRAIN_FAIL") == "1":
        print("FAKE_TRAIN_FAIL", file=sys.stderr)
        raise SystemExit(12)
    run.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "trustforge.paper01.paired-restoration-expectations.v1",
        "evaluator_seed": 42,
        "synthetic_pool_manifest_sha256": args.pool_manifest_sha256,
        "source_commit": args.expected_head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "train_pid": 111,
        "arms": {
            "real_only": {"manifest_sha256": "1" * 64},
            "real_plus_synthetic": {"manifest_sha256": "2" * 64},
        },
    }
    if os.environ.get("FAKE_BAD_RECEIPT") == "1":
        receipt["synthetic_pool_manifest_sha256"] = "0" * 64
    if os.environ.get("FAKE_SKIP_RECEIPT") != "1":
        (run / "expectations.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    print("CIC_GAN_SEED42_EVALUATION_TRAIN=PASS")
    raise SystemExit(0)

if args.phase == "restore":
    if os.environ.get("FAKE_RESTORE_FAIL") == "1":
        print("FAKE_RESTORE_FAIL", file=sys.stderr)
        raise SystemExit(13)
    receipt = run / "expectations.json"
    actual = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if args.receipt_sha256 != actual:
        print("FAKE_RECEIPT_MISMATCH", file=sys.stderr)
        raise SystemExit(14)
    print("CIC_GAN_SEED42_EVALUATION_RESTORE=PASS")
    print("CIC_GAN_SEED42_BOUNDED_EVALUATION=PASS")
    print("COMPLETE_EVALUATION=False")
    print("FULL_BENCHMARK_GO=NO")
    raise SystemExit(0)

raise SystemExit(15)
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Fixture:
    home: Path
    checkout: Path
    launcher: Path
    authorization: Path
    pool_manifest: Path
    dataset_config: Path
    weights: Path
    run_dir: Path
    log: Path
    head: str
    environment: dict


def _write_authorization(path, record):
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _authorization(fixture):
    return json.loads(fixture.authorization.read_text(encoding="utf-8"))


def _mutate(fixture, **changes):
    record = _authorization(fixture)
    record.update(changes)
    _write_authorization(fixture.authorization, record)
    return record


def _build_fixture(tmp_path):
    home = (tmp_path / "home").resolve()
    home.mkdir()
    checkout = (tmp_path / "checkout").resolve()
    checkout.mkdir()
    bin_dir = (tmp_path / "bin").resolve()
    bin_dir.mkdir()
    log = (tmp_path / "driver.jsonl").resolve()

    python_dir = home / "miniforge3/envs/gcs_py310/bin"
    python_dir.mkdir(parents=True)
    python_wrapper = python_dir / "python"
    python_wrapper.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n',
        encoding="utf-8",
    )
    python_wrapper.chmod(0o755)

    pool_dir = (
        home
        / "trustforge_revision_2026_artifacts/paper01/gan/cicmaldroid2020"
        / "seed42/pools/production/commit4324aee_attempt01"
    )
    pool_dir.mkdir(parents=True)
    pool_manifest = pool_dir / "manifest.json"
    pool_manifest.write_bytes(b'{"fixture": "pool"}\n')

    governed = home / "governed"
    governed.mkdir()
    dataset_config = governed / "dataset_config.json"
    dataset_config.write_bytes(b'{"fixture": "config"}\n')
    weights = governed / "inception.h5"
    weights.write_bytes(b"fixture-weights")

    run_parent = (
        home
        / "trustforge_revision_2026_artifacts/paper01/gan/evaluations"
        / "cicmaldroid2020/seed42/runs"
    )
    run_parent.mkdir(parents=True)
    run_dir = run_parent / f"attempt01-{JOB_ID}"

    authorization_dir = (
        home
        / "trustforge_revision_2026_artifacts/paper01/gan/evaluations"
        / "authorizations"
    )
    authorization_dir.mkdir(parents=True)
    authorization = (
        authorization_dir / "cicmaldroid2020_seed42_evaluation.json"
    )

    launcher = checkout / LAUNCHER_RELATIVE
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, launcher)
    driver = checkout / DRIVER_RELATIVE
    driver.parent.mkdir(parents=True)
    driver.write_text(FAKE_DRIVER, encoding="utf-8")
    (checkout / "src/trustforge").mkdir(parents=True)
    (checkout / "src/trustforge/__init__.py").write_text(
        "", encoding="utf-8",
    )
    (checkout / "README").write_text("fixture\n", encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(checkout),
            "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()

    record = {
        "schema": "trustforge.paper01.gan.cic-evaluation-authorization.v1",
        "paper": "paper01",
        "revision": "revision_2026",
        "family": "gan",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "authorization_status": "APPROVED",
        "authorization_id": "fixture-evaluation-authorization",
        "source_commit": head,
        "launcher_sha256": _sha(launcher),
        "driver_relative_path": DRIVER_RELATIVE,
        "pool_manifest_path": str(pool_manifest),
        "pool_manifest_sha256": POOL_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "recipe_sha256": RECIPE_SHA,
        "dataset_config_path": str(dataset_config),
        "dataset_config_sha256": CONFIG_SHA,
        "inception_weights_path": str(weights),
        "inception_weights_sha256": WEIGHTS_SHA,
        "run_dir": str(run_dir),
        "resources": {
            "partition": "talon",
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
    _write_authorization(authorization, record)

    scontrol = bin_dir / "scontrol"
    scontrol.write_text(FAKE_SCONTROL, encoding="utf-8")
    scontrol.chmod(0o755)

    environment = {
        "HOME": str(home),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "SLURM_JOB_ID": JOB_ID,
        "SLURM_JOB_NODELIST": "talon08",
        "SLURM_CPUS_PER_TASK": "8",
        "GAN_EVAL_EXECUTION_CHECKOUT": str(checkout),
        "GAN_EVAL_SOURCE_COMMIT": head,
        "GAN_EVAL_DATASET_CONFIG": str(dataset_config),
        "GAN_EVAL_WEIGHTS": str(weights),
        "GAN_EVAL_TEST_LOG": str(log),
    }
    return Fixture(
        home=home,
        checkout=checkout,
        launcher=launcher,
        authorization=authorization,
        pool_manifest=pool_manifest,
        dataset_config=dataset_config,
        weights=weights,
        run_dir=run_dir,
        log=log,
        head=head,
        environment=environment,
    )


@pytest.fixture
def fixture(tmp_path):
    return _build_fixture(tmp_path)


def _run(fixture, *, overrides=None, drop=()):
    environment = os.environ.copy()
    environment.update(fixture.environment)
    if overrides:
        environment.update(overrides)
    for name in drop:
        environment.pop(name, None)
    return subprocess.run(
        ["bash", str(fixture.launcher)],
        cwd=fixture.checkout,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )


def _entries(fixture):
    if not fixture.log.exists():
        return []
    return [
        json.loads(line)
        for line in fixture.log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _phases(fixture):
    return [entry["phase"] for entry in _entries(fixture)]


def _option(entry, name):
    argv = entry["argv"]
    return argv[argv.index(name) + 1]


# ---------------------------------------------------------------------------
# Positive orchestration
# ---------------------------------------------------------------------------


def test_launcher_declares_approved_resource_envelope():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --job-name=p1r1-gan-eval-s42" in text
    assert "#SBATCH --partition=talon" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --array" not in text
    assert "#SBATCH --requeue" not in text
    assert "sbatch " not in text
    assert "srun " not in text
    assert "gan-eval-s42-%j.out" in text
    assert "gan-eval-s42-%j.err" in text

    assert GENERATOR_SOURCE in text
    assert "RESEARCHER_DECISION_PENDING" in text
    assert "authorization_status" in text
    assert "APPROVED" in text
    assert "NOT execution authorization" in text


def test_successful_orchestration_preflight_train_receipt_restore(fixture):
    result = _run(fixture)

    assert result.returncode == 0, result.stderr
    assert _phases(fixture) == ["preflight", "train", "restore"]

    preflight, train, restore = _entries(fixture)
    assert _option(preflight, "--experimental-seed") == "42"
    assert _option(preflight, "--expected-head") == fixture.head
    assert _option(preflight, "--pool-manifest") == str(
        fixture.pool_manifest
    )
    assert _option(preflight, "--pool-manifest-sha256") == POOL_SHA
    assert _option(preflight, "--config-sha256") == CONFIG_SHA
    assert _option(train, "--phase") == "train"
    assert _option(train, "--run-dir") == str(fixture.run_dir)
    assert _option(restore, "--phase") == "restore"
    assert _option(restore, "--run-dir") == str(fixture.run_dir)

    receipt = fixture.run_dir / "expectations.json"
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert _option(restore, "--receipt-sha256") == digest
    assert f"EVALUATION_RECEIPT_SHA256={digest}" in result.stdout
    assert "GAN_CIC_SEED42_EVALUATION_LAUNCHER=PASS" in result.stdout
    assert "COMPLETE_EVALUATION=False" in result.stdout
    assert "FULL_BENCHMARK_GO=NO" in result.stdout


def test_runtime_environment_is_cpu_only_and_checkout_bound(fixture):
    result = _run(fixture)

    assert result.returncode == 0, result.stderr
    for entry in _entries(fixture):
        assert entry["cwd"] == str(fixture.checkout)
        assert entry["cuda"] == "-1"
        assert entry["hashseed"] == "42"
        assert entry["pythonpath"] == (
            f"{fixture.checkout}/src:{fixture.checkout}"
        )
        assert entry["omp"] == "8"


# ---------------------------------------------------------------------------
# Authorization and source identity rejection
# ---------------------------------------------------------------------------


def test_missing_authorization_blocks_every_phase(fixture):
    fixture.authorization.unlink()
    result = _run(fixture)

    assert result.returncode != 0
    assert "GAN_EVAL_LAUNCHER_BLOCKED" in result.stderr
    assert _entries(fixture) == []


def test_placeholder_authorization_id_blocks_every_phase(fixture):
    _mutate(fixture, authorization_id="RESEARCHER_DECISION_PENDING")
    result = _run(fixture)

    assert result.returncode != 0
    assert "not approved" in result.stderr
    assert _entries(fixture) == []


def test_source_commit_mismatch_blocks(fixture):
    result = _run(
        fixture, overrides={"GAN_EVAL_SOURCE_COMMIT": "f" * 40},
    )

    assert result.returncode != 0
    assert "HEAD differs" in result.stderr
    assert _entries(fixture) == []


def test_dirty_checkout_blocks(fixture):
    (fixture.checkout / "untracked.txt").write_text(
        "dirty\n", encoding="utf-8",
    )
    result = _run(fixture)

    assert result.returncode != 0
    assert "must be clean" in result.stderr
    assert _entries(fixture) == []


def test_launcher_digest_mismatch_blocks(fixture):
    _mutate(fixture, launcher_sha256="0" * 64)
    result = _run(fixture)

    assert result.returncode != 0
    assert "launcher digest mismatch" in result.stderr
    assert _entries(fixture) == []


def test_driver_path_mismatch_blocks(fixture):
    _mutate(fixture, driver_relative_path="scripts/other.py")
    result = _run(fixture)

    assert result.returncode != 0
    assert "field differs" in result.stderr
    assert _entries(fixture) == []


def test_wrong_pool_manifest_digest_blocks(fixture):
    _mutate(fixture, pool_manifest_sha256="0" * 64)
    result = _run(fixture)

    assert result.returncode != 0
    assert "digest differs" in result.stderr
    assert _entries(fixture) == []


def test_wrong_pool_path_blocks(fixture):
    _mutate(fixture, pool_manifest_path="/tmp/other/manifest.json")
    result = _run(fixture)

    assert result.returncode != 0
    assert "pool path" in result.stderr
    assert _entries(fixture) == []


def test_missing_dataset_config_blocks(fixture):
    fixture.dataset_config.unlink()
    result = _run(fixture)

    assert result.returncode != 0
    assert "dataset configuration" in result.stderr
    assert _entries(fixture) == []


def test_missing_weights_blocks(fixture):
    fixture.weights.unlink()
    result = _run(fixture)

    assert result.returncode != 0
    assert "Inception weights" in result.stderr
    assert _entries(fixture) == []


def test_unauthorized_seed_blocks(fixture):
    _mutate(fixture, experimental_seed=43)
    result = _run(fixture)

    assert result.returncode != 0
    assert "seed is not 42" in result.stderr
    assert _entries(fixture) == []


def test_unauthorized_retry_policy_blocks(fixture):
    record = _authorization(fixture)
    record["resources"] = {**record["resources"], "automatic_retries": 1}
    _write_authorization(fixture.authorization, record)
    result = _run(fixture)

    assert result.returncode != 0
    assert "resource differs" in result.stderr
    assert _entries(fixture) == []


# ---------------------------------------------------------------------------
# Slurm allocation rejection
# ---------------------------------------------------------------------------


def _allocation(*, partition="talon", cpus="8", time="02:00:00",
                requeue="0", tres="cpu=8,node=1,mem=32768M"):
    return (
        f"JobId={JOB_ID} JobName=p1r1-gan-eval-s42 Partition={partition} "
        f"JobState=RUNNING NumNodes=1 NumTasks=1 NumCPUs={cpus} "
        f"TRES={tres},billing=8 MinMemoryNode=32768M "
        f"TimeLimit={time} Requeue={requeue} Restarts=0 Gres=(null)"
    )


@pytest.mark.parametrize(
    "line,message",
    (
        (_allocation(partition="debug"), "partition differs"),
        (_allocation(cpus="4"), "NumCPUs"),
        (_allocation(time="01:00:00"), "walltime differs"),
        (_allocation(requeue="1"), "requeue policy differs"),
        (
            _allocation(tres="cpu=8,node=1,mem=32768M,gpu=1"),
            "requests a GPU",
        ),
    ),
)
def test_allocation_mismatch_blocks(fixture, line, message):
    result = _run(fixture, overrides={"FAKE_SCONTROL_LINE": line})

    assert result.returncode != 0
    assert message in result.stderr
    assert _entries(fixture) == []


def test_allocation_lookup_failure_blocks(fixture):
    result = _run(fixture, overrides={"FAKE_SCONTROL_FAIL": "1"})

    assert result.returncode != 0
    assert "allocation lookup failed" in result.stderr
    assert _entries(fixture) == []


def test_missing_slurm_identity_blocks_even_with_authorization(fixture):
    result = _run(fixture, drop=("SLURM_JOB_ID",))

    assert result.returncode != 0
    assert "Slurm allocation required" in result.stderr
    assert _entries(fixture) == []


# ---------------------------------------------------------------------------
# Phase failure and evidence preservation
# ---------------------------------------------------------------------------


def test_failed_preflight_stops_before_training(fixture):
    result = _run(fixture, overrides={"FAKE_PREFLIGHT_FAIL": "1"})

    assert result.returncode == 11
    assert _phases(fixture) == ["preflight"]
    assert not fixture.run_dir.exists()


def test_failed_train_prevents_restore(fixture):
    result = _run(fixture, overrides={"FAKE_TRAIN_FAIL": "1"})

    assert result.returncode == 12
    assert _phases(fixture) == ["preflight", "train"]
    assert not (fixture.run_dir / "expectations.json").exists()


def test_missing_receipt_prevents_restore(fixture):
    result = _run(fixture, overrides={"FAKE_SKIP_RECEIPT": "1"})

    assert result.returncode != 0
    assert _phases(fixture) == ["preflight", "train"]


def test_tampered_receipt_identity_prevents_restore(fixture):
    result = _run(fixture, overrides={"FAKE_BAD_RECEIPT": "1"})

    assert result.returncode != 0
    assert "training receipt identity mismatch" in result.stderr
    assert _phases(fixture) == ["preflight", "train"]


def test_failed_restore_is_not_converted_to_success(fixture):
    result = _run(fixture, overrides={"FAKE_RESTORE_FAIL": "1"})

    assert result.returncode == 13
    assert _phases(fixture) == ["preflight", "train", "restore"]
    assert "GAN_CIC_SEED42_EVALUATION_LAUNCHER=PASS" not in result.stdout
    assert (fixture.run_dir / "expectations.json").is_file()


def test_existing_run_directory_blocks(fixture):
    fixture.run_dir.mkdir()
    result = _run(fixture)

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert _entries(fixture) == []
