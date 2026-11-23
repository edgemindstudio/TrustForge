"""Fixture/static tests for the VAE seed-42 governed evaluation launcher.

No TensorFlow model, dataset content, Slurm submission or classifier
execution is involved. Fixture repositories, authorization records and
allocation fields are created in temporary directories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from trustforge.evaluation import vae_evaluation_launch_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = (
    REPO_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "vae_cic_seed42_bounded_evaluation.slurm"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("vae-launcher")
    checkout = root / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "slurm" / "paper01_revision_2026").mkdir(parents=True)
    (checkout / contract.DRIVER_RELATIVE).write_text(
        "# fixture VAE driver\n", encoding="utf-8",
    )
    (checkout / contract.LAUNCHER_RELATIVE).write_text(
        "# fixture VAE launcher\n", encoding="utf-8",
    )
    _git(checkout, "init", "-q")
    _git(
        checkout, "-c", "user.email=fixture@example.invalid",
        "-c", "user.name=Fixture", "add", "-A",
    )
    _git(
        checkout, "-c", "user.email=fixture@example.invalid",
        "-c", "user.name=Fixture", "commit", "-q", "-m", "fixture",
    )
    head = _git(checkout, "rev-parse", "HEAD")

    home = (root / "home").resolve()
    paths = {
        "authorization": home / contract.AUTH_RELATIVE,
        "pool": home / contract.POOL_RELATIVE,
        "dataset_config": home / contract.DATASET_CONFIG_RELATIVE,
        "weights": home / contract.WEIGHTS_RELATIVE,
        "run_dir": home / contract.EVAL_RELATIVE / "runs" / "attempt01",
    }
    for key in ("authorization", "pool", "dataset_config", "weights"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    paths["pool"].write_text('{"fixture": "pool"}\n', encoding="utf-8")
    paths["dataset_config"].write_text(
        '{"dataset_id": "cicmaldroid2020", "paths": {}}\n', encoding="utf-8",
    )
    paths["weights"].write_text("fixture-weights\n", encoding="utf-8")
    paths["run_dir"].parent.mkdir(parents=True, exist_ok=True)

    accepted = copy.deepcopy(contract.ACCEPTED)
    accepted.update({
        "pool_manifest_sha256": _sha(paths["pool"]),
        "checkpoint_generation_id": "0" * 32,
        "checkpoint_manifest_sha256": "1" * 64,
        "checkpoint_source_commit": "2" * 40,
        "synthesis_source_commit": "3" * 40,
        "synthesis_recipe_sha256": "4" * 64,
        "classifier_recipe_sha256": "5" * 64,
        "metric_recipe_sha256": "6" * 64,
        "dataset_config_sha256": _sha(paths["dataset_config"]),
        "inception_weights_sha256": _sha(paths["weights"]),
    })

    def record():
        return {
            "schema": contract.SCHEMA,
            "paper": contract.PAPER,
            "revision": contract.REVISION,
            "family": contract.FAMILY,
            "dataset_id": contract.DATASET_ID,
            "experimental_seed": contract.SEED,
            "attempt": contract.ATTEMPT,
            "authorization_status": "APPROVED",
            "authorization_id": "P1R2026-VAE-CIC-S42-EVAL-FIXTURE",
            "source_commit": head,
            "launcher_sha256": _sha(checkout / contract.LAUNCHER_RELATIVE),
            "driver_relative_path": contract.DRIVER_RELATIVE,
            "pool_manifest_path": str(paths["pool"]),
            "pool_manifest_sha256": accepted["pool_manifest_sha256"],
            "checkpoint_generation_id":
                accepted["checkpoint_generation_id"],
            "checkpoint_manifest_sha256":
                accepted["checkpoint_manifest_sha256"],
            "checkpoint_source_commit":
                accepted["checkpoint_source_commit"],
            "synthesis_source_commit":
                accepted["synthesis_source_commit"],
            "synthesis_recipe_sha256":
                accepted["synthesis_recipe_sha256"],
            "classifier_recipe_sha256":
                accepted["classifier_recipe_sha256"],
            "metric_recipe_sha256":
                accepted["metric_recipe_sha256"],
            "dataset_config_path": str(paths["dataset_config"]),
            "dataset_config_sha256":
                accepted["dataset_config_sha256"],
            "inception_weights_path": str(paths["weights"]),
            "inception_weights_sha256":
                accepted["inception_weights_sha256"],
            "run_dir": str(paths["run_dir"]),
            "resources": copy.deepcopy(accepted["resources"]),
        }

    return {
        "root": root,
        "checkout": checkout,
        "home": home.resolve(),
        "paths": paths,
        "head": head,
        "accepted": accepted,
        "record": record,
    }


def _allocation(fixture, **overrides):
    values = {
        "JobId": os.environ.get("SLURM_JOB_ID", "12345"),
        "JobName": contract.JOB_NAME,
        "Partition": "talon",
        "JobState": "RUNNING",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "TRES": "cpu=8,mem=32G,node=1",
        "MinMemoryNode": "32768M",
        "TimeLimit": "02:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "",
    }
    values.update(overrides)
    return values


def _write(fixture, document):
    fixture["paths"]["authorization"].write_text(
        json.dumps(document), encoding="utf-8",
    )


def _validate(fixture, document=None, *, allocation=None, **overrides):
    if document is not None:
        _write(fixture, document)
    arguments = {
        "authorization_path": fixture["paths"]["authorization"],
        "checkout": fixture["checkout"],
        "launcher_path": fixture["checkout"] / contract.LAUNCHER_RELATIVE,
        "pool_manifest": fixture["paths"]["pool"],
        "run_dir": fixture["paths"]["run_dir"],
        "dataset_config": fixture["paths"]["dataset_config"],
        "weights": fixture["paths"]["weights"],
        "source_commit": fixture["head"],
        "home": fixture["home"],
        "uid": os.getuid(),
        "accepted": fixture["accepted"],
        "allocation": _allocation(fixture) if allocation is None else allocation,
    }
    arguments.update(overrides)
    return contract.validate_authorization(**arguments)


@pytest.fixture(autouse=True)
def slurm_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")


def test_valid_record_is_accepted(fixture):
    record = _validate(fixture, fixture["record"]())
    assert record["family"] == "vae"
    assert record["experimental_seed"] == 42


def test_missing_authorization_rejected(fixture):
    fixture["paths"]["authorization"].unlink()
    with pytest.raises(contract.LaunchContractError, match="regular file"):
        _validate(fixture)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.__setitem__("authorization_status", "PENDING"),
         "not approved|field differs"),
        (lambda r: r.__setitem__("authorization_id", contract.PLACEHOLDER_ID),
         "not approved"),
        (lambda r: r.__setitem__("family", "gan"), "field differs: family"),
        (lambda r: r.__setitem__("dataset_id", "malex"),
         "field differs: dataset_id"),
        (lambda r: r.__setitem__("experimental_seed", 43), "seed is not 42"),
        (lambda r: r.__setitem__("attempt", 2), "attempt is not 1"),
        (lambda r: r.__setitem__("source_commit", "9" * 40),
         "source differs"),
        (lambda r: r.__setitem__("launcher_sha256", "0" * 64),
         "launcher digest mismatch"),
        (lambda r: r.__setitem__("pool_manifest_path", "/tmp/pool.json"),
         "pool path differs"),
        (lambda r: r.__setitem__("pool_manifest_sha256", "0" * 64),
         "digest differs: pool_manifest_sha256"),
        (lambda r: r.__setitem__("checkpoint_generation_id", "7" * 32),
         "digest differs: checkpoint_generation_id"),
        (lambda r: r.__setitem__("synthesis_source_commit", "0" * 40),
         "digest differs: synthesis_source_commit"),
        (lambda r: r.__setitem__("classifier_recipe_sha256", "0" * 64),
         "digest differs: classifier_recipe_sha256"),
        (lambda r: r.__setitem__("metric_recipe_sha256", "0" * 64),
         "digest differs: metric_recipe_sha256"),
        (lambda r: r.__setitem__("dataset_config_sha256", "0" * 64),
         "digest differs: dataset_config_sha256"),
        (lambda r: r.__setitem__("inception_weights_sha256", "0" * 64),
         "digest differs: inception_weights_sha256"),
        (lambda r: r.__setitem__("run_dir", "/tmp/elsewhere"), "run directory"),
        (lambda r: r["resources"].__setitem__("cpus_per_task", 4),
         "resource differs: cpus_per_task"),
        (lambda r: r["resources"].__setitem__("walltime_seconds", 3600),
         "resource differs: walltime_seconds"),
        (lambda r: r["resources"].__setitem__("gpu_enabled", True),
         "resource differs: gpu_enabled"),
        (lambda r: r.__setitem__("unexpected", True), "fields are invalid"),
    ],
)
def test_invalid_record_rejected(fixture, mutate, message):
    record = fixture["record"]()
    mutate(record)
    with pytest.raises(contract.LaunchContractError, match=message):
        _validate(fixture, record)


def test_wrong_checkout_head_rejected(fixture):
    record = fixture["record"]()
    record["source_commit"] = "8" * 40
    with pytest.raises(
        contract.LaunchContractError, match="HEAD differs|source differs",
    ):
        _validate(fixture, record, source_commit="8" * 40)


def test_missing_driver_rejected(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "scripts").mkdir(parents=True)
    (repo / "slurm" / "paper01_revision_2026").mkdir(parents=True)
    (repo / contract.LAUNCHER_RELATIVE).write_text(
        "# fixture launcher\n", encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(
        repo, "-c", "user.email=fixture@example.invalid",
        "-c", "user.name=Fixture", "add", "-A",
    )
    _git(
        repo, "-c", "user.email=fixture@example.invalid",
        "-c", "user.name=Fixture", "commit", "-q", "-m", "fixture",
    )
    head = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(contract.LaunchContractError, match="driver"):
        contract.validate_checkout(repo, head)


def test_existing_run_destination_rejected(fixture):
    run = fixture["paths"]["run_dir"]
    run.mkdir()
    try:
        with pytest.raises(contract.LaunchContractError, match="already exists"):
            _validate(fixture, fixture["record"]())
    finally:
        run.rmdir()


def test_wrong_dataset_config_path_rejected(fixture):
    record = fixture["record"]()
    record["dataset_config_path"] = "/tmp/other.json"
    with pytest.raises(contract.LaunchContractError, match="dataset configuration path"):
        _validate(fixture, record)


def test_wrong_weights_path_rejected(fixture):
    record = fixture["record"]()
    record["inception_weights_path"] = "/tmp/weights.h5"
    with pytest.raises(contract.LaunchContractError, match="Inception weights path"):
        _validate(fixture, record)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"Partition": "cpu"}, "partition differs"),
        ({"NumCPUs": "4"}, "field differs: NumCPUs"),
        ({"TRES": "cpu=4,mem=32G,node=1"}, "TRES CPU"),
        ({"TimeLimit": "01:00:00"}, "walltime differs"),
        ({"Requeue": "1"}, "requeue policy"),
        ({"Restarts": "1"}, "restarted"),
        ({"Gres": "gpu:1"}, "requests a GPU"),
        ({"JobName": "p1r1-gan-eval-s42"}, "job name differs"),
        ({"JobState": "PENDING"}, "not running"),
    ],
)
def test_wrong_allocation_rejected(fixture, overrides, message):
    with pytest.raises(contract.LaunchContractError, match=message):
        _validate(
            fixture, fixture["record"](),
            allocation=_allocation(fixture, **overrides),
        )


def test_duplicate_json_field_rejected(fixture):
    record = fixture["record"]()
    raw = json.dumps(record)
    raw = raw[:-1] + ', "family": "vae"}'
    fixture["paths"]["authorization"].write_text(raw, encoding="utf-8")
    with pytest.raises(contract.LaunchContractError, match="duplicate JSON field"):
        _validate(fixture)


# ---------------------------------------------------------------------------
# Static launcher contract
# ---------------------------------------------------------------------------


def _launcher_text():
    return LAUNCHER_PATH.read_text(encoding="utf-8")


def test_launcher_shell_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER_PATH)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_uses_vae_driver_argument_contract():
    text = _launcher_text()
    assert "--evaluator-seed 42" in text
    assert "--preflight-only" in text
    assert "--phase train" in text
    assert "--phase restore" in text
    assert "--receipt-sha256 \"$RECEIPT_SHA\"" in text
    assert "--experimental-seed" not in text
    assert "run_paper01_cic_vae_seed42_bounded_evaluation.py" in text


def test_launcher_sequence_preflight_train_receipt_restore():
    text = _launcher_text()
    preflight = text.index("--preflight-only")
    train = text.index("--phase train")
    receipt = text.index('RECEIPT="$RUN_DIR/expectations.json"')
    restore = text.index("--phase restore")
    assert preflight < train < receipt < restore
    assert text.index("vae_evaluation_launch_contract") < preflight
    assert "check_source" in text[:preflight]


def test_launcher_has_no_gan_or_maf_assumptions():
    text = _launcher_text().lower()
    for forbidden in (
        "gan_cic", "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1",
        "963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530",
        "experimental-seed", "maf_cic", "625 ordered batches",
    ):
        assert forbidden not in text


def test_launcher_declares_governed_resources():
    text = _launcher_text()
    assert "#SBATCH --partition=talon" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --no-requeue" in text
    assert "CUDA_VISIBLE_DEVICES=-1" in text
