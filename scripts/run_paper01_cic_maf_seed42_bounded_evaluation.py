#!/usr/bin/env python3
"""Bounded CICMalDroid2020 MAF seed-42 real-data evaluation driver.

Connects the accepted MAF/CIC seed-42 synthetic pool to the existing
Paper 1 evaluation components. It does not define new scientific policy,
new metrics, or a general orchestration framework.

Modes:
  --preflight-only   Read-only authentication; writes nothing.
  --phase train      Paired classifier training/publication; never touches test.
  --phase restore    Fresh-process restoration, test evaluation, fidelity and
                     unified validation.

Unified validation retains complete_evaluation=false. Generator-lineage
authentication is recorded as separate driver evidence because the unified
validator reports only the checks it performs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np


DATASET_ID = "cicmaldroid2020"
FAMILY = "maf"
EXPERIMENTAL_SEED = 42
CLASS_COUNT = 5
POOL_SAMPLES = 10_000
POOL_BATCHES = 1_250
SAMPLES_PER_CLASS = 2_000
TRAIN_SIZE = 8_117
VALIDATION_SIZE = 1_158
TEST_SIZE = 2_323
SPLIT_COUNTS = {
    "train": TRAIN_SIZE, "validation": VALIDATION_SIZE, "test": TEST_SIZE,
}
SPLIT_CLASS_COUNTS = {
    "train": (877, 1470, 2732, 1782, 1256),
    "validation": (125, 210, 390, 254, 179),
    "test": (251, 420, 782, 510, 360),
}

POOL_MANIFEST_SHA256 = (
    "db884e298ea324597bad2d488d03e2c86c205e7a53bbef2a11bd72f06da5651e"
)
POOL_CHECKPOINT_SHA256 = (
    "3604bd4009e2199400665b28ef156d5db8e6d2e75094e5eb1d193b090d89264d"
)
GENERATOR_RECIPE_SHA256 = (
    "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
)
CLASSIFIER_RECIPE_SHA256 = (
    "1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982"
)
METRIC_RECIPE_SHA256 = (
    "d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85"
)
DATASET_CONFIG_SHA256 = (
    "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
)
CHECKPOINT_SOURCE_COMMIT = (
    "913d86f9c64e6a3a1a0d009f112bcfa889a02ec8"
)
CHECKPOINT_GENERATION = "maf_cic_seed42_execution"
CHECKPOINT_FITTED_STATE_SHA256 = (
    "c177c31e8602e94b06dc965e14dafe8ec3374fd07816b93107df3f5c1fc7f5cf"
)
CHECKPOINT_EXECUTION_MANIFEST_SHA256 = (
    "d82709d879a173e3b0ada30d7e3db1b5f9dbfba823db64a5bc3ad41d52b3240f"
)
CHECKPOINT_EXPECTATIONS_SHA256 = (
    "2e8f64a58b8900c9d6c6fd5580794c8374353abc14e51fab49e8155d675526ec"
)
CHECKPOINT_COMPLETED_STEPS = 254
INCEPTION_WEIGHTS_SHA256 = (
    "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
)
INCEPTION_WEIGHTS_BYTES = 87_910_968

GENERATOR_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
)
CLASSIFIER_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
)
METRIC_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
)

RECEIPT_SCHEMA = "trustforge.paper01.paired-restoration-expectations.v1"
CHECKPOINT_EXPECTATIONS_SCHEMA = (
    "trustforge.paper01.maf-cic.execution-expectations.v1"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Generic evidence helpers
# ---------------------------------------------------------------------------


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(name + " requires a lowercase SHA-256")
    return value


def _require_commit(value, *, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise ValueError(name + " requires a 40-character commit")
    return value


def _read_regular(path, *, name, max_bytes=MAX_JSON_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(name + " must be a regular file")
    if path.stat().st_size > max_bytes:
        raise ValueError(name + " exceeds the bounded size limit")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(name + " must not be empty")
    return raw


def _write_exclusive_bytes(path, raw):
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return raw


def _write_exclusive_json(path, value):
    return _write_exclusive_bytes(path, _canonical(value))


def _write_exclusive_npy(path, array):
    if not isinstance(array, np.ndarray):
        raise TypeError("exact ndarray required")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return _write_exclusive_bytes(path, buffer.getvalue())


def _write_exclusive_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return _write_exclusive_bytes(path, buffer.getvalue())


def _require_empty_run_dir(path):
    run = Path(path).expanduser()
    if run.is_symlink() or run.exists():
        raise FileExistsError("evaluation run directory already exists")
    run.mkdir(mode=0o700)
    return run


def _require_existing_run_dir(path):
    run = Path(path).expanduser()
    if run.is_symlink() or not run.is_dir():
        raise ValueError("evaluation run directory unavailable")
    return run


# ---------------------------------------------------------------------------
# Source and input authentication
# ---------------------------------------------------------------------------


def _capture_source(root):
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )

    return capture_source_identity(root)


def _verify_source(root, expected_head):
    root = Path(root).expanduser().resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError("execution checkout mismatch")
    _require_commit(expected_head, name="expected source revision")
    head, sources = _capture_source(root)
    if head != expected_head:
        raise RuntimeError("unexpected source revision")
    status = subprocess.check_output(
        [
            "git", "-C", str(root), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked working tree must be clean")
    untracked = subprocess.check_output(
        [
            "git", "-C", str(root), "ls-files", "--others",
            "--exclude-standard", "--",
            "scripts", "src", "slurm", "common", "studies",
        ],
        text=True,
    )
    if untracked.strip():
        raise RuntimeError("untracked executable source must be resolved")
    return head, sources


def _verify_imports(root, sources):
    root = Path(root).resolve()
    count = 0
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "trustforge" or name.startswith("trustforge.")
            or name == "common" or name.startswith("common.")
        ):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            for directory in getattr(module, "__path__", ()):
                Path(directory).resolve().relative_to(root)
            continue
        path = Path(filename).resolve()
        relative = path.relative_to(root).as_posix()
        if sources.get(relative) != _sha(path.read_bytes()):
            raise RuntimeError("project import differs from captured source: " + name)
        count += 1
    return count


def _load_dataset_config(path, expected_sha):
    raw = _read_regular(path, name="dataset configuration")
    observed = _sha(raw)
    if observed != _require_digest(expected_sha, name="dataset configuration"):
        raise ValueError("dataset configuration digest mismatch")
    if observed != DATASET_CONFIG_SHA256:
        raise ValueError("dataset configuration differs from accepted evidence")
    config = json.loads(raw)
    if (
        type(config) is not dict
        or set(config) != {"dataset_id", "paths"}
        or config["dataset_id"] != DATASET_ID
        or type(config["paths"]) is not dict
        or set(config["paths"]) != {"source_csv", "feature_columns_path"}
        or any(
            type(value) is not str or not value.strip()
            for value in config["paths"].values()
        )
    ):
        raise ValueError("invalid CIC dataset configuration")
    return raw, config


def _load_recipes(root):
    root = Path(root)
    recipes = {
        "generator": _read_regular(
            root / GENERATOR_RECIPE_PATH, name="generator recipe",
        ),
        "classifier": _read_regular(
            root / CLASSIFIER_RECIPE_PATH, name="classifier recipe",
        ),
        "metric": _read_regular(
            root / METRIC_RECIPE_PATH, name="downstream metric recipe",
        ),
    }
    for name, expected in (
        ("generator", GENERATOR_RECIPE_SHA256),
        ("classifier", CLASSIFIER_RECIPE_SHA256),
        ("metric", METRIC_RECIPE_SHA256),
    ):
        if _sha(recipes[name]) != expected:
            raise ValueError(name + " recipe identity mismatch")
    return recipes


def _verify_split_identities(adapter, *, include_test):
    splits = ("train", "validation", "test") if include_test else (
        "train", "validation",
    )
    sizes, counts = {}, {}
    for split in splits:
        size = adapter.split_size(split)
        if type(size) is not int or size != SPLIT_COUNTS[split]:
            raise ValueError("governed split count mismatch: " + split)
        labels = np.asarray(adapter.labels(split))
        if labels.ndim != 1 or labels.size != size:
            raise ValueError("governed split label count mismatch: " + split)
        observed = tuple(
            int(value)
            for value in np.bincount(
                labels.astype(np.int64), minlength=CLASS_COUNT,
            ).tolist()
        )
        if observed != SPLIT_CLASS_COUNTS[split]:
            raise ValueError("governed split class counts mismatch: " + split)
        sizes[split], counts[split] = size, observed
    return {"split_sizes": sizes, "split_class_counts": counts}


def _test_identity(provenance_sha256):
    return {
        "dataset_id": DATASET_ID,
        "split": "test",
        "size": TEST_SIZE,
        "class_count": CLASS_COUNT,
        "declared_provenance_sha256": provenance_sha256,
    }


def _authenticate_pool_manifest_raw(
    raw, *, expected_sha, expected_checkpoint, expected_recipe,
):
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        synthesis_batch_seed,
    )

    observed = _sha(raw)
    if observed != _require_digest(expected_sha, name="pool manifest"):
        raise ValueError("pool manifest digest mismatch")
    if observed != POOL_MANIFEST_SHA256:
        raise ValueError("pool manifest differs from accepted evidence")
    manifest = json.loads(raw)
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("family") != FAMILY
        or manifest.get("experimental_seed") != EXPERIMENTAL_SEED
        or manifest.get("recipe_sha256") != expected_recipe
        or manifest.get("checkpoint_sha256") != expected_checkpoint
    ):
        raise ValueError("pool manifest identity mismatch")
    if expected_recipe != GENERATOR_RECIPE_SHA256:
        raise ValueError("pool recipe differs from accepted generator recipe")
    if expected_checkpoint != POOL_CHECKPOINT_SHA256:
        raise ValueError("pool checkpoint differs from accepted checkpoint")

    records = manifest.get("batches")
    if type(records) is not list or len(records) != POOL_BATCHES:
        raise ValueError("complete pool requires 1250 ordered batches")
    indexed = {}
    for record in records:
        if type(record) is not dict:
            raise ValueError("invalid pool batch record")
        class_id = record.get("class_id")
        batch_index = record.get("batch_index")
        if (
            type(class_id) is not int or not 0 <= class_id < CLASS_COUNT
            or type(batch_index) is not int
            or not 0 <= batch_index < SAMPLES_PER_CLASS // 8
        ):
            raise ValueError("invalid pool class or batch index")
        key = (class_id, batch_index)
        if key in indexed:
            raise ValueError("duplicate pool batch coverage")
        filename = record.get("filename")
        if (
            type(filename) is not str
            or not filename.endswith(".npy")
            or Path(filename).name != filename
        ):
            raise ValueError("invalid pool batch filename")
        _require_digest(record.get("sha256"), name="pool batch")
        seed = synthesis_batch_seed(
            dataset_id=DATASET_ID, family=FAMILY,
            experimental_seed=EXPERIMENTAL_SEED,
            class_id=class_id, batch_index=batch_index,
        )
        if record.get("sampling_seed") != seed:
            raise ValueError("pool batch sampling seed mismatch")
        indexed[key] = record
    required = {
        (class_id, batch_index)
        for class_id in range(CLASS_COUNT)
        for batch_index in range(SAMPLES_PER_CLASS // 8)
    }
    if set(indexed) != required:
        raise ValueError("incomplete pool batch coverage")
    return manifest


def _authenticate_expectations(raw, expected_sha, config_sha):
    observed = _sha(raw)
    if observed != _require_digest(
        expected_sha, name="checkpoint expectations",
    ):
        raise ValueError("checkpoint expectations digest mismatch")
    if observed != CHECKPOINT_EXPECTATIONS_SHA256:
        raise ValueError("checkpoint expectations differ from accepted evidence")
    receipt = json.loads(raw)
    if (
        type(receipt) is not dict
        or receipt.get("schema") != CHECKPOINT_EXPECTATIONS_SCHEMA
        or receipt.get("source_commit") != CHECKPOINT_SOURCE_COMMIT
        or receipt.get("generation_name") != CHECKPOINT_GENERATION
        or receipt.get("job_id") != "311675"
        or receipt.get("fitted_state_sha256")
        != CHECKPOINT_FITTED_STATE_SHA256
        or receipt.get("execution_manifest_file_sha256")
        != CHECKPOINT_EXECUTION_MANIFEST_SHA256
        or receipt.get("configuration_sha256") != config_sha
        or type(receipt.get("training_pid")) is not int
    ):
        raise ValueError("checkpoint execution receipt mismatch")
    return receipt


def _historical_source_files(root, commit):
    def git(*command):
        return subprocess.check_output(["git", "-C", str(root), *command])

    names = git("ls-tree", "-r", "--name-only", "-z", commit)
    return {
        name: hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest()
        for name in names.decode().split("\x00")
        if name.endswith(".py") or name == GENERATOR_RECIPE_PATH
    }


def _lineage_summary(
    *, pool_identity, checkpoint_manifest, best_raw, execution_manifest,
    receipt_sha256, fitted_state_sha256, source_commit,
):
    best_sha = _sha(best_raw)
    envelope_sha = _sha(_canonical(execution_manifest) + b"\n")
    if (
        source_commit != CHECKPOINT_SOURCE_COMMIT
        or fitted_state_sha256 != CHECKPOINT_FITTED_STATE_SHA256
        or receipt_sha256 != CHECKPOINT_EXPECTATIONS_SHA256
        or envelope_sha != CHECKPOINT_EXECUTION_MANIFEST_SHA256
        or best_sha != POOL_CHECKPOINT_SHA256
        or pool_identity.get("checkpoint_sha256") != best_sha
        or pool_identity.get("recipe_sha256") != GENERATOR_RECIPE_SHA256
        or pool_identity.get("manifest_sha256") != POOL_MANIFEST_SHA256
        or checkpoint_manifest.get("source_commit") != CHECKPOINT_SOURCE_COMMIT
        or checkpoint_manifest.get("completed_training_steps")
        != CHECKPOINT_COMPLETED_STEPS
        or checkpoint_manifest.get("artifacts", {}).get("best", {}).get("sha256")
        != best_sha
    ):
        raise ValueError("generator checkpoint lineage mismatch")
    summary = {
        "schema": "trustforge.paper01.cic-maf.seed42.lineage-evidence.v1",
        "dataset_id": DATASET_ID,
        "family": FAMILY,
        "experimental_seed": EXPERIMENTAL_SEED,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "pool_checkpoint_sha256": pool_identity["checkpoint_sha256"],
        "pool_recipe_sha256": pool_identity["recipe_sha256"],
        "checkpoint_generation": CHECKPOINT_GENERATION,
        "checkpoint_source_commit": source_commit,
        "checkpoint_best_state_sha256": best_sha,
        "checkpoint_fitted_state_sha256": fitted_state_sha256,
        "checkpoint_execution_manifest_sha256": envelope_sha,
        "checkpoint_expectations_sha256": receipt_sha256,
        "checkpoint_completed_training_steps": CHECKPOINT_COMPLETED_STEPS,
        "lineage_authentication": "DRIVER_VERIFIED",
        "semantic_class_fidelity": "NOT_ESTABLISHED",
        "complete_evaluation": False,
    }
    summary["sha256"] = _sha(_canonical(summary))
    return summary


# ---------------------------------------------------------------------------
# Component loading and input session
# ---------------------------------------------------------------------------


def _components():
    from trustforge.data.paper01_revision_2026.cicmaldroid import (
        CICMalDroidRevision2026,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
        read_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_dataset_preflight import (
        preflight_dataset,
    )
    from trustforge.evaluation.paper01_cic_pool import CICPoolView
    from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
    from trustforge.evaluation.paper01_inception import verify_weight_file
    from trustforge.evaluation.paper01_inputs import RealSplitView

    return {
        "adapter": CICMalDroidRevision2026,
        "preflight_dataset": preflight_dataset,
        "read_checkpoint_bundle": read_checkpoint_bundle,
        "pool": CICPoolView,
        "combined": CombinedTrainingView,
        "real_view": RealSplitView,
        "verify_weight_file": verify_weight_file,
    }


def _prepare_inputs(args, *, include_test_identity):
    root = Path(args.repository_root).expanduser().resolve()
    head, sources = _verify_source(root, args.expected_head)
    config_raw, config = _load_dataset_config(
        args.dataset_config, args.config_sha256,
    )
    recipes = _load_recipes(root)
    components = _components()
    adapter = components["adapter"](**config["paths"], validate_hashes=True)
    evidence = components["preflight_dataset"](
        family=FAMILY, dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED, adapter=adapter,
        recipe_raw=recipes["generator"],
    )
    identity = evidence["identity"]
    if (
        identity.get("family") != FAMILY
        or identity.get("dataset_id") != DATASET_ID
        or identity.get("experimental_seed") != EXPERIMENTAL_SEED
        or identity.get("class_count") != CLASS_COUNT
        or identity.get("class_mapping") != ["0", "1", "2", "3", "4"]
    ):
        raise ValueError("governed checkpoint identity mismatch")
    provenance_sha256 = _require_digest(
        identity.get("dataset_provenance_sha256"),
        name="dataset provenance",
    )
    split_evidence = _verify_split_identities(
        adapter, include_test=include_test_identity,
    )

    pool_raw = _read_regular(args.pool_manifest, name="pool manifest")
    manifest = _authenticate_pool_manifest_raw(
        pool_raw,
        expected_sha=args.pool_manifest_sha256,
        expected_checkpoint=POOL_CHECKPOINT_SHA256,
        expected_recipe=GENERATOR_RECIPE_SHA256,
    )
    if manifest["dataset_id"] != DATASET_ID:
        raise ValueError("pool dataset mismatch")
    pool = components["pool"](
        args.pool_manifest,
        manifest_sha256=args.pool_manifest_sha256,
        family=FAMILY, experimental_seed=EXPERIMENTAL_SEED,
        recipe_sha256=GENERATOR_RECIPE_SHA256,
        checkpoint_sha256=POOL_CHECKPOINT_SHA256,
    )
    if pool.identity["manifest_sha256"] != POOL_MANIFEST_SHA256:
        raise ValueError("constructed pool identity mismatch")

    expectations_raw = _read_regular(
        args.checkpoint_expectations, name="checkpoint expectations",
    )
    expectations = _authenticate_expectations(
        expectations_raw, args.checkpoint_expectations_sha256,
        _sha(config_raw),
    )
    publication_parent = Path(
        args.checkpoint_publication_parent,
    ).expanduser()
    if publication_parent.is_symlink() or not publication_parent.is_dir():
        raise ValueError("checkpoint publication parent unavailable")
    if args.checkpoint_generation_name != CHECKPOINT_GENERATION:
        raise ValueError("unexpected checkpoint generation")
    historical = _historical_source_files(root, CHECKPOINT_SOURCE_COMMIT)
    loaded = components["read_checkpoint_bundle"](
        publication_parent / args.checkpoint_generation_name,
        recipe_raw=recipes["generator"],
        expected_identity=identity,
        expected_source_commit=CHECKPOINT_SOURCE_COMMIT,
        expected_source_files=historical,
        expected_fitted_state_sha256=CHECKPOINT_FITTED_STATE_SHA256,
    )
    lineage = _lineage_summary(
        pool_identity=pool.identity,
        checkpoint_manifest=loaded["manifest"],
        best_raw=loaded["best_raw"],
        execution_manifest=loaded["execution_manifest"],
        receipt_sha256=_sha(expectations_raw),
        fitted_state_sha256=expectations["fitted_state_sha256"],
        source_commit=loaded["manifest"]["source_commit"],
    )
    components["verify_weight_file"](args.weights)
    _verify_imports(root, sources)
    return {
        "root": root,
        "head": head,
        "sources": sources,
        "config_raw": config_raw,
        "config": config,
        "recipes": recipes,
        "components": components,
        "adapter": adapter,
        "identity": identity,
        "provenance_sha256": provenance_sha256,
        "split_evidence": split_evidence,
        "pool": pool,
        "pool_raw": pool_raw,
        "pool_sha256": POOL_MANIFEST_SHA256,
        "expectations_raw": expectations_raw,
        "lineage": lineage,
    }


def _shared_identity(session, environment):
    return {
        "source_commit": session["head"],
        "source_files_sha256": _sha(_canonical(session["sources"])),
        "environment_sha256": _sha(_canonical(environment)),
    }


def _environment():
    import keras
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "hostname": platform.node(),
        "settings": {
            key: os.environ.get(key) for key in (
                "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
            )
        },
    }


def _build_training_views(session, components):
    real = components["real_view"](
        adapter=session["adapter"], dataset_id=DATASET_ID,
        split="train", provenance_sha256=session["provenance_sha256"],
    )
    validation = components["real_view"](
        adapter=session["adapter"], dataset_id=DATASET_ID,
        split="validation", provenance_sha256=session["provenance_sha256"],
    )
    augmented = components["combined"](
        real=real, synthetic=session["pool"],
    )
    return real, validation, augmented


def _require_training_evidence(paired, *, augmented_identity):
    if (
        type(paired) is not dict
        or paired.get("test_evaluated") is not False
        or paired.get("complete_evaluation") is not False
    ):
        raise ValueError("training evidence must remain pre-test and incomplete")
    arms = paired.get("arms")
    if type(arms) is not dict or set(arms) != {
        "real_only", "real_plus_synthetic",
    }:
        raise ValueError("exactly two paired classifier arms required")
    if arms["real_only"].get("pool_consumption") is not None:
        raise ValueError("real-only arm must not report pool consumption")
    consumption = arms["real_plus_synthetic"].get("pool_consumption")
    if type(consumption) is not dict or not consumption:
        raise ValueError("augmented pool consumption record unavailable")
    reports = {}
    for name, arm in arms.items():
        report = arm.get("report")
        if type(report) is not dict:
            raise ValueError("classifier training report required")
        _require_digest(
            report.get("selected_weights_sha256"),
            name="selected classifier weights",
        )
        reports[name] = report
    if (
        reports["real_only"].get("initial_weights_sha256")
        != reports["real_plus_synthetic"].get("initial_weights_sha256")
    ):
        raise ValueError("paired initial classifier weights differ")
    synthetic = augmented_identity.get("synthetic")
    if (
        type(synthetic) is not dict
        or synthetic.get("manifest_sha256") != POOL_MANIFEST_SHA256
        or synthetic.get("checkpoint_sha256") != POOL_CHECKPOINT_SHA256
        or synthetic.get("family") != FAMILY
        or synthetic.get("experimental_seed") != EXPERIMENTAL_SEED
    ):
        raise ValueError("augmented training view differs from accepted pool")
    return consumption


def _assert_incomplete_report(value):
    if (
        type(value) is not dict
        or value.get("complete_evaluation") is not False
        or value.get("full_benchmark_go") is not False
    ):
        raise ValueError(
            "evidence must preserve complete_evaluation=false and "
            "full_benchmark_go=false"
        )
    return value


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _run_preflight(args):
    session = _prepare_inputs(args, include_test_identity=True)
    pool_evidence = session["pool"].preflight()
    if (
        pool_evidence.get("verified_batches") != POOL_BATCHES
        or pool_evidence.get("verified_images") != POOL_SAMPLES
    ):
        raise RuntimeError("accepted pool preflight did not cover the pool")
    summary = {
        "scope": "CIC_MAF_SEED42_EVALUATION_PREFLIGHT",
        "source_commit": session["head"],
        "dataset_config_sha256": _sha(session["config_raw"]),
        "split_identity": session["split_evidence"],
        "generator_identity": session["identity"],
        "pool": {
            "manifest_sha256": session["pool_sha256"],
            "identity": session["pool"].identity,
            "verified_batches": pool_evidence["verified_batches"],
            "verified_images": pool_evidence["verified_images"],
        },
        "generator_lineage": session["lineage"],
        "recipes": {
            "generator_sha256": _sha(session["recipes"]["generator"]),
            "classifier_sha256": _sha(session["recipes"]["classifier"]),
            "metric_sha256": _sha(session["recipes"]["metric"]),
        },
        "feature_extractor": {
            "weights_sha256": INCEPTION_WEIGHTS_SHA256,
            "weights_bytes": INCEPTION_WEIGHTS_BYTES,
        },
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    _assert_incomplete_report(summary)
    print("CIC_MAF_SEED42_EVALUATION_PREFLIGHT=PASS")
    print(_canonical(summary).decode("utf-8"), flush=True)
    return summary


# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------


def _run_train(args):
    run = _require_empty_run_dir(args.run_dir)
    session = _prepare_inputs(args, include_test_identity=False)
    components = session["components"]
    from trustforge.evaluation.paper01_classifier_bundle import (
        publish_classifier_bundle,
    )
    from trustforge.evaluation.paper01_classifier_manifest import (
        canonical_manifest, prepare_classifier_manifest,
    )
    from trustforge.evaluation.paper01_classifier_state import (
        encode_classifier_weights,
    )
    from trustforge.evaluation.paper01_classifier_tensorflow import (
        prepare_paired_classifiers, require_compute_node,
    )

    require_compute_node()
    real, validation, augmented = _build_training_views(session, components)
    if augmented.identity.get("real") != real.identity:
        raise ValueError("augmented arm uses a different real training view")
    paired = prepare_paired_classifiers(
        real_training=real, augmented_training=augmented,
        validation=validation, evaluator_seed=EXPERIMENTAL_SEED,
        recipe_raw=session["recipes"]["classifier"],
    )
    consumption = _require_training_evidence(
        paired, augmented_identity=augmented.identity,
    )
    environment = _environment()
    shared = _shared_identity(session, environment)

    expectations, prepared = {}, {}
    for name, training_view in (
        ("real_only", real), ("real_plus_synthetic", augmented),
    ):
        arm = paired["arms"][name]
        context = {
            **shared,
            "dataset_id": DATASET_ID,
            "evaluator_seed": EXPERIMENTAL_SEED,
            "arm": name,
            "training_sample_count": training_view.size,
            "validation_sample_count": validation.size,
            "training_identity_sha256": _sha(
                _canonical(training_view.identity)
            ),
            "validation_identity_sha256": _sha(_canonical(validation.identity)),
        }
        selected = arm["report"]["selected_weights_sha256"]
        state_raw = encode_classifier_weights(
            weights=arm["selected_weights"], dataset_id=DATASET_ID,
            recipe_raw=session["recipes"]["classifier"],
            expected_selected_weights_sha256=selected,
        )
        manifest_raw = canonical_manifest(prepare_classifier_manifest(
            context=context, training_report=arm["report"],
            state_raw=state_raw, recipe_raw=session["recipes"]["classifier"],
        ))
        prepared[name] = (manifest_raw, state_raw)
        expectations[name] = {
            "context": context,
            "manifest_sha256": _sha(manifest_raw),
            "selected_weights_sha256": selected,
        }

    checkpoints = run / "checkpoints"
    checkpoints.mkdir(mode=0o700)
    for name, (manifest_raw, state_raw) in prepared.items():
        expected = expectations[name]
        publish_classifier_bundle(
            checkpoints, name,
            manifest_raw=manifest_raw, state_raw=state_raw,
            recipe_raw=session["recipes"]["classifier"],
            expected_manifest_sha256=expected["manifest_sha256"],
            expected_context=expected["context"],
            expected_selected_weights_sha256=expected["selected_weights_sha256"],
        )

    context_raw = _write_exclusive_json(run / "context_evidence.json", {
        "source_files": session["sources"],
        "environment": environment,
        "real_training_identity": real.identity,
        "augmented_training_identity": augmented.identity,
        "validation_identity": validation.identity,
        "synthetic_pool_identity": session["pool"].identity,
        "generator_lineage": session["lineage"],
        "dataset_config_sha256": _sha(session["config_raw"]),
    })
    _write_exclusive_json(run / "pool_consumption.json", consumption)
    _write_exclusive_json(run / "lineage_evidence.json", session["lineage"])

    test_identity = _test_identity(session["provenance_sha256"])
    train_report = {
        "scope": "CIC_MAF_SEED42_BOUNDED_REAL_DATA_TRAIN",
        "source_commit": session["head"],
        "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(),
        "arms": {
            name: arm["report"] for name, arm in paired["arms"].items()
        },
        "initial_weights_sha256": paired["initial_weights_sha256"],
        "pool_consumption_sha256": _sha(_canonical(consumption)),
        "generator_lineage": session["lineage"],
        "test_identity": test_identity,
        "test_accessed": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    _assert_incomplete_report(train_report)
    # Required training evidence must be durable before the restore-authorizing
    # receipt becomes discoverable.
    _write_exclusive_json(run / "train_report.json", train_report)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        **shared,
        "job_id": os.environ["SLURM_JOB_ID"],
        "train_pid": os.getpid(),
        "evaluator_seed": EXPERIMENTAL_SEED,
        "initial_weights_sha256": paired["initial_weights_sha256"],
        "classifier_recipe_sha256": _sha(session["recipes"]["classifier"]),
        "metric_recipe_sha256": _sha(session["recipes"]["metric"]),
        "test_identity": test_identity,
        "synthetic_pool_manifest_sha256": session["pool_sha256"],
        "generator_lineage_sha256": session["lineage"]["sha256"],
        "context_evidence_sha256": _sha(context_raw),
        "arms": expectations,
    }
    receipt_raw = _write_exclusive_json(run / "expectations.json", receipt)

    _verify_imports(session["root"], session["sources"])
    _verify_source(session["root"], args.expected_head)
    print("CIC_MAF_SEED42_EVALUATION_TRAIN=PASS")
    print("EXPECTATIONS_SHA256=" + _sha(receipt_raw), flush=True)
    return train_report


# ---------------------------------------------------------------------------
# Restore phase
# ---------------------------------------------------------------------------


def _authenticate_receipt(
    raw, expected_sha, *, shared, classifier_recipe_sha256,
    metric_recipe_sha256, test_identity, pool_manifest_sha256,
    lineage_sha256, job_id, pid,
):
    if _sha(raw) != _require_digest(expected_sha, name="training receipt"):
        raise ValueError("training receipt digest mismatch")
    receipt = json.loads(raw)
    if (
        type(receipt) is not dict
        or receipt.get("schema") != RECEIPT_SCHEMA
    ):
        raise ValueError("unsupported training receipt schema")
    for key, value in shared.items():
        if receipt.get(key) != value:
            raise ValueError("training receipt source/environment mismatch")
    if (
        receipt.get("job_id") != job_id
        or type(receipt.get("train_pid")) is not int
        or receipt["train_pid"] == pid
        or receipt.get("evaluator_seed") != EXPERIMENTAL_SEED
        or receipt.get("classifier_recipe_sha256") != classifier_recipe_sha256
        or receipt.get("metric_recipe_sha256") != metric_recipe_sha256
        or receipt.get("test_identity") != test_identity
        or receipt.get("synthetic_pool_manifest_sha256") != pool_manifest_sha256
        or receipt.get("generator_lineage_sha256") != lineage_sha256
    ):
        raise ValueError("training receipt identity mismatch")
    _require_digest(
        receipt.get("context_evidence_sha256"),
        name="receipt context evidence",
    )
    arms = receipt.get("arms")
    if type(arms) is not dict or set(arms) != {
        "real_only", "real_plus_synthetic",
    }:
        raise ValueError("training receipt requires both arms")
    for name, arm in arms.items():
        if type(arm) is not dict or type(arm.get("context")) is not dict:
            raise ValueError("invalid training receipt arm: " + name)
        _require_digest(arm.get("manifest_sha256"), name="classifier manifest")
        _require_digest(
            arm.get("selected_weights_sha256"), name="selected weights",
        )
    return receipt


def _require_verified_evaluation(summary):
    if (
        type(summary) is not dict
        or summary.get("complete_evaluation") is not False
        or summary.get("verified_group_count") != 7
        or summary.get("required_status_blockers") != []
        or summary.get("lineage_blockers") != []
        or type(summary.get("verified_pool_consumption")) is not dict
    ):
        raise ValueError(
            "unified evaluation must verify all groups and retain "
            "complete_evaluation=false"
        )
    return summary


def _run_restore(args):
    run = _require_existing_run_dir(args.run_dir)
    session = _prepare_inputs(args, include_test_identity=False)
    from trustforge.evaluation.paper01_classifier_control import state_digest
    from trustforge.evaluation.paper01_classifier_restoration import (
        restore_selected_classifier,
    )
    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node,
    )
    from trustforge.evaluation.paper01_evaluation_validation import (
        verify_evaluation,
    )
    from trustforge.evaluation.paper01_fid import (
        complete_class_fid, feature_fid,
    )
    from trustforge.evaluation.paper01_fidelity import prepare_features_and_kid
    from trustforge.evaluation.paper01_inception import InceptionFeatures
    from trustforge.evaluation.paper01_ms_ssim import ThreeScaleMSSSIM
    from trustforge.evaluation.paper01_ms_ssim_evaluation import (
        evaluate_ms_ssim_pairs,
    )
    from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
    from trustforge.evaluation.paper01_paired_results import link_paired_results
    from trustforge.evaluation.paper01_predictions import (
        collect_test_predictions,
    )
    from trustforge.evaluation.paper01_reference_selection import (
        select_balanced_references,
    )
    from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS

    require_compute_node()
    environment = _environment()
    shared = _shared_identity(session, environment)
    receipt_raw = _read_regular(run / "expectations.json", name="training receipt")
    receipt = _authenticate_receipt(
        receipt_raw, args.receipt_sha256,
        shared=shared,
        classifier_recipe_sha256=_sha(session["recipes"]["classifier"]),
        metric_recipe_sha256=_sha(session["recipes"]["metric"]),
        test_identity=_test_identity(session["provenance_sha256"]),
        pool_manifest_sha256=session["pool_sha256"],
        lineage_sha256=session["lineage"]["sha256"],
        job_id=os.environ["SLURM_JOB_ID"],
        pid=os.getpid(),
    )
    context_raw = _read_regular(
        run / "context_evidence.json", name="context evidence",
    )
    if _sha(context_raw) != receipt["context_evidence_sha256"]:
        raise ValueError("context evidence digest mismatch")
    context = json.loads(context_raw)
    consumption_raw = _read_regular(
        run / "pool_consumption.json", name="pool consumption",
    )
    consumption = json.loads(consumption_raw)
    if _sha(_canonical(consumption)) != _sha(consumption_raw):
        raise ValueError("pool consumption evidence is not canonical")

    # Test access begins only after receipt and context authentication.
    test = session["components"]["real_view"](
        adapter=session["adapter"], dataset_id=DATASET_ID,
        split="test", provenance_sha256=session["provenance_sha256"],
    )
    if test.identity != receipt["test_identity"]:
        raise ValueError("held-out test identity differs from training receipt")
    test_evidence = _verify_split_identities(
        session["adapter"], include_test=True,
    )

    arms, restorations = {}, {}
    for name in ("real_only", "real_plus_synthetic"):
        expected = receipt["arms"][name]
        restored = restore_selected_classifier(
            run / "checkpoints" / name,
            recipe_raw=session["recipes"]["classifier"],
            expected_manifest_sha256=expected["manifest_sha256"],
            expected_context=expected["context"],
            expected_selected_weights_sha256=expected["selected_weights_sha256"],
        )
        binding = restored["binding"]
        predictions = collect_test_predictions(
            test_view=test,
            expected_test_identity=receipt["test_identity"],
            expected_selected_weights_sha256=expected["selected_weights_sha256"],
            predict_batch=lambda images: binding["model"](
                images, training=False,
            ).numpy(),
            get_weights=binding["controls"]["get_weights"],
            get_execution_state=binding["controls"]["get_execution_state"],
            metric_recipe_raw=session["recipes"]["metric"],
        )
        _write_exclusive_npz(
            run / (name + "_predictions.npz"),
            labels=predictions["labels"],
            probabilities=predictions["probabilities"],
        )
        arms[name] = {
            "training": restored["manifest"]["training_report"],
            "predictions": predictions,
        }
        restorations[name] = restored["report"]
        del binding, restored

    linked = link_paired_results(
        arms=arms, evaluator_seed=EXPERIMENTAL_SEED,
        expected_test_identity=receipt["test_identity"],
        expected_initial_weights_sha256=receipt["initial_weights_sha256"],
        classifier_recipe_raw=session["recipes"]["classifier"],
        metric_recipe_raw=session["recipes"]["metric"],
    )
    selected = receipt["arms"]["real_plus_synthetic"]["selected_weights_sha256"]
    if (
        restorations["real_plus_synthetic"]["selected_weights_sha256"] != selected
        or linked["arms"]["real_plus_synthetic"]["prediction"][
            "selected_weights_sha256"
        ] != selected
    ):
        raise ValueError("restored or linked selected weights differ from receipt")

    validation = session["components"]["real_view"](
        adapter=session["adapter"], dataset_id=DATASET_ID,
        split="validation", provenance_sha256=session["provenance_sha256"],
    )
    if validation.identity != context["validation_identity"]:
        raise ValueError("validation identity differs from training context")
    real_labels = np.asarray(
        session["adapter"].labels("validation"), dtype=np.int64,
    )
    synthetic_labels = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64), SAMPLES_PER_CLASS,
    )
    selection = select_balanced_references(
        real_validation_labels=real_labels,
        synthetic_labels=synthetic_labels,
        class_count=CLASS_COUNT,
        real_validation_identity_sha256=_sha(_canonical(validation.identity)),
        synthetic_manifest_sha256=session["pool_sha256"],
    )
    selection_raw = _write_exclusive_json(run / "selection.json", selection)
    extractor = InceptionFeatures(weights_path=args.weights)
    features = prepare_features_and_kid(
        selection_raw=selection_raw,
        selection_sha256=_sha(selection_raw),
        real_validation_view=validation,
        synthetic_view=session["pool"],
        real_validation_labels=real_labels,
        synthetic_labels=synthetic_labels,
        extractor=extractor,
    )
    feature_sha = {}
    for key in ("real_features", "synthetic_features"):
        _write_exclusive_npy(run / (key + ".npy"), features[key])
        feature_sha[key] = _sha((run / (key + ".npy")).read_bytes())
    kid = features["report"]

    pair_selection = select_ms_ssim_pairs(
        labels=synthetic_labels, class_count=CLASS_COUNT,
        manifest_sha256=session["pool_sha256"],
    )
    pair_raw = _write_exclusive_json(run / "pair_selection.json", pair_selection)
    scorer = ThreeScaleMSSSIM()
    ms_ssim = evaluate_ms_ssim_pairs(
        selection_raw=pair_raw, selection_sha256=_sha(pair_raw),
        synthetic_view=session["pool"], labels=synthetic_labels, scorer=scorer,
    )
    pair_scores = np.asarray(
        [
            ms_ssim["per_class"][str(class_id)]["scores"]
            for class_id in range(CLASS_COUNT)
        ],
        dtype=np.float32,
    )
    _write_exclusive_npy(run / "pair_scores.npy", pair_scores)

    samples_per_class = selection["real_reference"]["samples_per_class"]
    fid_labels = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64), samples_per_class,
    )
    kid_raw = _canonical(kid)
    kid_configuration = {
        "selection_file_sha256": _sha(selection_raw),
        "feature_extractor": kid["feature_extractor"],
    }
    fid_configuration = {
        **kid_configuration, "kid_result_file_sha256": _sha(kid_raw),
    }
    groups = {
        "pooled_balanced_kid": (
            kid, kid_configuration, len(kid["kid"]["draws"]),
        ),
        "pooled_balanced_fid": (
            feature_fid(features["real_features"], features["synthetic_features"]),
            fid_configuration, 1,
        ),
        "class_fid": (
            complete_class_fid(
                features["real_features"], features["synthetic_features"],
                real_labels=fid_labels, synthetic_labels=fid_labels,
                class_count=CLASS_COUNT,
            ),
            fid_configuration, CLASS_COUNT,
        ),
        "intra_class_ms_ssim": (
            ms_ssim,
            {
                "selection_file_sha256": _sha(pair_raw),
                "scorer": scorer.identity,
            },
            1_000,
        ),
    }

    records, artifacts = [], {}

    def persist(name, value, configuration, coverage):
        result_raw = _canonical(value)
        configuration_raw = (
            configuration if isinstance(configuration, bytes)
            else _canonical(configuration)
        )
        _write_exclusive_bytes(run / (name + ".json"), result_raw)
        _write_exclusive_bytes(
            run / (name + ".configuration.json"), configuration_raw,
        )
        records.append({
            "result_id": name,
            "status": "SUCCEEDED",
            "reason": None,
            "result_file_sha256": _sha(result_raw),
            "configuration_sha256": _sha(configuration_raw),
            "coverage": {"expected": coverage, "computed": coverage},
        })
        artifacts[name] = {
            "result_raw": result_raw,
            "configuration_raw": configuration_raw,
        }

    for name, result_id in (
        ("real_only", "real_only_downstream"),
        ("real_plus_synthetic", "real_plus_synthetic_downstream"),
    ):
        persist(
            result_id, linked["arms"][name]["prediction"]["metrics"],
            session["recipes"]["metric"], TEST_SIZE,
        )
    persist(
        "paired_downstream", linked,
        {
            "classifier_recipe_sha256": linked["classifier_recipe_sha256"],
            "metric_recipe_sha256": linked["metric_recipe_sha256"],
        },
        2,
    )
    for name, (value, configuration, coverage) in groups.items():
        persist(name, value, configuration, coverage)
    if [record["result_id"] for record in records] != list(REQUIRED_RESULTS):
        raise RuntimeError("required result inventory mismatch")

    downstream_arguments = {
        "arms": arms,
        "evaluator_seed": EXPERIMENTAL_SEED,
        "expected_test_identity": receipt["test_identity"],
        "expected_initial_weights_sha256": receipt["initial_weights_sha256"],
        "classifier_recipe_raw": session["recipes"]["classifier"],
        "metric_recipe_raw": session["recipes"]["metric"],
    }
    fidelity_arguments = {
        "expected_synthetic_manifest_sha256": session["pool_sha256"],
        "kid_inputs": {
            "selection_raw": selection_raw,
            "expected_selection_file_sha256": _sha(selection_raw),
            "expected_real_validation_identity": validation.identity,
            "expected_synthetic_manifest_sha256": session["pool_sha256"],
            "real_validation_labels": real_labels,
            "synthetic_labels": synthetic_labels,
            "real_features": features["real_features"],
            "synthetic_features": features["synthetic_features"],
            "expected_feature_extractor": kid["feature_extractor"],
        },
        "ms_ssim_inputs": {
            "selection_raw": pair_raw,
            "expected_selection_file_sha256": _sha(pair_raw),
            "labels": synthetic_labels,
            "expected_manifest_sha256": session["pool_sha256"],
            "expected_scorer_identity": scorer.identity,
            "pair_scores": pair_scores,
            "expected_pair_scores_state_sha256": state_digest([pair_scores]),
        },
    }
    linkage = {
        "arm": "real_plus_synthetic",
        "dataset_id": DATASET_ID,
        "experimental_seed": EXPERIMENTAL_SEED,
        "family": FAMILY,
        "manifest_sha256": session["pool_sha256"],
        "checkpoint_sha256": POOL_CHECKPOINT_SHA256,
        "selected_weights_sha256": selected,
    }
    checkpoint_evidence = {
        "directory": run / "checkpoints/real_plus_synthetic",
        "recipe_raw": session["recipes"]["classifier"],
        "expectations_raw": receipt_raw,
        "expected_expectations_sha256": args.receipt_sha256,
        "context_evidence_raw": context_raw,
    }
    evaluation = verify_evaluation(
        dataset_id=DATASET_ID, experimental_seed=EXPERIMENTAL_SEED,
        records=records, artifact_bytes=artifacts,
        downstream_arguments=downstream_arguments,
        fidelity_arguments=fidelity_arguments,
        augmented_pool_linkage=linkage,
        augmented_pool_consumption=consumption,
        augmented_checkpoint_evidence=checkpoint_evidence,
    )
    _require_verified_evaluation(evaluation)
    _write_exclusive_json(run / "evaluation.json", evaluation)

    restore_report = {
        "scope": "CIC_MAF_SEED42_BOUNDED_REAL_DATA_EVALUATION",
        "source_commit": session["head"],
        "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(),
        "train_pid": receipt["train_pid"],
        "receipt_sha256": args.receipt_sha256,
        "test_identity": receipt["test_identity"],
        "split_identity": test_evidence,
        "generator_lineage": session["lineage"],
        "feature_file_sha256": feature_sha,
        "selection_file_sha256": _sha(selection_raw),
        "pair_selection_file_sha256": _sha(pair_raw),
        "pair_scores_file_sha256": _sha((run / "pair_scores.npy").read_bytes()),
        "restorations": restorations,
        "paired_result": linked,
        "verified_groups": evaluation["verified_groups"],
        "verified_pool_consumption": evaluation["verified_pool_consumption"],
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    _assert_incomplete_report(restore_report)
    _write_exclusive_json(run / "restore_report.json", restore_report)

    _verify_imports(session["root"], session["sources"])
    _verify_source(session["root"], args.expected_head)
    print("CIC_MAF_SEED42_EVALUATION_RESTORE=PASS")
    print("CIC_MAF_SEED42_BOUNDED_EVALUATION=PASS")
    print("COMPLETE_EVALUATION=False")
    print("FULL_BENCHMARK_GO=NO", flush=True)
    return restore_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(
        description="Bounded CIC MAF seed-42 real-data evaluation driver",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--phase", choices=("train", "restore"))
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--pool-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-publication-parent", required=True)
    parser.add_argument("--checkpoint-generation-name", required=True)
    parser.add_argument("--checkpoint-expectations", required=True)
    parser.add_argument("--checkpoint-expectations-sha256", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--receipt-sha256")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.preflight_only:
        if args.run_dir is not None or args.receipt_sha256 is not None:
            raise SystemExit("preflight does not accept run or receipt arguments")
        _run_preflight(args)
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required for a phase")
    if args.phase == "train":
        if args.receipt_sha256 is not None:
            raise SystemExit("train phase does not accept --receipt-sha256")
        _run_train(args)
        return
    if args.receipt_sha256 is None:
        raise SystemExit("restore phase requires --receipt-sha256")
    _run_restore(args)


if __name__ == "__main__":
    main()
