#!/usr/bin/env python3
"""Bounded CICMalDroid2020 GAN seed-42 real-data evaluation driver.

Connects the accepted GAN/CIC seed-42 synthetic pool to the existing
Paper 1 evaluation components. It does not define new scientific policy,
new metrics, or a general orchestration framework.

Modes:
  --preflight-only   Read-only authentication; writes nothing.
  --phase train      Paired classifier training/publication; never touches test.
  --phase restore    Fresh-process restoration, test evaluation, fidelity and
                     unified validation.

Unified validation retains complete_evaluation=false. The GAN pool and
checkpoint identity are authenticated separately because they do not use the
MAF checkpoint-bundle or fitted-state evidence contracts.
"""


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
FAMILY = "gan"

SEED_BINDINGS = {
    42: {
        "experimental_seed": 42,
        "pool_manifest_sha256":
            "b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1",
        "checkpoint_sha256":
            "963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530",
        "checkpoint_run_identity": "commit42c40fc_attempt01",
        "generator_source_commit":
            "4324aeecb45a47a72ac0b479d8cd994fe79dc1e0",
        "synthesis_recipe_sha256":
            "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9",
        "synthesis_job_id": "314574",
        "authorization_id": "P1R2026-GAN-CIC-S42-A01-20261116",
        "lineage_schema":
            "trustforge.paper01.cic-gan.seed42.lineage-evidence.v1",
    },
}

EXPERIMENTAL_SEED = SEED_BINDINGS[42]["experimental_seed"]
CLASS_COUNT = 5
POOL_SAMPLES = 10_000
POOL_BATCHES = 625
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

POOL_MANIFEST_SHA256 = SEED_BINDINGS[42]["pool_manifest_sha256"]
POOL_CHECKPOINT_SHA256 = SEED_BINDINGS[42]["checkpoint_sha256"]
CHECKPOINT_RUN_IDENTITY = SEED_BINDINGS[42]["checkpoint_run_identity"]
GENERATOR_SOURCE_COMMIT = SEED_BINDINGS[42]["generator_source_commit"]
SYNTHESIS_JOB_ID = SEED_BINDINGS[42]["synthesis_job_id"]
SYNTHESIS_AUTHORIZATION_ID = SEED_BINDINGS[42]["authorization_id"]
LINEAGE_SCHEMA = SEED_BINDINGS[42]["lineage_schema"]
GENERATOR_RECIPE_SHA256 = (
    SEED_BINDINGS[42]["synthesis_recipe_sha256"]
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
INCEPTION_WEIGHTS_SHA256 = (
    "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
)
INCEPTION_WEIGHTS_BYTES = 87_910_968

GENERATOR_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/synthesis_linkage_recipe.json"
)
CLASSIFIER_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
)
METRIC_RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
)

RECEIPT_SCHEMA = "trustforge.paper01.paired-restoration-expectations.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 2 * 1024 * 1024


def evaluation_binding(experimental_seed):
    """Return the accepted GAN cell binding for one experimental seed."""
    if type(experimental_seed) is not int:
        raise ValueError("unsupported experimental seed")
    binding = SEED_BINDINGS.get(experimental_seed)
    if binding is None:
        raise ValueError("unsupported experimental seed")
    return dict(binding)


def _apply_binding(experimental_seed):
    """Bind all seed-dependent module identities before artifact access."""
    binding = evaluation_binding(experimental_seed)
    global EXPERIMENTAL_SEED, POOL_MANIFEST_SHA256, POOL_CHECKPOINT_SHA256
    global CHECKPOINT_RUN_IDENTITY, GENERATOR_SOURCE_COMMIT
    global SYNTHESIS_JOB_ID, SYNTHESIS_AUTHORIZATION_ID, LINEAGE_SCHEMA
    EXPERIMENTAL_SEED = binding["experimental_seed"]
    POOL_MANIFEST_SHA256 = binding["pool_manifest_sha256"]
    POOL_CHECKPOINT_SHA256 = binding["checkpoint_sha256"]
    CHECKPOINT_RUN_IDENTITY = binding["checkpoint_run_identity"]
    GENERATOR_SOURCE_COMMIT = binding["generator_source_commit"]
    SYNTHESIS_JOB_ID = binding["synthesis_job_id"]
    SYNTHESIS_AUTHORIZATION_ID = binding["authorization_id"]
    LINEAGE_SCHEMA = binding["lineage_schema"]
    return binding


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


def _authenticate_gan_manifest(raw, *, expected_sha):
    """Authenticate the accepted GAN pool manifest, batch count and identity."""
    from trustforge.models.paper01_revision_2026.gan_cic_pool_publication import (
        GAN_POOL_SCHEMA,
    )

    observed = _sha(raw)
    if observed != _require_digest(expected_sha, name="pool manifest"):
        raise ValueError("pool manifest digest mismatch")
    if observed != POOL_MANIFEST_SHA256:
        raise ValueError("pool manifest differs from accepted evidence")

    manifest = json.loads(raw)

    if (
        type(manifest) is not dict
        or manifest.get("schema") != GAN_POOL_SCHEMA
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("family") != FAMILY
        or manifest.get("experimental_seed") != EXPERIMENTAL_SEED
        or manifest.get("class_mapping") != ["0", "1", "2", "3", "4"]
        or manifest.get("class_count") != CLASS_COUNT
        or manifest.get("checkpoint_role") != "terminal"
        or manifest.get("checkpoint_sha256") != POOL_CHECKPOINT_SHA256
        or manifest.get("checkpoint_run_identity") != CHECKPOINT_RUN_IDENTITY
        or manifest.get("executable_source_commit") != GENERATOR_SOURCE_COMMIT
        or manifest.get("recipe_sha256") != GENERATOR_RECIPE_SHA256
        or manifest.get("model_call_batch_size") != 16
        or manifest.get("quota_per_class") != SAMPLES_PER_CLASS
        or manifest.get("batches_per_class") != 125
        or manifest.get("expected_total") != POOL_SAMPLES
        or manifest.get("observed_total") != POOL_SAMPLES
        or manifest.get("complete_pool") is not True
        or manifest.get("status") != "COMPLETE"
        or manifest.get("authoritative_artifact") is not True
    ):
        raise ValueError("GAN pool manifest identity mismatch")

    records = manifest.get("batches")
    if type(records) is not list or len(records) != POOL_BATCHES:
        raise ValueError("complete GAN pool requires 625 ordered batches")

    return manifest


def _lineage_summary(*, pool_identity, manifest, source_commit):
    """Record the accepted GAN checkpoint/pool lineage for evaluation."""
    if (
        source_commit != GENERATOR_SOURCE_COMMIT
        or manifest.get("executable_source_commit") != GENERATOR_SOURCE_COMMIT
        or pool_identity.get("checkpoint_sha256") != POOL_CHECKPOINT_SHA256
        or pool_identity.get("recipe_sha256") != GENERATOR_RECIPE_SHA256
        or pool_identity.get("manifest_sha256") != POOL_MANIFEST_SHA256
        or manifest.get("checkpoint_run_identity") != CHECKPOINT_RUN_IDENTITY
        or manifest.get("checkpoint_sha256") != POOL_CHECKPOINT_SHA256
        or manifest.get("status") != "COMPLETE"
        or manifest.get("authoritative_artifact") is not True
    ):
        raise ValueError("GAN generator checkpoint lineage mismatch")

    summary = {
        "schema": LINEAGE_SCHEMA,
        "dataset_id": DATASET_ID,
        "family": FAMILY,
        "experimental_seed": EXPERIMENTAL_SEED,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "pool_checkpoint_sha256": pool_identity["checkpoint_sha256"],
        "pool_recipe_sha256": pool_identity["recipe_sha256"],
        "checkpoint_run_identity": CHECKPOINT_RUN_IDENTITY,
        "generator_source_commit": source_commit,
        "synthesis_job_id": SYNTHESIS_JOB_ID,
        "synthesis_authorization_id": SYNTHESIS_AUTHORIZATION_ID,
        "synthesis_record_authentication": "NOT_ARTIFACT_VERIFIED",
        "pool_batches": POOL_BATCHES,
        "pool_samples": POOL_SAMPLES,
        "lineage_authentication": "DRIVER_VERIFIED",
        "semantic_class_fidelity": "NOT_ESTABLISHED",
        "complete_evaluation": False,
    }
    summary["sha256"] = _sha(_canonical(summary))
    return summary


def _components():
    from trustforge.data.paper01_revision_2026.cicmaldroid import (
        CICMalDroidRevision2026,
    )
    from trustforge.evaluation.paper01_gan_pool import GANPoolView
    from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
    from trustforge.evaluation.paper01_inception import verify_weight_file
    from trustforge.evaluation.paper01_inputs import RealSplitView

    return {
        "adapter": CICMalDroidRevision2026,
        "pool": GANPoolView,
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

    from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
        require_gan_dataset_provenance,
    )
    from trustforge.models.paper01_revision_2026.gan_v3_provenance_adapter import (
        build_gan_v3_provenance,
    )

    verified = require_gan_dataset_provenance(
        dataset_id=DATASET_ID, adapter=adapter,
    )
    provenance = build_gan_v3_provenance(
        dataset_id=DATASET_ID, verified_preflight=verified,
    )
    provenance_sha256 = _sha(_canonical(provenance))

    split_evidence = _verify_split_identities(
        adapter, include_test=include_test_identity,
    )

    pool_raw = _read_regular(args.pool_manifest, name="pool manifest")
    manifest = _authenticate_gan_manifest(
        pool_raw, expected_sha=args.pool_manifest_sha256,
    )
    pool = components["pool"](
        args.pool_manifest,
        manifest_sha256=args.pool_manifest_sha256,
        family=FAMILY, experimental_seed=EXPERIMENTAL_SEED,
        recipe_sha256=GENERATOR_RECIPE_SHA256,
        checkpoint_sha256=POOL_CHECKPOINT_SHA256,
    )
    if pool.identity["manifest_sha256"] != POOL_MANIFEST_SHA256:
        raise ValueError("constructed GAN pool identity mismatch")
    pool_evidence = pool.preflight()
    if (
        pool_evidence.get("verified_batches") != POOL_BATCHES
        or pool_evidence.get("verified_images") != POOL_SAMPLES
    ):
        raise RuntimeError("accepted pool preflight did not cover the pool")

    identity = {
        "dataset_id": DATASET_ID,
        "family": FAMILY,
        "experimental_seed": EXPERIMENTAL_SEED,
        "class_mapping": ["0", "1", "2", "3", "4"],
        "class_count": CLASS_COUNT,
        "checkpoint_run_identity": CHECKPOINT_RUN_IDENTITY,
        "checkpoint_sha256": POOL_CHECKPOINT_SHA256,
        "executable_source_commit": GENERATOR_SOURCE_COMMIT,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "pool_recipe_sha256": GENERATOR_RECIPE_SHA256,
        "pool_batches": POOL_BATCHES,
        "pool_samples": POOL_SAMPLES,
    }
    lineage = _lineage_summary(
        pool_identity=pool.identity,
        manifest=manifest,
        source_commit=GENERATOR_SOURCE_COMMIT,
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
        "pool_evidence": pool_evidence,
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
        raise ValueError("augmented training view differs from accepted GAN pool")
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
    pool_evidence = session["pool_evidence"]
    if (
        pool_evidence.get("verified_batches") != POOL_BATCHES
        or pool_evidence.get("verified_images") != POOL_SAMPLES
    ):
        raise RuntimeError("accepted pool preflight did not cover the pool")
    summary = {
        "scope": "CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_EVALUATION_PREFLIGHT",
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
    print("CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_EVALUATION_PREFLIGHT=PASS")
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
                _canonical(training_view.identity) + b"\n"
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
        "scope": "CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_BOUNDED_REAL_DATA_TRAIN",
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
    print("CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_EVALUATION_TRAIN=PASS")
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
        "scope": "CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_BOUNDED_REAL_DATA_EVALUATION",
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
    print("CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_EVALUATION_RESTORE=PASS")
    print("CIC_GAN_SEED" + str(EXPERIMENTAL_SEED) + "_BOUNDED_EVALUATION=PASS")
    print("COMPLETE_EVALUATION=False")
    print("FULL_BENCHMARK_GO=NO", flush=True)
    return restore_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(
        description="Bounded CIC GAN seed-42 real-data evaluation driver",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--phase", choices=("train", "restore"))
    parser.add_argument(
        "--experimental-seed", type=int, choices=(42,), default=42,
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--pool-manifest-sha256", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--receipt-sha256")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    _apply_binding(args.experimental_seed)
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
