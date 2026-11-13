"""Exercise actual paired training and verify the augmented pool record."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket


POOL_MANIFEST_SHA256 = (
    "53896ef041c03eaa99ea385fb00068efbb189e994e89ea5a0464edfd79951c38"
)
POOL_SOURCE_JOB_ID = "311683"
RECORD_NAME = "pool_consumption.json"
REPORT_NAME = "report.json"
SEED = 42


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open("xb") as handle:
        handle.write(canonical(value))


def read(path):
    require(path.is_file() and not path.is_symlink(), "unavailable: " + str(path))
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node, prepare_paired_classifiers,
    )
    require_compute_node()
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )
    root = Path(args.repository_root).resolve()
    require(root == Path.cwd().resolve(), "checkout mismatch")
    head, sources = capture_source_identity(root)
    require(head == args.expected_head, "source mismatch")
    run = args.run_dir.expanduser().resolve(strict=True)
    require(run.is_dir() and not run.is_symlink(), "regular run directory required")
    try:
        run.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("run directory must be external to the checkout")
    for name in (RECORD_NAME, REPORT_NAME):
        require(not (run / name).exists() and not (run / name).is_symlink(),
                "destination exists: " + name)

    import numpy as np
    import tensorflow as tf
    import keras
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_inputs import RealSplitView
    from trustforge.evaluation.paper01_cic_pool import CICPoolView
    from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
    from trustforge.evaluation.paper01_classifier_control import state_digest
    from trustforge.evaluation.paper01_pool_consumption import (
        validate_pool_consumption,
    )

    class Adapter:
        def __init__(self):
            self.test_allowed = False
            self.test_batches = []

        def split_size(self, split):
            return {"train": 9, "validation": 5, "test": 257}[split]

        def canonical_batch(self, split, positions):
            self.split_size(split)
            if split == "test":
                require(self.test_allowed, "premature test access")
                self.test_batches.append(len(positions))
            classes = positions % 5
            values = classes.astype(np.float32) / np.float32(4)
            images = np.broadcast_to(
                values[:, None, None, None], (len(positions), 40, 40, 3),
            ).copy()
            return images, np.eye(5, dtype=np.float32)[classes]

    def view(adapter, split, provenance):
        return RealSplitView(
            adapter=adapter, dataset_id="cicmaldroid2020",
            split=split, provenance_sha256=provenance,
        )

    adapter = Adapter()
    provenance = sha(b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE")
    real = view(adapter, "train", provenance)
    validation = view(adapter, "validation", provenance)
    test = view(adapter, "test", provenance)

    base = root / "studies/paper01_benchmark/revision_2026"
    recipe = read(base / "classifier_recipe.json")
    pool_path = (
        Path.home()
        / "trustforge_revision_2026_artifacts/paper01/diagnostics"
        / "paired-classifier-ff6bbb4-6j3j8sX7/fixture_pool/manifest.json"
    )
    pool_raw = read(pool_path)
    pool_sha = sha(pool_raw)
    require(pool_sha == POOL_MANIFEST_SHA256, "fixture pool identity")
    pool = json.loads(pool_raw)
    synthetic = CICPoolView(
        pool_path, manifest_sha256=pool_sha,
        family="maf", experimental_seed=SEED,
        recipe_sha256=pool["recipe_sha256"],
        checkpoint_sha256=pool["checkpoint_sha256"],
    )
    augmented = CombinedTrainingView(real=real, synthetic=synthetic)
    require(augmented.size == 10009, "combined fixture size mismatch")

    print("POOL_CONSUMPTION_BEGIN", os.getpid(), flush=True)
    result = prepare_paired_classifiers(
        real_training=real, augmented_training=augmented,
        validation=validation, evaluator_seed=SEED, recipe_raw=recipe,
    )
    require(adapter.test_batches == [], "training accessed test")
    require(
        result["status"] == "PAIRED_WEIGHTS_PREPARED_NOT_EVALUATED"
        and result["test_evaluated"] is False
        and result["complete_evaluation"] is False,
        "unexpected paired training status",
    )

    arms = result["arms"]
    require(set(arms) == {"real_only", "real_plus_synthetic"}, "arm inventory")
    require("pool_consumption" not in arms["real_only"],
            "real-only arm returned a consumption record")
    require("pool_consumption" in arms["real_plus_synthetic"],
            "augmented arm omitted the consumption record")
    augmented_arm = arms["real_plus_synthetic"]
    record = augmented_arm["pool_consumption"]
    report = augmented_arm["report"]
    require("pool_consumption" not in report,
            "consumption record contaminated the training report")

    validate_pool_consumption(
        record, training_identity=augmented.identity,
        training_size=augmented.size, training_report=report,
    )
    for name, arm in arms.items():
        observed = state_digest(arm["selected_weights"])
        require(observed == arm["report"]["selected_weights_sha256"],
                name + " selected-weight snapshot mismatch")
    require(
        record["selected_weights_sha256"]
        == state_digest(augmented_arm["selected_weights"]),
        "record selected weights differ from returned weights",
    )

    selected = report["selected_epoch_index"]
    completed = report["completed_epochs"]
    history = report["history"]
    per_epoch = augmented.size - real.size
    require(record["synthetic_examples_per_epoch"] == per_epoch,
            "synthetic per-epoch count mismatch")
    require(
        record["selected_prefix"]["synthetic_examples"]
        == (selected + 1) * per_epoch,
        "selected synthetic sample count mismatch",
    )
    require(
        record["final_prefix"]["synthetic_examples"] == completed * per_epoch,
        "final synthetic sample count mismatch",
    )
    require(
        record["selected_prefix"]["optimizer_updates"]
        == history[selected]["optimizer_updates"],
        "selected optimizer update count mismatch",
    )
    require(
        record["final_prefix"]["optimizer_updates"]
        == report["optimizer_updates"],
        "final optimizer update count mismatch",
    )
    per_epoch_updates = (augmented.size + 127) // 128
    require(
        report["optimizer_updates"] == completed * per_epoch_updates,
        "completed optimizer accounting mismatch",
    )

    require(capture_source_identity(root) == (head, sources), "source changed")
    record_raw = canonical(record)
    record_sha = sha(record_raw)
    with (run / RECORD_NAME).open("xb") as handle:
        handle.write(record_raw)
    write_json(run / REPORT_NAME, {
        "scope": "AUGMENTED_POOL_CONSUMPTION_SYNTHETIC_FIXTURE",
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "pool_source_job_id": POOL_SOURCE_JOB_ID,
        "pool_manifest_sha256": pool_sha,
        "record_file": RECORD_NAME,
        "record_sha256": record_sha,
        "classifier_recipe_sha256": sha(recipe),
        "evaluator_seed": SEED,
        "selected_weights_sha256": record["selected_weights_sha256"],
        "selected_epoch_index": selected,
        "completed_epochs": completed,
        "synthetic_examples_per_epoch": per_epoch,
        "selected_prefix": record["selected_prefix"],
        "final_prefix": record["final_prefix"],
        "only_augmented_arm_recorded": True,
        "record_validated": True,
        "selected_weights_independently_hashed": True,
        "test_accessed": False,
        "real_data_used": False,
        "generator_lineage_authenticated": False,
        "synthetic_fixture_evidence": True,
        "complete_evaluation": False,
        "full_benchmark_go": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    })
    print("AUGMENTED_POOL_CONSUMPTION_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
