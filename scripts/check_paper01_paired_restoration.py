"""Paired classifier training/publication and fresh-process test fixture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform


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
    parser.add_argument("--phase", choices=("train", "restore"), required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256")
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
    run = args.run_dir.resolve(strict=True)

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
    from trustforge.evaluation.paper01_classifier_state import encode_classifier_weights
    from trustforge.evaluation.paper01_classifier_manifest import (
        prepare_classifier_manifest, canonical_manifest,
    )
    from trustforge.evaluation.paper01_classifier_bundle import publish_classifier_bundle
    from trustforge.evaluation.paper01_classifier_restoration import restore_selected_classifier
    from trustforge.evaluation.paper01_predictions import collect_test_predictions
    from trustforge.evaluation.paper01_paired_results import link_paired_results

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

    adapter = Adapter()
    provenance = sha(b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE")

    def view(split):
        return RealSplitView(
            adapter=adapter, dataset_id="cicmaldroid2020",
            split=split, provenance_sha256=provenance,
        )

    real, validation, test = view("train"), view("validation"), view("test")
    base = root / "studies/paper01_benchmark/revision_2026"
    recipe = read(base / "classifier_recipe.json")
    metric_recipe = read(base / "downstream_metric_recipe.json")
    environment = {
        "python": platform.python_version(), "numpy": np.__version__,
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "settings": {
            key: os.environ.get(key) for key in (
                "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
            )
        },
    }
    shared = {
        "source_commit": head,
        "source_files_sha256": sha(canonical(sources)),
        "environment_sha256": sha(canonical(environment)),
    }
    print("PAIRED_RESTORATION_BEGIN", args.phase, os.getpid(), flush=True)

    if args.phase == "train":
        for name in ("checkpoints", "expectations.json", "train_report.json",
                     "context_evidence.json"):
            require(not (run / name).exists() and not (run / name).is_symlink(),
                    "destination exists: " + name)
        pool_path = (
            Path.home()
            / "trustforge_revision_2026_artifacts/paper01/diagnostics"
            / "paired-classifier-ff6bbb4-6j3j8sX7/fixture_pool/manifest.json"
        )
        pool_raw = read(pool_path)
        pool_sha = sha(pool_raw)
        require(pool_sha ==
                "53896ef041c03eaa99ea385fb00068efbb189e994e89ea5a0464edfd79951c38",
                "fixture pool identity")
        pool = json.loads(pool_raw)
        synthetic = CICPoolView(
            pool_path, manifest_sha256=pool_sha,
            family="maf", experimental_seed=42,
            recipe_sha256=pool["recipe_sha256"],
            checkpoint_sha256=pool["checkpoint_sha256"],
        )
        augmented = CombinedTrainingView(real=real, synthetic=synthetic)
        paired = prepare_paired_classifiers(
            real_training=real, augmented_training=augmented,
            validation=validation, evaluator_seed=42, recipe_raw=recipe,
        )
        require(adapter.test_batches == [], "training accessed test")
        prepared, expectations = {}, {}
        for name, training_view in (
            ("real_only", real), ("real_plus_synthetic", augmented),
        ):
            arm = paired["arms"][name]
            context = {
                **shared,
                "dataset_id": "cicmaldroid2020", "evaluator_seed": 42,
                "arm": name, "training_sample_count": training_view.size,
                "validation_sample_count": validation.size,
                "training_identity_sha256": sha(canonical(training_view.identity)),
                "validation_identity_sha256": sha(canonical(validation.identity)),
            }
            selected = arm["report"]["selected_weights_sha256"]
            state_raw = encode_classifier_weights(
                weights=arm["selected_weights"], dataset_id="cicmaldroid2020",
                recipe_raw=recipe, expected_selected_weights_sha256=selected,
            )
            manifest_raw = canonical_manifest(prepare_classifier_manifest(
                context=context, training_report=arm["report"],
                state_raw=state_raw, recipe_raw=recipe,
            ))
            prepared[name] = (manifest_raw, state_raw)
            expectations[name] = {
                "context": context, "manifest_sha256": sha(manifest_raw),
                "selected_weights_sha256": selected,
            }
        write_json(run / "context_evidence.json", {
            "source_files": sources, "environment": environment,
            "real_training_identity": real.identity,
            "augmented_training_identity": augmented.identity,
            "validation_identity": validation.identity,
        })
        receipt = {
            "schema": "trustforge.paper01.paired-restoration-expectations.v1",
            **shared,
            "job_id": os.environ["SLURM_JOB_ID"], "train_pid": os.getpid(),
            "evaluator_seed": 42,
            "initial_weights_sha256": paired["initial_weights_sha256"],
            "classifier_recipe_sha256": sha(recipe),
            "metric_recipe_sha256": sha(metric_recipe),
            "test_identity": test.identity,
            "fixture_pool_manifest_sha256": pool_sha,
            "context_evidence_sha256": sha(read(run / "context_evidence.json")),
            "arms": expectations,
        }
        write_json(run / "expectations.json", receipt)
        parent = run / "checkpoints"
        parent.mkdir(mode=0o700)
        for name, (manifest_raw, state_raw) in prepared.items():
            expected = expectations[name]
            publish_classifier_bundle(
                parent, name, manifest_raw=manifest_raw, state_raw=state_raw,
                recipe_raw=recipe,
                expected_manifest_sha256=expected["manifest_sha256"],
                expected_context=expected["context"],
                expected_selected_weights_sha256=expected["selected_weights_sha256"],
            )
        write_json(run / "train_report.json", {
            "scope": "PAIRED_PUBLICATION_SYNTHETIC_FIXTURE",
            "source_commit": head, "job_id": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(),
            "arms": {name: arm["report"] for name, arm in paired["arms"].items()},
            "both_bundles_published": True, "test_accessed": False,
            "complete_evaluation": False, "full_benchmark_go": False,
        })
        print("PAIRED_CLASSIFIER_PUBLICATION=PASS")
    else:
        receipt_raw = read(run / "expectations.json")
        require(args.receipt_sha256 is not None
                and sha(receipt_raw) == args.receipt_sha256, "receipt mismatch")
        receipt = json.loads(receipt_raw)
        require(receipt["schema"] ==
                "trustforge.paper01.paired-restoration-expectations.v1",
                "receipt schema")
        require(all(receipt[key] == value for key, value in shared.items()),
                "source/environment mismatch")
        require(receipt["job_id"] == os.environ["SLURM_JOB_ID"], "job mismatch")
        require(receipt["train_pid"] != os.getpid(), "fresh process required")
        require(receipt["classifier_recipe_sha256"] == sha(recipe)
                and receipt["metric_recipe_sha256"] == sha(metric_recipe),
                "recipe mismatch")
        require(receipt["test_identity"] == test.identity, "test identity")
        require(sha(read(run / "context_evidence.json"))
                == receipt["context_evidence_sha256"], "context evidence")
        adapter.test_allowed = True
        arms, restorations = {}, {}
        for name in ("real_only", "real_plus_synthetic"):
            expected = receipt["arms"][name]
            restored = restore_selected_classifier(
                run / "checkpoints" / name, recipe_raw=recipe,
                expected_manifest_sha256=expected["manifest_sha256"],
                expected_context=expected["context"],
                expected_selected_weights_sha256=expected["selected_weights_sha256"],
            )
            binding = restored["binding"]
            predictions = collect_test_predictions(
                test_view=test, expected_test_identity=receipt["test_identity"],
                expected_selected_weights_sha256=expected["selected_weights_sha256"],
                predict_batch=lambda images: binding["model"](
                    images, training=False,
                ).numpy(),
                get_weights=binding["controls"]["get_weights"],
                get_execution_state=binding["controls"]["get_execution_state"],
                metric_recipe_raw=metric_recipe,
            )
            with (run / (name + "_predictions.npz")).open("xb") as handle:
                np.savez(handle, labels=predictions["labels"],
                         probabilities=predictions["probabilities"])
            arms[name] = {
                "training": restored["manifest"]["training_report"],
                "predictions": predictions,
            }
            restorations[name] = restored["report"]
            del binding, restored
        require(adapter.test_batches == [128, 128, 1] * 2, "test coverage")
        linked = link_paired_results(
            arms=arms, evaluator_seed=42,
            expected_test_identity=receipt["test_identity"],
            expected_initial_weights_sha256=receipt["initial_weights_sha256"],
            classifier_recipe_raw=recipe, metric_recipe_raw=metric_recipe,
        )
        write_json(run / "restore_report.json", {
            "scope": "PAIRED_RESTORED_PREDICTION_SYNTHETIC_FIXTURE",
            "source_commit": head, "job_id": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(), "train_pid": receipt["train_pid"],
            "receipt_sha256": args.receipt_sha256,
            "restorations": restorations, "paired_result": linked,
            "test_batch_sizes": adapter.test_batches,
            "prediction_file_sha256": {
                name: sha(read(run / (name + "_predictions.npz")))
                for name in arms
            },
            "real_data_used": False, "generator_lineage_authenticated": False,
            "complete_evaluation": False, "full_benchmark_go": False,
        })
        print("PAIRED_RESTORED_PREDICTION_LINKAGE=PASS")

    require(capture_source_identity(root) == (head, sources), "source changed")


if __name__ == "__main__":
    main()
