"""Two-process selected-classifier publication/restoration fixture."""

import argparse
import hashlib
import io
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
    ).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open("xb") as handle:
        handle.write(canonical(value))


def read_regular(path):
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
        require_compute_node, build_classifier_binding,
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
    from trustforge.evaluation.paper01_classifier_control import fit_classifier_control
    from trustforge.evaluation.paper01_predictions import collect_test_predictions
    from trustforge.evaluation.paper01_classifier_state import encode_classifier_weights
    from trustforge.evaluation.paper01_classifier_manifest import (
        prepare_classifier_manifest, canonical_manifest,
    )
    from trustforge.evaluation.paper01_classifier_bundle import publish_classifier_bundle
    from trustforge.evaluation.paper01_classifier_restoration import restore_selected_classifier

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__, "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "threads": {
            key: os.environ.get(key) for key in (
                "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
            )
        },
    }
    environment_sha = sha(canonical(environment))
    source_sha = sha(canonical(sources))
    base = root / "studies/paper01_benchmark/revision_2026"
    recipe = (base / "classifier_recipe.json").read_bytes()
    metric_recipe = (base / "downstream_metric_recipe.json").read_bytes()

    class Adapter:
        def split_size(self, split):
            return {"train": 9, "validation": 5, "test": 257}[split]

        def canonical_batch(self, split, positions):
            self.split_size(split)
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

    training_view, validation_view, test_view = (
        view("train"), view("validation"), view("test"),
    )
    context = {
        "dataset_id": "cicmaldroid2020", "evaluator_seed": 42,
        "arm": "real_only",
        "training_sample_count": training_view.size,
        "validation_sample_count": validation_view.size,
        "training_identity_sha256": sha(canonical(training_view.identity)),
        "validation_identity_sha256": sha(canonical(validation_view.identity)),
        "source_commit": head, "source_files_sha256": source_sha,
        "environment_sha256": environment_sha,
    }

    def predict(binding, expected_selected, expected_test):
        return collect_test_predictions(
            test_view=test_view, expected_test_identity=expected_test,
            expected_selected_weights_sha256=expected_selected,
            predict_batch=lambda images: binding["model"](
                images, training=False,
            ).numpy(),
            get_weights=binding["controls"]["get_weights"],
            get_execution_state=binding["controls"]["get_execution_state"],
            metric_recipe_raw=metric_recipe,
        )

    print("CLASSIFIER_RESTORATION_BEGIN", args.phase, os.getpid(), flush=True)
    if args.phase == "train":
        for name in (
            "checkpoints", "expectations.json", "train_report.json",
            "train_predictions.npz", "context_evidence.json",
        ):
            require(not (run / name).exists() and not (run / name).is_symlink(),
                    "destination exists: " + name)
        binding = build_classifier_binding(
            recipe_raw=recipe, class_count=5, evaluator_seed=42,
        )
        training = fit_classifier_control(
            training=training_view, validation=validation_view,
            evaluator_seed=42, recipe_raw=recipe,
            **binding["controls"],
        )
        selected = training["selected_weights_sha256"]
        state_raw = encode_classifier_weights(
            weights=binding["model"].get_weights(),
            dataset_id="cicmaldroid2020", recipe_raw=recipe,
            expected_selected_weights_sha256=selected,
        )
        manifest = prepare_classifier_manifest(
            context=context, training_report=training,
            state_raw=state_raw, recipe_raw=recipe,
        )
        manifest_raw = canonical_manifest(manifest)
        prediction = predict(binding, selected, test_view.identity)
        with (run / "train_predictions.npz").open("xb") as handle:
            np.savez(
                handle, labels=prediction["labels"],
                probabilities=prediction["probabilities"],
            )
        write_json(run / "context_evidence.json", {
            "source_files": sources, "environment": environment,
            "training_identity": training_view.identity,
            "validation_identity": validation_view.identity,
        })
        receipt = {
            "schema": "trustforge.paper01.classifier-fixture-expectations.v1",
            "context": context,
            "manifest_sha256": sha(manifest_raw),
            "selected_weights_sha256": selected,
            "test_identity": test_view.identity,
            "classifier_recipe_sha256": sha(recipe),
            "metric_recipe_sha256": sha(metric_recipe),
            "prediction_state_sha256":
                prediction["report"]["ordered_prediction_state_sha256"],
            "predictions_file_sha256":
                sha(read_regular(run / "train_predictions.npz")),
            "context_evidence_sha256":
                sha(read_regular(run / "context_evidence.json")),
            "train_pid": os.getpid(),
            "job_id": os.environ["SLURM_JOB_ID"],
        }
        # External expectations precede authoritative bundle publication.
        write_json(run / "expectations.json", receipt)
        parent = run / "checkpoints"
        parent.mkdir(mode=0o700)
        published = publish_classifier_bundle(
            parent, "selected_classifier",
            manifest_raw=manifest_raw, state_raw=state_raw, recipe_raw=recipe,
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_context=context,
            expected_selected_weights_sha256=selected,
        )
        write_json(run / "train_report.json", {
            "scope": "CLASSIFIER_PUBLICATION_SYNTHETIC_FIXTURE",
            "source_commit": head, "job_id": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(), "training": training,
            "prediction": prediction["report"],
            "publication_status": published["publication_status"],
            "complete_evaluation": False, "full_benchmark_go": False,
        })
        print("CLASSIFIER_FIXTURE_TRAIN_PUBLICATION=PASS")
    else:
        require(args.receipt_sha256 is not None, "external receipt digest required")
        receipt_raw = read_regular(run / "expectations.json")
        require(sha(receipt_raw) == args.receipt_sha256, "receipt mismatch")
        receipt = json.loads(receipt_raw)
        require(receipt["schema"]
                == "trustforge.paper01.classifier-fixture-expectations.v1",
                "receipt schema")
        require(receipt["context"] == context, "source/environment/input mismatch")
        require(receipt["classifier_recipe_sha256"] == sha(recipe)
                and receipt["metric_recipe_sha256"] == sha(metric_recipe),
                "recipe mismatch")
        require(receipt["job_id"] == os.environ["SLURM_JOB_ID"], "job mismatch")
        require(receipt["train_pid"] != os.getpid(), "fresh process required")
        require(
            sha(read_regular(run / "context_evidence.json"))
            == receipt["context_evidence_sha256"], "context evidence mismatch",
        )
        restored = restore_selected_classifier(
            run / "checkpoints/selected_classifier",
            recipe_raw=recipe,
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_context=receipt["context"],
            expected_selected_weights_sha256=receipt["selected_weights_sha256"],
        )
        prediction = predict(
            restored["binding"], receipt["selected_weights_sha256"],
            receipt["test_identity"],
        )
        archive_raw = read_regular(run / "train_predictions.npz")
        require(sha(archive_raw) == receipt["predictions_file_sha256"],
                "reference prediction archive mismatch")
        with np.load(io.BytesIO(archive_raw), allow_pickle=False) as archive:
            require(set(archive.files) == {"labels", "probabilities"},
                    "archive inventory")
            require(np.array_equal(archive["labels"], prediction["labels"]),
                    "restored ordered labels")
            require(np.array_equal(
                archive["probabilities"], prediction["probabilities"],
            ), "restored probability replay")
        require(
            prediction["report"]["ordered_prediction_state_sha256"]
            == receipt["prediction_state_sha256"], "prediction identity replay",
        )
        try:
            restore_selected_classifier(
                run / "checkpoints/selected_classifier", recipe_raw=recipe,
                expected_manifest_sha256=receipt["manifest_sha256"],
                expected_context=receipt["context"],
                expected_selected_weights_sha256="0" * 64,
            )
        except ValueError:
            incompatible_rejected = True
        else:
            raise RuntimeError("incompatible selected-weight expectation accepted")
        write_json(run / "restore_report.json", {
            "scope": "CLASSIFIER_RESTORATION_SYNTHETIC_FIXTURE",
            "source_commit": head, "job_id": os.environ["SLURM_JOB_ID"],
            "pid": os.getpid(), "train_pid": receipt["train_pid"],
            "receipt_sha256": args.receipt_sha256,
            "restoration": restored["report"],
            "prediction": prediction["report"],
            "probability_replay_byte_exact": True,
            "incompatible_expectation_rejected": incompatible_rejected,
            "real_data_used": False, "complete_evaluation": False,
            "full_benchmark_go": False,
        })
        print("CLASSIFIER_FIXTURE_FRESH_RESTORE=PASS")

    require(capture_source_identity(root) == (head, sources),
            "source changed during fixture")


if __name__ == "__main__":
    main()
