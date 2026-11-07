"""Actual EvalCNN training-to-test engineering fixture; no real-data results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    require(root == Path.cwd().resolve(), "checkout mismatch")

    def check_source():
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
        require(head == args.expected_head, "source mismatch")
        subprocess.run(["git", "diff", "--quiet"], check=True)
        subprocess.run(["git", "diff", "--cached", "--quiet"], check=True)

    check_source()
    run = args.run_dir.resolve(strict=True)
    for name in ("expectations.json", "predictions.npz", "report.json"):
        require(not (run / name).exists() and not (run / name).is_symlink(),
                "destination exists: " + name)

    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node, build_classifier_binding,
    )
    require_compute_node()
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_classifier_control import (
        fit_classifier_control, state_digest,
    )
    from trustforge.evaluation.paper01_inputs import RealSplitView
    from trustforge.evaluation.paper01_predictions import collect_test_predictions

    class Adapter:
        def __init__(self):
            self.test_allowed = False
            self.requests = {"train": [], "validation": [], "test": []}

        def split_size(self, split):
            return {"train": 9, "validation": 5, "test": 257}[split]

        def canonical_batch(self, split, positions):
            if split == "test":
                require(self.test_allowed, "test accessed before expectation")
            self.requests[split].append(len(positions))
            classes = positions % 5
            values = classes.astype(np.float32) / np.float32(4)
            images = np.broadcast_to(
                values[:, None, None, None], (len(positions), 40, 40, 3),
            ).copy()
            return images, np.eye(5, dtype=np.float32)[classes]

    adapter = Adapter()
    provenance = hashlib.sha256(
        b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE"
    ).hexdigest()

    def view(split):
        return RealSplitView(
            adapter=adapter, dataset_id="cicmaldroid2020",
            split=split, provenance_sha256=provenance,
        )

    base = root / "studies/paper01_benchmark/revision_2026"
    classifier_recipe = (base / "classifier_recipe.json").read_bytes()
    metric_recipe = (base / "downstream_metric_recipe.json").read_bytes()
    binding = build_classifier_binding(
        recipe_raw=classifier_recipe, class_count=5, evaluator_seed=42,
    )
    training = fit_classifier_control(
        training=view("train"), validation=view("validation"),
        evaluator_seed=42, recipe_raw=classifier_recipe,
        **binding["controls"],
    )
    require(adapter.requests["test"] == [], "training consumed test")
    selected_digest = state_digest(binding["model"].get_weights())
    test_view = view("test")
    expectation = {
        "selected_weights_sha256": selected_digest,
        "test_identity": test_view.identity,
        "classifier_recipe_sha256": hashlib.sha256(classifier_recipe).hexdigest(),
        "metric_recipe_sha256": hashlib.sha256(metric_recipe).hexdigest(),
    }
    write_json(run / "expectations.json", expectation)
    expectation_raw = (run / "expectations.json").read_bytes()
    persisted = json.loads(expectation_raw)
    adapter.test_allowed = True

    result = collect_test_predictions(
        test_view=test_view,
        expected_test_identity=persisted["test_identity"],
        expected_selected_weights_sha256=persisted["selected_weights_sha256"],
        predict_batch=lambda images: binding["model"](
            images, training=False,
        ).numpy(),
        get_weights=binding["controls"]["get_weights"],
        get_execution_state=binding["controls"]["get_execution_state"],
        metric_recipe_raw=metric_recipe,
    )
    require(adapter.requests["test"] == [128, 128, 1], "test batch coverage")
    np.testing.assert_array_equal(result["labels"], np.arange(257) % 5)
    require(result["probabilities"].shape == (257, 5), "prediction coverage")
    require(
        (run / "expectations.json").read_bytes() == expectation_raw,
        "expectation changed",
    )
    with (run / "predictions.npz").open("xb") as handle:
        np.savez(
            handle, labels=result["labels"],
            probabilities=result["probabilities"],
        )
    check_source()
    write_json(run / "report.json", {
        "scope": "TRAINED_EVALCNN_TEST_PREDICTION_SYNTHETIC_FIXTURE",
        "source_commit": args.expected_head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "tensorflow": tf.__version__,
        "expectations_sha256": hashlib.sha256(expectation_raw).hexdigest(),
        "predictions_file_sha256": hashlib.sha256(
            (run / "predictions.npz").read_bytes()
        ).hexdigest(),
        "expectation_persisted_before_test_access": True,
        "test_batch_sizes": adapter.requests["test"],
        "training": training,
        "prediction": result["report"],
        "real_data_used": False,
        "primary_scientific_result": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    })
    print("TRAINED_EVALCNN_TEST_PREDICTION_FIXTURE=PASS")


if __name__ == "__main__":
    main()
