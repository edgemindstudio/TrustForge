"""Compute-node RGB/Inception engineering fixture; no real references."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import require_compute_node
    require_compute_node()
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )
    root = Path(args.repository_root).resolve()
    require(Path.cwd().resolve() == root, "checkout mismatch")
    head, sources = capture_source_identity(root)
    require(head == args.expected_head, "revision mismatch")
    require(not args.output.exists(), "output already exists")

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_inception import (
        InceptionFeatures, preprocess_canonical_images,
    )
    from trustforge.evaluation.paper01_classifier_control import state_digest

    images = np.zeros((3, 40, 40, 3), dtype=np.float32)
    for index in range(3):
        images[index, :, :, index] = 1
    processed = preprocess_canonical_images(images).numpy()
    expected = -np.ones((3, 3), dtype=np.float32)
    np.fill_diagonal(expected, 1)
    require(processed.shape == (3, 299, 299, 3), "preprocessing shape")
    require(processed.dtype == np.float32, "preprocessing dtype")
    require(
        np.array_equal(
            processed,
            np.broadcast_to(expected[:, None, None, :], processed.shape),
        ),
        "RGB channels or preprocessing values changed",
    )

    extractor = InceptionFeatures(weights_path=args.weights)
    before = state_digest(extractor._model.get_weights())
    features = extractor.features(images)
    replay = extractor.features(images)
    require(np.array_equal(features, replay), "inference replay differs")
    require(not np.array_equal(features[0], features[1]), "RGB fixture features identical")
    require(
        state_digest(extractor._model.get_weights()) == before,
        "inference changed model weights",
    )

    class FixtureView:
        size = 17
        identity = {
            "dataset_id": "cicmaldroid2020",
            "scope": "synthetic_feature_iterator_fixture",
        }

        def __init__(self):
            self.requests = []

        def batch(self, positions):
            self.requests.append(positions.copy())
            return (
                positions.copy(),
                images[positions % 3],
                np.eye(5, dtype=np.float32)[positions % 5],
            )

    view = FixtureView()
    positions = np.arange(16, -1, -1, dtype=np.int64)
    batches = list(extractor.iter_selected(view=view, positions=positions))
    require([len(batch[0]) for batch in batches] == [16, 1], "iterator batch bounds")
    require(
        np.array_equal(np.concatenate([batch[0] for batch in batches]), positions),
        "reference order changed",
    )
    require(
        all(values.shape == (len(indices), 2048) for indices, values in batches),
        "iterator feature shape",
    )
    for invalid in (
        images.astype(np.float64),
        np.full_like(images, np.nan),
        np.full_like(images, 2),
        np.zeros((17, 40, 40, 3), dtype=np.float32),
    ):
        try:
            extractor.features(invalid)
        except ValueError:
            pass
        else:
            raise RuntimeError("invalid feature input accepted")

    require(capture_source_identity(root) == (head, sources), "source changed")
    imports = 0
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
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == sources.get(relative),
            "project import mismatch: " + name,
        )
        imports += 1

    report = {
        "scope": "INCEPTION_RGB_FEATURE_FIXTURE",
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "feature_extractor": extractor.identity,
        "rgb_preprocessing_checked": True,
        "inference_replay_passed": True,
        "inference_weights_unchanged": True,
        "invalid_inputs_rejected": True,
        "iterator_batch_sizes": [len(batch[0]) for batch in batches],
        "reference_position_order_preserved": True,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "fixture_features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "verified_project_import_count": imports,
        "real_references_used": False,
        "primary_fidelity_evaluated": False,
        "full_benchmark_go": False,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("INCEPTION_RGB_FEATURE_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
