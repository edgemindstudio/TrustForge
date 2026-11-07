"""Compute-node classifier binding fixture; no real data or test evaluation."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node, build_classifier_binding,
    )
    require_compute_node()

    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )
    root = Path(args.repository_root).resolve()
    require(Path.cwd().resolve() == root, "checkout location mismatch")
    head, sources = capture_source_identity(root)
    require(head == args.expected_head, "source revision mismatch")
    require(not args.output.exists(), "output already exists")

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_inputs import RealSplitView
    from trustforge.evaluation.paper01_classifier_control import (
        fit_classifier_control, state_digest,
    )

    recipe_raw = (
        root / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
    ).read_bytes()
    recipe = json.loads(recipe_raw)

    class FixtureAdapter:
        def __init__(self, train_count):
            self.train_count = train_count

        def split_size(self, split):
            if split == "train":
                return self.train_count
            if split == "validation":
                return 5
            raise RuntimeError("fixture prohibits held-out test access")

        def canonical_batch(self, split, positions):
            self.split_size(split)
            values = (positions % 5).astype(np.float32) / np.float32(4)
            images = np.broadcast_to(
                values[:, None, None, None], (len(positions), 40, 40, 3)
            ).copy()
            labels = np.eye(5, dtype=np.float32)[positions % 5]
            return images, labels

    def build(seed=42):
        return build_classifier_binding(
            recipe_raw=recipe_raw, class_count=5, evaluator_seed=seed,
        )

    probe = build()
    model = probe["model"]
    conv = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    pools = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.MaxPooling2D)
    ]
    require([layer.filters for layer in conv] == [32, 64, 128], "convolution widths")
    require(all(layer.kernel_size == (3, 3) for layer in conv), "kernel sizes")
    require(all(layer.padding == "same" for layer in conv), "convolution padding")
    require(all(layer.strides == (1, 1) for layer in conv), "convolution strides")
    require(
        all(layer.activation.__name__ == "relu" for layer in conv),
        "convolution activation",
    )
    require(
        len(pools) == 3
        and all(
            layer.pool_size == (2, 2)
            and layer.strides == (2, 2)
            and layer.padding == "valid"
            for layer in pools
        ),
        "pool configuration",
    )
    require(
        isinstance(model.layers[-3], tf.keras.layers.GlobalAveragePooling2D),
        "global pooling head",
    )
    require(
        isinstance(model.layers[-2], tf.keras.layers.Dropout)
        and model.layers[-2].rate == 0.1,
        "dropout configuration",
    )
    require(
        model.layers[-1].units == 5
        and model.layers[-1].activation.__name__ == "softmax",
        "output configuration",
    )
    require(model.output_shape == (None, 5), "output shape")
    require(
        all(layer.compute_dtype == "float32" for layer in model.layers),
        "compute dtype",
    )

    optimizer_config = probe["optimizer"].get_config()
    selected_optimizer = recipe["implementation_selection"]["optimizer"]
    for key in ("learning_rate", "beta_1", "beta_2", "epsilon"):
        require(
            np.isclose(
                optimizer_config[key], selected_optimizer[key],
                rtol=1e-6, atol=0,
            ),
            "optimizer control mismatch: " + key,
        )
    require(optimizer_config["amsgrad"] is False, "AMSGrad unexpectedly enabled")
    require(optimizer_config["use_ema"] is False, "EMA unexpectedly enabled")
    for key in (
        "weight_decay", "clipnorm", "clipvalue", "global_clipnorm",
        "gradient_accumulation_steps",
    ):
        require(optimizer_config.get(key) is None, "unexpected optimizer option: " + key)

    adapter = FixtureAdapter(9)
    images, labels = adapter.canonical_batch("train", np.arange(5, dtype=np.int64))
    initial_digest = probe["initial_weights_sha256"]
    probe["controls"]["train_step"](images, labels)
    require(probe["controls"]["optimizer_updates"]() == 1, "optimizer update count")
    require(state_digest(model.get_weights()) != initial_digest, "weights unchanged")

    before_validation = state_digest(probe["controls"]["get_execution_state"]())
    probe["controls"]["validation_step"](images, labels)
    require(
        state_digest(probe["controls"]["get_execution_state"]()) == before_validation,
        "validation mutated execution state",
    )
    del probe, model

    # Fresh construction after training must reproduce initial weights.
    replay = build()
    require(replay["initial_weights_sha256"] == initial_digest, "initialization replay")
    del replay
    alternate = build(43)
    require(alternate["initial_weights_sha256"] != initial_digest, "seed variation")
    del alternate

    arms = {}
    for name, train_count in (("fixture_real_only", 9), ("fixture_augmented", 14)):
        adapter = FixtureAdapter(train_count)
        training = RealSplitView(
            adapter=adapter, dataset_id="cicmaldroid2020", split="train",
            provenance_sha256="a" * 64,
        )
        validation = RealSplitView(
            adapter=adapter, dataset_id="cicmaldroid2020", split="validation",
            provenance_sha256="a" * 64,
        )
        binding = build()
        require(
            binding["initial_weights_sha256"] == initial_digest,
            "fixture arm initialization differs",
        )
        report = fit_classifier_control(
            training=training, validation=validation,
            evaluator_seed=42, recipe_raw=recipe_raw,
            **binding["controls"],
        )
        require(
            report["selected_weights_sha256"]
            == state_digest(binding["model"].get_weights()),
            "selected weights were not restored",
        )
        require(
            1 <= report["completed_epochs"] <= 20
            and report["optimizer_updates"] == report["completed_epochs"],
            "fixture training budget mismatch",
        )
        arms[name] = report
        del binding

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
            "project import source mismatch: " + name,
        )
        imports += 1

    report = {
        "scope": "SYNTHETIC_CLASSIFIER_BINDING_FIXTURE",
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "initial_weights_sha256": initial_digest,
        "architecture_checked": True,
        "optimizer_controls_checked": True,
        "initialization_replay_passed": True,
        "alternate_seed_changed_initialization": True,
        "optimizer_update_passed": True,
        "validation_state_unchanged": True,
        "arms": arms,
        "verified_project_import_count": imports,
        "production_paired_entry_point_exercised": False,
        "real_data_used": False,
        "test_evaluated": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("CLASSIFIER_TENSORFLOW_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
