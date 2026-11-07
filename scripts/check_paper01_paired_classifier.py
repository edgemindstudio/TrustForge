"""Paired-entry-point engineering fixture; no real or generator-produced data."""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node, prepare_paired_classifiers,
    )
    require_compute_node()
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity, RECIPE_PATH,
    )
    root = Path(args.repository_root).resolve()
    require(Path.cwd().resolve() == root, "checkout mismatch")
    head, sources = capture_source_identity(root)
    require(head == args.expected_head, "source revision mismatch")
    run = args.run_dir.resolve(strict=True)
    require(not (run / "fixture_pool").exists(), "fixture destination exists")
    require(not (run / "report.json").exists(), "report already exists")

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_inputs import RealSplitView
    from trustforge.evaluation.paper01_cic_pool import CICPoolView
    from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
    from trustforge.evaluation.paper01_classifier_control import state_digest
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        synthesis_batch_seed,
    )

    recipe_raw = (
        root / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
    ).read_bytes()

    class FixtureAdapter:
        def __init__(self):
            self.requests = {"train": 0, "validation": 0}
            self.maximum_requested_batch = 0

        def split_size(self, split):
            if split == "train":
                return 9
            if split == "validation":
                return 5
            raise RuntimeError("fixture prohibits held-out test access")

        def canonical_batch(self, split, positions):
            self.split_size(split)
            self.requests[split] += 1
            self.maximum_requested_batch = max(
                self.maximum_requested_batch, len(positions),
            )
            require(len(positions) <= 128, "oversized real-image request")
            values = (positions % 5).astype(np.float32) / np.float32(4)
            images = np.broadcast_to(
                values[:, None, None, None], (len(positions), 40, 40, 3)
            ).copy()
            return images, np.eye(5, dtype=np.float32)[positions % 5]

    pool_dir = run / "fixture_pool"
    pool_dir.mkdir(mode=0o700)
    first = pool_dir / "c0_b0.npy"
    np.save(first, np.zeros((8, 40, 40, 3), dtype=np.float32))
    image_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    records = []
    for cls in range(5):
        for batch in range(250):
            filename = f"c{cls}_b{batch}.npy"
            if (cls, batch) != (0, 0):
                os.link(first, pool_dir / filename)
            records.append({
                "class_id": cls,
                "batch_index": batch,
                "sampling_seed": synthesis_batch_seed(
                    dataset_id="cicmaldroid2020", family="maf",
                    experimental_seed=42, class_id=cls, batch_index=batch,
                ),
                "filename": filename,
                "sha256": image_digest,
            })
    generator_recipe_digest = hashlib.sha256(
        (root / RECIPE_PATH).read_bytes()
    ).hexdigest()
    fixture_checkpoint_digest = hashlib.sha256(
        b"ENGINEERING_FIXTURE_NO_GENERATOR_CHECKPOINT"
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_id": "cicmaldroid2020",
        "family": "maf",
        "experimental_seed": 42,
        "recipe_sha256": generator_recipe_digest,
        "checkpoint_sha256": fixture_checkpoint_digest,
        "batches": records,
    }
    manifest_raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path = pool_dir / "manifest.json"
    manifest_path.write_bytes(manifest_raw)
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    synthetic = CICPoolView(
        manifest_path,
        manifest_sha256=manifest_digest,
        family="maf", experimental_seed=42,
        recipe_sha256=generator_recipe_digest,
        checkpoint_sha256=fixture_checkpoint_digest,
    )
    adapter = FixtureAdapter()
    provenance_digest = hashlib.sha256(
        b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE"
    ).hexdigest()
    real = RealSplitView(
        adapter=adapter, dataset_id="cicmaldroid2020",
        split="train", provenance_sha256=provenance_digest,
    )
    validation = RealSplitView(
        adapter=adapter, dataset_id="cicmaldroid2020",
        split="validation", provenance_sha256=provenance_digest,
    )
    augmented = CombinedTrainingView(real=real, synthetic=synthetic)
    require(augmented.size == 10009, "combined fixture size mismatch")

    print("PAIRED_CLASSIFIER_FIXTURE_BEGIN", flush=True)
    result = prepare_paired_classifiers(
        real_training=real, augmented_training=augmented,
        validation=validation, evaluator_seed=42, recipe_raw=recipe_raw,
    )
    require(
        result["status"] == "PAIRED_WEIGHTS_PREPARED_NOT_EVALUATED",
        "unexpected paired status",
    )
    require(result["test_evaluated"] is False, "unexpected test evaluation")
    require(result["complete_evaluation"] is False, "unexpected completeness")
    reports = {}
    for name, sample_count in (
        ("real_only", 9), ("real_plus_synthetic", 10009),
    ):
        arm = result["arms"][name]
        report = arm["report"]
        require(
            report["initial_weights_sha256"] == result["initial_weights_sha256"],
            "paired initial weights differ",
        )
        require(
            state_digest(arm["selected_weights"]) == report["selected_weights_sha256"],
            "selected weight snapshot mismatch",
        )
        require(
            report["optimizer_updates"]
            == report["completed_epochs"] * ((sample_count + 127) // 128),
            "arm optimizer accounting mismatch",
        )
        require(
            all(
                row["training_examples"] == sample_count
                and row["validation_examples"] == 5
                for row in report["history"]
            ),
            "arm coverage mismatch",
        )
        reports[name] = report

    require(capture_source_identity(root) == (head, sources), "source changed")
    import_count = 0
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
        import_count += 1

    report = {
        "scope": "PAIRED_ENTRY_POINT_SYNTHETIC_FIXTURE",
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "pool_manifest_sha256": manifest_digest,
        "initial_weights_sha256": result["initial_weights_sha256"],
        "production_paired_entry_point_exercised": True,
        "fixture_real_train_count": 9,
        "fixture_validation_count": 5,
        "fixture_synthetic_count": 10000,
        "maximum_real_batch_requested": adapter.maximum_requested_batch,
        "real_adapter_requests": adapter.requests,
        "arms": reports,
        "verified_project_import_count": import_count,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "generator_lineage_authenticated": False,
        "real_data_used": False,
        "test_evaluated": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    with (run / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("PAIRED_CLASSIFIER_ENTRY_POINT_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
