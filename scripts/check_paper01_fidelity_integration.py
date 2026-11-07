"""Persisted-reference/actual-Inception/KID engineering fixture."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import require_compute_node
    require_compute_node()
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity, RECIPE_PATH,
    )
    root = Path(args.repository_root).resolve()
    require(Path.cwd().resolve() == root, "checkout mismatch")
    head, sources = capture_source_identity(root)
    require(head == args.expected_head, "revision mismatch")
    run = args.run_dir.resolve(strict=True)
    require(not (run / "fixture_pool").exists(), "fixture destination exists")
    require(not (run / "selection.json").exists(), "selection already exists")
    require(not (run / "report.json").exists(), "report already exists")

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_inputs import RealSplitView
    from trustforge.evaluation.paper01_cic_pool import CICPoolView
    from trustforge.evaluation.paper01_reference_selection import (
        select_balanced_references,
    )
    from trustforge.evaluation.paper01_inception import InceptionFeatures
    from trustforge.evaluation.paper01_fidelity import prepare_features_and_kid
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        synthesis_batch_seed,
    )

    real_labels = np.repeat(np.arange(5, dtype=np.int32), [3, 2, 2, 2, 2])
    synthetic_labels = np.repeat(np.arange(5, dtype=np.int32), 2000)

    class FixtureAdapter:
        def __init__(self):
            self.maximum_request = 0

        def split_size(self, split):
            require(split == "validation", "fixture allows validation only")
            return len(real_labels)

        def canonical_batch(self, split, positions):
            self.split_size(split)
            self.maximum_request = max(self.maximum_request, len(positions))
            labels = real_labels[positions]
            images = np.zeros((len(positions), 40, 40, 3), dtype=np.float32)
            images[:, :, :, 0] = (labels / np.float32(4))[:, None, None]
            images[:, :, :, 2] = (np.float32(1) - labels / np.float32(4))[:, None, None]
            return images, np.eye(5, dtype=np.float32)[labels]

    adapter = FixtureAdapter()
    real = RealSplitView(
        adapter=adapter, dataset_id="cicmaldroid2020", split="validation",
        provenance_sha256=hashlib.sha256(
            b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE"
        ).hexdigest(),
    )

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
                "class_id": cls, "batch_index": batch,
                "filename": filename, "sha256": image_digest,
                "sampling_seed": synthesis_batch_seed(
                    dataset_id="cicmaldroid2020", family="maf",
                    experimental_seed=42, class_id=cls, batch_index=batch,
                ),
            })
    generator_recipe = hashlib.sha256((root / RECIPE_PATH).read_bytes()).hexdigest()
    fixture_checkpoint = hashlib.sha256(
        b"ENGINEERING_FIXTURE_NO_GENERATOR_CHECKPOINT"
    ).hexdigest()
    manifest = {
        "schema_version": 1, "dataset_id": "cicmaldroid2020",
        "family": "maf", "experimental_seed": 42,
        "recipe_sha256": generator_recipe,
        "checkpoint_sha256": fixture_checkpoint,
        "batches": records,
    }
    manifest_raw = canonical(manifest)
    manifest_path = pool_dir / "manifest.json"
    manifest_path.write_bytes(manifest_raw)
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    synthetic = CICPoolView(
        manifest_path, manifest_sha256=manifest_digest,
        family="maf", experimental_seed=42,
        recipe_sha256=generator_recipe, checkpoint_sha256=fixture_checkpoint,
    )

    selection = select_balanced_references(
        real_validation_labels=real_labels,
        synthetic_labels=synthetic_labels,
        class_count=5,
        real_validation_identity_sha256=hashlib.sha256(
            canonical(real.identity)
        ).hexdigest(),
        synthetic_manifest_sha256=manifest_digest,
    )
    # Persist before constructing the extractor or examining features.
    selection_path = run / "selection.json"
    with selection_path.open("xb") as handle:
        handle.write(canonical(selection) + b"\n")
    selection_raw = selection_path.read_bytes()
    selection_digest = hashlib.sha256(selection_raw).hexdigest()
    require(selection["real_reference"]["samples_per_class"] == 2, "reference count")

    extractor = InceptionFeatures(weights_path=args.weights)
    result = prepare_features_and_kid(
        selection_raw=selection_raw, selection_sha256=selection_digest,
        real_validation_view=real, synthetic_view=synthetic,
        real_validation_labels=real_labels, synthetic_labels=synthetic_labels,
        extractor=extractor,
    )
    feature_files = {}
    for key in ("real_features", "synthetic_features"):
        values = result[key]
        require(
            values.shape == (10, 2048)
            and values.dtype == np.float32
            and np.isfinite(values).all(),
            "invalid integrated feature array",
        )
        path = run / (key + ".npy")
        with path.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
        feature_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    require(adapter.maximum_request <= 16, "unbounded selected image access")
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
        "scope": "REFERENCE_INCEPTION_KID_SYNTHETIC_FIXTURE",
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "selection_file_sha256": selection_digest,
        "pool_manifest_sha256": manifest_digest,
        "feature_file_sha256": feature_files,
        "maximum_real_image_request": adapter.maximum_request,
        "fidelity": result["report"],
        "verified_project_import_count": imports,
        "selection_persisted_before_extraction": True,
        "actual_inception_used": True,
        "real_data_used": False,
        "generator_lineage_authenticated": False,
        "primary_scientific_result": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    with (run / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("REFERENCE_INCEPTION_KID_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
