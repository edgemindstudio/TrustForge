"""Generate a declared CIC AR class and batch range from a verified checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ar", "maf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-id", type=int, choices=range(5), required=True)
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--batch-stop", type=int, required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--publication-parent", required=True)
    parser.add_argument("--generation-name", required=True)
    args = parser.parse_args()
    if not 0 <= args.batch_start < args.batch_stop <= 250:
        parser.error("require 0 <= batch-start < batch-stop <= 250")
    if args.family != "ar":
        parser.error("this partition diagnostic requires --family ar")

    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute node required")

    root = Path(args.repository_root).resolve()
    config_path = Path(args.dataset_config)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("configuration unavailable")
    config_raw = config_path.read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != args.config_sha256:
        raise ValueError("configuration digest mismatch")
    config = json.loads(config_raw)
    if (
        set(config) != {"dataset_id", "paths"}
        or config["dataset_id"] != "cicmaldroid2020"
        or type(config["paths"]) is not dict
        or set(config["paths"]) != {"source_csv", "feature_columns_path"}
        or any(
            type(value) is not str or not value.strip()
            for value in config["paths"].values()
        )
    ):
        raise ValueError("pilot requires explicit CIC source paths")

    from trustforge.models.paper01_revision_2026 import (
        ar_maf_training_runner as runner,
    )
    head, source_files = runner.capture_source_identity(root)
    if head != args.expected_head or runner.TRAINING_EXECUTION_ENABLED is not False:
        raise RuntimeError("source revision or general execution gate mismatch")

    def verify_source():
        if runner.capture_source_identity(root) != (head, source_files):
            raise RuntimeError("source changed during pilot")
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
            file = Path(filename).resolve()
            relative = file.relative_to(root).as_posix()
            if (
                relative not in source_files
                or hashlib.sha256(file.read_bytes()).hexdigest()
                != source_files[relative]
            ):
                raise RuntimeError("project import source mismatch: " + name)
            count += 1
        return count

    verify_source()

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only pilot required")

    from trustforge.data.paper01_revision_2026 import CICMalDroidRevision2026
    from trustforge.models.paper01_revision_2026.ar_maf_training_setup import (
        prepare_training_setup,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
        read_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
        restore_model_snapshot,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        canonical_validation_batches,
    )

    adapter = CICMalDroidRevision2026(
        **config["paths"], validate_hashes=True,
    )
    recipe_raw = (root / runner.RECIPE_PATH).read_bytes()
    verify_source()

    import math

    import numpy as np

    training_commit, best_sha, generation = {
        "ar": (
            "e4473abcdbc53f0de3a01ebe48858f2f3c7b841a",
            "d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0",
            "ar_cic_seed42",
        ),
        "maf": (
            "d33c0e9146caab11fb3b492361873daf059ce6ce",
            "3604bd4009e2199400665b28ef156d5db8e6d2e75094e5eb1d193b090d89264d",
            "maf_cic_seed42",
        ),
    }[args.family]
    if args.generation_name != generation:
        raise ValueError("unexpected checkpoint generation")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    def git(*command):
        return subprocess.check_output(["git", "-C", str(root), *command])

    names = git("ls-tree", "-r", "--name-only", "-z", training_commit)
    historical = {
        name: hashlib.sha256(git("show", f"{training_commit}:{name}")).hexdigest()
        for name in names.decode().split("\0")
        if name.endswith(".py") or name == runner.RECIPE_PATH
    }
    if historical.get(runner.RECIPE_PATH) != hashlib.sha256(recipe_raw).hexdigest():
        raise RuntimeError("historical recipe mismatch")

    setup = prepare_training_setup(
        family=args.family, dataset_id="cicmaldroid2020",
        experimental_seed=42, adapter=adapter, recipe_raw=recipe_raw,
    )
    identity = setup["evidence"]["identity"]
    if identity["class_mapping"] != ["0", "1", "2", "3", "4"]:
        raise RuntimeError("unexpected CIC class mapping")
    loaded = read_checkpoint_bundle(
        Path(args.publication_parent) / generation,
        recipe_raw=recipe_raw,
        expected_identity=identity,
        expected_source_commit=training_commit,
        expected_source_files=historical,
    )
    manifest = loaded["manifest"]
    if (
        manifest["completed_training_steps"] != 254
        or hashlib.sha256(loaded["best_raw"]).hexdigest() != best_sha
    ):
        raise RuntimeError("unexpected best checkpoint identity")
    model = setup["model"]
    restore_model_snapshot(
        model=model, specification=manifest["tensor_specification"],
        raw=loaded["best_raw"],
    )

    total, count = 0.0, 0
    for positions, images, labels in canonical_validation_batches(
        adapter=setup["adapter"], dataset_id="cicmaldroid2020",
    ):
        total += len(positions) * setup["validation_step"](images, labels)
        count += len(positions)
    epoch = manifest["best_epoch_index"]
    expected = manifest["history"][epoch]["validation_nll"]
    if count != 1158:
        raise RuntimeError("incomplete validation coverage")
    observed = total / count
    if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("restored best checkpoint validation NLL mismatch")


    from time import perf_counter

    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        synthesis_batch_seed,
    )

    class_id = args.class_id
    args.output.mkdir(mode=0o700)
    for batch_index in range(args.batch_start, args.batch_stop):
        class_ids = np.full(8, class_id, dtype=np.int32)
        sampling_seed = synthesis_batch_seed(
            dataset_id="cicmaldroid2020",
            family=args.family,
            experimental_seed=42,
            class_id=class_id,
            batch_index=batch_index,
        )
        if args.family == "ar":
            from trustforge.models.paper01_revision_2026.autoregressive_sampling import (
                sample_revision_pixelcnn,
            )

            def sample():
                return sample_revision_pixelcnn(
                    model, class_ids=class_ids.copy(),
                    img_shape=(40, 40, 3), num_classes=5, seed=sampling_seed,
                )
        else:
            conditioning = tf.constant(class_ids, dtype=tf.int32)

            def sample():
                return model.sample(conditioning, seed=sampling_seed)

        verify_source()
        print("CIC_AR_PARTITION_BATCH_BEGIN", args.family, flush=True)
        started = perf_counter()
        images = np.asarray(sample())
        elapsed = perf_counter() - started
        if (
            images.shape != (8, 40, 40, 3)
            or images.dtype != np.float32
            or not np.isfinite(images).all()
            or np.any(images < 0)
            or np.any(images > 1)
        ):
            raise RuntimeError("invalid canonical eight-image batch")
        import_count = verify_source()

        batch_output = args.output / f"batch-{batch_index:03d}"
        batch_output.mkdir(mode=0o700)
        archive = batch_output / "samples.npz"
        with archive.open("xb") as stream:
            np.savez(stream, images=images, class_ids=class_ids)

        report = {
            "scope": "checkpoint_backed_ar_partition_batch_diagnostic",
            "family": args.family,
            "dataset_id": "cicmaldroid2020",
            "experimental_seed": 42,
            "class_id": class_id,
            "batch_index": batch_index,
            "sampling_seed": sampling_seed,
            "sampling_wall_seconds": elapsed,
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "checkpoint_sha256": best_sha,
            "training_commit": training_commit,
            "synthesis_commit": head,
            "validation_examples": count,
            "expected_validation_nll": expected,
            "restored_validation_nll": observed,
            "config_sha256": args.config_sha256,
            "samples_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "host": socket.gethostname(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "verified_project_import_count": import_count,
            "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        }
        encoded = json.dumps(report, sort_keys=True, allow_nan=False)
        with (batch_output / "report.json").open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print("CIC_AR_PARTITION_BATCH_RESULT", encoded, flush=True)
    print("CIC_AR_RANGE_WORKER=PASS", class_id,
          args.batch_start, args.batch_stop, flush=True)


if __name__ == "__main__":
    main()
