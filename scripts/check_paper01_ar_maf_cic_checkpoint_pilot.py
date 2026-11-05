"""Explicit CIC seed42 checkpoint integration pilot; not benchmark scale-out."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ar", "maf"), required=True)
    parser.add_argument("--phase", choices=("train", "restore"), required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--publication-parent", required=True)
    parser.add_argument("--generation-name", required=True)
    args = parser.parse_args()

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
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_publication import (
        publish_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
        read_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
        encode_model_snapshot, restore_model_snapshot,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        canonical_validation_batches,
    )

    adapter = CICMalDroidRevision2026(
        **config["paths"], validate_hashes=True,
    )
    recipe_raw = (root / runner.RECIPE_PATH).read_bytes()
    verify_source()
    print("CIC_CHECKPOINT_PILOT_BEGIN", json.dumps({
        "family": args.family, "phase": args.phase,
        "head": head, "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(),
    }), flush=True)

    if args.phase == "train":
        prepared = runner.run_training_and_prepare_checkpoint(
            repository_root=root, family=args.family,
            dataset_id="cicmaldroid2020", experimental_seed=42,
            adapter=adapter, bounded_integration=True,
        )
        manifest = prepared["manifest"]
        if manifest["completed_training_steps"] != 254:
            raise RuntimeError("expected 254 complete-recipe optimizer updates")
        verify_source()
        published = publish_checkpoint_bundle(
            args.publication_parent, args.generation_name,
            manifest_raw=prepared["manifest_raw"],
            best_raw=prepared["best_raw"], last_raw=prepared["last_raw"],
            recipe_raw=recipe_raw,
            expected_identity=manifest["identity"],
            expected_source_commit=head,
            expected_source_files=source_files,
        )
        details = {
            "publication_status": published["publication_status"],
            "history": manifest["history"],
            "best_epoch_index": manifest["best_epoch_index"],
            "optimizer_updates": manifest["completed_training_steps"],
            "artifacts": manifest["artifacts"],
        }
    else:
        setup = prepare_training_setup(
            family=args.family, dataset_id="cicmaldroid2020",
            experimental_seed=42, adapter=adapter, recipe_raw=recipe_raw,
        )
        loaded = read_checkpoint_bundle(
            Path(args.publication_parent) / args.generation_name,
            recipe_raw=recipe_raw,
            expected_identity=setup["evidence"]["identity"],
            expected_source_commit=head,
            expected_source_files=source_files,
        )
        manifest = loaded["manifest"]
        if manifest["completed_training_steps"] != 254:
            raise RuntimeError("unexpected checkpoint training budget")
        losses = {}
        for role in ("best", "last"):
            restore_model_snapshot(
                model=setup["model"],
                specification=manifest["tensor_specification"],
                raw=loaded[role + "_raw"],
            )
            total, count = 0.0, 0
            for positions, images, labels in canonical_validation_batches(
                adapter=setup["adapter"], dataset_id="cicmaldroid2020",
            ):
                loss = setup["validation_step"](images, labels)
                total += len(positions) * loss
                count += len(positions)
            if count != 1158:
                raise RuntimeError("incomplete restoration validation")
            observed = total / count
            epoch = manifest["artifacts"][role]["epoch_index"]
            expected = manifest["history"][epoch]["validation_nll"]
            if not np.isclose(observed, expected, rtol=1e-6, atol=1e-6):
                raise RuntimeError("restored full-validation NLL mismatch: " + role)
            losses[role] = {"expected": expected, "observed": observed}
        if int(setup["optimizer"].iterations.numpy()) != 0:
            raise RuntimeError("restoration performed optimizer updates")
        details = {
            "fresh_process_restoration_passed": True,
            "validation_examples_per_role": 1158,
            "validation_losses": losses,
            "artifacts": manifest["artifacts"],
        }

    import_count = verify_source()
    print("CIC_CHECKPOINT_PILOT_RESULT", json.dumps({
        **details,
        "family": args.family, "phase": args.phase,
        "dataset_id": "cicmaldroid2020", "experimental_seed": 42,
        "head": head, "job_id": os.environ["SLURM_JOB_ID"],
        "pid": os.getpid(), "host": socket.gethostname(),
        "config_sha256": args.config_sha256,
        "generation_name": args.generation_name,
        "verified_project_import_count": import_count,
        "tensorflow": tf.__version__, "numpy": np.__version__,
        "general_training_enabled": False,
        "full_benchmark_ready": False,
    }, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
