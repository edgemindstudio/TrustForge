"""Bounded actual-source AR/MAF wiring diagnostic; no publication."""

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
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
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
        raise ValueError("dataset configuration unavailable")
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if type(config) is not dict or set(config) != {"dataset_id", "paths"}:
        raise ValueError("expected dataset_id and paths only")

    requirements = {
        "ustc_tfc2016": {"source_dir"},
        "cicmaldroid2020": {"source_csv", "feature_columns_path"},
        "malnet_image": {"image_root", "preparation_dir", "mapping_path"},
        "malex": {
            "label_archive", "benign_byteplot_root", "malware_byteplot_root",
        },
    }
    dataset = config["dataset_id"]
    paths = config["paths"]
    if (
        type(dataset) is not str or dataset not in requirements
        or type(paths) is not dict or set(paths) != requirements[dataset]
        or any(type(p) is not str or not p.strip() for p in paths.values())
    ):
        raise ValueError("invalid dataset paths or field inventory")

    from trustforge.models.paper01_revision_2026 import (
        ar_maf_training_runner as runner,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_training_setup import (
        prepare_training_setup,
    )
    head, source_files = runner.capture_source_identity(root)
    if head != args.expected_head:
        raise RuntimeError("unexpected source revision")
    if runner.TRAINING_EXECUTION_ENABLED is not False:
        raise RuntimeError("full training runner must remain disabled")

    def verify_imports():
        observed = {}
        for name, module in tuple(sys.modules.items()):
            if not (
                name == "trustforge" or name.startswith("trustforge.")
                or name == "common" or name.startswith("common.")
            ):
                continue
            filename = getattr(module, "__file__", None)
            if filename is None:
                # Namespace package paths must also resolve inside this checkout.
                for directory in getattr(module, "__path__", ()):
                    Path(directory).resolve().relative_to(root)
                continue
            file = Path(filename).resolve()
            relative = file.relative_to(root).as_posix()
            if relative not in source_files:
                raise RuntimeError("untracked project import: " + name)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != source_files[relative]:
                raise RuntimeError("project import differs from captured source")
            observed[name] = {"path": relative, "sha256": digest}
        return observed

    verify_imports()

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only diagnostic required")

    from trustforge.data.paper01_revision_2026 import (
        USTCRevision2026, CICMalDroidRevision2026,
        MalNetImageRevision2026, MaleXRevision2026,
    )
    constructors = {
        "ustc_tfc2016": USTCRevision2026,
        "cicmaldroid2020": CICMalDroidRevision2026,
        "malnet_image": MalNetImageRevision2026,
        "malex": MaleXRevision2026,
    }
    adapter = constructors[dataset](**paths)
    setup = prepare_training_setup(
        family=args.family, dataset_id=dataset, experimental_seed=42,
        adapter=adapter, recipe_raw=(root / runner.RECIPE_PATH).read_bytes(),
    )
    verify_imports()
    checked = setup["adapter"]
    train_size = checked.split_size("train")
    train_positions = (
        np.arange(64, dtype=np.int64),
        np.array([train_size - 1], dtype=np.int64),
    )
    losses = []
    for positions in train_positions:
        images, labels = checked.canonical_batch("train", positions)
        losses.append(setup["training_step"](images, labels))

    validation_positions = np.arange(3, dtype=np.int64)
    images, labels = checked.canonical_batch(
        "validation", validation_positions,
    )
    validation_loss = setup["validation_step"](images, labels)
    updates = int(setup["optimizer"].iterations.numpy())
    if updates != 2:
        raise RuntimeError("expected exactly two optimizer updates")
    if not all(np.isfinite(v) for v in [*losses, validation_loss]):
        raise RuntimeError("nonfinite diagnostic loss")
    if runner.capture_source_identity(root) != (head, source_files):
        raise RuntimeError("source changed during diagnostic")
    imports = verify_imports()

    record = {
        "purpose": "BOUNDED_ACTUAL_SOURCE_WIRING_ONLY",
        "family": args.family,
        "dataset_id": dataset,
        "experimental_seed": 42,
        "head": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "dataset_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "resolved_paths": {
            key: str(Path(value).expanduser().resolve())
            for key, value in paths.items()
        },
        "identity": setup["evidence"]["identity"],
        "train_positions": [p.tolist() for p in train_positions],
        "validation_positions": validation_positions.tolist(),
        "training_losses": losses,
        "validation_loss": validation_loss,
        "optimizer_updates": updates,
        "verified_project_imports": imports,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "full_training_runner_enabled": False,
        "checkpoint_published": False,
        "production_ready": False,
    }
    print("REAL_INPUT_INTEGRATION_RESULT", json.dumps(
        record, sort_keys=True, allow_nan=False
    ), flush=True)


if __name__ == "__main__":
    main()
