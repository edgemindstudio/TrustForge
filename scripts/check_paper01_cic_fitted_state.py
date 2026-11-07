"""Compute-node CIC fitted-state capture and independent-process verification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("capture", "restore"), required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()

    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute node required")

    root = Path(args.repository_root).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError("run from the specified repository root")

    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )
    head, source_files = capture_source_identity(root)
    if head != args.expected_head:
        raise RuntimeError("source revision mismatch")

    config_path = Path(args.dataset_config).expanduser()
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("configuration must be a regular source file")
    config_raw = config_path.read_bytes()
    if digest(config_raw) != args.config_sha256:
        raise ValueError("configuration digest mismatch")
    config = json.loads(config_raw)
    if (
        type(config) is not dict
        or set(config) != {"dataset_id", "paths"}
        or config["dataset_id"] != "cicmaldroid2020"
        or type(config["paths"]) is not dict
        or set(config["paths"]) != {"source_csv", "feature_columns_path"}
        or any(
            type(value) is not str or not value.strip()
            for value in config["paths"].values()
        )
    ):
        raise ValueError("invalid CIC configuration")

    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from trustforge.data.paper01_revision_2026 import CICMalDroidRevision2026
    from trustforge.data.paper01_revision_2026.cic_fitted_state import (
        capture_fitted_state, restore_fitted_state,
    )
    from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
        require_gan_dataset_provenance,
    )

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only diagnostic required")

    paths = {
        key: Path(value).expanduser().resolve()
        for key, value in config["paths"].items()
    }
    adapter = CICMalDroidRevision2026(**paths, validate_hashes=True)
    verified = require_gan_dataset_provenance(
        dataset_id="cicmaldroid2020", adapter=adapter,
    )

    context = {
        "source_preflight": verified,
        "configuration_sha256": digest(config_raw),
        "feature_columns": list(adapter.feature_columns),
        "partition": {
            "seed": adapter.partition_seed,
            "ratios": [
                adapter.train_fraction,
                adapter.validation_fraction,
                adapter.test_fraction,
            ],
        },
        "implementation": {
            "commit": head,
            "tracked_source_files": source_files,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "byteorder": sys.byteorder,
        },
    }
    record = capture_fitted_state(
        scaler=adapter.scaler,
        training_indices=adapter.partitions.train,
        context=context,
    )

    destination = Path(args.evidence_dir).expanduser()
    if args.phase == "capture":
        if destination.exists():
            raise FileExistsError("capture destination already exists")
    elif destination.is_symlink() or not destination.is_dir():
        raise ValueError("restore evidence directory unavailable")

    if args.phase == "restore":
        stored = json.loads(
            (destination / "fitted_state.json").read_text(encoding="utf-8")
        )
        restored = restore_fitted_state(stored)
        if stored != record:
            raise RuntimeError("fresh adapter evidence differs from captured state")
    else:
        restored = restore_fitted_state(record)

    # Train-only bounded probe; no validation/test inputs or model execution.
    positions = np.array([0, 1, 63, 64, 8116], dtype=np.int64)
    raw = adapter.arrays.features[adapter.partitions.train[positions]].copy()
    expected = adapter.scaler.transform(raw.copy())
    observed = restored.transform(raw.copy())
    if (
        expected.dtype != np.float32
        or observed.dtype != np.float32
        or not np.array_equal(expected, observed)
    ):
        raise RuntimeError("restored transform differs from live fitted scaler")

    # Bind imported project code to the captured tracked checkout.
    imports = {}
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
        if digest(path.read_bytes()) != source_files.get(relative):
            raise RuntimeError("project import differs from captured source: " + name)
        imports[name] = relative

    if capture_source_identity(root) != (head, source_files):
        raise RuntimeError("source changed during diagnostic")

    report = {
        "purpose": "BOUNDED_CIC_FITTED_STATE_EVIDENCE",
        "phase": args.phase,
        "source_commit": head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "evidence_sha256": record["sha256"],
        "probe_train_positions": positions.tolist(),
        "transform_equal": True,
        "verified_project_imports": imports,
        "production_provenance_integrated": False,
        "full_benchmark_go": False,
    }

    if args.phase == "capture":
        destination.mkdir(parents=False, exist_ok=False)
        with (destination / "fitted_state.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")

    with (destination / (args.phase + "_report.json")).open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print("CIC_FITTED_STATE_" + args.phase.upper() + "=PASS", flush=True)


if __name__ == "__main__":
    main()
