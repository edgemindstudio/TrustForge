"""Bounded CICMalDroid2020 MAF generator execution for seeds 42/43; no scale-out."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


BOUNDED_SEEDS = (42, 43)


def execution_generation(experimental_seed):
    """Return the seed-bound checkpoint generation name."""
    return "maf_cic_seed" + str(experimental_seed) + "_execution"


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "restore"), required=True)
    parser.add_argument(
        "--experimental-seed", type=int, choices=BOUNDED_SEEDS, default=42,
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--receipt-sha256")
    return parser


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    args = _parser().parse_args()

    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute node required")
    root = Path(args.repository_root).resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError("run from specified checkout")
    run = Path(args.run_dir).expanduser().resolve(strict=True)

    config_path = Path(args.dataset_config)
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError("configuration unavailable")
    config_raw = config_path.read_bytes()
    if sha(config_raw) != args.config_sha256:
        raise ValueError("configuration digest mismatch")
    config = json.loads(config_raw)
    if (
        type(config) is not dict
        or set(config) != {"dataset_id", "paths"}
        or config["dataset_id"] != "cicmaldroid2020"
        or type(config["paths"]) is not dict
        or set(config["paths"]) != {"source_csv", "feature_columns_path"}
        or any(type(v) is not str or not v.strip() for v in config["paths"].values())
    ):
        raise ValueError("explicit governed CIC paths required")

    from trustforge.models.paper01_revision_2026 import (
        ar_maf_training_runner as runner,
    )
    head, sources = runner.capture_source_identity(root)
    if head != args.expected_head or runner.TRAINING_EXECUTION_ENABLED is not False:
        raise RuntimeError("source or general authorization mismatch")

    def verify_source():
        if runner.capture_source_identity(root) != (head, sources):
            raise RuntimeError("source changed during execution")
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
            path = Path(filename).resolve()
            relative = path.relative_to(root).as_posix()
            if sha(path.read_bytes()) != sources.get(relative):
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
        raise RuntimeError("CPU-only execution required")

    from trustforge.data.paper01_revision_2026 import CICMalDroidRevision2026
    from trustforge.models.paper01_revision_2026.ar_maf_cic_execution import (
        prepare_cic_execution, publish_and_read_cic_execution, _capture,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_training_setup import (
        prepare_training_setup,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
        read_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
        restore_model_snapshot,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
        canonical_json_bytes,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
        canonical_validation_batches,
    )

    adapter = CICMalDroidRevision2026(
        **{
            key: Path(value).expanduser().resolve()
            for key, value in config["paths"].items()
        },
        validate_hashes=True,
    )
    parent = run / "checkpoints"
    generation = execution_generation(args.experimental_seed)
    receipt_path = run / "expectations.json"
    recipe_raw = (root / runner.RECIPE_PATH).read_bytes()
    verify_source()

    print("MAF_CIC_EXECUTION_BEGIN", args.phase, os.getpid(), flush=True)
    if args.phase == "train":
        if receipt_path.exists() or parent.exists():
            raise FileExistsError("run already contains execution output")
        prepared = prepare_cic_execution(
            repository_root=root, family="maf",
            experimental_seed=args.experimental_seed,
            adapter=adapter, dataset_config_raw=config_raw,
            bounded_integration=True,
        )
        verify_source()
        envelope = json.loads(prepared["execution_manifest_raw"])
        manifest = envelope["checkpoint_manifest"]
        if manifest["completed_training_steps"] != 254:
            raise RuntimeError("expected 254 optimizer updates")
        receipt = {
            "schema": "trustforge.paper01.maf-cic.execution-expectations.v1",
            "source_commit": head,
            "configuration_sha256": args.config_sha256,
            "generation_name": generation,
            "fitted_state_sha256":
                prepared["requested"]["expected_fitted_state_sha256"],
            "execution_manifest_file_sha256":
                sha(prepared["execution_manifest_raw"]),
            "training_pid": os.getpid(),
            "job_id": os.environ["SLURM_JOB_ID"],
        }
        # Capture expectations outside the checkpoint before publication.
        write_json(receipt_path, receipt)
        parent.mkdir(mode=0o700)
        result = publish_and_read_cic_execution(
            parent, generation, prepared=prepared,
        )
        details = {
            "publication_status": result["publication"]["publication_status"],
            "optimizer_updates": manifest["completed_training_steps"],
            "history": manifest["history"],
            "artifacts": manifest["artifacts"],
        }
    else:
        if not args.receipt_sha256:
            raise ValueError("external receipt digest required")
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("expectation receipt unavailable")
        receipt_raw = receipt_path.read_bytes()
        if sha(receipt_raw) != args.receipt_sha256:
            raise ValueError("expectation receipt digest mismatch")
        receipt = json.loads(receipt_raw)
        if (
            receipt["schema"]
            != "trustforge.paper01.maf-cic.execution-expectations.v1"
            or receipt["source_commit"] != head
            or receipt["configuration_sha256"] != args.config_sha256
            or receipt["generation_name"] != generation
            or receipt["job_id"] != os.environ["SLURM_JOB_ID"]
            or receipt["training_pid"] == os.getpid()
        ):
            raise RuntimeError("execution expectation mismatch")

        live = _capture(
            repository_root=root, adapter=adapter,
            dataset_config_raw=config_raw,
        )
        if live["sha256"] != receipt["fitted_state_sha256"]:
            raise RuntimeError("fresh adapter preprocessing differs from training")
        setup = prepare_training_setup(
            family="maf", dataset_id="cicmaldroid2020",
            experimental_seed=args.experimental_seed, adapter=adapter,
            recipe_raw=recipe_raw,
        )
        requested = {
            "recipe_raw": recipe_raw,
            "expected_identity": setup["evidence"]["identity"],
            "expected_source_commit": head,
            "expected_source_files": sources,
            "expected_fitted_state_sha256": receipt["fitted_state_sha256"],
        }
        loaded = read_checkpoint_bundle(parent / generation, **requested)
        envelope_raw = canonical_json_bytes(loaded["execution_manifest"]) + b"\n"
        if sha(envelope_raw) != receipt["execution_manifest_file_sha256"]:
            raise RuntimeError("published envelope differs from expectation")

        incompatible = dict(requested)
        incompatible["expected_fitted_state_sha256"] = (
            "0" * 64 if receipt["fitted_state_sha256"] != "0" * 64 else "1" * 64
        )
        try:
            read_checkpoint_bundle(parent / generation, **incompatible)
        except ValueError as error:
            if "requested fitted-state identity mismatch" not in str(error):
                raise
        else:
            raise RuntimeError("incompatible expectation was accepted")

        manifest = loaded["manifest"]
        if manifest["completed_training_steps"] != 254:
            raise RuntimeError("checkpoint budget mismatch")
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
                raise RuntimeError("restored validation NLL mismatch: " + role)
            losses[role] = {"expected": expected, "observed": observed}
        if int(setup["optimizer"].iterations.numpy()) != 0:
            raise RuntimeError("restoration performed optimizer updates")
        details = {
            "fresh_process_restoration_passed": True,
            "incompatible_expectation_rejected": True,
            "validation_examples_per_role": 1158,
            "validation_losses": losses,
            "artifacts": manifest["artifacts"],
        }

    import_count = verify_source()
    write_json(run / (args.phase + "_report.json"), {
        **details,
        "phase": args.phase,
        "source_commit": head,
        "family": "maf",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": args.experimental_seed,
        "pid": os.getpid(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "fitted_state_sha256": receipt["fitted_state_sha256"],
        "execution_manifest_file_sha256":
            receipt["execution_manifest_file_sha256"],
        "verified_project_import_count": import_count,
        "general_training_enabled": False,
        "full_benchmark_go": False,
    })
    print("MAF_CIC_EXECUTION_" + args.phase.upper() + "=PASS", flush=True)


if __name__ == "__main__":
    main()
