"""Synthetic checkpoint round trip; not a production publisher or loader."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ar", "maf"), required=True)
    parser.add_argument("--phase", choices=("write", "restore"), required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if socket.gethostname().split(".")[0].startswith("login"):
        raise RuntimeError("Compute node required")

    import tensorflow as tf
    from trustforge.models.paper01_revision_2026.ar_maf_family_steps import (
        bind_family_steps,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
        encode_model_snapshot, restore_model_snapshot,
    )
    from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
        open_checkpoint_directory, read_anchored_regular_file,
    )
    from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_file import (
        write_exclusive_vae_artifact,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
        MAX_ARCHIVE_BYTES,
    )

    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only diagnostic required")
    tf.keras.utils.set_random_seed(42 if args.phase == "write" else 43)

    recipe_raw = Path(
        "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
    ).read_bytes()
    recipe = json.loads(recipe_raw)
    if recipe["production_execution_authorized"] is not False:
        raise RuntimeError("Expected restricted integration recipe")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    identity = {
        "family": args.family,
        "head": head,
        "recipe_sha256": digest(recipe_raw),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
    }

    def fixture(seed, count):
        x = np.random.default_rng(seed).random(
            (count, 40, 40, 3), dtype=np.float32
        )
        x[0] = 0
        x[1] = 1
        y = np.eye(2, dtype=np.float32)[np.arange(count) % 2]
        return x, y

    train_x, train_y = fixture(1042, 4)
    val_x, val_y = fixture(2042, 3)
    if args.family == "ar":
        from trustforge.models.paper01_revision_2026.autoregressive import (
            build_revision_pixelcnn,
        )
        controls = recipe["autoregressive"]
        model = build_revision_pixelcnn(
            (40, 40, 3), 2, filters=controls["filters"],
            masked_layers=controls["masked_layers"],
        )
        model([
            tf.convert_to_tensor(train_x),
            tf.convert_to_tensor(train_y),
        ], training=False).numpy()
    else:
        from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
            ConditionalBlockMAF,
        )
        model = ConditionalBlockMAF(
            2, permutation_seeds=recipe["masked_autoflow"]["permutation_seeds"],
        )
        model.log_prob(
            tf.convert_to_tensor(train_x),
            tf.constant([0, 1, 0, 1], tf.int32), training=False,
        ).numpy()

    controls = recipe["optimizer"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=controls["learning_rate"],
        beta_1=controls["beta_1"], beta_2=controls["beta_2"],
        epsilon=controls["epsilon"], amsgrad=controls["amsgrad"],
    )
    train, validate = bind_family_steps(
        family=args.family, model=model, optimizer=optimizer,
    )
    directory = Path(args.directory)

    if args.phase == "write":
        directory.mkdir(mode=0o700, exist_ok=False)
        before = [v.copy() for v in model.get_weights()]
        train(train_x, train_y)
        if not any(
            not np.array_equal(a, b)
            for a, b in zip(before, model.get_weights())
        ):
            raise RuntimeError("Training did not change model state")
        expected_loss = validate(val_x, val_y)
        specification, raw = encode_model_snapshot(
            model=model, weights=model.get_weights(),
        )
        evidence = {
            "schema": "trustforge.ar-maf.diagnostic-roundtrip.v1",
            "identity": identity,
            "specification": specification,
            "state_sha256": digest(raw),
            "state_bytes": len(raw),
            "validation_nll": expected_loss,
            "writer_pid": os.getpid(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "production_ready": False,
        }
        encoded = (
            json.dumps(evidence, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with open_checkpoint_directory(directory) as fd:
            write_exclusive_vae_artifact(fd, "state.npz", raw)
            write_exclusive_vae_artifact(fd, "diagnostic.json", encoded)
            os.fsync(fd)
        print("CHECKPOINT_DIAGNOSTIC_WRITTEN", json.dumps({
            "family": args.family, "tensor_count": len(specification),
            "state_bytes": len(raw), "writer_pid": os.getpid(),
        }), flush=True)
    else:
        with open_checkpoint_directory(directory) as fd:
            evidence = json.loads(read_anchored_regular_file(
                fd, "diagnostic.json", max_bytes=1_000_000,
            ))
            raw = read_anchored_regular_file(
                fd, "state.npz", max_bytes=MAX_ARCHIVE_BYTES,
            )
        if (
            evidence["identity"] != identity
            or evidence["schema"] != "trustforge.ar-maf.diagnostic-roundtrip.v1"
            or evidence["job_id"] != os.environ["SLURM_JOB_ID"]
            or evidence["production_ready"] is not False
            or evidence["writer_pid"] == os.getpid()
            or evidence["state_bytes"] != len(raw)
            or evidence["state_sha256"] != digest(raw)
        ):
            raise RuntimeError("Diagnostic identity or artifact mismatch")

        initial_raw = encode_model_snapshot(
            model=model, weights=model.get_weights(),
        )[1]
        if initial_raw == raw:
            raise RuntimeError("Fresh initialization unexpectedly equals checkpoint")

        restore_model_snapshot(
            model=model, specification=evidence["specification"], raw=raw,
        )
        observed = validate(val_x, val_y)
        if not np.isclose(
            observed, evidence["validation_nll"], rtol=1e-6, atol=1e-6,
        ):
            raise RuntimeError("Fresh-process validation loss mismatch")
        print("CHECKPOINT_ROUNDTRIP_RESULT", json.dumps({
            **identity,
            "job_id": os.environ["SLURM_JOB_ID"],
            "writer_pid": evidence["writer_pid"],
            "reader_pid": os.getpid(),
            "tensor_count": len(evidence["specification"]),
            "state_sha256": digest(raw),
            "state_bytes": len(raw),
            "expected_validation_nll": evidence["validation_nll"],
            "restored_validation_nll": observed,
            "fresh_process_restoration_passed": True,
            "production_ready": False,
        }, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
