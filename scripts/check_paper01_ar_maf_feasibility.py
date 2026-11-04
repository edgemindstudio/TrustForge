"""Synthetic-only AR/MAF feasibility diagnostic; no production publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ar", "maf"), required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Slurm node identity required")
    if socket.gethostname().split(".")[0].startswith("login"):
        raise RuntimeError("Compute-node execution required")

    recipe_path = Path(
        "studies/paper01_benchmark/revision_2026/ar_maf_execution_candidate.json"
    )
    raw = recipe_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != (
        "d082853e377913c30ccb951ea31994e56720559ee59e4b0120932ef4d42c35a6"
    ):
        raise RuntimeError("Candidate identity mismatch")
    recipe = json.loads(raw)
    if recipe["production_execution_authorized"] is not False:
        raise RuntimeError("Expected restricted candidate")

    import numpy as np
    import tensorflow as tf

    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Expected CPU-only diagnostic")
    tf.keras.utils.set_random_seed(42)

    record = {
        "family": args.family,
        "purpose": "SYNTHETIC_FEASIBILITY_ONLY",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "host": socket.gethostname(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "threads": threads,
    }
    print("FEASIBILITY_BEGIN", json.dumps(record), flush=True)

    batch = recipe["training"]["candidate_batch_size"]
    classes = recipe["feasibility"]["class_count"]
    rng = np.random.default_rng(42)
    images = rng.random((batch, 40, 40, 3), dtype=np.float32)
    images[0] = 0.0
    images[1] = 1.0
    labels = np.arange(batch, dtype=np.int32) % classes
    onehot = tf.one_hot(labels, classes, dtype=tf.float32)
    controls = recipe["optimizer"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=controls["learning_rate"],
        beta_1=controls["beta_1"],
        beta_2=controls["beta_2"],
        epsilon=controls["epsilon"],
        amsgrad=controls["amsgrad"],
    )
    prefix = "trustforge.models.paper01_revision_2026"

    if args.family == "ar":
        from trustforge.models.paper01_revision_2026.autoregressive import (
            build_revision_pixelcnn,
        )
        from trustforge.models.paper01_revision_2026.autoregressive_training import (
            AutoregressiveTrainingConfig, train_step, validation_step,
        )
        from trustforge.models.paper01_revision_2026.autoregressive_sampling import (
            sample_revision_pixelcnn,
        )
        ar = recipe["autoregressive"]
        model = build_revision_pixelcnn(
            (40, 40, 3), classes,
            filters=ar["filters"], masked_layers=ar["masked_layers"],
        )
        config = AutoregressiveTrainingConfig(
            epochs=2, patience=2, save_every=1,
            min_log_scale=ar["min_log_scale"],
            max_log_scale=ar["max_log_scale"],
        )
        # The primitive requires these fields; this diagnostic does not call fit().
        model([images, onehot], training=False).numpy()
        step = lambda: train_step(
            model, optimizer, images, onehot, config=config
        )
        validate = lambda: validation_step(
            model, images, onehot, config=config
        )
        sample = lambda: sample_revision_pixelcnn(
            model, class_ids=np.array([0], dtype=np.int32),
            img_shape=(40, 40, 3), num_classes=classes,
            temperature=ar["sampling_temperature"],
            min_log_scale=ar["min_log_scale"],
            max_log_scale=ar["max_log_scale"], seed=42,
        )
    else:
        from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
            ConditionalBlockMAF,
        )
        from trustforge.models.paper01_revision_2026.masked_autoflow_training import (
            train_step, conditional_nll,
        )
        model = ConditionalBlockMAF(
            classes,
            permutation_seeds=recipe["masked_autoflow"]["permutation_seeds"],
        )
        model.log_prob(images, labels, training=False).numpy()
        step = lambda: train_step(model, optimizer, images, labels)["loss"]
        validate = lambda: conditional_nll(
            model, images, labels, training=False
        )
        sample = lambda: model.sample(
            tf.constant([0], dtype=tf.int32), seed=42
        ).numpy()

    record["parameter_count"] = int(sum(
        np.prod(v.shape) for v in model.weights
    ))
    before = [v.numpy().copy() for v in model.trainable_variables]
    timings, losses = [], []
    for index in range(3):
        start = time.perf_counter()
        loss = float(step().numpy())
        if not np.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        timings.append(time.perf_counter() - start)
        losses.append(loss)
        print("TRAIN_STEP", index + 1, loss, timings[-1], flush=True)

    changed = any(
        not np.array_equal(old, variable.numpy())
        for old, variable in zip(before, model.trainable_variables)
    )
    if not changed or int(optimizer.iterations.numpy()) != 3:
        raise RuntimeError("Optimizer update verification failed")
    validation_loss = float(validate().numpy())
    if not np.isfinite(validation_loss):
        raise RuntimeError("Nonfinite validation loss")
    record.update(
        training_step_seconds=timings,
        training_losses=losses,
        validation_loss=validation_loss,
        optimizer_updates=3,
        parameters_changed=True,
    )

    def timeout_handler(signum, frame):
        raise TimeoutError("Sampling exceeded diagnostic time limit")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    start = time.perf_counter()
    signal.alarm(120)
    try:
        generated = sample()
        if (
            generated.shape != (1, 40, 40, 3)
            or generated.dtype != np.float32
            or not np.isfinite(generated).all()
            or np.any(generated < 0) or np.any(generated > 1)
        ):
            raise RuntimeError("Invalid generated sample")
        record["sampling_status"] = "COMPLETE_ONE_SAMPLE"
    except TimeoutError:
        record["sampling_status"] = "TIME_LIMIT_INCOMPLETE"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    record["sampling_seconds"] = time.perf_counter() - start
    record["peak_process_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    record["production_ready"] = False
    print("FEASIBILITY_RESULT", json.dumps(record, allow_nan=False), flush=True)
    if record["sampling_status"] != "COMPLETE_ONE_SAMPLE":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
