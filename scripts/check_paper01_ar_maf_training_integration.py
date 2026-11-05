"""Compute-node synthetic integration check; no production publication."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ar", "maf"), required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if socket.gethostname().split(".")[0].startswith("login"):
        raise RuntimeError("Compute node required")

    import tensorflow as tf
    from trustforge.models.paper01_revision_2026.ar_maf_training_core import (
        run_training_core,
    )
    from trustforge.models.paper01_revision_2026.ar_maf_family_steps import (
        bind_family_steps,
    )

    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only diagnostic required")
    tf.keras.utils.set_random_seed(42)

    recipe_path = Path(
        "studies/paper01_benchmark/revision_2026/"
        "ar_maf_integration_recipe.json"
    )
    raw = recipe_path.read_bytes()
    recipe = json.loads(raw)
    if (
        recipe["production_execution_authorized"] is not False
        or recipe["training"]["batch_size"] != 64
        or recipe["training"]["epochs"] != 2
    ):
        raise RuntimeError("Unexpected integration recipe")

    class Fixture:
        def __init__(self):
            self.images = {}
            self.labels = {}
            for split, count, seed in (
                ("train", 65, 1042), ("validation", 3, 2042),
            ):
                x = np.random.default_rng(seed).random(
                    (count, 40, 40, 3), dtype=np.float32
                )
                x[0] = 0.0
                x[1] = 1.0
                self.images[split] = x
                self.labels[split] = np.eye(2, dtype=np.float32)[
                    np.arange(count) % 2
                ]

        def split_size(self, split):
            return len(self.images[split])

        def canonical_batch(self, split, positions):
            return self.images[split][positions], self.labels[split][positions]

    adapter = Fixture()
    x = tf.convert_to_tensor(adapter.images["train"][:2])
    y = tf.convert_to_tensor(adapter.labels["train"][:2])

    if args.family == "ar":
        from trustforge.models.paper01_revision_2026.autoregressive import (
            build_revision_pixelcnn,
        )
        controls = recipe["autoregressive"]
        model = build_revision_pixelcnn(
            (40, 40, 3), 2, filters=controls["filters"],
            masked_layers=controls["masked_layers"],
        )
        model([x, y], training=False).numpy()
    else:
        from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
            ConditionalBlockMAF,
        )
        model = ConditionalBlockMAF(
            2, permutation_seeds=recipe["masked_autoflow"]["permutation_seeds"],
        )
        model.log_prob(x, tf.constant([0, 1], tf.int32), training=False).numpy()

    controls = recipe["optimizer"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=controls["learning_rate"],
        beta_1=controls["beta_1"], beta_2=controls["beta_2"],
        epsilon=controls["epsilon"], amsgrad=controls["amsgrad"],
    )
    initial = [v.copy() for v in model.get_weights()]
    train, validate = bind_family_steps(
        family=args.family, model=model, optimizer=optimizer,
    )
    result = run_training_core(
        adapter=adapter, dataset_id="malex", experimental_seed=42,
        model=model, training_step=train, validation_step=validate,
    )
    if (
        int(optimizer.iterations.numpy()) != 4
        or result["completed_training_steps"] != 4
        or result["completed_epochs"] != 2
    ):
        raise RuntimeError("Incorrect complete-epoch update accounting")
    for row in result["history"]:
        if (
            row["training_examples"] != 65
            or row["validation_examples"] != 3
            or row["training_batches"] != 2
            or row["validation_batches"] != 1
        ):
            raise RuntimeError("Incorrect traversal accounting")
    if not any(
        not np.array_equal(a, b)
        for a, b in zip(initial, result["last_weights"])
    ):
        raise RuntimeError("Training did not change model weights")

    selected = min(
        range(2), key=lambda i: result["history"][i]["validation_nll"]
    )
    if result["best_epoch_index"] != selected:
        raise RuntimeError("Incorrect checkpoint selection")

    for role, epoch in (("best", selected), ("last", 1)):
        model.set_weights(result[role + "_weights"])
        observed = validate(
            adapter.images["validation"], adapter.labels["validation"],
        )
        expected = result["history"][epoch]["validation_nll"]
        if not np.isclose(observed, expected, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Restored validation loss disagrees: " + role)

    record = {
        "purpose": "SYNTHETIC_TRAINING_INTEGRATION_ONLY",
        "family": args.family,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "recipe_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "history": result["history"],
        "best_epoch_index": selected,
        "optimizer_updates": int(optimizer.iterations.numpy()),
        "validation_state_guards_passed": True,
        "in_memory_restoration_passed": True,
        "publication_status": result["publication_status"],
        "production_ready": False,
    }
    print("TRAINING_INTEGRATION_RESULT", json.dumps(
        record, sort_keys=True, allow_nan=False
    ), flush=True)


if __name__ == "__main__":
    main()
