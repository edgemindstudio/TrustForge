#!/usr/bin/env bash
set -euo pipefail
umask 077

expected="$1"
cd "$HOME/TrustForge"
[[ "$(git rev-parse HEAD)" == "$expected" ]]
git diff --quiet
git diff --cached --quiet

export PYTHONPATH="$PWD/src:$PWD"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=1

pilot="$HOME/trustforge_revision_2026_artifacts/paper01/diagnostics/ar-maf-cic-synthesis-cbcde60-Y7HczdR3"
"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u - "$pilot" "$expected" <<'PY'
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from trustforge.data.paper01_revision_2026 import CICMalDroidRevision2026
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    require_gan_dataset_provenance,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verified_bytes(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f"digest mismatch: {path}")
    return raw


pilot = Path(sys.argv[1])
config_sha = "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
config = json.loads(verified_bytes(pilot / "dataset_config.json", config_sha))
adapter = CICMalDroidRevision2026(**config["paths"], validate_hashes=True)
require_gan_dataset_provenance(dataset_id="cicmaldroid2020", adapter=adapter)

classes = np.arange(5, dtype=np.int32)
for split, count in (("train", 8117), ("validation", 1158), ("test", 2323)):
    require(adapter.split_size(split) == count, f"unexpected {split} size")


def real_arrays(split):
    batches = [
        adapter.canonical_batch(split, np.arange(start, min(start + 128, count)))
        for count in (adapter.split_size(split),)
        for start in range(0, count, 128)
    ]
    return tuple(np.concatenate(parts) for parts in zip(*batches))


x_train, y_train = real_arrays("train")
x_val, y_val = real_arrays("validation")
digests = {
    "ar": "4b45339fa539e3070dea7096821f6e7158a20de6dac97fec66cbe3461fc54a22",
    "maf": "251e4994d1c24ae4e529621d7255bc92822556df6f9fad2d26cc14b91b5e942a",
}
arms = {"real_only": (x_train, y_train)}
for family, digest in digests.items():
    raw = verified_bytes(pilot / f"{family}_synthesis" / "samples.npz", digest)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        images, ids = archive["images"], archive["class_ids"]
    require(images.shape == (5, 40, 40, 3), f"{family} shape")
    require(images.dtype == np.float32 and np.isfinite(images).all(), f"{family} dtype or finiteness")
    require(((0 <= images) & (images <= 1)).all(), f"{family} range")
    require(ids.dtype == np.int32 and np.array_equal(ids, classes), f"{family} class IDs")
    labels = np.eye(5, dtype=y_train.dtype)[ids]
    arms[f"real_plus_{family}"] = (
        np.concatenate((x_train, images)),
        np.concatenate((y_train, labels)),
    )


def build_classifier():
    layers = [tf.keras.Input(shape=(40, 40, 3))]
    for width in (32, 64, 128):
        layers.extend((
            tf.keras.layers.Conv2D(width, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
        ))
    layers.extend((
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(5, activation="softmax"),
    ))
    return tf.keras.Sequential(layers)


tf.config.experimental.enable_op_determinism()
initial_sha = None
trained = {}
results = {}
for arm, (images, labels) in arms.items():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = build_classifier()
    weight_sha = hashlib.sha256(
        b"".join(weight.tobytes() for weight in model.get_weights())
    ).hexdigest()
    if initial_sha is None:
        initial_sha = weight_sha
    require(weight_sha == initial_sha, f"{arm} initial weights differ")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
    )
    selector = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True,
    )
    history = model.fit(
        images, labels, validation_data=(x_val, y_val),
        epochs=2, batch_size=128, callbacks=[selector], verbose=2,
    ).history
    losses = history["val_loss"]
    require(len(losses) == 2 and np.isfinite(losses).all(), f"{arm} validation losses")
    require(
        all(np.array_equal(current, selected)
            for current, selected in zip(model.get_weights(), selector.best_weights)),
        f"{arm} validation-selected weights were not restored",
    )
    trained[arm] = model
    results[arm] = {
        "initial_weights_sha256": weight_sha,
        "train_count": len(images),
        "validation_loss": losses,
        "selected_epoch": int(np.argmin(losses)) + 1,
        "optimizer_updates": int(model.optimizer.iterations.numpy()),
    }

x_test, y_test = real_arrays("test")
for arm, model in trained.items():
    scores = model.predict(x_test, batch_size=128, verbose=0)
    require(scores.shape == (2323, 5) and np.isfinite(scores).all(), f"{arm} predictions")
    results[arm]["test_macro_f1"] = f1_score(
        y_test.argmax(axis=1), scores.argmax(axis=1),
        labels=classes, average="macro", zero_division=0,
    )

baseline = results["real_only"]["test_macro_f1"]
for arm in ("real_plus_ar", "real_plus_maf"):
    results[arm]["delta_macro_f1"] = results[arm]["test_macro_f1"] - baseline

print(json.dumps({
    "scope": "engineering_smoke_test",
    "job_id": os.environ["SLURM_JOB_ID"],
    "source_commit": sys.argv[2],
    "dataset_config_sha256": config_sha,
    "archive_sha256": digests,
    "evaluator_seed": 42,
    "tensorflow": tf.__version__,
    "validation_count": len(x_val),
    "test_count": len(x_test),
    "arms": results,
}, sort_keys=True, allow_nan=False))
PY

[[ "$(git rev-parse HEAD)" == "$expected" ]]
git diff --quiet
git diff --cached --quiet
echo 'CIC_PAIRED_EVALUATOR_SMOKE=PASS'
