#!/usr/bin/env python3
"""Read-only governed MaleX GMM initialization scale diagnosis."""

import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    initialization_logical_positions,
    _production_initial_scales,
    _positive_float64_ulp_distance,
    MAX_SCALE_ROUNDTRIP_ULP,
)

from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    MIN_SCALE,
    transformed_scale,
)

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    canonical_images_to_vectors,
)

repo = Path.home() / "TrustForge"
sys.path.insert(0, str(repo / "scripts"))

from train_paper01_rbm_revision_checkpoint import (
    _build_adapter,
    _class_count,
)

from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
    _fetch_class_batch,
)

DATASET_ID = "malex"
CLASS_ID = 0
EXPECTED_TRAIN_CLASS_COUNT = 125794
INITIALIZATION_COUNT = 4096

print("===== REAL-DATA GMM SCALE DIAGNOSIS =====", flush=True)

adapter = _build_adapter(
    dataset_id=DATASET_ID,
    repo_root=repo,
)

count = _class_count(
    dataset_id=DATASET_ID,
    adapter=adapter,
)

if count != 2:
    raise RuntimeError("unexpected MaleX class count")

split_size = int(adapter.split_size("train"))

matched = []

for start in range(0, split_size, 8192):
    stop = min(start + 8192, split_size)
    positions = np.arange(start, stop, dtype=np.int64)

    labels = np.asarray(
        adapter.labels("train", positions)
    )

    if (
        labels.shape != positions.shape
        or labels.dtype.kind not in ("i", "u")
        or np.any(labels < 0)
        or np.any(labels >= count)
    ):
        raise RuntimeError("invalid governed MaleX labels")

    selected = positions[labels == CLASS_ID]

    if selected.size:
        matched.append(selected)

train_positions = np.concatenate(matched)

if train_positions.size != EXPECTED_TRAIN_CLASS_COUNT:
    raise RuntimeError("training class count differs from job 310386")

relative = initialization_logical_positions(
    int(train_positions.size)
)

selected_positions = train_positions[relative]

images = _fetch_class_batch(
    adapter=adapter,
    split="train",
    positions=selected_positions,
    class_id=CLASS_ID,
    context="GMM initialization scale diagnosis",
)

vectors = canonical_images_to_vectors(
    tf.convert_to_tensor(images)
).numpy()

if vectors.shape != (INITIALIZATION_COUNT, 4800):
    raise RuntimeError("unexpected canonical initialization matrix")

variance = np.var(
    vectors,
    axis=0,
    ddof=0,
    dtype=np.float64,
)

floored, raw, promoted = _production_initial_scales(
    variance
)

actual = transformed_scale(
    tf.constant(raw, dtype=tf.float64)
).numpy()

expected = np.sqrt(
    floored,
    dtype=np.float64,
)

floor = np.float64(MIN_SCALE)

expected[expected == floor] = np.nextafter(
    floor,
    np.float64(np.inf),
)

distance = _positive_float64_ulp_distance(
    actual,
    expected,
)

violations = np.flatnonzero(
    distance > MAX_SCALE_ROUNDTRIP_ULP
)

worst = np.argsort(distance)[-10:][::-1]

print("GMM_SCALE_DIAGNOSTIC "
      + json.dumps({
          "dataset_id": DATASET_ID,
          "class_id": CLASS_ID,
          "train_class_count": int(train_positions.size),
          "initialization_count": int(vectors.shape[0]),
          "canonical_dimension": int(vectors.shape[1]),
          "floor_promotions": int(promoted),
          "max_ulp_distance": int(distance.max()),
          "violating_dimensions": int(violations.size),
          "worst_dimensions": [
              {
                  "dimension": int(i),
                  "variance": float(variance[i]),
                  "expected_scale": float(expected[i]),
                  "actual_scale": float(actual[i]),
                  "ulp_distance": int(distance[i]),
              }
              for i in worst
          ],
      }, sort_keys=True, allow_nan=False),
      flush=True)

print("===== REAL-DATA SCALE DIAGNOSIS COMPLETE =====", flush=True)
