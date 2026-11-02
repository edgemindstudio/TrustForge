"""Bounded real-dataset integration test for revision-2026 GAN."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from trustforge.models.paper01_revision_2026.gan_data_integration import (
    GANTrainingData,
)
from trustforge.models.paper01_revision_2026.gan_training_bridge import (
    GANTrainingBridge,
)
from trustforge.models.paper01_revision_2026.gan_image_domain import (
    gan_to_canonical,
)


DATASET_IDS = (
    "ustc_tfc2016",
    "cicmaldroid2020",
    "malnet_image",
    "malex",
)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real-data integration requires Slurm")

    # Reuse the governed revision dataset configuration.
    from scripts.train_paper01_rbm_revision_checkpoint import (
        _build_adapter,
        _class_count,
    )

    root = Path(__file__).resolve().parents[1]

    for dataset_id in DATASET_IDS:
        print(f"DATASET={dataset_id} START", flush=True)

        adapter = _build_adapter(
            dataset_id=dataset_id,
            repo_root=root,
        )

        class_count = _class_count(
            dataset_id=dataset_id,
            adapter=adapter,
        )

        data = GANTrainingData(
            dataset=adapter,
            class_count=class_count,
            seed=42,
        )
        bridge = GANTrainingBridge(data, seed=42)

        batch = bridge.draw(4)

        assert batch.real_images.shape == (4, 40, 40, 3)
        assert batch.real_images.dtype == np.float32
        assert np.isfinite(batch.real_images).all()
        assert np.all(batch.real_images >= -1.0)
        assert np.all(batch.real_images <= 1.0)

        assert batch.real_labels.shape == (4, class_count)
        assert batch.fake_labels.shape == (4, class_count)
        assert batch.generator_labels.shape == (4, class_count)

        assert batch.train_positions.shape == (4,)
        assert np.all(batch.train_positions >= 0)
        assert np.all(
            batch.train_positions < adapter.split_size("train")
        )

        recovered = gan_to_canonical(batch.real_images)
        assert recovered.shape == (4, 40, 40, 3)

        images, labels = adapter.canonical_batch(
            "train",
            batch.train_positions,
        )

        np.testing.assert_allclose(
            recovered,
            images,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            labels,
            batch.real_labels,
        )

        print(
            f"DATASET={dataset_id} "
            f"CLASSES={class_count} "
            "REAL_DATA_GAN_BRIDGE=PASS",
            flush=True,
        )

    print("GAN_REAL_DATA_INTEGRATION=PASS", flush=True)


if __name__ == "__main__":
    main()
