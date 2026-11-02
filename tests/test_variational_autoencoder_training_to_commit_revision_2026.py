"""Synthetic VAE training-to-commit integration.

Uses disposable fixture provenance, not real dataset evidence.
No production training or authoritative publication.
"""

from copy import deepcopy
import os

import numpy as np
import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026.variational_autoencoder_training_runner import (
    run_synthetic_vae_training,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_preparation import (
    DERIVED_FIELDS,
    prepare_vae_checkpoint_artifacts,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_unpublished_transaction import (
    prepare_unpublished_vae_generation,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_publisher import (
    publish_candidate_vae_commit,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_committed_generation import (
    verify_committed_vae_generation,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
)

from tests.test_variational_autoencoder_checkpoint_unpublished_transaction_filesystem_revision_2026 import (
    transaction_inputs,
    open_parent,
    GENERATION_ID,
)


class SyntheticAdapter:
    def __init__(self, class_count):
        self.class_count = class_count
        rng = np.random.default_rng(42)

        self.images = {}
        self.labels = {}

        for split, count in (
            ("train", 4),
            ("validation", 2),
        ):
            self.images[split] = rng.uniform(
                0.0,
                1.0,
                size=(count, 40, 40, 3),
            ).astype(np.float32)

            self.labels[split] = np.eye(
                class_count,
                dtype=np.float32,
            )[np.arange(count) % class_count]

    def split_size(self, split):
        return len(self.images[split])

    def canonical_batch(self, split, positions):
        return (
            self.images[split][positions].copy(),
            self.labels[split][positions].copy(),
        )


def test_trained_vae_reaches_verified_synthetic_commit(
    transaction_inputs,
):
    generations, fixture_prepared = transaction_inputs
    metadata = fixture_prepared["metadata"]
    class_mapping = metadata["class_mapping"]
    class_count = len(class_mapping)
    seed = metadata["experimental_seed"]

    assert 0 <= seed <= 2**31 - 1

    # Stage 1: real optimizer updates on disposable synthetic data.
    trained = run_synthetic_vae_training(
        adapter=SyntheticAdapter(class_count),
        num_classes=class_count,
        experimental_seed=seed,
        epochs=1,
        batch_size=2,
        learning_rate=0.0001,
        development_mode="synthetic_only",
    )

    assert trained["epochs_completed"] == 1
    assert trained["completed_updates"] == 2
    assert trained["publication_status"] == "NOT_PUBLISHED"

    # Stage 2: reuse existing synthetic test provenance.
    # No dataset-origin authenticity is claimed.
    evidence = deepcopy({
        key: value
        for key, value in metadata.items()
        if key not in DERIVED_FIELDS
    })

    evidence["completed_epochs"] = trained["epochs_completed"]
    evidence["completed_updates"] = trained["completed_updates"]
    evidence["validation_criterion"]["value"] = float(
        trained["history"][-1]["validation"]["total_loss"]
    )

    prepared = prepare_vae_checkpoint_artifacts(
        encoder=trained["encoder"],
        decoder=trained["decoder"],
        manifest_evidence=evidence,
        dataset_provenance=fixture_prepared["dataset_provenance"],
    )

    assert prepared["metadata"]["completed_updates"] == 2

    # Stage 3: prepare and publish a disposable generation.
    commits = generations.parent / "commits"
    commits.mkdir(mode=0o700)
    commits.chmod(0o700)

    generations_fd = open_parent(generations)
    commits_fd = open_parent(commits)
    generation_fd = None

    try:
        unpublished = prepare_unpublished_vae_generation(
            generations_fd,
            GENERATION_ID,
            prepared=prepared,
            requested_dataset_id=metadata["dataset_id"],
            requested_experimental_seed=seed,
            requested_class_mapping=class_mapping,
        )

        generation_fd = unpublished["generation_fd"]

        assert unpublished["publication_status"] == "UNPUBLISHED"

        published = publish_candidate_vae_commit(
            generations_fd,
            commits_fd,
            generation_fd,
            GENERATION_ID,
            prepared=prepared,
            requested_dataset_id=metadata["dataset_id"],
            requested_experimental_seed=seed,
            requested_class_mapping=class_mapping,
        )

        assert published["outcome"] == "committed_synced"
        assert published["crash_durability_verified"] is False

        # Stage 4: independently verify the published checkpoint.
        verified = verify_committed_vae_generation(
            generations.parent,
            generation_id=GENERATION_ID,
            requested_dataset_id=metadata["dataset_id"],
            requested_experimental_seed=seed,
            requested_class_mapping=class_mapping,
        )

        assert verified is not None

        marker = commits / published["commit_filename"]
        assert marker.is_file()

        assert (
            unpublished["verified"]["metadata"]["completed_updates"]
            == 2
        )

        os.fstat(generation_fd)
        os.fstat(generations_fd)
        os.fstat(commits_fd)

    finally:
        if generation_fd is not None:
            os.close(generation_fd)
        os.close(commits_fd)
        os.close(generations_fd)
