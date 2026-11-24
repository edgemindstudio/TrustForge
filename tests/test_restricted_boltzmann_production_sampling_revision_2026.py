"""Focused tests for the approved T=10 RBM production sampler.

These tests use synthetic fixtures only.  They demonstrate the bounded Stage 3
sampling contract and do not execute training, production synthesis, real-data
sampling, or pool publication.
"""

from __future__ import annotations

from dataclasses import replace
import inspect
from typing import Any

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_production_sampling as production,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    RBMPersistedCheckpoint,
    persist_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    SharedConditionalRBM,
    canonical_images_to_vectors,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_sampling import (
    RBMProductionSamplingPolicy,
    sample_production_images,
    visible_vectors_to_canonical_images,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    train_shared_conditional_rbm,
)


DATASET_ID = "tiny-rbm-production-sampling-test"
EXPERIMENTAL_SEED = 42
CLASS_MAPPING = (
    "class_0",
    "class_1",
)


class TinyAdapter:
    def __init__(self):
        self.train_images = np.stack(
            [
                np.full((40, 40, 3), 0.20, dtype=np.float32),
                np.full((40, 40, 3), 0.30, dtype=np.float32),
                np.full((40, 40, 3), 0.70, dtype=np.float32),
                np.full((40, 40, 3), 0.80, dtype=np.float32),
            ],
            axis=0,
        )

        self.train_labels = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.validation_images = np.stack(
            [
                np.full((40, 40, 3), 0.25, dtype=np.float32),
                np.full((40, 40, 3), 0.75, dtype=np.float32),
            ],
            axis=0,
        )

        self.validation_labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def split_size(self, split):
        if split == "train":
            return 4

        if split == "validation":
            return 2

        raise AssertionError(split)

    def canonical_batch(self, split, logical_positions):
        positions = np.asarray(logical_positions, dtype=np.int64)

        if split == "train":
            return (
                self.train_images[positions],
                self.train_labels[positions],
            )

        if split == "validation":
            return (
                self.validation_images[positions],
                self.validation_labels[positions],
            )

        raise AssertionError(split)


@pytest.fixture(scope="module")
def training_result():
    return train_shared_conditional_rbm(
        adapter=TinyAdapter(),
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        train_class_positions={
            0: np.array([0, 1], dtype=np.int64),
            1: np.array([2, 3], dtype=np.int64),
        },
    )


@pytest.fixture
def checkpoint(tmp_path, training_result):
    return persist_rbm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_mapping=CLASS_MAPPING,
        training_result=training_result,
    )


def _labels(checkpoint):
    return tuple(
        entry["class_label"]
        for entry in checkpoint.metadata["class_mapping"]
    )


def _policy(checkpoint, **overrides):
    configuration: dict[str, Any] = {
        "dataset_id": checkpoint.metadata["dataset_id"],
        "experimental_seed": checkpoint.metadata["experimental_seed"],
        "class_mapping": list(_labels(checkpoint)),
        "checkpoint_state_sha256": checkpoint.metadata["state_sha256"],
    }
    configuration.update(overrides)
    return RBMProductionSamplingPolicy.from_mapping(configuration)


def test_approved_policy_defaults_are_bound(checkpoint):
    policy = _policy(checkpoint)

    assert policy.transition_count == 10
    assert policy.burn_in == 0
    assert policy.initialization == production.INITIALIZATION
    assert policy.class_conditioning == production.CLASS_CONDITIONING
    assert policy.chain_semantics == "independent_chain_per_sample"
    assert policy.internal_dtype == "float64"
    assert policy.canonical_shape == (40, 40, 3)
    assert policy.canonical_dtype == "float32"
    assert policy.canonical_range == (0.0, 1.0)
    assert policy.primary_samples_per_class == 2000
    assert policy.thinning == "none"
    assert policy.continuation_chains is False
    assert policy.adaptive_stopping is False


@pytest.mark.parametrize("transition_count", [0, 1, 9, 11, 100])
def test_incorrect_transition_count_is_rejected(checkpoint, transition_count):
    with pytest.raises(ValueError, match="transition count"):
        _policy(checkpoint, transition_count=transition_count)


@pytest.mark.parametrize(
    "override",
    [
        {"burn_in": 1},
        {"initialization": "empirical_data"},
        {"class_conditioning": "adaptive_requested_class"},
        {"chain_semantics": "continuation_chain"},
        {"internal_dtype": "float32"},
        {"canonical_shape": (40, 40, 1)},
        {"canonical_dtype": "float64"},
        {"canonical_range": (0.0, 255.0)},
        {"primary_samples_per_class": 1000},
        {"thinning": "interval"},
        {"continuation_chains": True},
        {"adaptive_stopping": True},
        {"schema": "wrong"},
        {"paper_revision": "wrong"},
        {"model_family": "wrong"},
    ],
)
def test_conflicting_policy_settings_fail_closed(checkpoint, override):
    with pytest.raises(ValueError):
        _policy(checkpoint, **override)


def test_unknown_configuration_field_fails_closed(checkpoint):
    configuration = {
        "dataset_id": checkpoint.metadata["dataset_id"],
        "experimental_seed": checkpoint.metadata["experimental_seed"],
        "class_mapping": list(_labels(checkpoint)),
        "checkpoint_state_sha256": checkpoint.metadata["state_sha256"],
        "unexpected": True,
    }

    with pytest.raises(ValueError, match="unknown"):
        RBMProductionSamplingPolicy.from_mapping(configuration)


def test_missing_checkpoint_identity_fails_closed(checkpoint):
    configuration = {
        "dataset_id": checkpoint.metadata["dataset_id"],
        "experimental_seed": checkpoint.metadata["experimental_seed"],
        "class_mapping": list(_labels(checkpoint)),
    }

    with pytest.raises(ValueError, match="missing"):
        RBMProductionSamplingPolicy.from_mapping(configuration)


def test_direct_construction_rejects_wrong_transition_count(checkpoint):
    policy = _policy(checkpoint)

    with pytest.raises(ValueError, match="transition count"):
        replace(policy, transition_count=5)


def test_policy_digest_is_stable_and_binds_checkpoint(checkpoint):
    policy = _policy(checkpoint)
    other = _policy(checkpoint, checkpoint_state_sha256="0" * 64)

    assert policy.digest() == policy.digest()
    assert policy.digest() != other.digest()


def test_each_sample_receives_one_fresh_initialization(
    checkpoint,
    monkeypatch,
):
    policy = _policy(checkpoint)
    calls = []
    real_initialize = production.initialize_chain

    def initialize(**kwargs):
        calls.append(kwargs["sample_index"])
        return real_initialize(**kwargs)

    monkeypatch.setattr(production, "initialize_chain", initialize)

    result = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=0,
        sample_indices=(3, 5),
    )

    assert calls == [3, 5]
    assert result.provenance.sample_indices == (3, 5)
    assert result.provenance.sample_count == 2


def test_exactly_ten_transitions_per_sample(
    checkpoint,
    monkeypatch,
):
    policy = _policy(checkpoint)
    calls = []
    real_transition = production.gibbs_transition

    def transition(**kwargs):
        calls.append(
            (
                kwargs["sample_index"],
                kwargs["state"].transition_count,
            )
        )
        return real_transition(**kwargs)

    monkeypatch.setattr(production, "gibbs_transition", transition)

    result = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=1,
        sample_indices=(3, 5),
    )

    assert result.provenance.transition_count == 10
    assert len(calls) == 20

    for sample_index in (3, 5):
        observed = [
            transition_count
            for observed_index, transition_count in calls
            if observed_index == sample_index
        ]
        assert observed == list(range(10))


def test_duplicate_sample_indices_are_rejected(checkpoint):
    policy = _policy(checkpoint)

    with pytest.raises(ValueError, match="unique"):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(1, 1),
        )


@pytest.mark.parametrize("sample_index", [-1, 2000, 5000])
def test_sample_indices_outside_budget_are_rejected(
    checkpoint,
    sample_index,
):
    policy = _policy(checkpoint)

    with pytest.raises((TypeError, ValueError)):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(sample_index,),
        )


def test_empty_sample_indices_are_rejected(checkpoint):
    policy = _policy(checkpoint)

    with pytest.raises(ValueError):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(),
        )


@pytest.mark.parametrize("class_id", [-1, 2, True])
def test_invalid_class_identity_is_rejected(checkpoint, class_id):
    policy = _policy(checkpoint)

    with pytest.raises((TypeError, ValueError)):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=class_id,
            sample_indices=(0,),
        )


def test_identical_request_is_deterministic(checkpoint):
    policy = _policy(checkpoint)

    first = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=0,
        sample_indices=(0, 1),
    )
    second = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=0,
        sample_indices=(0, 1),
    )

    np.testing.assert_array_equal(first.images, second.images)
    assert first.provenance.to_mapping() == second.provenance.to_mapping()
    assert (
        first.provenance.checkpoint_state_sha256
        == checkpoint.metadata["state_sha256"]
    )


def test_different_sample_identity_changes_output(checkpoint):
    policy = _policy(checkpoint)

    first = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=0,
        sample_indices=(0,),
    )
    second = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=0,
        sample_indices=(1,),
    )

    assert not np.array_equal(first.images, second.images)


def test_output_contract_is_float32_canonical(checkpoint):
    policy = _policy(checkpoint)

    result = sample_production_images(
        checkpoint=checkpoint,
        policy=policy,
        class_id=1,
        sample_indices=(0, 1),
    )

    images = result.images

    assert images.dtype == np.float32
    assert images.shape == (2, *production.CANONICAL_SHAPE)
    assert np.all(np.isfinite(images))
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0

    provenance = result.provenance

    assert provenance.class_id == 1
    assert provenance.class_label == "class_1"
    assert provenance.transition_count == 10
    assert provenance.burn_in == 0
    assert provenance.chain_semantics == "independent_chain_per_sample"
    assert provenance.internal_dtype == "float64"
    assert provenance.canonical_dtype == "float32"
    assert provenance.checkpoint_state_sha256 == checkpoint.metadata["state_sha256"]
    assert provenance.policy_digest == policy.digest()


def test_float32_visible_input_is_rejected():
    with pytest.raises(TypeError, match="float64"):
        visible_vectors_to_canonical_images(
            np.zeros((2, 4800), dtype=np.float32)
        )


def test_canonical_conversion_matches_forward_flattening():
    rng = np.random.default_rng(7)
    images = rng.random(
        (2, 40, 40, 3),
        dtype=np.float64,
    ).astype(np.float32)

    vectors = canonical_images_to_vectors(images).numpy()
    converted = visible_vectors_to_canonical_images(vectors)

    assert vectors.dtype == np.float64
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, images)


def test_checkpoint_hash_mismatch_is_rejected(checkpoint):
    policy = _policy(
        checkpoint,
        checkpoint_state_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="state hash"):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


def test_checkpoint_tampering_is_rejected(checkpoint):
    policy = _policy(checkpoint)
    checkpoint.state_path.write_bytes(
        checkpoint.state_path.read_bytes() + b"\x00"
    )

    with pytest.raises(ValueError, match="state file"):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


def test_mutated_in_memory_model_is_rejected(checkpoint):
    policy = _policy(checkpoint)
    checkpoint.model.W.assign(
        checkpoint.model.W + 0.01
    )

    with pytest.raises(ValueError, match="in-memory model"):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


def test_substituted_model_is_rejected(checkpoint):
    policy = _policy(checkpoint)
    state = checkpoint.model.state_dict()

    substituted_model = SharedConditionalRBM(
        W=state["W"] + 0.01,
        a=state["a"],
        b=state["b"],
        U=state["U"],
    )

    substituted = RBMPersistedCheckpoint(
        directory=checkpoint.directory,
        state_path=checkpoint.state_path,
        metadata_path=checkpoint.metadata_path,
        metadata=checkpoint.metadata,
        model=substituted_model,
    )

    with pytest.raises(ValueError, match="in-memory model"):
        sample_production_images(
            checkpoint=substituted,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


def test_checkpoint_path_binding_is_enforced(checkpoint):
    policy = _policy(checkpoint)

    misplaced = RBMPersistedCheckpoint(
        directory=checkpoint.directory,
        state_path=checkpoint.directory / "other_state.npz",
        metadata_path=checkpoint.metadata_path,
        metadata=checkpoint.metadata,
        model=checkpoint.model,
    )

    with pytest.raises(ValueError, match="state path"):
        sample_production_images(
            checkpoint=misplaced,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"dataset_id": "other-dataset"},
        {"experimental_seed": 43},
        {"class_mapping": ["class_1", "class_0"]},
        {"class_mapping": ["class_0"]},
    ],
)
def test_checkpoint_dataset_seed_and_mapping_mismatch_are_rejected(
    checkpoint,
    override,
):
    policy = _policy(checkpoint, **override)

    with pytest.raises(ValueError):
        sample_production_images(
            checkpoint=checkpoint,
            policy=policy,
            class_id=0,
            sample_indices=(0,),
        )


def test_no_cd1_or_diagnostic_runner_dependency():
    source = inspect.getsource(production)

    assert "CD_K" not in source
    assert "restricted_boltzmann_cd" not in source
    assert "run_paper01_rbm_chain_audit" not in source
    assert "run_trajectory" not in source
    assert "AUDIT_DEPTHS" not in source


def test_module_constants_bind_approved_policy():
    assert production.PRODUCTION_TRANSITION_COUNT == 10
    assert production.PRODUCTION_BURN_IN == 0
    assert production.PRIMARY_SAMPLES_PER_CLASS == 2000
