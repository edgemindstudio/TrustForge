"""Fail-closed GAN checkpoint manifest regression tests."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
    validate_gan_checkpoint_manifest,
)


def valid_manifest(dataset_id="ustc_tfc2016"):
    classes = DATASET_CLASS_COUNTS[dataset_id]

    return {
        "schema_version": "2.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": dataset_id,
        "experimental_seed": 42,
        "class_mapping": [
            f"class_{index}" for index in range(classes)
        ],
        "class_count": classes,
        "generator_iteration": 570,
        "discriminator_updates": 1140,
        "planned_generator_iterations": 570,
        "checkpoint_selection": "terminal_governed_iteration",
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "c" * 64,
        "state_files": {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
    }


@pytest.mark.parametrize(
    "dataset_id",
    tuple(DATASET_CLASS_COUNTS),
)
def test_governed_datasets_accepted(dataset_id):
    metadata = valid_manifest(dataset_id)
    result = validate_gan_checkpoint_manifest(metadata)

    assert result["dataset_id"] == dataset_id
    assert result["class_count"] == DATASET_CLASS_COUNTS[dataset_id]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "1.0"),
        ("model_family", "restricted_boltzmann"),
        ("dataset_id", "unrecognized"),
        ("experimental_seed", -1),
        ("experimental_seed", 2**32),
        ("experimental_seed", True),
        ("class_count", 999),
        ("generator_iteration", 569),
        ("discriminator_updates", 1139),
        ("planned_generator_iterations", 571),
        ("checkpoint_selection", "best_fid"),
        ("image_shape", [40, 40, 1]),
        ("latent_dim", 64),
        ("learning_rate", 0.001),
        ("beta_1", 0.9),
        ("class_sampling_alpha", 1.0),
        ("batch_size", 128),
    ],
)
def test_invalid_metadata_rejected(field, value):
    metadata = valid_manifest()
    metadata[field] = value

    with pytest.raises((TypeError, ValueError)):
        validate_gan_checkpoint_manifest(metadata)


def test_missing_field_rejected():
    metadata = valid_manifest()
    del metadata["experimental_seed"]

    with pytest.raises(ValueError):
        validate_gan_checkpoint_manifest(metadata)


def test_extra_field_rejected():
    metadata = valid_manifest()
    metadata["unauthorized_field"] = True

    with pytest.raises(ValueError):
        validate_gan_checkpoint_manifest(metadata)


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        ["duplicate"] * 9,
        [""] + [f"class_{i}" for i in range(8)],
        [" class_0"] + [f"class_{i}" for i in range(1, 9)],
        list(range(9)),
    ],
)
def test_invalid_class_mapping_rejected(mapping):
    metadata = valid_manifest()
    metadata["class_mapping"] = mapping

    with pytest.raises((TypeError, ValueError)):
        validate_gan_checkpoint_manifest(metadata)


@pytest.mark.parametrize(
    "inventory",
    [
        {},
        {"model_state.ckpt.index": "a" * 64},
        {
            "model_state.ckpt.data-00000-of-00001": "a" * 64
        },
        {
            "model_state.ckpt.index": "invalid",
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        {
            "model_state.ckpt.index": "a" * 64,
            "../model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-malformed": "b" * 64,
        },
    ],
)
def test_invalid_state_inventory_rejected(inventory):
    metadata = valid_manifest()
    metadata["state_files"] = inventory

    with pytest.raises(ValueError):
        validate_gan_checkpoint_manifest(metadata)


def test_validation_does_not_modify_input():
    metadata = valid_manifest()
    before = deepcopy(metadata)

    validate_gan_checkpoint_manifest(metadata)

    assert metadata == before


def test_complete_two_shard_inventory_accepted():
    metadata = valid_manifest()
    metadata["state_files"] = {
        "model_state.ckpt.index": "a" * 64,
        "model_state.ckpt.data-00000-of-00002": "b" * 64,
        "model_state.ckpt.data-00001-of-00002": "c" * 64,
    }

    validated = validate_gan_checkpoint_manifest(metadata)
    assert validated["state_files"] == metadata["state_files"]


@pytest.mark.parametrize(
    "shards",
    [
        ["model_state.ckpt.data-00000-of-00002"],
        [
            "model_state.ckpt.data-00000-of-00002",
            "model_state.ckpt.data-00001-of-00003",
        ],
        [
            "model_state.ckpt.data-00000-of-00002",
            "model_state.ckpt.data-00002-of-00002",
        ],
        ["model_state.ckpt.data-00000-of-00000"],
    ],
)
def test_incomplete_or_inconsistent_shards_rejected(shards):
    metadata = valid_manifest()
    metadata["state_files"] = {
        "model_state.ckpt.index": "a" * 64,
        **{name: "b" * 64 for name in shards},
    }

    with pytest.raises(ValueError):
        validate_gan_checkpoint_manifest(metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exposure_evidence_file", "other.json"),
        ("exposure_evidence_file", "../gan_exposure_evidence.json"),
        ("exposure_evidence_sha256", "xyz"),
        ("exposure_evidence_sha256", "C" * 64),
        ("exposure_evidence_sha256", None),
    ],
)
def test_v2_invalid_evidence_reference_rejected(field, value):
    metadata = valid_manifest()
    metadata[field] = value
    with pytest.raises((TypeError, ValueError)):
        validate_gan_checkpoint_manifest(metadata)


@pytest.mark.parametrize(
    "field",
    ["exposure_evidence_file", "exposure_evidence_sha256"],
)
def test_v2_missing_evidence_reference_rejected(field):
    metadata = valid_manifest()
    del metadata[field]
    with pytest.raises(ValueError):
        validate_gan_checkpoint_manifest(metadata)


def test_v1_format_not_silently_accepted_as_v2():
    metadata = valid_manifest()
    metadata["schema_version"] = "1.0"
    with pytest.raises((TypeError, ValueError)):
        validate_gan_checkpoint_manifest(metadata)
