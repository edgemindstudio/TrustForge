"""Focused tests for the approved S2-02 synthesis-linkage package."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    BATCH_SCHEMA,
    DATASET_CLASS_COUNTS,
    LATENT_STREAM_CONTRACT,
    POOL_SCHEMA,
    SynthesisRequest,
    derive_latent_vector,
    derive_latent_vectors,
    governed_batch_size,
    latent_entropy,
    load_synthesis_recipe,
    rng_metadata,
    synthesize_diffusion_batch,
    synthesize_gan_batch,
    synthesize_vae_batch,
    validate_checkpoint_identity,
    validate_complete_pool_manifest,
    validate_generated_batch,
    validate_synthesis_recipe,
    validate_synthesis_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "cicmaldroid2020"
CLASS_MAPPING = tuple(str(index) for index in range(5))
CHECKPOINT_SHA = "a" * 64


@pytest.fixture(scope="module")
def recipe():
    return load_synthesis_recipe(REPO_ROOT)


def identity(family, *, role=None, with_hash=True):
    result = {
        "dataset_id": DATASET,
        "experimental_seed": 42,
        "class_mapping": list(CLASS_MAPPING),
        "checkpoint_role": role
        or ("terminal" if family == "gan" else "best"),
    }
    if with_hash:
        result["checkpoint_sha256"] = CHECKPOINT_SHA
    return result


def request(family, *, indices=(0, 1), scope="fixture", class_id=0):
    return SynthesisRequest(
        dataset_id=DATASET,
        family=family,
        experimental_seed=42,
        class_id=class_id,
        sample_indices=tuple(indices),
        class_mapping=CLASS_MAPPING,
        scope=scope,
    )


class StandInGAN:
    def __init__(self, value=0.0):
        self.value = value
        self.calls = []

    def __call__(self, inputs, training=False):
        latents, labels = inputs
        assert training is False
        assert latents.shape == (len(latents), 100)
        assert labels.shape == (len(labels), 5)
        self.calls.append(
            {
                "latents": np.asarray(latents).copy(),
                "labels": np.asarray(labels).copy(),
            }
        )
        return np.full(
            (len(latents), 40, 40, 3),
            self.value,
            dtype=np.float32,
        )


class StandInVAE:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, training=False):
        latents, labels = inputs
        assert training is False
        assert latents.shape == (len(latents), 100)
        self.calls.append(
            {
                "latents": np.asarray(latents).copy(),
                "labels": np.asarray(labels).copy(),
            }
        )
        return np.zeros(
            (len(latents), 40, 40, 3),
            dtype=np.float32,
        )


class StandInDiffusion:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, training=False):
        images, labels, times = inputs
        assert training is False
        assert images.shape[0] == 1
        self.calls.append(
            {
                "labels": np.asarray(labels).copy(),
                "timesteps": np.asarray(times).copy(),
            }
        )
        return np.zeros_like(images, dtype=np.float32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_recipe_binds_approved_s2_02_controls(recipe):
    assert recipe.sha256 == hashlib.sha256(recipe.raw).hexdigest()
    assert recipe.data["schema"].endswith("synthesis-linkage-recipe.v1")
    assert recipe.data["experimental_seeds"] == list(range(42, 49))
    assert recipe.data["primary_synthetic_samples_per_class"] == 2000
    assert governed_batch_size(recipe, "gan") == 16
    assert governed_batch_size(recipe, "vae") == 32
    assert governed_batch_size(recipe, "diffusion") == 1
    assert recipe.data["families"]["gan"]["checkpoint_role"] == "terminal"
    assert recipe.data["families"]["vae"]["checkpoint_role"] == "best"
    assert (
        recipe.data["families"]["diffusion"]["checkpoint_role"] == "best"
    )
    assert (
        recipe.data["families"]["vae"]["stream_contract"]
        == LATENT_STREAM_CONTRACT
    )
    assert recipe.data["families"]["vae"]["observation_noise"] is False
    assert (
        recipe.data["families"]["diffusion"]["model_call_batch_size"] == 1
    )


def test_recipe_sha_is_pinned(recipe):
    assert recipe.sha256 == (
        "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
    )


def test_unknown_recipe_field_rejected(recipe):
    bad = deepcopy(recipe.data)
    bad["families"]["gan"]["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_synthesis_recipe(bad)


def test_missing_recipe_field_rejected(recipe):
    bad = deepcopy(recipe.data)
    del bad["families"]["vae"]["model_call_batch_size"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_synthesis_recipe(bad)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("gan", "model_call_batch_size", 17),
        ("vae", "checkpoint_role", "last"),
        ("diffusion", "stream_contract", "wrong"),
        ("gan", "canonical_conversion", "wrong"),
    ],
)
def test_conflicting_recipe_controls_rejected(recipe, family, field, value):
    bad = deepcopy(recipe.data)
    bad["families"][family][field] = value
    with pytest.raises(ValueError):
        validate_synthesis_recipe(bad)


def test_authorization_must_remain_false(recipe):
    bad = deepcopy(recipe.data)
    bad["authorization"]["synthesis_execution_authorized"] = True
    with pytest.raises(ValueError, match="must remain false"):
        validate_synthesis_recipe(bad)


# ---------------------------------------------------------------------------
# RNG contract
# ---------------------------------------------------------------------------


def test_exact_entropy_ordering():
    assert latent_entropy(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=3,
        sample_index=17,
    ) == [0x54534631, 2, 1, 42, 3, 17, 1]


def test_golden_vector_local_numpy():
    # Generated with the approved contract under local NumPy 2.4.2.
    # The recorded baseline is NumPy 2.2.6; pinned-version replay is
    # reported separately because that environment is unavailable here.
    expected = np.array(
        [
            0.8484367132186890,
            -0.01919245906174183,
            2.2974236011505127,
            0.17771393060684204,
            0.7904958128929138,
            -0.04613681882619858,
            0.6744997501373291,
            1.99544095993042,
        ],
        dtype=np.float32,
    )
    observed = derive_latent_vector(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_index=0,
    )
    np.testing.assert_array_equal(observed[:8], expected)


def test_same_identity_is_byte_identical():
    first = derive_latent_vector(
        dataset_id=DATASET,
        family="vae",
        experimental_seed=42,
        class_id=1,
        sample_index=9,
    )
    second = derive_latent_vector(
        dataset_id=DATASET,
        family="vae",
        experimental_seed=42,
        class_id=1,
        sample_index=9,
    )
    np.testing.assert_array_equal(first, second)


def test_dataset_family_and_sample_domains_are_separate():
    base = derive_latent_vector(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_index=0,
    )
    other_dataset = derive_latent_vector(
        dataset_id="ustc_tfc2016",
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_index=0,
    )
    other_family = derive_latent_vector(
        dataset_id=DATASET,
        family="vae",
        experimental_seed=42,
        class_id=0,
        sample_index=0,
    )
    other_sample = derive_latent_vector(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_index=1,
    )
    assert not np.array_equal(base, other_dataset)
    assert not np.array_equal(base, other_family)
    assert not np.array_equal(base, other_sample)


def test_latent_derivation_does_not_mutate_global_rng():
    np.random.seed(2026)
    expected = np.random.random(4)
    np.random.seed(2026)
    derive_latent_vectors(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_indices=(0, 1, 2),
    )
    observed = np.random.random(4)
    np.testing.assert_array_equal(observed, expected)


def test_batching_and_order_do_not_change_latents():
    first = derive_latent_vectors(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_indices=(2, 0, 1),
    )
    second = derive_latent_vectors(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_indices=(0, 1, 2),
    )
    np.testing.assert_array_equal(first[1], second[0])
    np.testing.assert_array_equal(first[2], second[1])
    np.testing.assert_array_equal(first[0], second[2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experimental_seed": 41},
        {"experimental_seed": 49},
        {"class_id": 5},
        {"class_id": -1},
        {"sample_index": 2000},
        {"sample_index": -1},
        {"purpose": "training"},
    ],
)
def test_invalid_latent_identities_rejected(kwargs):
    arguments = {
        "dataset_id": DATASET,
        "family": "gan",
        "experimental_seed": 42,
        "class_id": 0,
        "sample_index": 0,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        derive_latent_vector(**arguments)


def test_invalid_dataset_or_family_rejected():
    with pytest.raises(ValueError):
        latent_entropy(
            dataset_id="unknown",
            family="gan",
            experimental_seed=42,
            class_id=0,
            sample_index=0,
        )
    with pytest.raises(ValueError):
        latent_entropy(
            dataset_id=DATASET,
            family="diffusion",
            experimental_seed=42,
            class_id=0,
            sample_index=0,
        )


def test_duplicate_sample_indices_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        derive_latent_vectors(
            dataset_id=DATASET,
            family="gan",
            experimental_seed=42,
            class_id=0,
            sample_indices=(1, 1),
        )


def test_rng_metadata_records_contract(recipe):
    metadata = rng_metadata()
    assert metadata["contract"] == LATENT_STREAM_CONTRACT
    assert metadata["bit_generator"] == "PCG64"
    assert metadata["numpy_baseline"] == "2.2.6"
    assert metadata["numpy_observed"] == np.__version__
    assert metadata["entropy_order"][0] == "domain"
    assert metadata["latent_dimension"] == 100


# ---------------------------------------------------------------------------
# Request and checkpoint identity
# ---------------------------------------------------------------------------


def test_complete_class_request_requires_exact_quota():
    complete = request(
        "gan",
        indices=tuple(range(2000)),
        scope="production_complete_class",
    )
    assert validate_synthesis_request(complete) == tuple(range(2000))
    short = request(
        "gan",
        indices=tuple(range(1999)),
        scope="production_complete_class",
    )
    with pytest.raises(ValueError, match="2,000"):
        validate_synthesis_request(short)


def test_duplicate_request_identity_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_synthesis_request(request("gan", indices=(1, 1)))


def test_wrong_class_mapping_length_rejected():
    bad = SynthesisRequest(
        dataset_id=DATASET,
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_indices=(0,),
        class_mapping=("0", "1"),
    )
    with pytest.raises(ValueError, match="mapping length"):
        validate_synthesis_request(bad)


def test_out_of_range_class_rejected():
    with pytest.raises(ValueError, match="class_id"):
        validate_synthesis_request(request("gan", class_id=5))


@pytest.mark.parametrize(
    ("family", "accepted", "rejected"),
    [
        ("gan", "terminal", "best"),
        ("vae", "best", "last"),
        ("diffusion", "best", "last"),
    ],
)
def test_checkpoint_role_contract(recipe, family, accepted, rejected):
    good = identity(family, role=accepted)
    validate_checkpoint_identity(
        good, request=request(family), recipe=recipe
    )
    bad = identity(family, role=rejected)
    with pytest.raises(ValueError, match="role"):
        validate_checkpoint_identity(
            bad, request=request(family), recipe=recipe
        )


def test_production_checkpoint_hash_required(recipe):
    no_hash = identity("gan", with_hash=False)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint_identity(
            no_hash,
            request=request("gan", scope="production_batch"),
            recipe=recipe,
        )
    validate_checkpoint_identity(
        identity("gan"),
        request=request("gan", scope="production_batch"),
        recipe=recipe,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "ustc_tfc2016"),
        ("experimental_seed", 43),
        ("class_mapping", ["0", "1", "2", "3"]),
    ],
)
def test_checkpoint_identity_mismatch_rejected(recipe, field, value):
    bad = identity("gan")
    bad[field] = value
    with pytest.raises(ValueError):
        validate_checkpoint_identity(
            bad, request=request("gan"), recipe=recipe
        )


def test_invalid_checkpoint_hash_rejected(recipe):
    bad = identity("gan")
    bad["checkpoint_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint_identity(
            bad, request=request("gan"), recipe=recipe
        )


# ---------------------------------------------------------------------------
# GAN generation
# ---------------------------------------------------------------------------


def test_gan_fixture_generation_uses_approved_controls(recipe):
    generator = StandInGAN(value=0.0)
    result = synthesize_gan_batch(
        generator=generator,
        request=request("gan", indices=(0, 1, 2)),
        recipe=recipe,
        checkpoint_identity=identity("gan"),
    )
    assert result["images"].shape == (3, 40, 40, 3)
    assert result["images"].dtype == np.float32
    np.testing.assert_array_equal(result["images"], 0.5)
    assert result["record"]["schema"] == BATCH_SCHEMA
    assert result["record"]["model_call_batch_size"] == 16
    assert result["record"]["model_call_batch_sizes"] == [3]
    assert result["record"]["authoritative_artifact"] is False
    assert result["record"]["status"] == "FIXTURE_GENERATED"
    assert len(generator.calls) == 1
    np.testing.assert_array_equal(
        generator.calls[0]["latents"],
        derive_latent_vectors(
            dataset_id=DATASET,
            family="gan",
            experimental_seed=42,
            class_id=0,
            sample_indices=(0, 1, 2),
        ),
    )
    np.testing.assert_array_equal(
        generator.calls[0]["labels"][:, 0],
        np.ones(3, dtype=np.float32),
    )


def test_gan_partial_final_batch_is_explicit(recipe):
    generator = StandInGAN(value=0.0)
    result = synthesize_gan_batch(
        generator=generator,
        request=request("gan", indices=tuple(range(20))),
        recipe=recipe,
        checkpoint_identity=identity("gan"),
    )
    assert result["record"]["model_call_batch_sizes"] == [16, 4]
    assert [len(call["latents"]) for call in generator.calls] == [16, 4]
    assert result["images"].shape == (20, 40, 40, 3)


def test_production_batch_record_is_not_authoritative(recipe):
    generator = StandInGAN(value=0.0)
    result = synthesize_gan_batch(
        generator=generator,
        request=request("gan", indices=(0, 1), scope="production_batch"),
        recipe=recipe,
        checkpoint_identity=identity("gan"),
    )
    record = result["record"]
    assert record["status"] == (
        "PRODUCTION_BATCH_GENERATED_NOT_POOL_VERIFIED"
    )
    assert record["authoritative_artifact"] is False
    assert record["production_execution_authorized"] is False


def test_gan_wrong_checkpoint_role_rejected_before_generation(recipe):
    generator = StandInGAN()
    with pytest.raises(ValueError, match="role"):
        synthesize_gan_batch(
            generator=generator,
            request=request("gan"),
            recipe=recipe,
            checkpoint_identity=identity("gan", role="best"),
        )
    assert generator.calls == []


def test_gan_nonfinite_output_rejected(recipe):
    generator = StandInGAN(value=float("nan"))
    with pytest.raises((ValueError, TypeError)):
        synthesize_gan_batch(
            generator=generator,
            request=request("gan"),
            recipe=recipe,
            checkpoint_identity=identity("gan"),
        )


# ---------------------------------------------------------------------------
# VAE generation
# ---------------------------------------------------------------------------


def test_vae_fixture_generation_uses_decoder_mean(recipe):
    pytest.importorskip("tensorflow")
    decoder = StandInVAE()
    result = synthesize_vae_batch(
        decoder=decoder,
        request=request("vae", indices=(0, 1, 2)),
        recipe=recipe,
        checkpoint_identity=identity("vae"),
    )
    assert result["images"].shape == (3, 40, 40, 3)
    np.testing.assert_array_equal(result["images"], 0.5)
    assert result["record"]["model_call_batch_size"] == 32
    assert result["record"]["model_call_batch_sizes"] == [3]
    assert result["record"]["status"] == "FIXTURE_GENERATED"
    assert len(decoder.calls) == 1
    np.testing.assert_array_equal(
        decoder.calls[0]["labels"][:, 0],
        np.ones(3, dtype=np.float32),
    )


def test_vae_last_checkpoint_rejected_for_primary_synthesis(recipe):
    with pytest.raises(ValueError, match="role"):
        validate_checkpoint_identity(
            identity("vae", role="last"),
            request=request("vae"),
            recipe=recipe,
        )


# ---------------------------------------------------------------------------
# Diffusion generation
# ---------------------------------------------------------------------------


def test_diffusion_fixture_reuses_governed_sampler(recipe):
    model = StandInDiffusion()
    result = synthesize_diffusion_batch(
        model=model,
        request=request("diffusion", indices=(0, 1)),
        recipe=recipe,
        checkpoint_identity=identity("diffusion"),
    )
    assert result["images"].shape == (2, 40, 40, 3)
    assert result["images"].dtype == np.float32
    assert np.isfinite(result["images"]).all()
    assert np.all((result["images"] >= 0.0) & (result["images"] <= 1.0))
    assert result["record"]["model_call_batch_size"] == 1
    assert result["record"]["model_call_batch_sizes"] == [1, 1]
    assert len(model.calls) == 2000
    assert all(
        int(np.argmax(call["labels"][0])) == 0
        for call in model.calls[:2]
    )


def test_diffusion_last_checkpoint_rejected(recipe):
    with pytest.raises(ValueError, match="role"):
        validate_checkpoint_identity(
            identity("diffusion", role="last"),
            request=request("diffusion"),
            recipe=recipe,
        )


# ---------------------------------------------------------------------------
# Output validation and pool completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((1, 40, 40), dtype=np.float32),
        np.zeros((1, 40, 40, 3), dtype=np.float64),
        np.full((1, 40, 40, 3), np.nan, dtype=np.float32),
        np.full((1, 40, 40, 3), -0.1, dtype=np.float32),
        np.full((1, 40, 40, 3), 1.1, dtype=np.float32),
    ],
)
def test_invalid_generated_output_rejected(images):
    with pytest.raises((TypeError, ValueError)):
        validate_generated_batch(
            images,
            class_ids=[0],
            sample_indices=(0,),
            request=request("gan", indices=(0,)),
        )


def test_class_condition_mismatch_rejected():
    images = np.zeros((1, 40, 40, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="class condition"):
        validate_generated_batch(
            images,
            class_ids=[1],
            sample_indices=(0,),
            request=request("gan", indices=(0,), class_id=0),
        )


def pool_manifest(recipe, *, count=2000, classes=None):
    class_count = DATASET_CLASS_COUNTS[DATASET]
    records = classes if classes is not None else [
        {
            "class_id": class_id,
            "sample_indices": list(range(count)),
        }
        for class_id in range(class_count)
    ]
    return {
        "schema": POOL_SCHEMA,
        "dataset_id": DATASET,
        "family": "gan",
        "experimental_seed": 42,
        "class_mapping": list(CLASS_MAPPING),
        "checkpoint_role": "terminal",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "recipe_sha256": recipe.sha256,
        "model_call_batch_size": 16,
        "quota_per_class": 2000,
        "complete_pool": True,
        "classes": records,
        "expected_total": class_count * count,
        "observed_total": class_count * count,
        "status": "COMPLETE_METADATA_VALIDATED",
        "authoritative_artifact": False,
    }


def test_complete_pool_metadata_validation(recipe):
    result = validate_complete_pool_manifest(
        pool_manifest(recipe), recipe=recipe
    )
    assert result["total"] == 10000
    assert result["class_count"] == 5
    assert result["authoritative_artifact"] is False


def test_pool_missing_class_rejected(recipe):
    manifest = pool_manifest(recipe)
    manifest["classes"] = manifest["classes"][:-1]
    with pytest.raises(ValueError):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_pool_short_class_rejected(recipe):
    with pytest.raises(ValueError):
        validate_complete_pool_manifest(
            pool_manifest(recipe, count=1999), recipe=recipe
        )


def test_pool_duplicate_class_rejected(recipe):
    manifest = pool_manifest(recipe)
    manifest["classes"][1]["class_id"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_incomplete_output_cannot_claim_complete(recipe):
    manifest = pool_manifest(recipe)
    manifest["observed_total"] = 9999
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_fixture_pool_cannot_be_authoritative(recipe):
    manifest = pool_manifest(recipe)
    manifest["authoritative_artifact"] = True
    with pytest.raises(ValueError, match="authoritative"):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_pool_boolean_index_rejected(recipe):
    manifest = pool_manifest(recipe)
    manifest["classes"][0]["sample_indices"][0] = True
    with pytest.raises(TypeError):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_pool_recipe_mismatch_rejected(recipe):
    manifest = pool_manifest(recipe)
    manifest["recipe_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="recipe identity"):
        validate_complete_pool_manifest(manifest, recipe=recipe)


def test_pool_batch_or_role_mismatch_rejected(recipe):
    manifest = pool_manifest(recipe)
    manifest["model_call_batch_size"] = 32
    with pytest.raises(ValueError, match="batch"):
        validate_complete_pool_manifest(manifest, recipe=recipe)
    manifest = pool_manifest(recipe)
    manifest["checkpoint_role"] = "best"
    with pytest.raises(ValueError, match="role"):
        validate_complete_pool_manifest(manifest, recipe=recipe)
