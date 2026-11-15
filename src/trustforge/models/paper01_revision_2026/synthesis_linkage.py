"""Stage 2 GAN/VAE/Diffusion synthesis linkage.

This module binds the approved S2-02 synthesis controls to executable
consumers without performing production execution.  It reuses the existing
family checkpoint loaders, image-domain converters and Diffusion sampler.

No training, checkpoint publication, synthetic-pool publication or runtime
activation occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .ar_maf_execution_inputs import DATASET_CLASSES
from .gan_image_domain import gan_to_canonical


SCHEMA = "trustforge.paper01.synthesis-linkage-recipe.v1"
POOL_SCHEMA = "trustforge.paper01.synthesis-pool-manifest.v1"
BATCH_SCHEMA = "trustforge.paper01.synthesis-batch-record.v1"
LATENT_STREAM_CONTRACT = "trustforge.paper01.synthesis-latent-stream.v1"
DIFFUSION_STREAM_CONTRACT = (
    "trustforge.paper01.diffusion-governed-sampling.v1"
)
RECIPE_RELATIVE_PATH = (
    "studies/paper01_benchmark/revision_2026/"
    "synthesis_linkage_recipe.json"
)

DATASET_CLASS_COUNTS = dict(DATASET_CLASSES)

DOMAIN = 0x54534631
DATASET_CODES = {
    "ustc_tfc2016": 1,
    "cicmaldroid2020": 2,
    "malnet_image": 3,
    "malex": 4,
}
FAMILY_CODES = {"gan": 1, "vae": 2}
PURPOSE_CODES = {"production_latent": 1}
EXPERIMENTAL_SEEDS = tuple(range(42, 49))
PRIMARY_QUOTA = 2000
LATENT_DIMENSION = 100
CANONICAL_SHAPE = (40, 40, 3)
CANONICAL_OUTPUT = {
    "shape": [40, 40, 3],
    "layout": "NHWC",
    "dtype": "float32",
    "range": [0.0, 1.0],
}
FAMILIES = ("gan", "vae", "diffusion")
CHECKPOINT_ROLES = {
    "gan": "terminal",
    "vae": "best",
    "diffusion": "best",
}
MODEL_CALL_BATCH_SIZES = {
    "gan": 16,
    "vae": 32,
    "diffusion": 1,
}
GENERATION_RULES = {
    "gan": "conditional_dcgan_terminal",
    "vae": "conditional_vae_decoder_mean",
    "diffusion": "conditional_ddpm_reverse",
}
CANONICAL_CONVERSIONS = {
    "gan": "gan_to_canonical",
    "vae": "internal_to_canonical",
    "diffusion": "sampler_canonical_output",
}
STREAM_CONTRACTS = {
    "gan": LATENT_STREAM_CONTRACT,
    "vae": LATENT_STREAM_CONTRACT,
    "diffusion": DIFFUSION_STREAM_CONTRACT,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

RECIPE_SHA256 = (
    "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
)

_TOP_LEVEL_FIELDS = frozenset({
    "schema",
    "status",
    "decision_date",
    "datasets",
    "experimental_seeds",
    "primary_synthetic_samples_per_class",
    "canonical_output",
    "families",
    "authorization",
})
_FAMILY_COMMON_FIELDS = frozenset({
    "datasets",
    "checkpoint_role",
    "generation",
    "stream_contract",
    "stream_contract_version",
    "rng_baseline",
    "bit_generator",
    "model_call_batch_size",
    "quota_per_class",
    "class_conditioning",
    "canonical_conversion",
    "production_execution_authorized",
})
_FAMILY_EXTRA_FIELDS = {
    "gan": frozenset({
        "latent_dimension",
        "latent_distribution",
    }),
    "vae": frozenset({
        "latent_dimension",
        "prior",
        "observation_noise",
    }),
    "diffusion": frozenset({
        "timesteps",
        "beta_start",
        "beta_end",
        "noise_identity",
        "reverse_variance",
    }),
}
_AUTHORIZATION_FIELDS = frozenset({
    "full_benchmark_go",
    "synthesis_execution_authorized",
    "production_artifact_publication_authorized",
    "slurm_submission_authorized",
})
_POOL_FIELDS = frozenset({
    "schema",
    "dataset_id",
    "family",
    "experimental_seed",
    "class_mapping",
    "checkpoint_role",
    "checkpoint_sha256",
    "recipe_sha256",
    "model_call_batch_size",
    "quota_per_class",
    "complete_pool",
    "classes",
    "expected_total",
    "observed_total",
    "status",
    "authoritative_artifact",
})


@dataclass(frozen=True)
class SynthesisRecipe:
    """Validated recipe bytes and their canonical file identity."""

    data: dict
    raw: bytes
    sha256: str


@dataclass(frozen=True)
class SynthesisRequest:
    """One bounded per-class synthesis request."""

    dataset_id: str
    family: str
    experimental_seed: int
    class_id: int
    sample_indices: tuple[int, ...]
    class_mapping: tuple[str | int, ...]
    scope: str = "fixture"


def _mapping(value, *, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_int(value, *, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _string(value, *, name):
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _bool(value, *, name):
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value


def _number(value, *, name):
    if isinstance(value, (bool, np.bool_)) or type(value) not in (
        int,
        float,
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_keys(value, expected, *, name):
    value = _mapping(value, name=name)
    if set(value) != set(expected):
        raise ValueError(f"{name} fields mismatch")
    return value


def _dataset_list(value, *, name):
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    result = list(value)
    if result != list(DATASET_CODES):
        raise ValueError(f"{name} must list the four governed datasets")
    return result


def _seed_list(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError("experimental_seeds must be a sequence")
    result = [
        _exact_int(item, name="experimental_seed")
        for item in value
    ]
    if tuple(result) != EXPERIMENTAL_SEEDS:
        raise ValueError("experimental_seeds must be 42-48")
    return result


def _class_mapping(value):
    if not isinstance(value, (list, tuple)):
        raise TypeError("class_mapping must be a sequence")
    mapping = tuple(value)
    if not mapping:
        raise ValueError("class_mapping must be nonempty")
    for item in mapping:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise TypeError("class_mapping entries must be str or int")
        if isinstance(item, str) and not item:
            raise ValueError("class_mapping strings must be nonempty")
    if len(set(mapping)) != len(mapping):
        raise ValueError("class_mapping must be unique")
    return mapping


def validate_synthesis_recipe(recipe):
    """Validate the approved recipe exactly; unknown fields fail closed."""
    recipe = _exact_keys(
        recipe, _TOP_LEVEL_FIELDS, name="synthesis recipe"
    )
    if recipe["schema"] != SCHEMA:
        raise ValueError("unsupported synthesis recipe schema")
    if recipe["status"] != "APPROVED_STAGE2_LINKAGE":
        raise ValueError("synthesis recipe is not approved")
    _string(recipe["decision_date"], name="decision_date")
    _dataset_list(recipe["datasets"], name="datasets")
    _seed_list(recipe["experimental_seeds"])
    if _exact_int(
        recipe["primary_synthetic_samples_per_class"],
        name="primary_synthetic_samples_per_class",
    ) != PRIMARY_QUOTA:
        raise ValueError("primary synthesis quota must be 2000")

    canonical = _exact_keys(
        recipe["canonical_output"],
        frozenset(CANONICAL_OUTPUT),
        name="canonical_output",
    )
    if canonical != CANONICAL_OUTPUT:
        raise ValueError("canonical output contract mismatch")

    families = _exact_keys(
        recipe["families"], frozenset(FAMILIES), name="families"
    )
    for family in FAMILIES:
        entry = _exact_keys(
            families[family],
            _FAMILY_COMMON_FIELDS | _FAMILY_EXTRA_FIELDS[family],
            name=f"{family} recipe",
        )
        _dataset_list(entry["datasets"], name=f"{family}.datasets")
        if entry["checkpoint_role"] != CHECKPOINT_ROLES[family]:
            raise ValueError(f"{family} checkpoint role mismatch")
        if entry["generation"] != GENERATION_RULES[family]:
            raise ValueError(f"{family} generation rule mismatch")
        if entry["stream_contract"] != STREAM_CONTRACTS[family]:
            raise ValueError(f"{family} stream contract mismatch")
        if _exact_int(
            entry["stream_contract_version"],
            name=f"{family}.stream_contract_version",
        ) != 1:
            raise ValueError(f"{family} stream contract version mismatch")
        if entry["rng_baseline"] != "numpy==2.2.6":
            raise ValueError(f"{family} RNG baseline mismatch")
        if _exact_int(
            entry["model_call_batch_size"],
            name=f"{family}.model_call_batch_size",
            minimum=1,
        ) != MODEL_CALL_BATCH_SIZES[family]:
            raise ValueError(f"{family} synthesis batch size mismatch")
        if _exact_int(
            entry["quota_per_class"],
            name=f"{family}.quota_per_class",
        ) != PRIMARY_QUOTA:
            raise ValueError(f"{family} quota mismatch")
        if entry["class_conditioning"] != "one_hot_requested_class":
            raise ValueError(f"{family} class conditioning mismatch")
        if entry["canonical_conversion"] != CANONICAL_CONVERSIONS[family]:
            raise ValueError(f"{family} canonical conversion mismatch")
        if entry["production_execution_authorized"] is not False:
            raise ValueError(
                f"{family} production execution must remain unauthorized"
            )

        if family == "gan":
            if entry["bit_generator"] != "PCG64":
                raise ValueError("GAN bit generator must be PCG64")
            if _exact_int(
                entry["latent_dimension"],
                name="gan.latent_dimension",
            ) != LATENT_DIMENSION:
                raise ValueError("GAN latent dimension mismatch")
            if entry["latent_distribution"] != "standard_normal":
                raise ValueError("GAN latent distribution mismatch")
        elif family == "vae":
            if entry["bit_generator"] != "PCG64":
                raise ValueError("VAE bit generator must be PCG64")
            if _exact_int(
                entry["latent_dimension"],
                name="vae.latent_dimension",
            ) != LATENT_DIMENSION:
                raise ValueError("VAE latent dimension mismatch")
            if entry["prior"] != "standard_normal":
                raise ValueError("VAE prior mismatch")
            if entry["observation_noise"] is not False:
                raise ValueError("VAE observation noise must be disabled")
        else:
            if entry["bit_generator"] != "numpy_default_rng":
                raise ValueError(
                    "Diffusion bit generator must preserve default_rng"
                )
            if _exact_int(
                entry["timesteps"], name="diffusion.timesteps"
            ) != 1000:
                raise ValueError("Diffusion timestep count mismatch")
            if _number(
                entry["beta_start"], name="diffusion.beta_start"
            ) != 0.0001:
                raise ValueError("Diffusion beta_start mismatch")
            if _number(
                entry["beta_end"], name="diffusion.beta_end"
            ) != 0.02:
                raise ValueError("Diffusion beta_end mismatch")
            if list(entry["noise_identity"]) != [
                "experimental_seed",
                "class_id",
                "sample_index",
                "step",
                "stream",
            ]:
                raise ValueError("Diffusion noise identity mismatch")
            if entry["reverse_variance"] != "ddpm_posterior_variance":
                raise ValueError("Diffusion reverse variance mismatch")

    authorization = _exact_keys(
        recipe["authorization"],
        _AUTHORIZATION_FIELDS,
        name="authorization",
    )
    for field in _AUTHORIZATION_FIELDS:
        if _bool(authorization[field], name=field) is not False:
            raise ValueError(f"{field} must remain false")
    return None


def load_synthesis_recipe(repository_root):
    """Load, validate and SHA-pin the approved recipe."""
    root = Path(repository_root).expanduser().resolve()
    path = root / RECIPE_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("synthesis recipe is unavailable")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid synthesis recipe JSON") from exc
    validate_synthesis_recipe(data)
    digest = hashlib.sha256(raw).hexdigest()
    if RECIPE_SHA256 == "0" * 64:
        raise RuntimeError("synthesis recipe SHA has not been pinned")
    if digest != RECIPE_SHA256:
        raise ValueError("synthesis recipe identity mismatch")
    return SynthesisRecipe(data=data, raw=raw, sha256=digest)


def _family_entry(recipe, family):
    if family not in FAMILIES:
        raise ValueError("unsupported synthesis family")
    if not isinstance(recipe, SynthesisRecipe):
        raise TypeError("recipe must be a SynthesisRecipe")
    if hashlib.sha256(recipe.raw).hexdigest() != recipe.sha256:
        raise ValueError("recipe bytes do not match recipe identity")
    validate_synthesis_recipe(recipe.data)
    return recipe.data["families"][family]


def governed_batch_size(recipe, family):
    return int(_family_entry(recipe, family)["model_call_batch_size"])


def latent_entropy(
    *,
    dataset_id,
    family,
    experimental_seed,
    class_id,
    sample_index,
    purpose="production_latent",
):
    """Return the exact approved entropy list for one latent identity."""
    if dataset_id not in DATASET_CODES:
        raise ValueError("unsupported governed dataset")
    if family not in FAMILY_CODES:
        raise ValueError("latent streams support only GAN and VAE")
    if purpose not in PURPOSE_CODES:
        raise ValueError("unsupported latent stream purpose")
    seed = _exact_int(
        experimental_seed, name="experimental_seed"
    )
    if seed not in EXPERIMENTAL_SEEDS:
        raise ValueError("experimental seed must be 42-48")
    cls = _exact_int(class_id, name="class_id")
    if not 0 <= cls < DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError("class_id outside authoritative dataset mapping")
    index = _exact_int(sample_index, name="sample_index")
    if not 0 <= index < PRIMARY_QUOTA:
        raise ValueError("sample_index must be 0-1999")
    return [
        DOMAIN,
        DATASET_CODES[dataset_id],
        FAMILY_CODES[family],
        seed,
        cls,
        index,
        PURPOSE_CODES[purpose],
    ]


def derive_latent_vector(
    *,
    dataset_id,
    family,
    experimental_seed,
    class_id,
    sample_index,
    purpose="production_latent",
):
    """Derive one independently seeded float32 latent vector."""
    entropy = latent_entropy(
        dataset_id=dataset_id,
        family=family,
        experimental_seed=experimental_seed,
        class_id=class_id,
        sample_index=sample_index,
        purpose=purpose,
    )
    seed_sequence = np.random.SeedSequence(entropy)
    bit_generator = np.random.PCG64(seed_sequence)
    rng = np.random.Generator(bit_generator)
    latent = rng.standard_normal(LATENT_DIMENSION).astype(np.float32)
    if latent.shape != (LATENT_DIMENSION,):
        raise RuntimeError("unexpected latent dimension")
    if not np.isfinite(latent).all():
        raise RuntimeError("nonfinite latent vector")
    return latent


def derive_latent_vectors(
    *,
    dataset_id,
    family,
    experimental_seed,
    class_id,
    sample_indices,
):
    """Derive all latents before any model-call batching."""
    indices = tuple(sample_indices)
    if not indices:
        raise ValueError("sample_indices must be nonempty")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate sample identity")
    latents = [
        derive_latent_vector(
            dataset_id=dataset_id,
            family=family,
            experimental_seed=experimental_seed,
            class_id=class_id,
            sample_index=index,
        )
        for index in indices
    ]
    result = np.stack(latents, axis=0)
    if result.dtype != np.float32:
        raise RuntimeError("latent batch must be float32")
    return result


def rng_metadata():
    """Return the exact RNG implementation metadata for records."""
    return {
        "contract": LATENT_STREAM_CONTRACT,
        "domain": DOMAIN,
        "domain_hex": "0x54534631",
        "entropy_order": [
            "domain",
            "dataset_code",
            "family_code",
            "experimental_seed",
            "class_id",
            "sample_index",
            "purpose_code",
        ],
        "dataset_codes": dict(DATASET_CODES),
        "family_codes": dict(FAMILY_CODES),
        "purpose_codes": dict(PURPOSE_CODES),
        "bit_generator": "PCG64",
        "generator": "numpy.random.Generator",
        "numpy_baseline": "2.2.6",
        "numpy_observed": np.__version__,
        "latent_dimension": LATENT_DIMENSION,
        "latent_distribution": "standard_normal",
    }


def validate_synthesis_request(request):
    """Validate a bounded per-class request without generating anything."""
    if not isinstance(request, SynthesisRequest):
        raise TypeError("request must be a SynthesisRequest")
    if request.dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")
    if request.family not in FAMILIES:
        raise ValueError("unsupported generator family")
    seed = _exact_int(
        request.experimental_seed, name="experimental_seed"
    )
    if seed not in EXPERIMENTAL_SEEDS:
        raise ValueError("experimental seed must be 42-48")
    mapping = _class_mapping(request.class_mapping)
    if len(mapping) != DATASET_CLASS_COUNTS[request.dataset_id]:
        raise ValueError("class mapping length mismatch")
    cls = _exact_int(request.class_id, name="class_id")
    if not 0 <= cls < len(mapping):
        raise ValueError("class_id outside authoritative class mapping")
    if request.scope not in {
        "fixture",
        "production_batch",
        "production_complete_class",
    }:
        raise ValueError("unsupported request scope")
    if not isinstance(request.sample_indices, (tuple, list)):
        raise TypeError("sample_indices must be a sequence")
    indices = tuple(
        _exact_int(item, name="sample_index")
        for item in request.sample_indices
    )
    if not indices:
        raise ValueError("sample_indices must be nonempty")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate sample identity")
    if any(not 0 <= index < PRIMARY_QUOTA for index in indices):
        raise ValueError("sample_index must be 0-1999")
    if request.scope == "production_complete_class":
        if indices != tuple(range(PRIMARY_QUOTA)):
            raise ValueError(
                "complete production class requires all 2,000 sample identities"
            )
    return indices


def validate_checkpoint_identity(
    identity,
    *,
    request,
    recipe,
    require_hash=None,
):
    """Validate checkpoint role and identity before model consumption."""
    validate_synthesis_request(request)
    entry = _family_entry(recipe, request.family)
    identity = _mapping(identity, name="checkpoint identity")
    allowed = {
        "dataset_id",
        "experimental_seed",
        "class_mapping",
        "checkpoint_role",
        "checkpoint_sha256",
        "source_commit",
    }
    required = {
        "dataset_id",
        "experimental_seed",
        "class_mapping",
        "checkpoint_role",
    }
    if not required <= set(identity) or set(identity) - allowed:
        raise ValueError("checkpoint identity fields mismatch")
    if identity["dataset_id"] != request.dataset_id:
        raise ValueError("checkpoint dataset identity mismatch")
    if _exact_int(
        identity["experimental_seed"],
        name="checkpoint experimental_seed",
    ) != request.experimental_seed:
        raise ValueError("checkpoint experimental seed mismatch")
    if _class_mapping(identity["class_mapping"]) != tuple(
        request.class_mapping
    ):
        raise ValueError("checkpoint class mapping mismatch")
    if identity["checkpoint_role"] != entry["checkpoint_role"]:
        raise ValueError("checkpoint role mismatch")
    if require_hash is None:
        require_hash = request.scope != "fixture"
    if require_hash and "checkpoint_sha256" not in identity:
        raise ValueError("checkpoint SHA-256 required for production request")
    if "checkpoint_sha256" in identity:
        digest = identity["checkpoint_sha256"]
        if type(digest) is not str or HEX64.fullmatch(digest) is None:
            raise ValueError("invalid checkpoint SHA-256")
    if "source_commit" in identity:
        commit = identity["source_commit"]
        if type(commit) is not str or HEX40.fullmatch(commit) is None:
            raise ValueError("invalid source commit")
    return entry


def validate_generated_batch(
    images,
    *,
    class_ids,
    sample_indices,
    request,
):
    """Validate one completed per-class output batch."""
    indices = validate_synthesis_request(request)
    if not isinstance(images, np.ndarray):
        raise TypeError("generated images must be numpy.ndarray")
    if images.shape != (len(indices), *CANONICAL_SHAPE):
        raise ValueError("generated image shape mismatch")
    if images.dtype != np.float32:
        raise TypeError("generated images must be float32")
    if not np.isfinite(images).all():
        raise ValueError("generated images contain nonfinite values")
    if np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("generated images escaped canonical range")
    if len(class_ids) != len(indices):
        raise ValueError("generated class count mismatch")
    if len(tuple(sample_indices)) != len(indices):
        raise ValueError("generated sample identity count mismatch")
    if tuple(sample_indices) != indices:
        raise ValueError("generated sample identity order mismatch")
    for class_id in class_ids:
        if _exact_int(class_id, name="class_id") != request.class_id:
            raise ValueError("generated class condition mismatch")
    return {
        "sample_count": len(indices),
        "class_id": request.class_id,
        "sample_indices": list(indices),
    }


def _one_hot(class_id, class_count, count):
    labels = np.zeros((count, class_count), dtype=np.float32)
    labels[:, class_id] = 1.0
    return labels


def _to_numpy(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    return array


def _batch_slices(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield tuple(indices[start:start + batch_size])


def _batch_record(
    *,
    request,
    recipe,
    checkpoint_identity,
    observed_count,
    batch_sizes,
):
    entry = _family_entry(recipe, request.family)
    status = (
        "FIXTURE_GENERATED"
        if request.scope == "fixture"
        else "PRODUCTION_BATCH_GENERATED_NOT_POOL_VERIFIED"
    )
    return {
        "schema": BATCH_SCHEMA,
        "scope": request.scope,
        "dataset_id": request.dataset_id,
        "family": request.family,
        "experimental_seed": request.experimental_seed,
        "class_id": request.class_id,
        "class_mapping": list(request.class_mapping),
        "sample_indices": list(request.sample_indices),
        "expected_sample_count": len(request.sample_indices),
        "observed_sample_count": int(observed_count),
        "checkpoint_role": entry["checkpoint_role"],
        "checkpoint_sha256": checkpoint_identity.get(
            "checkpoint_sha256"
        ),
        "recipe_sha256": recipe.sha256,
        "stream_contract": entry["stream_contract"],
        "model_call_batch_size": entry["model_call_batch_size"],
        "model_call_batch_sizes": list(batch_sizes),
        "quota_per_class": entry["quota_per_class"],
        "canonical_output": dict(CANONICAL_OUTPUT),
        "rng": rng_metadata(),
        "status": status,
        "authoritative_artifact": False,
        "production_execution_authorized": False,
    }


def synthesize_gan_batch(
    *,
    generator,
    request,
    recipe,
    checkpoint_identity,
):
    """Generate one bounded GAN batch through the accepted terminal model."""
    indices = validate_synthesis_request(request)
    entry = validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "gan":
        raise ValueError("GAN adapter requires the GAN family")
    if not callable(generator):
        raise TypeError("generator must be callable")

    latents = derive_latent_vectors(
        dataset_id=request.dataset_id,
        family="gan",
        experimental_seed=request.experimental_seed,
        class_id=request.class_id,
        sample_indices=indices,
    )
    batch_size = int(entry["model_call_batch_size"])
    outputs = []
    batch_sizes = []
    for batch_indices in _batch_slices(indices, batch_size):
        start = indices.index(batch_indices[0])
        stop = start + len(batch_indices)
        z = latents[start:stop]
        labels = _one_hot(
            request.class_id, len(request.class_mapping), len(batch_indices)
        )
        generated = generator([z, labels], training=False)
        generated = gan_to_canonical(_to_numpy(generated))
        validate_generated_batch(
            generated,
            class_ids=[request.class_id] * len(batch_indices),
            sample_indices=batch_indices,
            request=SynthesisRequest(
                dataset_id=request.dataset_id,
                family=request.family,
                experimental_seed=request.experimental_seed,
                class_id=request.class_id,
                sample_indices=batch_indices,
                class_mapping=request.class_mapping,
                scope=(
                    "fixture"
                    if request.scope == "fixture"
                    else "production_batch"
                ),
            ),
        )
        outputs.append(generated)
        batch_sizes.append(len(batch_indices))

    images = np.concatenate(outputs, axis=0)
    validate_generated_batch(
        images,
        class_ids=[request.class_id] * len(indices),
        sample_indices=indices,
        request=request,
    )
    return {
        "images": images,
        "record": _batch_record(
            request=request,
            recipe=recipe,
            checkpoint_identity=checkpoint_identity,
            observed_count=len(images),
            batch_sizes=batch_sizes,
        ),
    }


def synthesize_vae_batch(
    *,
    decoder,
    request,
    recipe,
    checkpoint_identity,
):
    """Generate one bounded VAE decoder-mean batch."""
    import tensorflow as tf

    from .variational_autoencoder_observation import (
        internal_to_canonical,
    )

    indices = validate_synthesis_request(request)
    entry = validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "vae":
        raise ValueError("VAE adapter requires the VAE family")
    if not callable(decoder):
        raise TypeError("decoder must be callable")

    latents = derive_latent_vectors(
        dataset_id=request.dataset_id,
        family="vae",
        experimental_seed=request.experimental_seed,
        class_id=request.class_id,
        sample_indices=indices,
    )
    batch_size = int(entry["model_call_batch_size"])
    outputs = []
    batch_sizes = []
    for batch_indices in _batch_slices(indices, batch_size):
        start = indices.index(batch_indices[0])
        stop = start + len(batch_indices)
        z = latents[start:stop]
        labels = _one_hot(
            request.class_id, len(request.class_mapping), len(batch_indices)
        )
        internal = decoder([z, labels], training=False)
        internal = tf.convert_to_tensor(_to_numpy(internal))
        generated = _to_numpy(internal_to_canonical(internal))
        validate_generated_batch(
            generated,
            class_ids=[request.class_id] * len(batch_indices),
            sample_indices=batch_indices,
            request=SynthesisRequest(
                dataset_id=request.dataset_id,
                family=request.family,
                experimental_seed=request.experimental_seed,
                class_id=request.class_id,
                sample_indices=batch_indices,
                class_mapping=request.class_mapping,
                scope=(
                    "fixture"
                    if request.scope == "fixture"
                    else "production_batch"
                ),
            ),
        )
        outputs.append(generated)
        batch_sizes.append(len(batch_indices))

    images = np.concatenate(outputs, axis=0)
    validate_generated_batch(
        images,
        class_ids=[request.class_id] * len(indices),
        sample_indices=indices,
        request=request,
    )
    return {
        "images": images,
        "record": _batch_record(
            request=request,
            recipe=recipe,
            checkpoint_identity=checkpoint_identity,
            observed_count=len(images),
            batch_sizes=batch_sizes,
        ),
    }


def synthesize_diffusion_batch(
    *,
    model,
    request,
    recipe,
    checkpoint_identity,
):
    """Generate one bounded Diffusion batch with model-call batch size one."""
    from .diffusion_ddpm_math import linear_ddpm_schedule
    from .diffusion_governed_sampling import sample_governed_diffusion

    indices = validate_synthesis_request(request)
    entry = validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "diffusion":
        raise ValueError("Diffusion adapter requires the Diffusion family")
    if not callable(model):
        raise TypeError("model must be callable")
    if int(entry["model_call_batch_size"]) != 1:
        raise ValueError("Diffusion model-call batch size must be 1")

    schedule = linear_ddpm_schedule(
        timesteps=entry["timesteps"],
        beta_start=entry["beta_start"],
        beta_end=entry["beta_end"],
    )
    images = []
    batch_sizes = []
    for index in indices:
        generated = sample_governed_diffusion(
            model=model,
            class_ids=[request.class_id],
            sample_indices=[index],
            num_classes=len(request.class_mapping),
            image_shape=CANONICAL_SHAPE,
            experimental_seed=request.experimental_seed,
            schedule=schedule,
        )
        generated = np.asarray(generated, dtype=np.float32)
        validate_generated_batch(
            generated,
            class_ids=[request.class_id],
            sample_indices=(index,),
            request=SynthesisRequest(
                dataset_id=request.dataset_id,
                family=request.family,
                experimental_seed=request.experimental_seed,
                class_id=request.class_id,
                sample_indices=(index,),
                class_mapping=request.class_mapping,
                scope=(
                    "fixture"
                    if request.scope == "fixture"
                    else "production_batch"
                ),
            ),
        )
        images.append(generated[0])
        batch_sizes.append(1)

    result = np.stack(images, axis=0).astype(np.float32)
    validate_generated_batch(
        result,
        class_ids=[request.class_id] * len(indices),
        sample_indices=indices,
        request=request,
    )
    return {
        "images": result,
        "record": _batch_record(
            request=request,
            recipe=recipe,
            checkpoint_identity=checkpoint_identity,
            observed_count=len(result),
            batch_sizes=batch_sizes,
        ),
    }


def load_gan_generator_for_synthesis(
    *,
    checkpoint_directory,
    request,
    recipe,
    checkpoint_identity,
):
    """Thin wrapper around the accepted terminal GAN loader."""
    validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "gan":
        raise ValueError("GAN loader requires the GAN family")
    from .gan_checkpoint_v3_restoration import (
        load_terminal_gan_checkpoint_v3,
    )

    loaded = load_terminal_gan_checkpoint_v3(
        checkpoint_directory,
        requested_dataset_id=request.dataset_id,
        requested_experimental_seed=request.experimental_seed,
        requested_class_mapping=list(request.class_mapping),
    )
    return loaded["generator"]


def load_vae_decoder_for_synthesis(
    *,
    checkpoint_directory,
    request,
    recipe,
    checkpoint_identity,
):
    """Thin wrapper around the accepted VAE loader requiring best role."""
    validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "vae":
        raise ValueError("VAE loader requires the VAE family")
    from .variational_autoencoder_checkpoint_loader import (
        load_governed_vae_checkpoint,
    )

    loaded = load_governed_vae_checkpoint(
        checkpoint_directory,
        requested_dataset_id=request.dataset_id,
        requested_experimental_seed=request.experimental_seed,
        requested_class_mapping=list(request.class_mapping),
    )
    if loaded["metadata"]["checkpoint_role"] != "best":
        raise ValueError("VAE primary synthesis requires the best checkpoint")
    return loaded["decoder"]


def restore_diffusion_model_for_synthesis(
    *,
    prepared,
    model_factory,
    request,
    recipe,
    checkpoint_identity,
):
    """Restore an unpublished Diffusion best state for Stage 2 fixtures."""
    validate_checkpoint_identity(
        checkpoint_identity,
        request=request,
        recipe=recipe,
    )
    if request.family != "diffusion":
        raise ValueError("Diffusion loader requires the Diffusion family")
    from .diffusion_checkpoint_state import (
        restore_diffusion_checkpoint_state,
    )

    return restore_diffusion_checkpoint_state(
        prepared=prepared,
        model_factory=model_factory,
        requested_dataset_id=request.dataset_id,
        requested_experimental_seed=request.experimental_seed,
        requested_class_mapping=list(request.class_mapping),
        requested_role="best",
    )


def validate_complete_pool_manifest(manifest, *, recipe):
    """Validate complete-pool metadata without claiming generated bytes."""
    if not isinstance(recipe, SynthesisRecipe):
        raise TypeError("recipe must be a SynthesisRecipe")
    manifest = _exact_keys(
        manifest, _POOL_FIELDS, name="synthesis pool manifest"
    )
    if manifest["schema"] != POOL_SCHEMA:
        raise ValueError("unsupported synthesis pool schema")
    if manifest["dataset_id"] not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")
    family = manifest["family"]
    if family not in FAMILIES:
        raise ValueError("unsupported generator family")
    entry = _family_entry(recipe, family)
    seed = _exact_int(
        manifest["experimental_seed"], name="experimental_seed"
    )
    if seed not in EXPERIMENTAL_SEEDS:
        raise ValueError("experimental seed must be 42-48")
    mapping = _class_mapping(manifest["class_mapping"])
    if len(mapping) != DATASET_CLASS_COUNTS[manifest["dataset_id"]]:
        raise ValueError("class mapping length mismatch")
    if manifest["checkpoint_role"] != entry["checkpoint_role"]:
        raise ValueError("checkpoint role mismatch")
    digest = manifest["checkpoint_sha256"]
    if type(digest) is not str or HEX64.fullmatch(digest) is None:
        raise ValueError("invalid checkpoint SHA-256")
    if manifest["recipe_sha256"] != recipe.sha256:
        raise ValueError("pool recipe identity mismatch")
    if _exact_int(
        manifest["model_call_batch_size"],
        name="model_call_batch_size",
    ) != entry["model_call_batch_size"]:
        raise ValueError("pool batch size mismatch")
    if _exact_int(
        manifest["quota_per_class"], name="quota_per_class"
    ) != PRIMARY_QUOTA:
        raise ValueError("pool quota mismatch")
    if manifest["complete_pool"] is not True:
        raise ValueError("manifest does not claim a complete pool")
    if manifest["status"] != "COMPLETE_METADATA_VALIDATED":
        raise ValueError("pool status mismatch")
    if manifest["authoritative_artifact"] is not False:
        raise ValueError("fixture pool cannot be authoritative")

    classes = manifest["classes"]
    if not isinstance(classes, list) or len(classes) != len(mapping):
        raise ValueError("complete pool requires every authoritative class")
    seen = set()
    for record in classes:
        record = _exact_keys(
            record,
            frozenset({"class_id", "sample_indices"}),
            name="pool class record",
        )
        class_id = _exact_int(record["class_id"], name="class_id")
        if not 0 <= class_id < len(mapping):
            raise ValueError("pool class outside authoritative mapping")
        if class_id in seen:
            raise ValueError("duplicate pool class")
        seen.add(class_id)
        if not isinstance(record["sample_indices"], list):
            raise TypeError("pool sample_indices must be a list")
        if len(record["sample_indices"]) != PRIMARY_QUOTA:
            raise ValueError(
                "complete pool requires 2,000 sample indices per class"
            )
        for expected_index, value in enumerate(
            record["sample_indices"]
        ):
            if _exact_int(value, name="pool sample_index") != expected_index:
                raise ValueError(
                    "complete pool requires ordered sample indices 0-1999"
                )
    if seen != set(range(len(mapping))):
        raise ValueError("incomplete authoritative class coverage")
    total = len(mapping) * PRIMARY_QUOTA
    if _exact_int(
        manifest["expected_total"], name="expected_total"
    ) != total:
        raise ValueError("expected total mismatch")
    if _exact_int(
        manifest["observed_total"], name="observed_total"
    ) != total:
        raise ValueError("incomplete production output")
    return {
        "dataset_id": manifest["dataset_id"],
        "family": family,
        "experimental_seed": seed,
        "class_count": len(mapping),
        "total": total,
        "status": "COMPLETE_METADATA_VALIDATED",
        "authoritative_artifact": False,
    }


__all__ = [
    "BATCH_SCHEMA",
    "CANONICAL_OUTPUT",
    "DATASET_CLASS_COUNTS",
    "DATASET_CODES",
    "DIFFUSION_STREAM_CONTRACT",
    "FAMILY_CODES",
    "LATENT_STREAM_CONTRACT",
    "POOL_SCHEMA",
    "PRIMARY_QUOTA",
    "SynthesisRecipe",
    "SynthesisRequest",
    "derive_latent_vector",
    "derive_latent_vectors",
    "governed_batch_size",
    "latent_entropy",
    "load_gan_generator_for_synthesis",
    "load_synthesis_recipe",
    "load_vae_decoder_for_synthesis",
    "restore_diffusion_model_for_synthesis",
    "rng_metadata",
    "synthesize_diffusion_batch",
    "synthesize_gan_batch",
    "synthesize_vae_batch",
    "validate_checkpoint_identity",
    "validate_complete_pool_manifest",
    "validate_generated_batch",
    "validate_synthesis_recipe",
    "validate_synthesis_request",
]
