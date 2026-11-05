"""Pure AR/MAF checkpoint identity and dataset-evidence binding.

Validates declared records, not actual historical training inputs.
No filesystem access, TensorFlow import, publication or authorization.
"""

from copy import deepcopy
import hashlib
import json

from .ar_maf_execution_inputs import DATASET_CLASSES, SEEDS, require_integer
from .variational_autoencoder_checkpoint_provenance import (
    validate_vae_checkpoint_provenance,
)

SELECTED_RECIPE_SHA256 = (
    "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
)


def canonical_json_bytes(value):
    """Encode deterministic JSON; reject nonfinite or non-JSON values."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def build_checkpoint_identity(
    *, family, dataset_id, experimental_seed, class_mapping,
    recipe_raw, dataset_provenance,
):
    """Validate the selected protocol and return independent identity records.

    Class labels must follow authoritative model-ID order. Their semantic
    agreement with the actual adapter is an outer execution-boundary check.
    """
    if type(family) is not str or family not in ("ar", "maf"):
        raise ValueError("unsupported checkpoint family")
    if type(dataset_id) is not str or dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported checkpoint dataset")
    seed = require_integer(experimental_seed, "experimental_seed")
    if seed not in SEEDS:
        raise ValueError("unsupported experimental seed")

    if not isinstance(class_mapping, (list, tuple)):
        raise TypeError("class mapping must be an ordered sequence")
    mapping = list(class_mapping)
    if (
        len(mapping) != DATASET_CLASSES[dataset_id]
        or any(type(label) is not str or not label.strip() for label in mapping)
        or len(set(mapping)) != len(mapping)
    ):
        raise ValueError("invalid authoritative class mapping")

    if type(recipe_raw) is not bytes:
        raise TypeError("recipe must be exact bytes")
    recipe_digest = hashlib.sha256(recipe_raw).hexdigest()
    if recipe_digest != SELECTED_RECIPE_SHA256:
        raise ValueError("recipe differs from selected integration protocol")
    recipe = json.loads(recipe_raw)
    if (
        recipe["schema"] != "trustforge.paper01.ar-maf.integration-recipe.v1"
        or recipe["production_execution_authorized"] is not False
    ):
        raise ValueError("unexpected recipe identity or execution status")

    # Reuse the existing strict dataset-specific evidence contract.
    # Its historical GAN schema identifier remains unchanged.
    provenance = validate_vae_checkpoint_provenance(
        deepcopy(dataset_provenance), expected_dataset_id=dataset_id,
    )
    provenance_raw = canonical_json_bytes(provenance)

    family_key = "autoregressive" if family == "ar" else "masked_autoflow"
    controls = deepcopy(recipe[family_key])
    controls.pop("required_integration_changes", None)

    return {
        "schema": "trustforge.paper01.ar-maf.checkpoint-identity.v1",
        "family": family,
        "dataset_id": dataset_id,
        "experimental_seed": seed,
        "class_mapping": mapping,
        "class_count": len(mapping),
        "canonical_shape": deepcopy(recipe["canonical_shape"]),
        "recipe_sha256": recipe_digest,
        "family_controls": controls,
        "training_controls": deepcopy(recipe["training"]),
        "optimizer_controls": deepcopy(recipe["optimizer"]),
        "validation_controls": deepcopy(recipe["validation"]),
        "dataset_provenance_sha256": hashlib.sha256(provenance_raw).hexdigest(),
        "dataset_provenance": deepcopy(provenance),
        "checkpoint_use": "evaluation_and_synthesis",
        "training_resume_supported": False,
    }
