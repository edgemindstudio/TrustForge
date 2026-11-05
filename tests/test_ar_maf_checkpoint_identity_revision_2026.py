from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_identity as identity


@pytest.fixture
def inputs(monkeypatch):
    calls = []

    def validator(evidence, *, expected_dataset_id):
        calls.append((deepcopy(evidence), expected_dataset_id))
        if evidence.get("reject"):
            raise ValueError("provenance rejected")
        return evidence

    monkeypatch.setattr(identity, "validate_vae_checkpoint_provenance", validator)
    recipe = (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
    ).read_bytes()
    return {
        "family": "ar",
        "dataset_id": "malex",
        "experimental_seed": 42,
        "class_mapping": ["benign", "malware"],
        "recipe_raw": recipe,
        "dataset_provenance": {"fixture": {"value": 1}},
    }, calls


@pytest.mark.parametrize("family", ["ar", "maf"])
def test_identity_binds_recipe_controls_and_provenance(inputs, family):
    values, calls = inputs
    values["family"] = family
    result = identity.build_checkpoint_identity(**values)
    assert calls == [(values["dataset_provenance"], "malex")]
    assert result["family"] == family
    assert result["dataset_id"] == "malex"
    assert result["experimental_seed"] == 42
    assert result["class_mapping"] == ["benign", "malware"]
    assert result["class_count"] == 2
    assert result["canonical_shape"] == [40, 40, 3]
    assert result["recipe_sha256"] == hashlib.sha256(
        values["recipe_raw"]
    ).hexdigest()
    assert result["dataset_provenance_sha256"] == hashlib.sha256(
        identity.canonical_json_bytes(result["dataset_provenance"])
    ).hexdigest()
    assert result["training_controls"]["epochs"] == 2
    assert result["optimizer_controls"]["learning_rate"] == 0.001
    assert result["training_resume_supported"] is False
    assert "required_integration_changes" not in result["family_controls"]


def test_result_does_not_alias_caller_inputs(inputs):
    values, _ = inputs
    result = identity.build_checkpoint_identity(**values)
    values["class_mapping"][0] = "changed"
    values["dataset_provenance"]["fixture"]["value"] = 99
    assert result["class_mapping"][0] == "benign"
    assert result["dataset_provenance"]["fixture"]["value"] == 1


@pytest.mark.parametrize("key,value", [
    ("family", "gan"),
    ("dataset_id", "unknown"),
    ("experimental_seed", True),
    ("experimental_seed", 42.0),
    ("experimental_seed", 41),
    ("class_mapping", "benign,malware"),
    ("class_mapping", ["benign"]),
    ("class_mapping", ["benign", "benign"]),
    ("class_mapping", ["benign", " "]),
    ("class_mapping", ["benign", 1]),
])
def test_invalid_identity_rejected_before_provenance(inputs, key, value):
    values, calls = inputs
    values[key] = value
    with pytest.raises((ValueError, TypeError)):
        identity.build_checkpoint_identity(**values)
    assert calls == []


@pytest.mark.parametrize("change", ["bytes", "type"])
def test_recipe_mismatch_rejected_before_provenance(inputs, change):
    values, calls = inputs
    if change == "bytes":
        values["recipe_raw"] += b"\n"
    else:
        values["recipe_raw"] = values["recipe_raw"].decode()
    with pytest.raises((ValueError, TypeError)):
        identity.build_checkpoint_identity(**values)
    assert calls == []


def test_provenance_failure_propagates(inputs):
    values, calls = inputs
    values["dataset_provenance"] = {"reject": True}
    with pytest.raises(ValueError, match="provenance rejected"):
        identity.build_checkpoint_identity(**values)
    assert len(calls) == 1


def test_validator_cannot_mutate_callers_provenance(inputs, monkeypatch):
    values, _ = inputs

    def mutating_validator(evidence, *, expected_dataset_id):
        evidence["fixture"]["value"] = 7
        return evidence

    monkeypatch.setattr(
        identity, "validate_vae_checkpoint_provenance", mutating_validator,
    )
    identity.build_checkpoint_identity(**values)
    assert values["dataset_provenance"]["fixture"]["value"] == 1


def test_canonical_json_is_order_independent_and_rejects_nan():
    assert identity.canonical_json_bytes({"b": 2, "a": 1}) == (
        identity.canonical_json_bytes({"a": 1, "b": 2})
    )
    with pytest.raises(ValueError):
        identity.canonical_json_bytes({"invalid": float("nan")})
