"""Bounded seed-43 MAF/CIC generator scope and identity-binding checks."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_paper01_maf_cic_execution as caller
from trustforge.models.paper01_revision_2026 import (
    ar_maf_cic_execution as execution,
)
from trustforge.models.paper01_revision_2026 import (
    ar_maf_training_runner as runner,
)


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
)


def _manifest(*, seed):
    return {
        "identity": {
            "family": "maf",
            "dataset_id": "cicmaldroid2020",
            "experimental_seed": seed,
        },
        "source_commit": "a" * 40,
        "source_files": {"src/fixture.py": "b" * 64},
    }


@pytest.mark.parametrize("family,dataset,seed,allowed", [
    ("ar", "cicmaldroid2020", 42, True),
    ("maf", "cicmaldroid2020", 42, True),
    ("maf", "cicmaldroid2020", 43, True),
    ("ar", "cicmaldroid2020", 43, False),
    ("maf", "cicmaldroid2020", 44, False),
    ("maf", "malex", 43, False),
    ("gan", "cicmaldroid2020", 42, False),
    ("maf", "cicmaldroid2020", True, False),
    ("maf", "cicmaldroid2020", 42.0, False),
])
def test_bounded_scope_matrix(family, dataset, seed, allowed):
    assert runner.bounded_cic_request_allowed(
        family=family, dataset_id=dataset, experimental_seed=seed,
    ) is allowed


def test_seed42_scope_is_preserved():
    assert runner.BOUNDED_CIC_SCOPE == {
        ("ar", 42), ("maf", 42), ("maf", 43),
    }


def test_runner_reaches_source_for_maf_seed43(monkeypatch):
    def stop_at_source(*args, **kwargs):
        raise ValueError("source boundary reached")

    monkeypatch.setattr(runner, "TRAINING_EXECUTION_ENABLED", False)
    monkeypatch.setattr(runner, "capture_source_identity", stop_at_source)
    with pytest.raises(ValueError, match="source boundary reached"):
        runner.run_training_and_prepare_checkpoint(
            repository_root="unused", family="maf",
            dataset_id="cicmaldroid2020", experimental_seed=43,
            adapter=object(), bounded_integration=True,
        )


def _patch_execution(monkeypatch, *, identity_seed, observed):
    def train(**kwargs):
        observed.update(kwargs)
        return {
            "manifest": _manifest(seed=identity_seed),
            "manifest_raw": b"{}",
            "recipe_raw": b"recipe",
            "best_raw": b"best",
            "last_raw": b"last",
        }

    monkeypatch.setattr(
        execution, "_capture", lambda **kwargs: {"sha256": "c" * 64},
    )
    monkeypatch.setattr(
        execution.runner, "run_training_and_prepare_checkpoint", train,
    )
    monkeypatch.setattr(
        execution, "prepare_cic_execution_manifest",
        lambda **kwargs: b"envelope",
    )


def test_execution_forwards_seed43_into_request(monkeypatch):
    observed = {}
    _patch_execution(monkeypatch, identity_seed=43, observed=observed)
    prepared = execution.prepare_cic_execution(
        repository_root="fixture", family="maf", experimental_seed=43,
        adapter=object(), dataset_config_raw=b"fixture",
        bounded_integration=True,
    )
    assert observed["family"] == "maf"
    assert observed["experimental_seed"] == 43
    assert observed["bounded_integration"] is True
    assert prepared["requested"]["expected_identity"][
        "experimental_seed"
    ] == 43


def test_execution_rejects_mismatched_checkpoint_seed(monkeypatch):
    observed = {}
    _patch_execution(monkeypatch, identity_seed=42, observed=observed)
    with pytest.raises(
        RuntimeError, match="differs from requested execution",
    ):
        execution.prepare_cic_execution(
            repository_root="fixture", family="maf", experimental_seed=43,
            adapter=object(), dataset_config_raw=b"fixture",
            bounded_integration=True,
        )


def test_execution_rejects_ar_seed43_before_capture(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("unauthorized request crossed capture boundary")

    monkeypatch.setattr(execution, "_capture", forbidden)
    with pytest.raises(RuntimeError, match="bounded CIC execution scope"):
        execution.prepare_cic_execution(
            repository_root="fixture", family="ar", experimental_seed=43,
            adapter=object(), dataset_config_raw=b"fixture",
            bounded_integration=True,
        )


BASE_ARGUMENTS = (
    "--phase", "train",
    "--repository-root", ".",
    "--expected-head", "a" * 40,
    "--dataset-config", "dataset.json",
    "--config-sha256", "d" * 64,
    "--run-dir", "run",
)


def test_caller_defaults_to_seed42():
    arguments = caller._parser().parse_args(BASE_ARGUMENTS)
    assert arguments.experimental_seed == 42


def test_caller_accepts_seed43():
    arguments = caller._parser().parse_args(
        BASE_ARGUMENTS + ("--experimental-seed", "43")
    )
    assert arguments.experimental_seed == 43


@pytest.mark.parametrize("seed", ["41", "44", "48"])
def test_caller_rejects_other_seeds(seed):
    with pytest.raises(SystemExit):
        caller._parser().parse_args(
            BASE_ARGUMENTS + ("--experimental-seed", seed)
        )


def test_generation_names_are_seed_bound():
    assert caller.execution_generation(42) == "maf_cic_seed42_execution"
    assert caller.execution_generation(43) == "maf_cic_seed43_execution"
    assert caller.execution_generation(42) != caller.execution_generation(43)


def test_accepted_recipe_and_permutation_seeds_are_unchanged():
    raw = RECIPE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
    )
    recipe = json.loads(raw)
    assert tuple(recipe["masked_autoflow"]["permutation_seeds"]) == (
        42, 43, 44, 45, 46,
    )
    assert recipe["experimental_seeds"] == [42, 43, 44, 45, 46, 47, 48]


def test_checkpoint_identity_binds_seed43():
    from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
        build_checkpoint_identity, canonical_json_bytes,
    )
    from trustforge.models.paper01_revision_2026.gan_v3_provenance_adapter import (
        build_gan_v3_provenance,
    )

    preflight = {
        "dataset_id": "cicmaldroid2020",
        "source_contract": "cic_authoritative_source_revision_2026",
        "integrity_settings_verified": True,
        "verified_sha256": {
            "source_csv":
                "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00",
            "feature_columns_path":
                "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f",
        },
    }
    provenance = build_gan_v3_provenance(
        dataset_id="cicmaldroid2020", verified_preflight=preflight,
    )
    identity = {}
    for seed in (42, 43):
        identity[seed] = build_checkpoint_identity(
            family="maf", dataset_id="cicmaldroid2020",
            experimental_seed=seed,
            class_mapping=[str(index) for index in range(5)],
            recipe_raw=RECIPE.read_bytes(), dataset_provenance=provenance,
        )
    assert identity[42]["experimental_seed"] == 42
    assert identity[43]["experimental_seed"] == 43
    assert canonical_json_bytes(identity[42]) != canonical_json_bytes(
        identity[43]
    )
    assert identity[42]["recipe_sha256"] == identity[43]["recipe_sha256"]
