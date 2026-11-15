"""Bounded CIC MAF synthesis binding and stream-identity checks (seeds 43/44)."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_paper01_ar_cic_range_worker as worker
from trustforge.evaluation.paper01_cic_pool import CICPoolView
from trustforge.models.paper01_revision_2026.ar_maf_cic_partition_finalizer import (
    finalize_cic_partitioned_pool,
)
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
)
RECIPE_SHA256 = "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
CHECKPOINT_43 = (
    "57e378c44c870af58e4b8eff38cf1cc976516b0dade249194bb5962c28a724f7"
)
CHECKPOINT_44 = (
    "cc58fe65abaa0656e4f09d6d17d83e586f048f50a6a858d01e1f1d7b9191055a"
)


def batch_seed(seed, class_id, batch_index):
    return synthesis_batch_seed(
        dataset_id="cicmaldroid2020", family="maf",
        experimental_seed=seed, class_id=class_id,
        batch_index=batch_index,
    )


def test_binding_table_covers_exactly_the_authorized_cells():
    assert set(worker.BINDINGS) == {
        ("ar", 42), ("maf", 42), ("maf", 43), ("maf", 44),
    }
    maf43 = worker.BINDINGS[("maf", 43)]
    assert maf43["training_commit"] == (
        "58f965c39aad103dc056f7ac690de90578ed2932"
    )
    assert maf43["best_sha256"] == CHECKPOINT_43
    assert maf43["generation"] == "maf_cic_seed43_execution"
    assert maf43["training_job"] == "314075"
    maf44 = worker.BINDINGS[("maf", 44)]
    assert maf44["training_commit"] == (
        "8d4a281e04ed2232a134a6da10ff20c56d67316e"
    )
    assert maf44["best_sha256"] == CHECKPOINT_44
    assert maf44["generation"] == "maf_cic_seed44_execution"
    assert maf44["training_job"] == "314269"
    assert worker.BINDINGS[("maf", 42)]["training_job"] == "311675"
    assert worker.BINDINGS[("ar", 42)]["generation"] == "ar_cic_seed42"


@pytest.mark.parametrize("family,seed", [
    ("ar", 42), ("maf", 42), ("maf", 43), ("maf", 44),
])
def test_authorized_bindings_return_expected_generation(family, seed):
    binding = worker.execution_binding(family, seed)
    assert binding["generation"] == {
        ("ar", 42): "ar_cic_seed42",
        ("maf", 42): "maf_cic_seed42_execution",
        ("maf", 43): "maf_cic_seed43_execution",
        ("maf", 44): "maf_cic_seed44_execution",
    }[(family, seed)]


def test_previous_synthesis_bindings_are_unchanged():
    assert worker.BINDINGS[("ar", 42)] == {
        "training_commit": "e4473abcdbc53f0de3a01ebe48858f2f3c7b841a",
        "best_sha256":
            "d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0",
        "generation": "ar_cic_seed42",
        "training_job": None,
    }
    assert worker.BINDINGS[("maf", 42)] == {
        "training_commit": "913d86f9c64e6a3a1a0d009f112bcfa889a02ec8",
        "best_sha256":
            "3604bd4009e2199400665b28ef156d5db8e6d2e75094e5eb1d193b090d89264d",
        "generation": "maf_cic_seed42_execution",
        "training_job": "311675",
    }


def test_seed44_binding_differs_from_every_earlier_cell():
    maf44 = worker.BINDINGS[("maf", 44)]
    for cell, binding in worker.BINDINGS.items():
        if cell == ("maf", 44):
            continue
        assert maf44["generation"] != binding["generation"]
        assert maf44["best_sha256"] != binding["best_sha256"]


@pytest.mark.parametrize("family,seed", [
    ("ar", 43), ("ar", 44), ("maf", 45), ("gan", 42), ("gan", 44),
    ("maf", True), ("maf", 42.0), ("maf", "43"), ("maf", "44"),
])
def test_unauthorized_bindings_rejected(family, seed):
    with pytest.raises(
        ValueError, match="unsupported synthesis family or experimental seed",
    ):
        worker.execution_binding(family, seed)


BASE_ARGUMENTS = (
    "--family", "maf",
    "--output", "out",
    "--class-id", "0",
    "--batch-start", "0",
    "--batch-stop", "2",
    "--repository-root", ".",
    "--expected-head", "a" * 40,
    "--dataset-config", "config.json",
    "--config-sha256", "b" * 64,
    "--publication-parent", "checkpoints",
    "--generation-name", "maf_cic_seed43_execution",
    "--expectations", "expectations.json",
    "--expectations-sha256", "c" * 64,
)


def test_parser_defaults_to_seed42_for_existing_callers():
    arguments = worker._parser().parse_args(BASE_ARGUMENTS)
    assert arguments.experimental_seed == 42


def test_parser_accepts_seed43():
    arguments = worker._parser().parse_args(
        BASE_ARGUMENTS + ("--experimental-seed", "43")
    )
    assert arguments.experimental_seed == 43


def test_parser_accepts_seed44():
    arguments = worker._parser().parse_args(
        BASE_ARGUMENTS + ("--experimental-seed", "44")
    )
    assert arguments.experimental_seed == 44


@pytest.mark.parametrize("seed", ["41", "45", "48"])
def test_parser_rejects_other_seeds(seed):
    with pytest.raises(SystemExit):
        worker._parser().parse_args(
            BASE_ARGUMENTS + ("--experimental-seed", seed)
        )


def test_seed43_class0_batch_seeds_match_accepted_derivation():
    assert batch_seed(43, 0, 0) == 129251
    assert batch_seed(43, 0, 1) == 129252
    assert batch_seed(43, 0, 249) == 129500
    assert batch_seed(43, 4, 249) == 130500
    assert batch_seed(42, 0, 0) == 35251
    assert batch_seed(42, 0, 1) == 35252


def test_seed44_batch_seeds_match_accepted_derivation():
    assert batch_seed(44, 0, 0) == 223251
    assert batch_seed(44, 0, 1) == 223252
    assert batch_seed(44, 0, 249) == 223500
    assert batch_seed(44, 4, 249) == 224500
    seeds = {
        batch_seed(44, class_id, batch_index)
        for class_id in range(5) for batch_index in range(250)
    }
    assert len(seeds) == 1250
    assert min(seeds) == 223251
    assert max(seeds) == 224500


def test_seed43_stream_is_disjoint_from_seed42():
    seed42 = {
        batch_seed(42, class_id, batch_index)
        for class_id in range(5) for batch_index in range(250)
    }
    seed43 = {
        batch_seed(43, class_id, batch_index)
        for class_id in range(5) for batch_index in range(250)
    }
    assert len(seed42) == len(seed43) == 1250
    assert seed42.isdisjoint(seed43)


def test_seed44_stream_is_disjoint_from_seeds42_and_43():
    streams = {
        seed: {
            batch_seed(seed, class_id, batch_index)
            for class_id in range(5) for batch_index in range(250)
        }
        for seed in (42, 43, 44)
    }
    assert all(len(values) == 1250 for values in streams.values())
    assert streams[44].isdisjoint(streams[42])
    assert streams[44].isdisjoint(streams[43])
    assert streams[42].isdisjoint(streams[43])


def test_seed44_budget_matches_full_pool_structure():
    # Five classes x 250 batches x 8 images = 10,000 images, 2,000 per class.
    per_class = {
        class_id: {
            batch_seed(44, class_id, batch_index) for batch_index in range(250)
        }
        for class_id in range(5)
    }
    assert all(len(values) == 250 for values in per_class.values())
    assert per_class[0].isdisjoint(per_class[4])
    assert batch_seed(44, 0, 0) == 223251
    assert batch_seed(44, 4, 0) == 224251
    assert batch_seed(44, 0, 249) == 223500
    assert batch_seed(44, 4, 249) == 224500


def test_recipe_and_sampling_policy_unchanged():
    raw = RECIPE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECIPE_SHA256
    recipe = json.loads(raw)
    assert recipe["experimental_seeds"] == [42, 43, 44, 45, 46, 47, 48]
    assert recipe["sampling"]["batch_size"] == 8
    assert recipe["sampling"]["samples_per_class"] == 2000


def _batch_report(*, seed, sampling_seed, checkpoint_sha256=CHECKPOINT_43,
                  synthesis_commit="d" * 40):
    return {
        "dataset_id": "cicmaldroid2020",
        "family": "maf",
        "experimental_seed": seed,
        "class_id": 0,
        "batch_index": 0,
        "sampling_seed": sampling_seed,
        "recipe_sha256": RECIPE_SHA256,
        "checkpoint_sha256": checkpoint_sha256,
        "synthesis_commit": synthesis_commit,
        "config_sha256": "e" * 64,
        "samples_sha256": "f" * 64,
    }


def _write_batch(tmp_path, report):
    batch = tmp_path / "task-0" / "batch-000"
    batch.mkdir(parents=True)
    (batch / "report.json").write_text(json.dumps(report) + "\n")
    return batch


def test_finalizer_rejects_seed42_batch_under_seed43_scope(tmp_path):
    batch = tmp_path / "task-0" / "batch-000"
    batch.mkdir(parents=True)
    (batch / "report.json").write_text(
        json.dumps(_batch_report(seed=42, sampling_seed=35251)) + "\n"
    )
    with pytest.raises(ValueError, match="worker batch identity mismatch"):
        finalize_cic_partitioned_pool(
            tmp_path / "pool",
            batch_directories=[batch],
            family="maf", experimental_seed=43,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_43,
            synthesis_commit="d" * 40,
            config_sha256="e" * 64,
        )
    assert not (tmp_path / "pool").exists()


def test_finalizer_rejects_seed43_batch_under_seed44_scope(tmp_path):
    batch = _write_batch(
        tmp_path, _batch_report(seed=43, sampling_seed=129251),
    )
    with pytest.raises(ValueError, match="worker batch identity mismatch"):
        finalize_cic_partitioned_pool(
            tmp_path / "pool",
            batch_directories=[batch],
            family="maf", experimental_seed=44,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_44,
            synthesis_commit="d" * 40,
            config_sha256="e" * 64,
        )
    assert not (tmp_path / "pool").exists()


def test_finalizer_rejects_seed43_checkpoint_under_seed44_scope(tmp_path):
    batch = _write_batch(
        tmp_path,
        _batch_report(
            seed=44, sampling_seed=223251, checkpoint_sha256=CHECKPOINT_43,
        ),
    )
    with pytest.raises(ValueError, match="worker batch identity mismatch"):
        finalize_cic_partitioned_pool(
            tmp_path / "pool",
            batch_directories=[batch],
            family="maf", experimental_seed=44,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_44,
            synthesis_commit="d" * 40,
            config_sha256="e" * 64,
        )
    assert not (tmp_path / "pool").exists()


def test_finalizer_rejects_seed44_source_mismatch(tmp_path):
    batch = _write_batch(
        tmp_path,
        _batch_report(
            seed=44, sampling_seed=223251,
            checkpoint_sha256=CHECKPOINT_44, synthesis_commit="a" * 40,
        ),
    )
    with pytest.raises(ValueError, match="worker batch identity mismatch"):
        finalize_cic_partitioned_pool(
            tmp_path / "pool",
            batch_directories=[batch],
            family="maf", experimental_seed=44,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_44,
            synthesis_commit="d" * 40,
            config_sha256="e" * 64,
        )
    assert not (tmp_path / "pool").exists()


def test_finalizer_rejects_seed43_stream_under_seed44_scope(tmp_path):
    batch = _write_batch(
        tmp_path,
        _batch_report(
            seed=44, sampling_seed=129251, checkpoint_sha256=CHECKPOINT_44,
        ),
    )
    with pytest.raises(ValueError, match="worker batch seed mismatch"):
        finalize_cic_partitioned_pool(
            tmp_path / "pool",
            batch_directories=[batch],
            family="maf", experimental_seed=44,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_44,
            synthesis_commit="d" * 40,
            config_sha256="e" * 64,
        )
    assert not (tmp_path / "pool").exists()


def test_pool_view_rejects_seed42_manifest_under_seed43_scope(tmp_path):
    manifest = {
        "schema_version": 1,
        "dataset_id": "cicmaldroid2020",
        "family": "maf",
        "experimental_seed": 42,
        "recipe_sha256": RECIPE_SHA256,
        "checkpoint_sha256": CHECKPOINT_43,
        "batches": [],
    }
    path = tmp_path / "manifest.json"
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="CIC pool identity mismatch"):
        CICPoolView(
            path, manifest_sha256=hashlib.sha256(raw).hexdigest(),
            family="maf", experimental_seed=43,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_43,
        )


def test_pool_view_rejects_seed43_manifest_under_seed44_scope(tmp_path):
    manifest = {
        "schema_version": 1,
        "dataset_id": "cicmaldroid2020",
        "family": "maf",
        "experimental_seed": 43,
        "recipe_sha256": RECIPE_SHA256,
        "checkpoint_sha256": CHECKPOINT_44,
        "batches": [],
    }
    path = tmp_path / "manifest.json"
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="CIC pool identity mismatch"):
        CICPoolView(
            path, manifest_sha256=hashlib.sha256(raw).hexdigest(),
            family="maf", experimental_seed=44,
            recipe_sha256=RECIPE_SHA256,
            checkpoint_sha256=CHECKPOINT_44,
        )
