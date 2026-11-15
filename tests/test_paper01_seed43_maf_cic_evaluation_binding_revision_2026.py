"""Bounded seed-43 MAF/CIC paired-evaluation binding checks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import runpy

from scripts import run_paper01_cic_maf_seed42_bounded_evaluation as driver
from trustforge.evaluation import paper01_augmented_checkpoint
from trustforge.evaluation import paper01_classifier_tensorflow
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)


HELPERS = runpy.run_path(str(Path(__file__).with_name(
    "test_cic_maf_seed42_bounded_evaluation_revision_2026.py",
)))

SEED43 = {
    "experimental_seed": 43,
    "pool_manifest_sha256":
        "95b1d78e90cb8794573b88a195bb1e89fbbfcfcfc7d9562a5210a9971527354f",
    "pool_checkpoint_sha256":
        "57e378c44c870af58e4b8eff38cf1cc976516b0dade249194bb5962c28a724f7",
    "checkpoint_source_commit":
        "58f965c39aad103dc056f7ac690de90578ed2932",
    "checkpoint_generation": "maf_cic_seed43_execution",
    "checkpoint_fitted_state_sha256":
        "d1b2c34f9354fb7cd57b595d55825d17d6c6401f05fb9ab337b0f3931163d7ab",
    "checkpoint_execution_manifest_sha256":
        "cfd37e07a58d1acdd07344288b9d899026f5c18921cbe63fe69ed86783bd3711",
    "checkpoint_expectations_sha256":
        "aabb7ccf389000565f923e2ddc80eb5d40dc4a2fcedcc4c23daffd829916607e",
    "checkpoint_training_job": "314075",
    "lineage_schema":
        "trustforge.paper01.cic-maf.seed43.lineage-evidence.v1",
}


def _binding(seed):
    try:
        return driver._apply_binding(seed)
    except BaseException:
        driver._apply_binding(42)
        raise


def _receipt(seed=43):
    values = driver.SEED_BINDINGS[seed]
    return {
        "schema": driver.CHECKPOINT_EXPECTATIONS_SCHEMA,
        "source_commit": values["checkpoint_source_commit"],
        "generation_name": values["checkpoint_generation"],
        "job_id": values["checkpoint_training_job"],
        "fitted_state_sha256": values["checkpoint_fitted_state_sha256"],
        "execution_manifest_file_sha256":
            values["checkpoint_execution_manifest_sha256"],
        "configuration_sha256": driver.DATASET_CONFIG_SHA256,
        "training_pid": 1234,
    }


def _manifest(seed=43, *, checkpoint=None):
    values = driver.SEED_BINDINGS[seed]
    records = []
    for class_id in range(5):
        for batch_index in range(250):
            records.append({
                "class_id": class_id,
                "batch_index": batch_index,
                "sampling_seed": synthesis_batch_seed(
                    dataset_id="cicmaldroid2020", family="maf",
                    experimental_seed=seed,
                    class_id=class_id, batch_index=batch_index,
                ),
                "filename": f"c{class_id}-b{batch_index:03d}.npy",
                "sha256": "a" * 64,
            })
    return {
        "schema_version": 1,
        "dataset_id": "cicmaldroid2020",
        "family": "maf",
        "experimental_seed": seed,
        "recipe_sha256": driver.GENERATOR_RECIPE_SHA256,
        "checkpoint_sha256": (
            values["pool_checkpoint_sha256"] if checkpoint is None
            else checkpoint
        ),
        "batches": records,
    }


def test_binding_table_is_exactly_the_authorized_cells():
    assert set(driver.SEED_BINDINGS) == {42, 43}
    assert driver.SEED_BINDINGS[43] == SEED43
    assert driver.SEED_BINDINGS[42]["experimental_seed"] == 42
    assert driver.SEED_BINDINGS[42]["checkpoint_generation"] == (
        "maf_cic_seed42_execution"
    )
    assert driver.SEED_BINDINGS[42]["checkpoint_training_job"] == "311675"


def test_seed42_defaults_are_preserved():
    assert driver.EXPERIMENTAL_SEED == 42
    assert driver.POOL_MANIFEST_SHA256 == driver.SEED_BINDINGS[42][
        "pool_manifest_sha256"
    ]
    assert driver.POOL_CHECKPOINT_SHA256 == driver.SEED_BINDINGS[42][
        "pool_checkpoint_sha256"
    ]
    assert driver.CHECKPOINT_SOURCE_COMMIT == driver.SEED_BINDINGS[42][
        "checkpoint_source_commit"
    ]
    assert driver.CHECKPOINT_GENERATION == "maf_cic_seed42_execution"
    assert driver.CHECKPOINT_TRAINING_JOB == "311675"
    assert driver.LINEAGE_SCHEMA.endswith("seed42.lineage-evidence.v1")


@pytest.mark.parametrize("seed", [41, 44, 48, True, 42.0, "43"])
def test_unsupported_seeds_rejected(seed):
    with pytest.raises(ValueError, match="unsupported experimental seed"):
        driver.evaluation_binding(seed)


def test_apply_binding_43_sets_all_identities():
    try:
        binding = driver._apply_binding(43)
        assert binding == SEED43
        assert driver.EXPERIMENTAL_SEED == 43
        assert driver.POOL_MANIFEST_SHA256 == SEED43["pool_manifest_sha256"]
        assert driver.POOL_CHECKPOINT_SHA256 == SEED43[
            "pool_checkpoint_sha256"
        ]
        assert driver.CHECKPOINT_SOURCE_COMMIT == SEED43[
            "checkpoint_source_commit"
        ]
        assert driver.CHECKPOINT_GENERATION == "maf_cic_seed43_execution"
        assert driver.CHECKPOINT_FITTED_STATE_SHA256 == SEED43[
            "checkpoint_fitted_state_sha256"
        ]
        assert driver.CHECKPOINT_EXECUTION_MANIFEST_SHA256 == SEED43[
            "checkpoint_execution_manifest_sha256"
        ]
        assert driver.CHECKPOINT_EXPECTATIONS_SHA256 == SEED43[
            "checkpoint_expectations_sha256"
        ]
        assert driver.CHECKPOINT_TRAINING_JOB == "314075"
        assert driver.LINEAGE_SCHEMA == SEED43["lineage_schema"]
    finally:
        driver._apply_binding(42)
    assert driver.EXPERIMENTAL_SEED == 42
    assert driver.CHECKPOINT_GENERATION == "maf_cic_seed42_execution"


BASE_ARGUMENTS = (
    "--phase", "train",
    "--repository-root", ".",
    "--expected-head", "a" * 40,
    "--dataset-config", "config.json",
    "--config-sha256", "b" * 64,
    "--pool-manifest", "pool.json",
    "--pool-manifest-sha256", "c" * 64,
    "--checkpoint-publication-parent", "checkpoints",
    "--checkpoint-generation-name", "maf_cic_seed43_execution",
    "--checkpoint-expectations", "expectations.json",
    "--checkpoint-expectations-sha256", "d" * 64,
    "--weights", "weights.h5",
    "--run-dir", "run",
)


def test_parser_defaults_to_seed42():
    assert driver._parser().parse_args(BASE_ARGUMENTS).experimental_seed == 42


def test_parser_accepts_seed43():
    arguments = driver._parser().parse_args(
        BASE_ARGUMENTS + ("--experimental-seed", "43")
    )
    assert arguments.experimental_seed == 43


@pytest.mark.parametrize("seed", ["41", "44", "48"])
def test_parser_rejects_other_seeds(seed):
    with pytest.raises(SystemExit):
        driver._parser().parse_args(
            BASE_ARGUMENTS + ("--experimental-seed", seed)
        )


def test_generator_receipt_requires_seed43_identity(monkeypatch):
    try:
        driver._apply_binding(43)
        raw = (json.dumps(_receipt(43), sort_keys=True) + "\n").encode()
        digest = driver._sha(raw)
        monkeypatch.setattr(driver, "CHECKPOINT_EXPECTATIONS_SHA256", digest)
        assert driver._authenticate_expectations(
            raw, digest, driver.DATASET_CONFIG_SHA256,
        )["generation_name"] == "maf_cic_seed43_execution"

        seed42 = (json.dumps(_receipt(42), sort_keys=True) + "\n").encode()
        seed42_digest = driver._sha(seed42)
        monkeypatch.setattr(
            driver, "CHECKPOINT_EXPECTATIONS_SHA256", seed42_digest,
        )
        with pytest.raises(
            ValueError, match="checkpoint execution receipt mismatch",
        ):
            driver._authenticate_expectations(
                seed42, seed42_digest, driver.DATASET_CONFIG_SHA256,
            )

        wrong_job = _receipt(43)
        wrong_job["job_id"] = "311675"
        wrong_raw = (json.dumps(wrong_job, sort_keys=True) + "\n").encode()
        wrong_digest = driver._sha(wrong_raw)
        monkeypatch.setattr(driver, "CHECKPOINT_EXPECTATIONS_SHA256", wrong_digest)
        with pytest.raises(
            ValueError, match="checkpoint execution receipt mismatch",
        ):
            driver._authenticate_expectations(
                wrong_raw, wrong_digest, driver.DATASET_CONFIG_SHA256,
            )
    finally:
        driver._apply_binding(42)


def test_pool_manifest_requires_seed43_identity(monkeypatch):
    try:
        driver._apply_binding(43)
        manifest = _manifest(43)
        raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        digest = driver._sha(raw)
        monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", digest)
        assert driver._authenticate_pool_manifest_raw(
            raw,
            expected_sha=digest,
            expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )["experimental_seed"] == 43

        seed42 = _manifest(42)
        seed42_raw = (json.dumps(seed42, sort_keys=True) + "\n").encode()
        seed42_digest = driver._sha(seed42_raw)
        monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", seed42_digest)
        with pytest.raises(ValueError, match="pool manifest identity mismatch"):
            driver._authenticate_pool_manifest_raw(
                seed42_raw,
                expected_sha=seed42_digest,
                expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
                expected_recipe=driver.GENERATOR_RECIPE_SHA256,
            )

        wrong = _manifest(43, checkpoint="0" * 64)
        wrong_raw = (json.dumps(wrong, sort_keys=True) + "\n").encode()
        wrong_digest = driver._sha(wrong_raw)
        monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", wrong_digest)
        with pytest.raises(ValueError, match="pool checkpoint differs"):
            driver._authenticate_pool_manifest_raw(
                wrong_raw,
                expected_sha=wrong_digest,
                expected_checkpoint="0" * 64,
                expected_recipe=driver.GENERATOR_RECIPE_SHA256,
            )
    finally:
        driver._apply_binding(42)


def test_train_propagates_seed43_and_keeps_shared_initialization(
    tmp_path, monkeypatch,
):
    try:
        driver._apply_binding(43)
        session = HELPERS["_fake_session"]({})
        session["lineage"]["schema"] = driver.LINEAGE_SCHEMA
        written = []
        paired, _published = HELPERS["_patch_train"](
            monkeypatch, session, written,
        )
        captured = {}

        def prepare(**kwargs):
            captured.update(kwargs)
            return paired

        monkeypatch.setattr(
            paper01_classifier_tensorflow,
            "prepare_paired_classifiers", prepare,
        )
        run = tmp_path / "run"
        driver._run_train(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
        ))
        assert captured["evaluator_seed"] == 43
        receipt = json.loads((run / "expectations.json").read_text())
        context = json.loads((run / "context_evidence.json").read_text())
        train_report = json.loads((run / "train_report.json").read_text())
        assert receipt["synthetic_pool_manifest_sha256"] == SEED43[
            "pool_manifest_sha256"
        ]
        assert train_report["scope"].startswith("CIC_MAF_SEED43_")
        assert context["synthetic_pool_identity"]["checkpoint_sha256"] == (
            SEED43["pool_checkpoint_sha256"]
        )
        for name in ("real_only", "real_plus_synthetic"):
            arm = receipt["arms"][name]["context"]
            assert arm["evaluator_seed"] == 43
            identity = context[
                "real_training_identity" if name == "real_only"
                else "augmented_training_identity"
            ]
            assert arm["training_identity_sha256"] == (
                paper01_augmented_checkpoint._sha(
                    paper01_augmented_checkpoint._canonical(identity)
                )
            )
        reports = train_report["arms"]
        assert reports["real_only"]["initial_weights_sha256"] == (
            reports["real_plus_synthetic"]["initial_weights_sha256"]
        )
    finally:
        driver._apply_binding(42)


def test_seven_required_groups_unchanged():
    assert set(REQUIRED_RESULTS) == {
        "real_only_downstream",
        "real_plus_synthetic_downstream",
        "paired_downstream",
        "pooled_balanced_kid",
        "pooled_balanced_fid",
        "class_fid",
        "intra_class_ms_ssim",
    }
    assert len(REQUIRED_RESULTS) == 7


def test_recipe_and_validator_modules_untouched_by_binding():
    recipe = Path(
        "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
    ).read_bytes()
    assert hashlib.sha256(recipe).hexdigest() == (
        "1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982"
    )
    assert driver.GENERATOR_RECIPE_SHA256 == (
        "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
    )
    assert driver.METRIC_RECIPE_SHA256 == (
        "d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85"
    )
