"""Caller orchestration checks; training and live capture use stand-ins.

Envelope validation and filesystem publication/loading use real components.
These checks do not establish real-model execution through the new caller.
"""

from copy import deepcopy
import json

import pytest

from test_ar_maf_cic_execution_manifest_revision_2026 import fixture
from trustforge.models.paper01_revision_2026 import (
    ar_maf_cic_execution as execution,
)


def setup_caller(fixture, monkeypatch, *, change=False):
    arguments = fixture[0]
    record = json.loads(arguments["fitted_state_raw"])
    calls = []

    def capture(**kwargs):
        calls.append("capture")
        result = deepcopy(record)
        if change and calls.count("capture") == 2:
            result["sha256"] = "e" * 64
        return result

    def train(**kwargs):
        calls.append("train")
        assert kwargs["dataset_id"] == "cicmaldroid2020"
        assert kwargs["bounded_integration"] is True
        return {
            "manifest": json.loads(arguments["checkpoint_manifest_raw"]),
            "manifest_raw": arguments["checkpoint_manifest_raw"],
            "recipe_raw": arguments["recipe_raw"],
            "best_raw": arguments["best_raw"],
            "last_raw": arguments["last_raw"],
        }

    monkeypatch.setattr(execution, "_capture", capture)
    monkeypatch.setattr(
        execution.runner, "run_training_and_prepare_checkpoint", train,
    )
    return calls


def prepare():
    return execution.prepare_cic_execution(
        repository_root="fixture",
        family="ar", experimental_seed=42, adapter=object(),
        dataset_config_raw=b"fixture",
        bounded_integration=True,
    )


def test_caller_enforces_capture_train_capture_and_publication(
    fixture, monkeypatch, tmp_path,
):
    calls = setup_caller(fixture, monkeypatch)
    prepared = prepare()
    assert calls == ["capture", "train", "capture"]
    assert prepared["publication_status"] == "NOT_PUBLISHED"
    tmp_path.chmod(0o700)
    result = execution.publish_and_read_cic_execution(
        tmp_path, "execution", prepared=prepared,
    )
    assert result["verified"]["fitted_state_raw"] == prepared["fitted_state_raw"]


def test_changed_preprocessing_blocks_preparation(fixture, monkeypatch):
    calls = setup_caller(fixture, monkeypatch, change=True)
    with pytest.raises(RuntimeError, match="changed during training"):
        prepare()
    assert calls == ["capture", "train", "capture"]


def test_default_authorization_stays_disabled(monkeypatch):
    monkeypatch.setattr(execution.runner, "TRAINING_EXECUTION_ENABLED", False)

    def forbidden(**kwargs):
        raise AssertionError("capture must not run")

    monkeypatch.setattr(execution, "_capture", forbidden)
    with pytest.raises(RuntimeError, match="execution remains disabled"):
        execution.prepare_cic_execution(
            repository_root="fixture", family="ar",
            experimental_seed=42, adapter=object(),
            dataset_config_raw=b"fixture",
        )


def test_bounded_scope_not_expanded():
    with pytest.raises(RuntimeError, match="bounded CIC seed42"):
        execution.prepare_cic_execution(
            repository_root="fixture", family="ar",
            experimental_seed=43, adapter=object(),
            dataset_config_raw=b"fixture", bounded_integration=True,
        )


def test_publication_requires_fitted_expectation():
    with pytest.raises(ValueError, match="expectation required"):
        execution.publish_and_read_cic_execution(
            "unused", "unused", prepared={"requested": {}},
        )
