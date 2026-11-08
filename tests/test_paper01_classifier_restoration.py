from pathlib import Path
import runpy
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_bundle import publish_classifier_bundle
from trustforge.evaluation.paper01_classifier_state import decode_classifier_weights
import trustforge.evaluation.paper01_classifier_restoration as restoration

HELPERS = runpy.run_path(str(
    Path(__file__).with_name("test_paper01_classifier_bundle.py")
))


class Model:
    def __init__(self, weights, fault=None):
        self.weights = [np.zeros_like(value) for value in weights]
        self.fault = fault
        self.assignments = 0
        if fault == "shape":
            self.weights[0] = self.weights[0].reshape(-1)

    def get_weights(self):
        return [value.copy() for value in self.weights]

    def set_weights(self, values):
        self.assignments += 1
        if self.fault != "assignment":
            self.weights = [value.copy() for value in values]


def setup(tmp_path, monkeypatch, fault=None):
    prepared = HELPERS["package"]()
    parent = tmp_path / "publications"
    parent.mkdir(mode=0o700)
    publish_classifier_bundle(parent, "generation", **prepared)
    weights = decode_classifier_weights(
        raw=prepared["state_raw"],
        dataset_id="cicmaldroid2020",
        recipe_raw=prepared["recipe_raw"],
        expected_selected_weights_sha256=prepared["expected_selected_weights_sha256"],
    )
    model = Model(weights, fault)
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {
            "model": model,
            "controls": {
                "optimizer_updates": lambda: 1 if fault == "optimizer" else 0,
            },
        }

    monkeypatch.setattr(restoration, "require_compute_node", lambda: None)
    monkeypatch.setattr(restoration, "build_classifier_binding", build)
    return parent / "generation", prepared, model, calls


def restore(directory, prepared):
    return restoration.restore_selected_classifier(
        directory,
        **{
            key: value for key, value in prepared.items()
            if key not in ("manifest_raw", "state_raw")
        },
    )


def test_verified_assignment_and_inference_only_scope(tmp_path, monkeypatch):
    directory, prepared, model, calls = setup(tmp_path, monkeypatch)
    result = restore(directory, prepared)
    assert model.assignments == 1
    assert calls[0]["class_count"] == 5
    assert calls[0]["evaluator_seed"] == 42
    assert result["report"]["selected_weight_assignment_verified"] is True
    assert result["report"]["optimizer_state_restored"] is False
    assert result["report"]["training_continuation_authorized"] is False
    assert result["report"]["complete_evaluation"] is False


@pytest.mark.parametrize("fault", ["expectation", "state"])
def test_invalid_bundle_rejected_before_model_construction(
    tmp_path, monkeypatch, fault,
):
    directory, prepared, model, calls = setup(tmp_path, monkeypatch)
    if fault == "expectation":
        prepared["expected_selected_weights_sha256"] = "0" * 64
    else:
        (directory / "selected_state.npz").write_bytes(b"invalid")
    with pytest.raises(ValueError):
        restore(directory, prepared)
    assert calls == []
    assert model.assignments == 0


def test_compute_guard_precedes_model_construction(tmp_path, monkeypatch):
    directory, prepared, model, calls = setup(tmp_path, monkeypatch)

    def reject():
        raise RuntimeError("fixture allocation absent")

    monkeypatch.setattr(restoration, "require_compute_node", reject)
    with pytest.raises(RuntimeError):
        restore(directory, prepared)
    assert calls == []
    assert model.assignments == 0


@pytest.mark.parametrize("fault", ["shape", "assignment", "optimizer"])
def test_assignment_contract_failures_rejected(tmp_path, monkeypatch, fault):
    directory, prepared, model, calls = setup(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError)):
        restore(directory, prepared)
    assert len(calls) == 1
    if fault == "shape":
        assert model.assignments == 0


def test_restoration_uses_returned_verified_bytes(tmp_path, monkeypatch):
    directory, prepared, model, calls = setup(tmp_path, monkeypatch)
    original = restoration.read_classifier_bundle

    def read_then_remove(*args, **kwargs):
        loaded = original(*args, **kwargs)
        # Temporary fixture only: prove assignment does not reopen this path.
        (directory / "selected_state.npz").unlink()
        return loaded

    monkeypatch.setattr(restoration, "read_classifier_bundle", read_then_remove)
    result = restore(directory, prepared)
    assert result["report"]["selected_weight_assignment_verified"] is True
    assert model.assignments == 1
