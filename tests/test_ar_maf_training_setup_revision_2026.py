import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_training_setup as setup
from trustforge.models.paper01_revision_2026 import ar_maf_family_steps as steps


def recipe_bytes():
    return (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026/"
          "ar_maf_integration_recipe.json"
    ).read_bytes()


def test_preflight_failure_prevents_tensorflow_import(monkeypatch):
    def fail(**kwargs):
        raise ValueError("source preflight failed")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise AssertionError("TensorFlow reached before successful preflight")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(setup, "preflight_and_bind_inputs", fail)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="source preflight failed"):
        setup.prepare_training_setup(
            family="ar", dataset_id="malex", experimental_seed=42,
            adapter=object(), recipe_raw=recipe_bytes(),
        )


@pytest.fixture
def environment(monkeypatch):
    events = []
    original_adapter = object()
    checked_adapter = object()
    evidence = {"identity": {"class_count": 2}}
    options = {"gpu": False, "iterations": 0}

    def preflight(**kwargs):
        assert kwargs["adapter"] is original_adapter
        events.append(("preflight",))
        return checked_adapter, evidence

    def forbidden_step(*args, **kwargs):
        raise AssertionError("setup executed a batch step")

    model = SimpleNamespace(
        log_prob=lambda *args, **kwargs: (
            events.append(("build_maf", kwargs)),
            SimpleNamespace(numpy=lambda: 0.0),
        )[1],
    )

    class ARModel:
        def __call__(self, *args, **kwargs):
            events.append(("build_ar", kwargs))
            return SimpleNamespace(numpy=lambda: 0.0)

    def construct_ar(*args, **kwargs):
        events.append(("ar_constructor", args, kwargs))
        return ARModel()

    def construct_maf(*args, **kwargs):
        events.append(("maf_constructor", args, kwargs))
        return model

    def adam(**kwargs):
        events.append(("optimizer", kwargs))
        return SimpleNamespace(iterations=SimpleNamespace(
            numpy=lambda: options["iterations"],
        ))

    def bind(**kwargs):
        events.append(("bind", kwargs))
        return forbidden_step, forbidden_step

    fake_tf = SimpleNamespace(
        float32="float32", int32="int32",
        zeros=lambda *args, **kwargs: "images",
        one_hot=lambda *args, **kwargs: "onehot",
        constant=lambda *args, **kwargs: "integer_labels",
        config=SimpleNamespace(
            list_physical_devices=lambda kind: ["GPU"] if options["gpu"] else [],
        ),
        keras=SimpleNamespace(
            utils=SimpleNamespace(
                set_random_seed=lambda seed: events.append(("seed", seed)),
            ),
            optimizers=SimpleNamespace(Adam=adam),
        ),
    )
    prefix = "trustforge.models.paper01_revision_2026."
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(
        sys.modules, prefix + "autoregressive",
        SimpleNamespace(build_revision_pixelcnn=construct_ar),
    )
    monkeypatch.setitem(
        sys.modules, prefix + "masked_autoflow_model",
        SimpleNamespace(ConditionalBlockMAF=construct_maf),
    )
    monkeypatch.setattr(setup, "preflight_and_bind_inputs", preflight)
    monkeypatch.setattr(steps, "bind_family_steps", bind)
    return events, options, original_adapter, checked_adapter, evidence


@pytest.mark.parametrize("family", ["ar", "maf"])
def test_selected_setup_without_training(environment, family):
    events, _, original, checked, evidence = environment
    raw = recipe_bytes()
    recipe = json.loads(raw)
    result = setup.prepare_training_setup(
        family=family, dataset_id="malex", experimental_seed=43,
        adapter=original, recipe_raw=raw,
    )
    assert events[0] == ("preflight",)
    assert ("seed", 43) in events
    assert result["adapter"] is checked
    assert result["evidence"] is evidence
    assert result["status"] == "PREPARED_NOT_TRAINED"
    optimizer = next(event[1] for event in events if event[0] == "optimizer")
    assert optimizer == {
        key: recipe["optimizer"][key]
        for key in ("learning_rate", "beta_1", "beta_2", "epsilon", "amsgrad")
    }
    constructor = next(
        event for event in events if event[0] == family + "_constructor"
    )
    if family == "ar":
        assert constructor[1] == ((40, 40, 3), 2)
        assert constructor[2] == {"filters": 64, "masked_layers": 6}
    else:
        assert constructor[1] == (2,)
        assert constructor[2] == {"permutation_seeds": [42, 43, 44, 45, 46]}
    bound = next(event[1] for event in events if event[0] == "bind")
    assert bound["model"] is result["model"]
    assert bound["optimizer"] is result["optimizer"]


def test_gpu_rejected_before_model_construction(environment):
    events, options, original, _, _ = environment
    options["gpu"] = True
    with pytest.raises(RuntimeError, match="CPU-only"):
        setup.prepare_training_setup(
            family="ar", dataset_id="malex", experimental_seed=42,
            adapter=original, recipe_raw=recipe_bytes(),
        )
    assert events == [("preflight",)]


def test_nonzero_new_optimizer_rejected_before_binding(environment):
    events, options, original, _, _ = environment
    options["iterations"] = 1
    with pytest.raises(RuntimeError, match="already has updates"):
        setup.prepare_training_setup(
            family="maf", dataset_id="malex", experimental_seed=42,
            adapter=original, recipe_raw=recipe_bytes(),
        )
    assert not any(event[0] == "bind" for event in events)
