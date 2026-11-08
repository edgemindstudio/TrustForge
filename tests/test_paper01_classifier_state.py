from pathlib import Path
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_classifier_state import (
    classifier_specification, encode_classifier_weights, decode_classifier_weights,
)

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
).read_bytes()


def weights():
    return [
        np.full(entry["shape"], index / 10, dtype=np.float32)
        for index, entry in enumerate(classifier_specification(
            dataset_id="cicmaldroid2020", recipe_raw=RECIPE,
        ))
    ]


def encode(values, **overrides):
    args = {
        "weights": values,
        "dataset_id": "cicmaldroid2020",
        "recipe_raw": RECIPE,
        "expected_selected_weights_sha256": state_digest(values),
    }
    args.update(overrides)
    return encode_classifier_weights(**args)


def test_recipe_derived_shapes():
    spec = classifier_specification(dataset_id="cicmaldroid2020", recipe_raw=RECIPE)
    assert [entry["shape"] for entry in spec] == [
        [3, 3, 3, 32], [32], [3, 3, 32, 64], [64],
        [3, 3, 64, 128], [128], [128, 5], [5],
    ]


def test_deterministic_roundtrip_and_no_aliasing():
    values = weights()
    raw = encode(values)
    assert raw == encode(values)
    restored = decode_classifier_weights(
        raw=raw, dataset_id="cicmaldroid2020", recipe_raw=RECIPE,
        expected_selected_weights_sha256=state_digest(values),
    )
    for original, loaded in zip(values, restored):
        np.testing.assert_array_equal(original, loaded)
        assert not np.shares_memory(original, loaded)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite", "missing"])
def test_invalid_weight_inventory_rejected(fault):
    values = weights()
    if fault == "shape":
        values[0] = values[0].reshape(-1)
    elif fault == "dtype":
        values[0] = values[0].astype(np.float64)
    elif fault == "nonfinite":
        values[0].flat[0] = np.nan
    else:
        values.pop()
    with pytest.raises((ValueError, TypeError)):
        encode(values)


def test_wrong_selected_digest_rejected():
    with pytest.raises(ValueError):
        encode(weights(), expected_selected_weights_sha256="0" * 64)


def test_wrong_dataset_architecture_rejected_on_decode():
    values = weights()
    with pytest.raises(ValueError):
        decode_classifier_weights(
            raw=encode(values), dataset_id="malex", recipe_raw=RECIPE,
            expected_selected_weights_sha256=state_digest(values),
        )


def test_trailing_archive_bytes_rejected():
    values = weights()
    with pytest.raises(ValueError):
        decode_classifier_weights(
            raw=encode(values) + b"extra",
            dataset_id="cicmaldroid2020", recipe_raw=RECIPE,
            expected_selected_weights_sha256=state_digest(values),
        )


def test_changed_recipe_rejected():
    with pytest.raises(ValueError):
        encode(weights(), recipe_raw=RECIPE + b" ")
