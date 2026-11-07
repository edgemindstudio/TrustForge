"""Orchestration tests with stand-in features; no TensorFlow execution."""

import hashlib
import json

import numpy as np
import pytest

from trustforge.evaluation.paper01_fidelity import prepare_features_and_kid
from trustforge.evaluation.paper01_reference_selection import select_balanced_references
from trustforge.evaluation.paper01_inception import WEIGHTS_SHA256


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


class View:
    def __init__(self, labels, real=False):
        self.labels = labels
        self.size = len(labels)
        self.identity = {"dataset_id": "malex"}
        if real:
            self.identity.update(split="validation", class_count=2)
        else:
            self.identity["manifest_sha256"] = "b" * 64
        self.ready = real
        self.wrong_labels = False

    def preflight(self):
        self.ready = True

    def batch(self, positions):
        assert self.ready
        ids = self.labels[positions]
        images = np.broadcast_to(
            (ids.astype(np.float32) / 2)[:, None, None, None],
            (len(ids), 40, 40, 3),
        ).copy()
        if self.wrong_labels:
            ids = 1 - ids
        return positions.copy(), images, np.eye(2, dtype=np.float32)[ids]


class Extractor:
    identity = {
        "architecture": "Keras_InceptionV3", "include_top": False,
        "pooling": "avg", "weights_sha256": WEIGHTS_SHA256,
        "feature_dimension": 2048, "preserve_rgb": True, "training": False,
        "resize": {"shape": [299, 299], "method": "bilinear", "antialias": False},
        "preprocessing": "multiply_255_then_inception_v3_preprocess_input",
        "scope": "STAND_IN_FEATURES_NOT_AUTHENTICATED_INCEPTION",
    }

    def __init__(self, fault=None):
        self.fault = fault

    def iter_selected(self, *, view, positions):
        if self.fault == "missing":
            return
        for start in range(0, len(positions), 16):
            selected = positions[start:start + 16]
            _, images, _ = view.batch(selected)
            features = np.zeros((len(selected), 2048), dtype=np.float32)
            features[:, 0] = images[:, 0, 0, 0]
            if self.fault == "dtype":
                features = features.astype(np.float64)
            if self.fault == "order":
                selected = selected[::-1]
            yield selected, features


def inputs():
    real_labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    synthetic_labels = np.repeat(np.arange(2, dtype=np.int32), 2000)
    real = View(real_labels, real=True)
    synthetic = View(synthetic_labels)
    selection = select_balanced_references(
        real_validation_labels=real_labels,
        synthetic_labels=synthetic_labels,
        class_count=2,
        real_validation_identity_sha256=hashlib.sha256(
            canonical(real.identity)
        ).hexdigest(),
        synthetic_manifest_sha256=synthetic.identity["manifest_sha256"],
    )
    raw = canonical(selection)
    return {
        "selection_raw": raw,
        "selection_sha256": hashlib.sha256(raw).hexdigest(),
        "real_validation_view": real,
        "synthetic_view": synthetic,
        "real_validation_labels": real_labels,
        "synthetic_labels": synthetic_labels,
        "extractor": Extractor(),
    }


def test_complete_reference_feature_kid_orchestration():
    arguments = inputs()
    result = prepare_features_and_kid(**arguments)
    assert result["real_features"].shape == (4, 2048)
    assert result["synthetic_features"].shape == (4, 2048)
    assert result["report"]["samples_per_class"] == 2
    assert result["report"]["kid"]["subset_count"] == 10
    assert result["report"]["complete_evaluation"] is False
    assert arguments["synthetic_view"].ready


def test_changed_selection_bytes_rejected():
    arguments = inputs()
    arguments["selection_raw"] += b"\n"
    with pytest.raises(ValueError, match="selection file identity"):
        prepare_features_and_kid(**arguments)


def test_rehashed_changed_selection_rejected():
    arguments = inputs()
    selection = json.loads(arguments["selection_raw"])
    selection["real_reference"]["positions_by_class"][0][0] = 4
    raw = canonical(selection)
    arguments.update(selection_raw=raw, selection_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="governed selection"):
        prepare_features_and_kid(**arguments)


def test_selected_image_label_disagreement_rejected():
    arguments = inputs()
    arguments["synthetic_view"].wrong_labels = True
    with pytest.raises(RuntimeError, match="image labels differ"):
        prepare_features_and_kid(**arguments)


@pytest.mark.parametrize("fault", ["missing", "dtype", "order"])
def test_invalid_feature_stream_rejected(fault):
    arguments = inputs()
    arguments["extractor"] = Extractor(fault)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_features_and_kid(**arguments)
