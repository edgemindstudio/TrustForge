"""Reference-to-feature/KID orchestration.

Checks declared identities and runtime batch agreement. Original real-data
provenance, generator lineage and reference persistence timing remain outer
requirements. Does not establish complete evaluation.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .paper01_inputs import CLASS_COUNTS, validate_batch
from .paper01_reference_selection import select_balanced_references
from .paper01_kid import kid_subsets
from .paper01_inception import WEIGHTS_SHA256


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


class _LabelCheckedView:
    def __init__(self, view, labels, classes):
        self._view = view
        self._labels = labels.copy()
        self._classes = classes
        self._identity = deepcopy(view.identity)
        self._size = view.size

    @property
    def size(self):
        return self._size

    @property
    def identity(self):
        if self._view.identity != self._identity or self._view.size != self._size:
            raise RuntimeError("feature view identity changed")
        return deepcopy(self._identity)

    def batch(self, positions):
        self.identity
        observed, images, labels = self._view.batch(positions.copy())
        if not np.array_equal(observed, positions):
            raise RuntimeError("feature view changed position order")
        validate_batch(images, labels, count=len(positions), class_count=self._classes)
        if not np.array_equal(labels.argmax(axis=1), self._labels[positions]):
            raise RuntimeError("selected image labels differ from reference labels")
        return observed, images, labels


def _collect(extractor, view, positions):
    result = np.empty((len(positions), 2048), dtype=np.float32)
    offset = 0
    for observed, features in extractor.iter_selected(view=view, positions=positions):
        count = len(observed)
        if (
            not 1 <= count <= 16
            or offset + count > len(positions)
            or not np.array_equal(observed, positions[offset:offset + count])
            or not isinstance(features, np.ndarray)
            or features.shape != (count, 2048)
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise ValueError("invalid or misordered feature batch")
        result[offset:offset + count] = features
        offset += count
    if offset != len(positions):
        raise RuntimeError("incomplete selected feature coverage")
    return result


def prepare_features_and_kid(
    *, selection_raw, selection_sha256,
    real_validation_view, synthetic_view,
    real_validation_labels, synthetic_labels, extractor,
):
    """Consume an externally persisted selection and return features/KID."""
    if (
        type(selection_raw) is not bytes
        or type(selection_sha256) is not str
        or hashlib.sha256(selection_raw).hexdigest() != selection_sha256
    ):
        raise ValueError("persisted selection file identity mismatch")
    selection = json.loads(selection_raw)
    real_identity = deepcopy(real_validation_view.identity)
    synthetic_identity = deepcopy(synthetic_view.identity)
    dataset = real_identity.get("dataset_id")
    if (
        dataset not in CLASS_COUNTS
        or real_identity.get("split") != "validation"
        or synthetic_identity.get("dataset_id") != dataset
    ):
        raise ValueError("incompatible reference input views")
    classes = CLASS_COUNTS[dataset]
    if (
        real_identity.get("class_count") != classes
        or real_validation_view.size != len(real_validation_labels)
        or synthetic_view.size != len(synthetic_labels)
    ):
        raise ValueError("reference view class/count mismatch")

    recomputed = select_balanced_references(
        real_validation_labels=real_validation_labels,
        synthetic_labels=synthetic_labels,
        class_count=classes,
        real_validation_identity_sha256=hashlib.sha256(
            _canonical(real_identity)
        ).hexdigest(),
        synthetic_manifest_sha256=synthetic_identity["manifest_sha256"],
    )
    if _canonical(selection) != _canonical(recomputed):
        raise ValueError("persisted reference differs from governed selection")

    feature_identity = deepcopy(extractor.identity)
    required = {
        "architecture": "Keras_InceptionV3",
        "include_top": False,
        "pooling": "avg",
        "weights_sha256": WEIGHTS_SHA256,
        "feature_dimension": 2048,
        "preserve_rgb": True,
        "training": False,
        "resize": {
            "shape": [299, 299], "method": "bilinear", "antialias": False,
        },
        "preprocessing": "multiply_255_then_inception_v3_preprocess_input",
    }
    if any(
        type(feature_identity.get(key)) is not type(value)
        or feature_identity[key] != value
        for key, value in required.items()
    ):
        raise ValueError("feature extractor differs from selected contract")

    # Verify the complete pool before selected image access.
    synthetic_view.preflight()
    real_checked = _LabelCheckedView(
        real_validation_view, real_validation_labels, classes,
    )
    synthetic_checked = _LabelCheckedView(synthetic_view, synthetic_labels, classes)

    def positions(reference):
        return np.asarray(reference["positions_by_class"], dtype=np.int64).reshape(-1)

    real_positions = positions(selection["real_reference"])
    synthetic_positions = positions(selection["synthetic_reference"])
    real_features = _collect(extractor, real_checked, real_positions)
    synthetic_features = _collect(extractor, synthetic_checked, synthetic_positions)
    # Exhaustion and final identity checks occur before the estimator.
    real_checked.identity
    synthetic_checked.identity
    if extractor.identity != feature_identity:
        raise RuntimeError("feature extractor identity changed")
    kid = kid_subsets(real_features, synthetic_features)

    report = {
        "metric": "pooled_class_balanced_kid",
        "selection_file_sha256": selection_sha256,
        "real_reference_sha256": selection["real_reference_sha256"],
        "synthetic_reference_sha256": selection["synthetic_reference_sha256"],
        "dataset_id": dataset,
        "class_count": classes,
        "samples_per_class": selection["real_reference"]["samples_per_class"],
        "feature_extractor": feature_identity,
        "feature_shape": list(real_features.shape),
        "real_features_sha256": hashlib.sha256(
            real_features.tobytes(order="C")
        ).hexdigest(),
        "synthetic_features_sha256": hashlib.sha256(
            synthetic_features.tobytes(order="C")
        ).hexdigest(),
        "kid": kid,
        "complete_evaluation": False,
    }
    return {
        "real_features": real_features,
        "synthetic_features": synthetic_features,
        "report": report,
    }
