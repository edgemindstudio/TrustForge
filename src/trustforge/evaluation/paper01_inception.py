"""Verified-weight, bounded RGB Inception features for Paper 1.

No downloads, source-data authentication, reference selection or metric
completeness claims. Iterator consumers must exhaust the stream and persist
selected sample/feature identities.
"""

from copy import deepcopy
import hashlib
from importlib import metadata
from pathlib import Path

import numpy as np

from .paper01_classifier_tensorflow import require_compute_node
from .paper01_inputs import CLASS_COUNTS, validate_batch


WEIGHTS_SHA256 = (
    "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
)
MAX_FEATURE_BATCH = 16


def verify_weight_file(path):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular weight artifact required")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError("weight artifact changed while reading")
    if after.st_size != 87910968 or digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("weight artifact identity mismatch")
    return path


def preprocess_canonical_images(images):
    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.float32
        or images.ndim != 4
        or images.shape[1:] != (40, 40, 3)
        or not 1 <= len(images) <= MAX_FEATURE_BATCH
        or not np.isfinite(images).all()
        or np.any(images < 0) or np.any(images > 1)
    ):
        raise ValueError("invalid bounded canonical RGB images")
    require_compute_node()
    import tensorflow as tf
    resized = tf.image.resize(
        tf.convert_to_tensor(images), (299, 299),
        method="bilinear", antialias=False,
    )
    return tf.keras.applications.inception_v3.preprocess_input(
        resized * tf.constant(255.0, dtype=tf.float32),
        data_format="channels_last",
    )


class InceptionFeatures:
    def __init__(self, *, weights_path):
        require_compute_node()
        path = verify_weight_file(weights_path)

        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("selected benchmark requires CPU-only execution")
        if (
            tf.keras.backend.image_data_format() != "channels_last"
            or tf.keras.mixed_precision.global_policy().name != "float32"
        ):
            raise RuntimeError("channels-last float32 policy required")
        tf.config.experimental.enable_op_determinism()
        # A verified explicit path prevents an implicit weight download.
        self._model = tf.keras.applications.InceptionV3(
            include_top=False, pooling="avg",
            input_shape=(299, 299, 3), weights=str(path),
        )
        self._model.trainable = False
        verify_weight_file(path)
        self._identity = {
            "schema": "trustforge.paper01.inception-features.v1",
            "architecture": "Keras_InceptionV3",
            "include_top": False,
            "pooling": "avg",
            "weights_sha256": WEIGHTS_SHA256,
            "weights_bytes": 87910968,
            "tensorflow": tf.__version__,
            "keras": metadata.version("keras"),
            "input": "float32_NHWC_40x40x3_[0,1]",
            "resize": {
                "shape": [299, 299],
                "method": "bilinear",
                "antialias": False,
            },
            "preprocessing": "multiply_255_then_inception_v3_preprocess_input",
            "preserve_rgb": True,
            "training": False,
            "feature_dimension": 2048,
            "maximum_batch_size": MAX_FEATURE_BATCH,
        }

    @property
    def identity(self):
        return deepcopy(self._identity)

    def features(self, images):
        values = self._model(
            preprocess_canonical_images(images), training=False,
        ).numpy()
        if (
            values.shape != (len(images), 2048)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise RuntimeError("invalid Inception feature output")
        return values

    def iter_selected(self, *, view, positions):
        """Yield requested positions/features in exact reference order."""
        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or positions.ndim != 1
            or not 1 <= len(positions) <= 47 * 200
            or np.unique(positions).size != len(positions)
            or type(view.size) is not int
            or np.any(positions < 0) or np.any(positions >= view.size)
        ):
            raise ValueError("invalid selected reference positions")
        selected = positions.copy()
        identity = deepcopy(view.identity)
        dataset = identity.get("dataset_id")
        if dataset not in CLASS_COUNTS:
            raise ValueError("unsupported feature input dataset")
        for start in range(0, len(selected), MAX_FEATURE_BATCH):
            if view.identity != identity:
                raise RuntimeError("feature input identity changed")
            requested = selected[start:start + MAX_FEATURE_BATCH]
            observed, images, labels = view.batch(requested.copy())
            if not np.array_equal(observed, requested):
                raise RuntimeError("feature input changed requested order")
            validate_batch(
                images, labels, count=len(requested),
                class_count=CLASS_COUNTS[dataset],
            )
            yield requested.copy(), self.features(images)
        if view.identity != identity:
            raise RuntimeError("feature input identity changed")
