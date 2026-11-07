"""Selected three-scale MS-SSIM scoring for bounded canonical RGB pairs.

No image conversion, pair selection, pool authentication or completeness
authorization. TensorFlow's channel reduction semantics are preserved.
"""

from copy import deepcopy
import numpy as np

from .paper01_classifier_tensorflow import require_compute_node


MAX_PAIR_BATCH = 16
SETTINGS = {
    "metric": "ms_ssim_3scale_40x40_v1",
    "input": "float32_NHWC_40x40x3_[0,1]",
    "max_val": 1.0,
    "filter_size": 7,
    "filter_sigma": 1.5,
    "k1": 0.01,
    "k2": 0.03,
    "power_factors": [1 / 3, 1 / 3, 1 / 3],
    "channel_reduction": "tensorflow_ssim_multiscale",
    "maximum_pair_batch": MAX_PAIR_BATCH,
}


def _images(value):
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.ndim != 4
        or value.shape[1:] != (40, 40, 3)
        or not 1 <= len(value) <= MAX_PAIR_BATCH
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or np.any(value > 1)
    ):
        raise ValueError("bounded canonical float32 RGB batch required")
    return value


class ThreeScaleMSSSIM:
    def __init__(self):
        require_compute_node()
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            raise RuntimeError("selected benchmark requires CPU-only execution")
        tf.config.experimental.enable_op_determinism()
        self._identity = {
            **deepcopy(SETTINGS),
            "tensorflow": tf.__version__,
        }

    @property
    def identity(self):
        return deepcopy(self._identity)

    def score_pairs(self, first, second):
        first, second = _images(first), _images(second)
        if first.shape != second.shape:
            raise ValueError("paired image batch shapes differ")
        require_compute_node()
        import tensorflow as tf

        result = tf.image.ssim_multiscale(
            first,
            second,
            max_val=1.0,
            power_factors=(1 / 3, 1 / 3, 1 / 3),
            filter_size=7,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
        ).numpy()
        if (
            result.shape != (len(first),)
            or result.dtype != np.float32
            or not np.isfinite(result).all()
        ):
            raise ValueError("invalid or nonfinite MS-SSIM scores")
        return result.copy()
