"""In-memory VAE best-weight selection for synthetic development.

No filesystem checkpoint writes, publication, or real-data authorization.
"""

from __future__ import annotations

import math
import numpy as np


def capture_vae_weights(*, encoder, decoder):
    """Capture detached, finite copies of both model weight lists."""
    snapshot = {}

    for name, model in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        getter = getattr(model, "get_weights", None)

        if not callable(getter):
            raise TypeError(f"{name} must implement get_weights")

        weights = getter()

        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError(f"{name} has no valid weights")

        copied = []

        for weight in weights:
            array = np.asarray(weight)

            if not np.issubdtype(array.dtype, np.number):
                raise TypeError("model weight must be numeric")

            if not np.all(np.isfinite(array)):
                raise ValueError("model weight must be finite")

            cloned = array.copy()
            cloned.setflags(write=False)
            copied.append(cloned)

        snapshot[name] = tuple(copied)

    return snapshot


def consider_vae_best_weights(
    *,
    current,
    epoch,
    validation_total_loss,
    encoder,
    decoder,
):
    """Capture only strict improvements; earliest epoch wins ties."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")

    if type(validation_total_loss) not in (int, float):
        raise TypeError("validation loss must be numeric")

    loss = float(validation_total_loss)

    if not math.isfinite(loss):
        raise ValueError("validation loss must be finite")

    if current is not None:
        if not isinstance(current, dict):
            raise TypeError("current must be a selection dictionary")

        previous_epoch = current["best_epoch"]
        previous_loss = current["best_validation_total_loss"]

        if type(previous_epoch) is not int or epoch <= previous_epoch:
            raise ValueError("observations must advance in epoch order")

        if not math.isfinite(previous_loss):
            raise ValueError("previous loss must be finite")

        if loss >= previous_loss:
            return current

    weights = capture_vae_weights(
        encoder=encoder,
        decoder=decoder,
    )

    return {
        "best_epoch": epoch,
        "best_validation_total_loss": loss,
        "encoder_weights": weights["encoder"],
        "decoder_weights": weights["decoder"],
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


__all__ = [
    "capture_vae_weights",
    "consider_vae_best_weights",
]
