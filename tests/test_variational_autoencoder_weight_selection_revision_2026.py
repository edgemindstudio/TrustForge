"""Independent VAE weight-selection regression tests."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.variational_autoencoder_weight_selection import (
    capture_vae_weights,
    consider_vae_best_weights,
)


class FakeModel:
    def __init__(self, value):
        self.weights = [
            np.array([value], dtype=np.float32)
        ]

    def get_weights(self):
        return self.weights


def observe(current, epoch, loss, encoder, decoder):
    return consider_vae_best_weights(
        current=current,
        epoch=epoch,
        validation_total_loss=loss,
        encoder=encoder,
        decoder=decoder,
    )


def test_snapshot_detached_from_live_weights():
    encoder = FakeModel(1)
    decoder = FakeModel(2)

    snapshot = capture_vae_weights(
        encoder=encoder,
        decoder=decoder,
    )

    encoder.weights[0][0] = 9
    decoder.weights[0][0] = 8

    assert snapshot["encoder"][0][0] == 1
    assert snapshot["decoder"][0][0] == 2


def test_best_epoch_weights_survive_later_training():
    encoder = FakeModel(1)
    decoder = FakeModel(2)

    best = observe(None, 0, 3.0, encoder, decoder)

    encoder.weights[0][0] = 7
    decoder.weights[0][0] = 8

    best = observe(best, 1, 4.0, encoder, decoder)

    assert best["best_epoch"] == 0
    assert best["encoder_weights"][0][0] == 1
    assert best["decoder_weights"][0][0] == 2


def test_strict_improvement_replaces_best_snapshot():
    encoder = FakeModel(1)
    decoder = FakeModel(2)

    best = observe(None, 0, 3.0, encoder, decoder)

    encoder.weights[0][0] = 5
    decoder.weights[0][0] = 6

    best = observe(best, 1, 2.0, encoder, decoder)

    assert best["best_epoch"] == 1
    assert best["encoder_weights"][0][0] == 5
    assert best["decoder_weights"][0][0] == 6


def test_tie_preserves_earlier_weights():
    encoder = FakeModel(1)
    decoder = FakeModel(2)

    best = observe(None, 0, 2.0, encoder, decoder)

    encoder.weights[0][0] = 7

    selected = observe(best, 1, 2.0, encoder, decoder)

    assert selected is best
    assert selected["best_epoch"] == 0


def test_nonfinite_loss_rejected():
    for loss in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            observe(None, 0, loss, FakeModel(1), FakeModel(2))


def test_nonfinite_weights_rejected():
    with pytest.raises(ValueError):
        capture_vae_weights(
            encoder=FakeModel(float("nan")),
            decoder=FakeModel(2),
        )


def test_out_of_order_epoch_rejected():
    best = observe(None, 2, 3.0, FakeModel(1), FakeModel(2))

    with pytest.raises(ValueError):
        observe(best, 1, 2.0, FakeModel(3), FakeModel(4))


def test_snapshot_arrays_are_read_only():
    snapshot = capture_vae_weights(
        encoder=FakeModel(1),
        decoder=FakeModel(2),
    )

    assert not snapshot["encoder"][0].flags.writeable
    assert not snapshot["decoder"][0].flags.writeable
