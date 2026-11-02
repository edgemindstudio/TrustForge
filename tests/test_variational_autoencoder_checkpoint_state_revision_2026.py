"""Checkpoint state-integrity tests for Paper 1 revised VAE."""

from copy import deepcopy

import numpy as np
import pytest

import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as checkpoint,
)


def make_case(num_classes=9):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=num_classes,
    )

    state = checkpoint.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )
    return encoder, decoder, state


def snapshot(encoder, decoder):
    return [
        v.numpy().copy()
        for v in list(encoder.variables) + list(decoder.variables)
    ]


def assert_unchanged(before, encoder, decoder):
    after = snapshot(encoder, decoder)
    assert len(before) == len(after)

    for left, right in zip(before, after):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("num_classes", [2, 5, 9, 47])
def test_complete_capture_for_all_class_counts(num_classes):
    encoder, decoder, state = make_case(num_classes)

    assert set(state) == {"encoder", "decoder"}
    assert len(state["encoder"]) == 16
    assert len(state["decoder"]) == 8

    validated = checkpoint.validate_vae_state(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    for role in ("encoder", "decoder"):
        assert set(validated[role]) == set(state[role])

        for key in state[role]:
            np.testing.assert_array_equal(
                validated[role][key],
                state[role][key],
            )


def test_batchnorm_nontrainable_state_included():
    encoder, decoder, state = make_case()

    specification = checkpoint.variable_specification(
        encoder,
        role="encoder",
    )

    assert specification[6]["key"] == "encoder_0006"
    assert specification[7]["key"] == "encoder_0007"
    assert specification[6]["trainable"] is False
    assert specification[7]["trainable"] is False

    for key in ("encoder_0006", "encoder_0007"):
        assert key in state["encoder"]
        assert state["encoder"][key].shape == (64,)


def test_capture_and_validation_do_not_mutate_models():
    encoder, decoder, state = make_case()
    before = snapshot(encoder, decoder)

    checkpoint.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )
    checkpoint.validate_vae_state(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    assert_unchanged(before, encoder, decoder)


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_missing_tensor_rejected(role):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    bad[role].pop(next(iter(bad[role])))

    with pytest.raises(ValueError, match="inventory mismatch"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_extra_tensor_rejected(role):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    bad[role][f"{role}_9999"] = np.zeros(1, np.float32)

    with pytest.raises(ValueError, match="inventory mismatch"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_wrong_dtype_rejected(role):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    key = next(iter(bad[role]))
    bad[role][key] = bad[role][key].astype(np.float64)

    with pytest.raises(TypeError, match="dtype mismatch"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_wrong_shape_rejected(role):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    key = next(iter(bad[role]))
    bad[role][key] = bad[role][key].reshape(-1)

    with pytest.raises(ValueError, match="shape mismatch"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


@pytest.mark.parametrize("role", ["encoder", "decoder"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_tensor_rejected(role, invalid):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    key = next(iter(bad[role]))
    bad[role][key].flat[0] = invalid

    with pytest.raises(ValueError, match="nonfinite"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


@pytest.mark.parametrize("missing_role", ["encoder", "decoder"])
def test_incomplete_pair_rejected(missing_role):
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    bad.pop(missing_role)

    with pytest.raises(ValueError, match="encoder and decoder"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


def test_unexpected_pair_role_rejected():
    encoder, decoder, state = make_case()
    bad = deepcopy(state)
    bad["optimizer"] = {}

    with pytest.raises(ValueError, match="encoder and decoder"):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )


def test_corrupt_decoder_rejection_does_not_modify_encoder():
    encoder, decoder, state = make_case()
    before = snapshot(encoder, decoder)

    bad = deepcopy(state)
    bad["decoder"]["decoder_0000"].flat[0] = float("nan")

    with pytest.raises(ValueError):
        checkpoint.validate_vae_state(
            encoder=encoder,
            decoder=decoder,
            state=bad,
        )

    assert_unchanged(before, encoder, decoder)


def test_validation_returns_detached_copies():
    encoder, decoder, state = make_case()

    validated = checkpoint.validate_vae_state(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    key = "encoder_0000"
    original = state["encoder"][key].flat[0].copy()

    validated["encoder"][key].flat[0] += 1.0

    assert state["encoder"][key].flat[0] == original


def test_class_count_shape_mismatch_rejected():
    encoder9, decoder9, state9 = make_case(num_classes=9)

    encoder5, decoder5, _ = make_case(num_classes=5)

    with pytest.raises(ValueError, match="shape mismatch"):
        checkpoint.validate_vae_state(
            encoder=encoder5,
            decoder=decoder5,
            state=state9,
        )


def test_invalid_network_rejected():
    with pytest.raises(TypeError):
        checkpoint.variable_specification(
            object(),
            role="encoder",
        )


def test_invalid_role_rejected():
    encoder, _, _ = make_case()

    with pytest.raises(ValueError):
        checkpoint.variable_specification(
            encoder,
            role="generator",
        )
