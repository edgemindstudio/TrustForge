"""Synthetic-only VAE restoration preparation regression tests.

Preparation must validate both networks without changing model
variables, publishing checkpoints, or performing training.
"""

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as state_module,
    variational_autoencoder_checkpoint_restore as restore,
)


@pytest.fixture
def prepared_models():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=9,
    )

    state = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    return encoder, decoder, state


def snapshot_variables(network):
    return tuple(
        variable.numpy().copy()
        for variable in network.variables
    )


def assert_variables_unchanged(network, before):
    assert len(network.variables) == len(before)

    for variable, expected in zip(
        network.variables,
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(
            variable.numpy(),
            expected,
        )


def test_valid_paired_state_preparation(prepared_models):
    encoder, decoder, state = prepared_models

    prepared = restore.prepare_vae_checkpoint_restoration(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    assert set(prepared) == {"encoder", "decoder"}

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        actual = prepared[role]

        assert actual["specification"] == specification
        assert len(actual["values"]) == len(specification)

        for entry, array in zip(
            specification,
            actual["values"],
            strict=True,
        ):
            np.testing.assert_array_equal(
                array,
                state[role][entry["key"]],
            )


def test_preparation_does_not_modify_models(prepared_models):
    encoder, decoder, state = prepared_models

    encoder_before = snapshot_variables(encoder)
    decoder_before = snapshot_variables(decoder)

    restore.prepare_vae_checkpoint_restoration(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    assert_variables_unchanged(
        encoder,
        encoder_before,
    )
    assert_variables_unchanged(
        decoder,
        decoder_before,
    )


def test_prepared_values_are_detached(prepared_models):
    encoder, decoder, state = prepared_models

    prepared = restore.prepare_vae_checkpoint_restoration(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    for role in ("encoder", "decoder"):
        for entry, array in zip(
            prepared[role]["specification"],
            prepared[role]["values"],
            strict=True,
        ):
            original = state[role][entry["key"]]

            assert not np.shares_memory(array, original)

            np.testing.assert_array_equal(
                array,
                original,
            )


@pytest.mark.parametrize(
    "missing_role",
    ["encoder", "decoder"],
)
def test_missing_network_state_rejected(
    prepared_models,
    missing_role,
):
    encoder, decoder, state = prepared_models

    incomplete = {
        role: tensors
        for role, tensors in state.items()
        if role != missing_role
    }

    with pytest.raises(ValueError):
        restore.prepare_vae_checkpoint_restoration(
            encoder=encoder,
            decoder=decoder,
            state=incomplete,
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf],
)
def test_invalid_decoder_rejected_without_mutation(
    prepared_models,
    bad_value,
):
    encoder, decoder, state = prepared_models

    encoder_before = snapshot_variables(encoder)
    decoder_before = snapshot_variables(decoder)

    damaged = {
        role: {
            key: array.copy()
            for key, array in tensors.items()
        }
        for role, tensors in state.items()
    }

    key = next(iter(damaged["decoder"]))
    damaged["decoder"][key].flat[0] = bad_value

    with pytest.raises(ValueError):
        restore.prepare_vae_checkpoint_restoration(
            encoder=encoder,
            decoder=decoder,
            state=damaged,
        )

    assert_variables_unchanged(
        encoder,
        encoder_before,
    )
    assert_variables_unchanged(
        decoder,
        decoder_before,
    )


def test_decoder_inventory_failure_does_not_mutate_encoder(
    prepared_models,
):
    encoder, decoder, state = prepared_models

    encoder_before = snapshot_variables(encoder)
    decoder_before = snapshot_variables(decoder)

    incomplete_decoder = dict(state["decoder"])
    incomplete_decoder.pop(next(iter(incomplete_decoder)))

    with pytest.raises(ValueError):
        restore.prepare_vae_checkpoint_restoration(
            encoder=encoder,
            decoder=decoder,
            state={
                "encoder": state["encoder"],
                "decoder": incomplete_decoder,
            },
        )

    assert_variables_unchanged(
        encoder,
        encoder_before,
    )
    assert_variables_unchanged(
        decoder,
        decoder_before,
    )


def test_invalid_encoder_shape_rejected(prepared_models):
    encoder, decoder, state = prepared_models

    damaged_encoder = dict(state["encoder"])
    key = next(iter(damaged_encoder))
    damaged_encoder[key] = np.zeros(
        (1,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        restore.prepare_vae_checkpoint_restoration(
            encoder=encoder,
            decoder=decoder,
            state={
                "encoder": damaged_encoder,
                "decoder": state["decoder"],
            },
        )


def test_invalid_decoder_dtype_rejected(prepared_models):
    encoder, decoder, state = prepared_models

    damaged_decoder = dict(state["decoder"])
    key = next(iter(damaged_decoder))
    damaged_decoder[key] = damaged_decoder[key].astype(
        np.float64
    )

    with pytest.raises(TypeError):
        restore.prepare_vae_checkpoint_restoration(
            encoder=encoder,
            decoder=decoder,
            state={
                "encoder": state["encoder"],
                "decoder": damaged_decoder,
            },
        )
