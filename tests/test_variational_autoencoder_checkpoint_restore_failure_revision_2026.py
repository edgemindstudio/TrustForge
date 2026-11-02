"""Failure-containment tests for isolated VAE restoration.

Only synthetic, untrained model states are used.
"""

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as state_module,
    variational_autoencoder_checkpoint_restore as restore,
)


@pytest.fixture
def source_state():
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

    return state


def snapshot(network):
    return tuple(
        variable.numpy().copy()
        for variable in network.variables
    )


def assert_unchanged(network, expected):
    assert len(network.variables) == len(expected)

    for variable, original in zip(
        network.variables,
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(
            variable.numpy(),
            original,
        )


def test_failed_restore_does_not_modify_existing_models(
    source_state,
    monkeypatch,
):
    operational_encoder, operational_decoder = (
        model.build_models(
            latent_dim=100,
            num_classes=9,
        )
    )

    before_encoder = snapshot(operational_encoder)
    before_decoder = snapshot(operational_decoder)

    original_factory = model.build_models

    class InjectedAssignmentFailure(RuntimeError):
        pass

    def failing_factory(*, latent_dim, num_classes):
        encoder, decoder = original_factory(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        target = decoder.variables[0]

        def fail_assign(value):
            raise InjectedAssignmentFailure(
                "injected decoder assignment failure"
            )

        monkeypatch.setattr(target, "assign", fail_assign)

        return encoder, decoder

    monkeypatch.setattr(
        model,
        "build_models",
        failing_factory,
    )

    with pytest.raises(
        InjectedAssignmentFailure,
        match="injected decoder assignment failure",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=source_state,
        )

    assert_unchanged(
        operational_encoder,
        before_encoder,
    )
    assert_unchanged(
        operational_decoder,
        before_decoder,
    )


def test_readback_mismatch_prevents_success(
    source_state,
    monkeypatch,
):
    original_capture = state_module.capture_vae_state
    call_count = 0

    def corrupted_capture(*, encoder, decoder):
        nonlocal call_count
        call_count += 1

        captured = original_capture(
            encoder=encoder,
            decoder=decoder,
        )

        key = next(iter(captured["decoder"]))

        captured["decoder"][key].flat[0] += np.float32(1.0)

        return captured

    monkeypatch.setattr(
        state_module,
        "capture_vae_state",
        corrupted_capture,
    )

    with pytest.raises(AssertionError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=source_state,
        )

    assert call_count == 1


def test_invalid_state_rejected_before_assignments(
    source_state,
    monkeypatch,
):
    original_factory = model.build_models
    assignment_calls = []

    def tracking_factory(*, latent_dim, num_classes):
        encoder, decoder = original_factory(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        for variable in (
            *encoder.variables,
            *decoder.variables,
        ):
            original_assign = variable.assign

            def tracking_assign(
                value,
                *,
                _original=original_assign,
            ):
                assignment_calls.append(True)
                return _original(value)

            monkeypatch.setattr(
                variable,
                "assign",
                tracking_assign,
            )

        return encoder, decoder

    monkeypatch.setattr(
        model,
        "build_models",
        tracking_factory,
    )

    damaged = {
        role: {
            key: value.copy()
            for key, value in tensors.items()
        }
        for role, tensors in source_state.items()
    }

    key = next(iter(damaged["decoder"]))
    damaged["decoder"][key].flat[0] = np.nan

    with pytest.raises(ValueError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=damaged,
        )

    assert assignment_calls == []
