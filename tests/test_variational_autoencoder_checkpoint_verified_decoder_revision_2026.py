"""Verified-byte VAE restoration callback regressions.

All models and state archives are disposable and untrained.
No real datasets, training, or checkpoint publication.
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
    variational_autoencoder_checkpoint_npz as npz,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)


DATASET_ID = "ustc_tfc2016"
CLASS_COUNT = 9


def make_archives():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=CLASS_COUNT,
    )

    state = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    archives = {}

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        archives[role] = npz.encode_network_state(
            role=role,
            state=state[role],
            specification=specification,
        )

    return state, archives


def track_factory(monkeypatch):
    original = model.build_models
    constructed = []
    assignments = []

    def factory(*, latent_dim, num_classes):
        encoder, decoder = original(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        constructed.append((encoder, decoder))

        for role, network in (
            ("encoder", encoder),
            ("decoder", decoder),
        ):
            for index, variable in enumerate(network.variables):
                original_assign = variable.assign

                def tracking_assign(
                    value,
                    *,
                    _assign=original_assign,
                    _role=role,
                    _index=index,
                ):
                    assignments.append((_role, _index))
                    return _assign(value)

                monkeypatch.setattr(
                    variable,
                    "assign",
                    tracking_assign,
                )

        return encoder, decoder

    monkeypatch.setattr(model, "build_models", factory)

    return constructed, assignments


def validate_architecture(*, encoder, decoder):
    return registry.validate_vae_architecture_fingerprint(
        dataset_id=DATASET_ID,
        encoder=encoder,
        decoder=decoder,
    )


def decode_archives(archives, *, encoder, decoder):
    result = {}

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        result[role] = npz.decode_network_state(
            role=role,
            raw=archives[role],
            specification=specification,
        )

    return result


def test_fingerprint_rejection_prevents_decoder_call(monkeypatch):
    _, archives = make_archives()
    constructed, assignments = track_factory(monkeypatch)
    decoder_calls = []

    def reject(*, encoder, decoder):
        raise ValueError("injected fingerprint rejection")

    def forbidden_decoder(*, encoder, decoder):
        decoder_calls.append(True)
        raise AssertionError("decoder must not run")

    with pytest.raises(
        ValueError,
        match="injected fingerprint rejection",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=CLASS_COUNT,
            pre_assignment_validator=reject,
            verified_state_decoder=forbidden_decoder,
        )

    assert len(constructed) == 1
    assert decoder_calls == []
    assert assignments == []


def test_decoder_exception_prevents_assignment(monkeypatch):
    _, archives = make_archives()
    constructed, assignments = track_factory(monkeypatch)
    events = []

    def validate(*, encoder, decoder):
        events.append("fingerprint")
        validate_architecture(
            encoder=encoder,
            decoder=decoder,
        )

    def fail_decode(*, encoder, decoder):
        events.append("decode")
        raise ValueError("injected decoder failure")

    with pytest.raises(
        ValueError,
        match="injected decoder failure",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=CLASS_COUNT,
            pre_assignment_validator=validate,
            verified_state_decoder=fail_decode,
        )

    assert len(constructed) == 1
    assert events == ["fingerprint", "decode"]
    assert assignments == []


def test_malformed_decoded_state_prevents_assignment(monkeypatch):
    _, archives = make_archives()
    constructed, assignments = track_factory(monkeypatch)

    def malformed(*, encoder, decoder):
        result = decode_archives(
            archives,
            encoder=encoder,
            decoder=decoder,
        )
        result["decoder"] = {}
        return result

    with pytest.raises(ValueError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=CLASS_COUNT,
            pre_assignment_validator=validate_architecture,
            verified_state_decoder=malformed,
        )

    assert len(constructed) == 1
    assert assignments == []


def test_state_and_decoder_mutually_exclusive(monkeypatch):
    state, _ = make_archives()
    constructed, assignments = track_factory(monkeypatch)

    with pytest.raises(
        ValueError,
        match="exactly one of state or verified_state_decoder",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=CLASS_COUNT,
            state=state,
            pre_assignment_validator=validate_architecture,
            verified_state_decoder=lambda **kwargs: state,
        )

    assert constructed == []
    assert assignments == []


def test_decoder_requires_validator(monkeypatch):
    _, archives = make_archives()
    constructed, assignments = track_factory(monkeypatch)

    with pytest.raises(
        ValueError,
        match="requires a pre-assignment validator",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=CLASS_COUNT,
            verified_state_decoder=lambda **kwargs: {},
        )

    assert constructed == []
    assert assignments == []


def test_canonical_archives_restore_exactly(monkeypatch):
    expected, archives = make_archives()
    constructed, assignments = track_factory(monkeypatch)
    events = []

    def validate(*, encoder, decoder):
        events.append("fingerprint")
        return validate_architecture(
            encoder=encoder,
            decoder=decoder,
        )

    def decode(*, encoder, decoder):
        events.append("decode")
        assert len(constructed) == 1
        assert encoder is constructed[0][0]
        assert decoder is constructed[0][1]
        assert assignments == []

        return decode_archives(
            archives,
            encoder=encoder,
            decoder=decoder,
        )

    encoder, decoder = restore.restore_vae_state_into_isolated_models(
        num_classes=CLASS_COUNT,
        pre_assignment_validator=validate,
        verified_state_decoder=decode,
    )

    assert len(constructed) == 1
    assert events == ["fingerprint", "decode"]
    assert len(assignments) == 24

    restored = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    for role in ("encoder", "decoder"):
        for key, value in expected[role].items():
            np.testing.assert_array_equal(
                restored[role][key],
                value,
            )
