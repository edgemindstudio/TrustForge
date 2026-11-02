"""Strict VAE pre-assignment restoration gate regressions.

Disposable, untrained model states only.
No real malware datasets, training, or checkpoint publication.
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
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)


def make_state(class_count=9):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    return state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )


def instrument_factory(monkeypatch):
    original = model.build_models
    assignments = []
    constructed = []

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
                assign = variable.assign

                def tracking_assign(
                    value,
                    *,
                    _assign=assign,
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

    monkeypatch.setattr(
        model,
        "build_models",
        factory,
    )

    return constructed, assignments


def test_rejected_validator_prevents_all_assignments(monkeypatch):
    state = make_state()
    constructed, assignments = instrument_factory(monkeypatch)
    seen = []

    class RejectedArchitecture(ValueError):
        pass

    def reject(*, encoder, decoder):
        seen.append((encoder, decoder))
        raise RejectedArchitecture("architecture rejected")

    with pytest.raises(
        RejectedArchitecture,
        match="architecture rejected",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=state,
            pre_assignment_validator=reject,
        )

    assert len(constructed) == 1
    assert len(seen) == 1
    assert seen[0][0] is constructed[0][0]
    assert seen[0][1] is constructed[0][1]
    assert assignments == []


def test_valid_fingerprint_allows_exact_restoration(monkeypatch):
    state = make_state()
    constructed, assignments = instrument_factory(monkeypatch)
    seen = []

    def accept(*, encoder, decoder):
        seen.append((encoder, decoder))

        return registry.validate_vae_architecture_fingerprint(
            dataset_id="ustc_tfc2016",
            encoder=encoder,
            decoder=decoder,
        )

    encoder, decoder = restore.restore_vae_state_into_isolated_models(
        num_classes=9,
        state=state,
        pre_assignment_validator=accept,
    )

    assert len(constructed) == 1
    assert len(seen) == 1
    assert encoder is constructed[0][0]
    assert decoder is constructed[0][1]
    assert len(assignments) == 24

    restored = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    for role in ("encoder", "decoder"):
        for key, expected in state[role].items():
            np.testing.assert_array_equal(
                restored[role][key],
                expected,
            )


def test_noncallable_validator_rejected_before_assignment(monkeypatch):
    state = make_state()
    constructed, assignments = instrument_factory(monkeypatch)

    with pytest.raises(
        TypeError,
        match="pre_assignment_validator must be callable",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=state,
            pre_assignment_validator="not-callable",
        )

    assert len(constructed) == 0
    assert assignments == []


def test_invalid_state_rejected_before_validator(monkeypatch):
    state = make_state()
    constructed, assignments = instrument_factory(monkeypatch)
    validator_calls = []

    damaged = {
        role: {
            key: value.copy()
            for key, value in tensors.items()
        }
        for role, tensors in state.items()
    }

    key = next(iter(damaged["decoder"]))
    damaged["decoder"][key].flat[0] = np.nan

    def validator(*, encoder, decoder):
        validator_calls.append(True)

    with pytest.raises(ValueError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=damaged,
            pre_assignment_validator=validator,
        )

    assert len(constructed) == 1
    assert validator_calls == []
    assert assignments == []
