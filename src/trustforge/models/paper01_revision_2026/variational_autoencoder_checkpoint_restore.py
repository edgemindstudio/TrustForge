"""Prepared-state validation for VAE checkpoint restoration.

Paper 1 revision 2026.

This module prepares an already decoded paired checkpoint
state for a separately governed restoration operation.

It performs no model-variable assignments, filesystem
operations, training, checkpoint publication, or promotion.

Architecture identity and restoration authorization must
be established separately.
"""

from __future__ import annotations

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_state import (
    ROLE_ENCODER,
    ROLE_DECODER,
    validate_vae_state,
    variable_specification,
)


def prepare_vae_checkpoint_restoration(
    *,
    encoder,
    decoder,
    state,
):
    """Validate both complete states without modifying models.

    Returns detached arrays ordered by the target models'
    positional variable specifications.

    The result is an engineering preparation artifact,
    not permission to assign variables.
    """
    validated = validate_vae_state(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    prepared = {}

    for role, network in (
        (ROLE_ENCODER, encoder),
        (ROLE_DECODER, decoder),
    ):
        specification = variable_specification(
            network,
            role=role,
        )

        network_state = validated[role]

        prepared[role] = {
            "specification": tuple(
                {
                    "key": entry["key"],
                    "shape": tuple(entry["shape"]),
                    "dtype": entry["dtype"],
                    "trainable": entry["trainable"],
                }
                for entry in specification
            ),
            "values": tuple(
                network_state[entry["key"]].copy()
                for entry in specification
            ),
        }

    return prepared


def restore_vae_state_into_isolated_models(
    *,
    num_classes,
    state=None,
    latent_dim=100,
    pre_assignment_validator=None,
    verified_state_decoder=None,
):
    """Restore decoded state into private, newly built VAE models.

    Both states are validated before any assignment.

    The model pair is returned only after all assignments
    and exact readback verification succeed.

    This is an engineering restoration operation, not
    checkpoint provenance verification, deployment approval,
    publication authorization, or training authorization.
    """
    import numpy as np

    from trustforge.models.paper01_revision_2026.variational_autoencoder_model import (
        build_models,
    )
    from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_state import (
        capture_vae_state,
    )

    if (state is None) == (verified_state_decoder is None):
        raise ValueError(
            "exactly one of state or verified_state_decoder "
            "must be supplied"
        )

    if (
        pre_assignment_validator is not None
        and not callable(pre_assignment_validator)
    ):
        raise TypeError(
            "pre_assignment_validator must be callable"
        )

    if verified_state_decoder is not None:
        if not callable(verified_state_decoder):
            raise TypeError(
                "verified_state_decoder must be callable"
            )

        if pre_assignment_validator is None:
            raise ValueError(
                "verified-byte decoding requires "
                "a pre-assignment validator"
            )

    encoder, decoder = build_models(
        latent_dim=latent_dim,
        num_classes=num_classes,
    )

    if verified_state_decoder is not None:
        # Validate the actual model architecture before decoding.
        pre_assignment_validator(
            encoder=encoder,
            decoder=decoder,
        )

        state = verified_state_decoder(
            encoder=encoder,
            decoder=decoder,
        )

    prepared = prepare_vae_checkpoint_restoration(
        encoder=encoder,
        decoder=decoder,
        state=state,
    )

    if (
        verified_state_decoder is None
        and pre_assignment_validator is not None
    ):
        # Preserve the original decoded-state validation order.
        pre_assignment_validator(
            encoder=encoder,
            decoder=decoder,
        )

    for role, network in (
        (ROLE_ENCODER, encoder),
        (ROLE_DECODER, decoder),
    ):
        variables = tuple(network.variables)
        values = prepared[role]["values"]

        if len(variables) != len(values):
            raise RuntimeError(
                f"restoration variable-count mismatch: {role}"
            )

        for variable, value in zip(
            variables,
            values,
            strict=True,
        ):
            variable.assign(value)

    restored = capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    for role in (ROLE_ENCODER, ROLE_DECODER):
        specification = prepared[role]["specification"]

        for entry, expected in zip(
            specification,
            prepared[role]["values"],
            strict=True,
        ):
            np.testing.assert_array_equal(
                restored[role][entry["key"]],
                expected,
                err_msg=(
                    f"restoration verification failed: "
                    f"{role}:{entry['key']}"
                ),
            )

    return encoder, decoder


__all__ = [
    "prepare_vae_checkpoint_restoration",
    "restore_vae_state_into_isolated_models",
]
