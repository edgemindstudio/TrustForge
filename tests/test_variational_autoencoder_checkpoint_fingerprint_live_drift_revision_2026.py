"""Live-network VAE architecture drift regression tests.

Disposable, untrained Keras models only.
No real dataset reads, training, or checkpoint publication.
"""

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)


def make_models():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    return model.build_models(
        latent_dim=100,
        num_classes=9,
    )


def clone_with_changed_activation(
    network,
    *,
    layer_name,
    previous,
    replacement,
):
    def clone_layer(layer):
        config = layer.get_config()

        if layer.name == layer_name:
            assert config["activation"] == previous
            config["activation"] = replacement

        return layer.__class__.from_config(config)

    result = tf.keras.models.clone_model(
        network,
        clone_function=clone_layer,
    )

    assert (
        result.get_layer(layer_name).activation.__name__
        == replacement
    )

    assert [
        tuple(variable.shape)
        for variable in network.weights
    ] == [
        tuple(variable.shape)
        for variable in result.weights
    ]

    return result


@pytest.mark.parametrize(
    "role,layer_name,previous,replacement",
    [
        (
            "encoder",
            "encoder_conv1",
            "relu",
            "linear",
        ),
        (
            "decoder",
            "decoder_mean",
            "tanh",
            "sigmoid",
        ),
    ],
)
def test_live_architecture_drift_rejected(
    role,
    layer_name,
    previous,
    replacement,
):
    encoder, decoder = make_models()

    baseline = registry.validate_vae_architecture_fingerprint(
        dataset_id="ustc_tfc2016",
        encoder=encoder,
        decoder=decoder,
    )

    if role == "encoder":
        altered_encoder = clone_with_changed_activation(
            encoder,
            layer_name=layer_name,
            previous=previous,
            replacement=replacement,
        )
        altered_decoder = decoder
    else:
        altered_encoder = encoder
        altered_decoder = clone_with_changed_activation(
            decoder,
            layer_name=layer_name,
            previous=previous,
            replacement=replacement,
        )

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        registry.validate_vae_architecture_fingerprint(
            dataset_id="ustc_tfc2016",
            encoder=altered_encoder,
            decoder=altered_decoder,
        )

    assert (
        registry.validate_vae_architecture_fingerprint(
            dataset_id="ustc_tfc2016",
            encoder=encoder,
            decoder=decoder,
        )
        == baseline
    )
