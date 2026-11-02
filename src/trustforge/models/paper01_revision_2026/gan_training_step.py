"""Governed explicit-gradient GAN training iteration.

Historical conditional DCGAN architecture is preserved.

One iteration performs:
1. discriminator real-batch optimizer update;
2. discriminator fake-batch optimizer update;
3. generator optimizer update.

A failed iteration is not a valid checkpoint boundary.
"""

from __future__ import annotations


def gan_training_step(
    *,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    real_images,
    real_labels,
    fake_labels,
    generator_labels,
    discriminator_noise,
    generator_noise,
):
    """Execute one adversarial iteration with strict input and gradient checks."""

    import tensorflow as tf

    image_shape = (40, 40, 3)

    def tensor(name, value, *, rank, trailing_shape=None):
        result = tf.convert_to_tensor(value, dtype=tf.float32)

        if result.shape.rank != rank:
            raise ValueError(f"{name}: invalid tensor rank")

        if trailing_shape is not None:
            if tuple(result.shape[1:]) != tuple(trailing_shape):
                raise ValueError(f"{name}: invalid tensor shape")

        tf.debugging.assert_all_finite(result, f"{name}: nonfinite values")
        return result

    real_images = tensor(
        "real_images",
        real_images,
        rank=4,
        trailing_shape=image_shape,
    )

    if bool(tf.reduce_any(real_images < -1.0).numpy()) or bool(
        tf.reduce_any(real_images > 1.0).numpy()
    ):
        raise ValueError("real_images outside internal [-1,1] domain")

    num_classes = int(generator.inputs[1].shape[-1])
    latent_dim = int(generator.inputs[0].shape[-1])
    batch_size = int(tf.shape(real_images)[0].numpy())

    if batch_size <= 0:
        raise ValueError("empty real batch")

    if int(discriminator.inputs[1].shape[-1]) != num_classes:
        raise ValueError("generator/discriminator class-count mismatch")

    def one_hot(name, value, expected_batch):
        result = tensor(
            name,
            value,
            rank=2,
            trailing_shape=(num_classes,),
        )

        if int(tf.shape(result)[0].numpy()) != expected_batch:
            raise ValueError(f"{name}: batch-size mismatch")

        if not bool(tf.reduce_all(
            tf.logical_or(result == 0.0, result == 1.0)
        ).numpy()):
            raise ValueError(f"{name}: non-binary one-hot encoding")

        if not bool(tf.reduce_all(
            tf.reduce_sum(result, axis=1) == 1.0
        ).numpy()):
            raise ValueError(f"{name}: invalid one-hot rows")

        return result

    real_labels = one_hot("real_labels", real_labels, batch_size)

    discriminator_noise = tensor(
        "discriminator_noise",
        discriminator_noise,
        rank=2,
        trailing_shape=(latent_dim,),
    )

    generator_noise = tensor(
        "generator_noise",
        generator_noise,
        rank=2,
        trailing_shape=(latent_dim,),
    )

    fake_batch_size = int(tf.shape(discriminator_noise)[0].numpy())
    generator_batch_size = int(tf.shape(generator_noise)[0].numpy())

    if fake_batch_size <= 0 or generator_batch_size <= 0:
        raise ValueError("noise batches must be nonempty")

    fake_labels = one_hot(
        "fake_labels", fake_labels, fake_batch_size
    )
    generator_labels = one_hot(
        "generator_labels", generator_labels, generator_batch_size
    )

    if fake_batch_size != batch_size or generator_batch_size != batch_size:
        raise ValueError("GAN iteration requires equal batch sizes")

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    def checked_update(name, loss, tape, variables, optimizer):
        tf.debugging.assert_all_finite(
            loss, f"{name}: nonfinite loss"
        )

        gradients = tape.gradient(loss, variables)

        if len(gradients) != len(variables):
            raise RuntimeError(f"{name}: gradient count mismatch")

        if any(gradient is None for gradient in gradients):
            raise RuntimeError(f"{name}: disconnected gradient")

        for gradient in gradients:
            tf.debugging.assert_all_finite(
                tf.convert_to_tensor(gradient),
                f"{name}: nonfinite gradient",
            )

        optimizer.apply_gradients(zip(gradients, variables))

    # Discriminator update 1: real batch.
    with tf.GradientTape() as tape:
        real_scores = discriminator(
            [real_images, real_labels], training=True
        )
        real_loss = loss_fn(tf.ones_like(real_scores), real_scores)

    checked_update(
        "discriminator_real",
        real_loss,
        tape,
        discriminator.trainable_variables,
        discriminator_optimizer,
    )

    # Discriminator update 2: fake batch.
    fake_images = tf.stop_gradient(
        generator(
            [discriminator_noise, fake_labels],
            training=False,
        )
    )

    tf.debugging.assert_all_finite(
        fake_images, "nonfinite generated discriminator images"
    )

    with tf.GradientTape() as tape:
        fake_scores = discriminator(
            [fake_images, fake_labels], training=True
        )
        fake_loss = loss_fn(tf.zeros_like(fake_scores), fake_scores)

    checked_update(
        "discriminator_fake",
        fake_loss,
        tape,
        discriminator.trainable_variables,
        discriminator_optimizer,
    )

    # Generator update: D participates in gradient propagation,
    # but its optimizer and moving statistics are not updated.
    with tf.GradientTape() as tape:
        generated = generator(
            [generator_noise, generator_labels],
            training=True,
        )
        scores = discriminator(
            [generated, generator_labels],
            training=False,
        )
        generator_loss = loss_fn(tf.ones_like(scores), scores)

    checked_update(
        "generator",
        generator_loss,
        tape,
        generator.trainable_variables,
        generator_optimizer,
    )

    return {
        "discriminator_real_loss": float(real_loss.numpy()),
        "discriminator_fake_loss": float(fake_loss.numpy()),
        "generator_loss": float(generator_loss.numpy()),
    }
