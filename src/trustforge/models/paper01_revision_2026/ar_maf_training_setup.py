"""Prepare selected AR/MAF training objects after dataset preflight.

No optimizer steps, checkpoint publication or execution authorization.
Not a production training entry point.
"""

import json

from .ar_maf_verified_inputs import preflight_and_bind_inputs


def prepare_training_setup(
    *, family, dataset_id, experimental_seed, adapter, recipe_raw,
):
    """Bind verified inputs before constructing any model or optimizer."""
    checked, evidence = preflight_and_bind_inputs(
        family=family, dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        adapter=adapter, recipe_raw=recipe_raw,
    )

    # Preflight validates exact selected recipe bytes and experimental identity.
    recipe = json.loads(recipe_raw)
    identity = evidence["identity"]
    classes = identity["class_count"]

    import tensorflow as tf
    from .ar_maf_family_steps import bind_family_steps

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("selected integration requires CPU-only execution")
    tf.keras.utils.set_random_seed(experimental_seed)

    # Build with a small canonical fixture, not a selected training batch.
    x = tf.zeros((1, 40, 40, 3), dtype=tf.float32)
    if family == "ar":
        from .autoregressive import build_revision_pixelcnn
        controls = recipe["autoregressive"]
        model = build_revision_pixelcnn(
            (40, 40, 3), classes,
            filters=controls["filters"],
            masked_layers=controls["masked_layers"],
        )
        y = tf.one_hot([0], classes, dtype=tf.float32)
        model([x, y], training=False).numpy()
    elif family == "maf":
        from .masked_autoflow_model import ConditionalBlockMAF
        model = ConditionalBlockMAF(
            classes,
            permutation_seeds=recipe["masked_autoflow"]["permutation_seeds"],
        )
        model.log_prob(
            x, tf.constant([0], dtype=tf.int32), training=False,
        ).numpy()
    else:
        raise ValueError("unsupported family")

    controls = recipe["optimizer"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=controls["learning_rate"],
        beta_1=controls["beta_1"],
        beta_2=controls["beta_2"],
        epsilon=controls["epsilon"],
        amsgrad=controls["amsgrad"],
    )
    if int(optimizer.iterations.numpy()) != 0:
        raise RuntimeError("new optimizer already has updates")

    train, validate = bind_family_steps(
        family=family, model=model, optimizer=optimizer,
    )
    return {
        "adapter": checked,
        "evidence": evidence,
        "model": model,
        "optimizer": optimizer,
        "training_step": train,
        "validation_step": validate,
        "status": "PREPARED_NOT_TRAINED",
    }
