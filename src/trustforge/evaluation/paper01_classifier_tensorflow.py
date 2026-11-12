"""Compute-node EvalCNN binding and paired training preparation.

No dataset provenance authentication, test evaluation, artifact publication,
scheduler submission or complete-evaluation authorization.
"""

import os
import socket

import numpy as np

from .paper01_classifier_control import (
    selected_recipe, fit_classifier_control, state_digest, snapshot,
)
from .paper01_inputs import CLASS_COUNTS, validate_batch


def require_compute_node():
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute-node classifier execution required")


def build_classifier_binding(*, recipe_raw, class_count, evaluator_seed):
    recipe = selected_recipe(recipe_raw)
    if type(class_count) is not int or class_count not in set(CLASS_COUNTS.values()):
        raise ValueError("unsupported authoritative class count")
    if type(evaluator_seed) is not int or evaluator_seed not in recipe["experimental_seeds"]:
        raise ValueError("unsupported evaluator seed")
    require_compute_node()

    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("selected benchmark requires CPU-only execution")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(evaluator_seed)
    tf.config.experimental.enable_op_determinism()

    architecture = recipe["architecture"]
    selection = recipe["implementation_selection"]
    network = selection["network"]
    dtype = network["compute_dtype"]

    def kernel():
        return tf.keras.initializers.GlorotUniform()

    layers = [tf.keras.Input(
        shape=tuple(recipe["canonical_input"]["shape"]), dtype=dtype,
    )]
    for width in architecture["convolution_widths"]:
        layers.extend((
            tf.keras.layers.Conv2D(
                width,
                tuple(architecture["convolution_kernel"]),
                strides=tuple(network["convolution_strides"]),
                padding=architecture["convolution_padding"],
                activation=architecture["convolution_activation"],
                use_bias=network["biases_enabled"],
                kernel_initializer=kernel(),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                dtype=dtype,
            ),
            tf.keras.layers.MaxPooling2D(
                pool_size=tuple(architecture["pool_after_each_convolution"]["size"]),
                strides=tuple(architecture["pool_after_each_convolution"]["strides"]),
                padding=network["pool_padding"],
                dtype=dtype,
            ),
        ))
    layers.extend((
        tf.keras.layers.GlobalAveragePooling2D(dtype=dtype),
        tf.keras.layers.Dropout(
            architecture["dropout_rate"], seed=evaluator_seed, dtype=dtype,
        ),
        tf.keras.layers.Dense(
            class_count, activation="softmax",
            use_bias=network["biases_enabled"],
            kernel_initializer=kernel(),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, dtype=dtype,
        ),
    ))
    model = tf.keras.Sequential(layers)
    model(tf.zeros((1, 40, 40, 3), dtype=tf.float32), training=False)

    controls = selection["optimizer"]
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=controls["learning_rate"],
        beta_1=controls["beta_1"],
        beta_2=controls["beta_2"],
        epsilon=controls["epsilon"],
        amsgrad=controls["amsgrad"],
        weight_decay=controls["weight_decay"],
        clipnorm=None, clipvalue=None, global_clipnorm=None,
        use_ema=controls["use_ema"],
        gradient_accumulation_steps=None,
    )
    # Build slots before validation-state snapshots.
    optimizer.build(model.trainable_variables)
    loss_controls = selection["loss"]
    loss_function = tf.keras.losses.CategoricalCrossentropy(
        from_logits=loss_controls["from_logits"],
        label_smoothing=loss_controls["label_smoothing"],
        reduction="none",
    )

    def inputs(images, labels):
        validate_batch(
            images, labels, count=len(images), class_count=class_count,
        )
        return (
            tf.convert_to_tensor(images, dtype=tf.float32),
            tf.convert_to_tensor(labels, dtype=tf.float32),
        )

    def mean_loss(images, labels, training):
        predictions = model(images, training=training)
        tf.debugging.assert_type(predictions, tf.float32)
        tf.debugging.assert_all_finite(predictions, "nonfinite classifier scores")
        losses = loss_function(labels, predictions)
        tf.debugging.assert_all_finite(losses, "nonfinite classifier losses")
        return tf.reduce_mean(losses)

    def train_step(images, labels):
        images, labels = inputs(images, labels)
        with tf.GradientTape() as tape:
            loss = mean_loss(images, labels, True)
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(value is None for value in gradients):
            raise RuntimeError("missing classifier gradient")
        for value in gradients:
            tf.debugging.assert_all_finite(value, "nonfinite classifier gradient")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return float(loss.numpy())

    def validation_step(images, labels):
        images, labels = inputs(images, labels)
        return float(mean_loss(images, labels, False).numpy())

    def execution_state():
        return [
            variable.numpy()
            for variable in (*model.variables, *optimizer.variables)
        ]

    return {
        "model": model,
        "optimizer": optimizer,
        "initial_weights_sha256": state_digest(model.get_weights()),
        "controls": {
            "train_step": train_step,
            "validation_step": validation_step,
            "get_weights": model.get_weights,
            "set_weights": model.set_weights,
            "get_execution_state": execution_state,
            "optimizer_updates": lambda: int(optimizer.iterations.numpy()),
        },
    }


def prepare_paired_classifiers(
    *, real_training, augmented_training, validation,
    evaluator_seed, recipe_raw,
):
    """Return selected arm weights/reports; held-out test is not an input.

    Initial implementation requires the existing CIC combined-input contract.
    Provenance, lineage and primary-pool validation remain outer requirements.
    """
    from .paper01_inputs import RealSplitView
    from .paper01_combined_inputs import CombinedTrainingView

    recipe = selected_recipe(recipe_raw)
    require_compute_node()
    if (
        not isinstance(real_training, RealSplitView)
        or not isinstance(validation, RealSplitView)
        or not isinstance(augmented_training, CombinedTrainingView)
    ):
        raise TypeError("governed paired input views required")
    if augmented_training.identity["real"] != real_training.identity:
        raise ValueError("augmented arm uses a different real training view")
    if (
        validation.identity["split"] != "validation"
        or real_training.identity["split"] != "train"
        or validation.identity["dataset_id"] != real_training.identity["dataset_id"]
        or validation.identity["declared_provenance_sha256"]
        != real_training.identity["declared_provenance_sha256"]
        or augmented_training.identity["synthetic"]["experimental_seed"]
        != evaluator_seed
    ):
        raise ValueError("paired dataset, provenance or seed mismatch")

    classes = real_training.identity["class_count"]
    result = {}
    initial_digest = None
    for arm, training in (
        ("real_only", real_training),
        ("real_plus_synthetic", augmented_training),
    ):
        binding = build_classifier_binding(
            recipe_raw=recipe_raw, class_count=classes,
            evaluator_seed=evaluator_seed,
        )
        observed_digest = binding["initial_weights_sha256"]
        if initial_digest is None:
            initial_digest = observed_digest
        if observed_digest != initial_digest:
            raise RuntimeError("paired initial classifier weights differ")
        consumption = {} if arm == "real_plus_synthetic" else None
        report = fit_classifier_control(
            training=training, validation=validation,
            evaluator_seed=evaluator_seed, recipe_raw=recipe_raw,
            pool_consumption=consumption, **binding["controls"],
        )
        if report["initial_weights_sha256"] != initial_digest:
            raise RuntimeError("classifier changed before controlled training")
        result[arm] = {
            "report": report,
            "selected_weights": snapshot(binding["model"].get_weights()),
        }
        if consumption is not None:
            result[arm]["pool_consumption"] = consumption
        del binding

    return {
        "status": "PAIRED_WEIGHTS_PREPARED_NOT_EVALUATED",
        "evaluator_seed": evaluator_seed,
        "initial_weights_sha256": initial_digest,
        "class_count": classes,
        "maximum_epochs": recipe["training"]["maximum_epochs"],
        "arms": result,
        "test_evaluated": False,
        "complete_evaluation": False,
    }
