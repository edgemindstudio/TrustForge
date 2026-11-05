"""Family step bindings; no dataset authorization or checkpoint publication."""

import numpy as np

from .ar_maf_training_core import _loss


def _iteration(optimizer):
    value = np.asarray(optimizer.iterations.numpy())
    if value.shape != () or value.dtype.kind not in "iu":
        raise RuntimeError("invalid optimizer iteration counter")
    result = int(value)
    if result < 0:
        raise RuntimeError("negative optimizer iteration counter")
    return result


def _state(variables):
    result = []
    for variable in variables:
        value = np.asarray(variable.numpy())
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise RuntimeError("invalid model or optimizer state")
        result.append(np.array(value, copy=True))
    return tuple(result)


def _unchanged(before, variables):
    after = _state(variables)
    return len(before) == len(after) and all(
        a.shape == b.shape and a.dtype == b.dtype
        and np.array_equal(a, b)
        for a, b in zip(before, after)
    )


def checked_steps(*, model, optimizer, training_step, validation_step):
    """Wrap eager batch functions with update and validation integrity checks."""

    def train(images, labels):
        before = _iteration(optimizer)
        loss = _loss(training_step(images, labels))
        if _iteration(optimizer) != before + 1:
            raise RuntimeError("training must increment optimizer exactly once")
        _state(model.weights)
        _state(optimizer.variables)
        return loss

    def validate(images, labels):
        iteration = _iteration(optimizer)
        model_before = _state(model.weights)
        optimizer_before = _state(optimizer.variables)
        loss = _loss(validation_step(images, labels))
        if (
            _iteration(optimizer) != iteration
            or not _unchanged(model_before, model.weights)
            or not _unchanged(optimizer_before, optimizer.variables)
        ):
            raise RuntimeError("validation changed model or optimizer state")
        return loss

    return train, validate


def bind_family_steps(*, family, model, optimizer):
    """Bind the selected likelihood primitives to canonical batch inputs.

    The caller must construct and build the selected architecture, configure
    its optimizer, and validate canonical batches before invoking these steps.
    No fit() loop, early stopping, saving, or sampling occurs here.
    """
    if family not in ("ar", "maf"):
        raise ValueError("unsupported family")

    import tensorflow as tf

    if family == "ar":
        from .autoregressive_training import (
            AutoregressiveTrainingConfig, train_step, validation_step,
        )
        config = AutoregressiveTrainingConfig(
            epochs=2, patience=2, save_every=1,
            min_log_scale=-7.0, max_log_scale=3.0,
        )
        config.validate()
        # Only likelihood bounds are consumed by these batch primitives.
        # The shared core governs epochs; legacy fit() is never called.
        def train(images, labels):
            return train_step(
                model, optimizer,
                tf.convert_to_tensor(images, dtype=tf.float32),
                tf.convert_to_tensor(labels, dtype=tf.float32),
                config=config,
            )

        def validate(images, labels):
            return validation_step(
                model,
                tf.convert_to_tensor(images, dtype=tf.float32),
                tf.convert_to_tensor(labels, dtype=tf.float32),
                config=config,
            )
    else:
        from .masked_autoflow_training import train_step, conditional_nll

        def inputs(images, labels):
            # Canonical batch validation establishes strict one-hot labels.
            return (
                tf.convert_to_tensor(images, dtype=tf.float32),
                tf.convert_to_tensor(
                    np.argmax(labels, axis=1), dtype=tf.int32,
                ),
            )

        def train(images, labels):
            x, y = inputs(images, labels)
            return train_step(model, optimizer, x, y)["loss"]

        def validate(images, labels):
            x, y = inputs(images, labels)
            return conditional_nll(model, x, y, training=False)

    return checked_steps(
        model=model, optimizer=optimizer,
        training_step=train, validation_step=validate,
    )
