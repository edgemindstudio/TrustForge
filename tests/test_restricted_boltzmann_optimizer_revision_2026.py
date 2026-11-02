"""Focused tests for governed revision-2026 RBM manual SGD."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_cd import (
    RBMCD1Result,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_initialization import (
    initialize_shared_conditional_rbm,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_optimizer import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    RBMOptimizerStepDiagnostics,
    apply_manual_sgd_update,
)


def _model(
    *,
    class_count: int = 3,
):
    model, _ = initialize_shared_conditional_rbm(
        class_count=class_count,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )
    return model


def _cd_result(
    model,
    *,
    W_value: float = 0.25,
    a_value: float = -0.50,
    b_value: float = 0.75,
    U_value: float = -0.125,
):
    state = model.state_dict()

    return RBMCD1Result(
        positive_hidden_probabilities=np.zeros(
            (1, 128),
            dtype=np.float64,
        ),
        sampled_hidden_state=np.zeros(
            (1, 128),
            dtype=np.float64,
        ),
        negative_visible_state=np.zeros(
            (1, 4800),
            dtype=np.float64,
        ),
        negative_hidden_probabilities=np.zeros(
            (1, 128),
            dtype=np.float64,
        ),
        delta_W=np.full_like(
            state["W"],
            W_value,
            dtype=np.float64,
        ),
        delta_a=np.full_like(
            state["a"],
            a_value,
            dtype=np.float64,
        ),
        delta_b=np.full_like(
            state["b"],
            b_value,
            dtype=np.float64,
        ),
        delta_U=np.full_like(
            state["U"],
            U_value,
            dtype=np.float64,
        ),
    )


def test_governed_optimizer_constants():
    assert LEARNING_RATE == 5.0e-4
    assert WEIGHT_DECAY == 0.0


def test_manual_sgd_updates_all_parameter_blocks_exactly():
    model = _model()
    result = _cd_result(
        model
    )

    before = model.state_dict()

    diagnostics = apply_manual_sgd_update(
        model=model,
        cd_result=result,
    )

    after = model.state_dict()

    np.testing.assert_array_equal(
        after["W"],
        before["W"]
        + LEARNING_RATE * result.delta_W,
    )
    np.testing.assert_array_equal(
        after["a"],
        before["a"]
        + LEARNING_RATE * result.delta_a,
    )
    np.testing.assert_array_equal(
        after["b"],
        before["b"]
        + LEARNING_RATE * result.delta_b,
    )
    np.testing.assert_array_equal(
        after["U"],
        before["U"]
        + LEARNING_RATE * result.delta_U,
    )

    assert isinstance(
        diagnostics,
        RBMOptimizerStepDiagnostics,
    )


def test_optimizer_preserves_float64_model_state():
    model = _model()

    apply_manual_sgd_update(
        model=model,
        cd_result=_cd_result(
            model
        ),
    )

    for value in model.state_dict().values():
        assert value.dtype == np.float64


def test_zero_statistics_produce_no_parameter_change():
    model = _model()
    state = model.state_dict()

    result = _cd_result(
        model,
        W_value=0.0,
        a_value=0.0,
        b_value=0.0,
        U_value=0.0,
    )

    apply_manual_sgd_update(
        model=model,
        cd_result=result,
    )

    after = model.state_dict()

    for name in state:
        np.testing.assert_array_equal(
            after[name],
            state[name],
        )


def test_diagnostics_record_fixed_optimizer_configuration():
    model = _model()

    diagnostics = apply_manual_sgd_update(
        model=model,
        cd_result=_cd_result(
            model
        ),
    )

    assert diagnostics.learning_rate == 5.0e-4
    assert diagnostics.weight_decay == 0.0


def test_diagnostics_norms_match_cd_statistics():
    model = _model()
    result = _cd_result(
        model
    )

    diagnostics = apply_manual_sgd_update(
        model=model,
        cd_result=result,
    )

    assert diagnostics.delta_W_norm == pytest.approx(
        np.linalg.norm(
            result.delta_W
        )
    )
    assert diagnostics.delta_a_norm == pytest.approx(
        np.linalg.norm(
            result.delta_a
        )
    )
    assert diagnostics.delta_b_norm == pytest.approx(
        np.linalg.norm(
            result.delta_b
        )
    )
    assert diagnostics.delta_U_norm == pytest.approx(
        np.linalg.norm(
            result.delta_U
        )
    )


def test_repeated_identical_updates_are_deterministic():
    first_model = _model()
    second_model = _model()

    first_result = _cd_result(
        first_model
    )
    second_result = _cd_result(
        second_model
    )

    apply_manual_sgd_update(
        model=first_model,
        cd_result=first_result,
    )
    apply_manual_sgd_update(
        model=second_model,
        cd_result=second_result,
    )

    first_state = first_model.state_dict()
    second_state = second_model.state_dict()

    for name in first_state:
        np.testing.assert_array_equal(
            first_state[name],
            second_state[name],
        )


def test_optimizer_rejects_wrong_model_type():
    model = _model()

    with pytest.raises(TypeError):
        apply_manual_sgd_update(
            model=object(),
            cd_result=_cd_result(
                model
            ),
        )


def test_optimizer_rejects_wrong_cd_result_type():
    with pytest.raises(TypeError):
        apply_manual_sgd_update(
            model=_model(),
            cd_result=object(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "delta_W",
        "delta_a",
        "delta_b",
        "delta_U",
    ],
)
def test_optimizer_rejects_non_float64_statistics(
    field_name,
):
    model = _model()
    result = _cd_result(
        model
    )

    bad = getattr(
        result,
        field_name,
    ).astype(
        np.float32
    )

    result = replace(
        result,
        **{
            field_name: bad
        },
    )

    with pytest.raises(TypeError):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "delta_W",
        "delta_a",
        "delta_b",
        "delta_U",
    ],
)
def test_optimizer_rejects_nonfinite_statistics(
    field_name,
):
    model = _model()
    result = _cd_result(
        model
    )

    bad = np.array(
        getattr(
            result,
            field_name,
        ),
        dtype=np.float64,
        copy=True,
    )

    bad.flat[0] = np.nan

    result = replace(
        result,
        **{
            field_name: bad
        },
    )

    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


def test_optimizer_rejects_wrong_W_statistic_shape():
    model = _model()
    result = _cd_result(
        model
    )

    result = replace(
        result,
        delta_W=np.zeros(
            (4799, 128),
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


def test_optimizer_rejects_wrong_a_statistic_shape():
    model = _model()
    result = _cd_result(
        model
    )

    result = replace(
        result,
        delta_a=np.zeros(
            (4799,),
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


def test_optimizer_rejects_wrong_b_statistic_shape():
    model = _model()
    result = _cd_result(
        model
    )

    result = replace(
        result,
        delta_b=np.zeros(
            (127,),
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


def test_optimizer_rejects_wrong_U_statistic_shape():
    model = _model(
        class_count=3
    )
    result = _cd_result(
        model
    )

    result = replace(
        result,
        delta_U=np.zeros(
            (2, 128),
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )


def test_failed_update_does_not_mutate_model_state():
    model = _model()
    before = model.state_dict()

    result = _cd_result(
        model
    )

    bad_delta_W = np.array(
        result.delta_W,
        dtype=np.float64,
        copy=True,
    )
    bad_delta_W.flat[0] = np.inf

    result = replace(
        result,
        delta_W=bad_delta_W,
    )

    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        apply_manual_sgd_update(
            model=model,
            cd_result=result,
        )

    after = model.state_dict()

    for name in before:
        np.testing.assert_array_equal(
            before[name],
            after[name],
        )
