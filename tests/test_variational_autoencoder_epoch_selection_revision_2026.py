"""Regression tests for deterministic VAE epoch selection."""

import copy

import pytest

from trustforge.models.paper01_revision_2026.variational_autoencoder_epoch_selection import (
    select_best_vae_epoch,
)


def record(epoch, loss, updates):
    return {
        "epoch": epoch,
        "training_examples": 64,
        "completed_updates": updates,
        "validation": {
            "count": 16,
            "total_loss": loss,
        },
    }


def test_selects_minimum_validation_loss():
    history = [
        record(0, 12.0, 1),
        record(1, 8.0, 2),
        record(2, 10.0, 3),
    ]

    result = select_best_vae_epoch(history)

    assert result["best_epoch"] == 1
    assert result["best_validation_total_loss"] == 8.0
    assert result["best_completed_updates"] == 2
    assert result["last_epoch"] == 2
    assert result["last_completed_updates"] == 3


def test_tie_selects_earliest_epoch():
    history = [
        record(0, 7.0, 1),
        record(1, 7.0, 2),
        record(2, 8.0, 3),
    ]

    result = select_best_vae_epoch(history)

    assert result["best_epoch"] == 0
    assert result["best_completed_updates"] == 1


@pytest.mark.parametrize(
    "history",
    [
        [],
        [record(1, 2.0, 1)],
        [record(0, 2.0, 1), record(2, 1.0, 2)],
        [record(0, 2.0, 1), record(1, 1.0, 1)],
        [record(0, float("nan"), 1)],
        [record(0, float("inf"), 1)],
        [record(0, -float("inf"), 1)],
    ],
)
def test_invalid_histories_rejected(history):
    with pytest.raises((TypeError, ValueError)):
        select_best_vae_epoch(history)


def test_rejects_missing_validation_evidence():
    history = [record(0, 1.0, 1)]
    del history[0]["validation"]

    with pytest.raises(TypeError):
        select_best_vae_epoch(history)


def test_rejects_invalid_example_count():
    history = [record(0, 1.0, 1)]
    history[0]["training_examples"] = 0

    with pytest.raises(ValueError):
        select_best_vae_epoch(history)


def test_rejects_boolean_loss():
    history = [record(0, True, 1)]

    with pytest.raises(TypeError):
        select_best_vae_epoch(history)


def test_selection_does_not_mutate_history():
    history = [
        record(0, 3.0, 1),
        record(1, 2.0, 2),
    ]
    original = copy.deepcopy(history)

    select_best_vae_epoch(history)

    assert history == original


def test_selection_metadata():
    result = select_best_vae_epoch(
        [record(0, 1.0, 1)]
    )

    assert result["selection_metric"] == "validation_total_loss"
    assert result["selection_direction"] == "minimize"
    assert result["tie_break"] == "earliest_epoch"
    assert result["checkpoint_status"] == "NOT_PREPARED"
    assert result["publication_status"] == "NOT_PUBLISHED"
