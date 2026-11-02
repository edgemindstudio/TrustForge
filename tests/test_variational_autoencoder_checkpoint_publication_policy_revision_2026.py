"""Pure VAE checkpoint-publication outcome policy regressions.

No filesystem publication, datasets, or training.
"""

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_policy as policy,
)


@pytest.mark.parametrize(
    "linked,synced,expected",
    [
        (False, False, policy.NOT_COMMITTED),
        (True, False, policy.INDETERMINATE),
        (True, True, policy.COMMITTED_SYNCED),
    ],
)
def test_publication_outcome_classification(linked, synced, expected):
    assert policy.classify_publication_outcome(
        commit_link_created=linked,
        commit_directory_synced=synced,
    ) == expected


def test_sync_without_commit_rejected():
    with pytest.raises(
        ValueError,
        match="commit synchronization without commit link",
    ):
        policy.classify_publication_outcome(
            commit_link_created=False,
            commit_directory_synced=True,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit_link_created", None),
        ("commit_link_created", 0),
        ("commit_link_created", "False"),
        ("commit_directory_synced", None),
        ("commit_directory_synced", 1),
        ("commit_directory_synced", "True"),
    ],
)
def test_non_boolean_publication_events_rejected(field, value):
    arguments = {
        "commit_link_created": True,
        "commit_directory_synced": False,
    }
    arguments[field] = value

    with pytest.raises(TypeError, match="must be bool"):
        policy.classify_publication_outcome(**arguments)


def test_outcome_inventory_is_exact():
    assert policy.OUTCOMES == frozenset({
        "not_committed",
        "committed_synced",
        "indeterminate",
    })


def test_indeterminate_is_not_success_or_rollback():
    outcome = policy.classify_publication_outcome(
        commit_link_created=True,
        commit_directory_synced=False,
    )

    assert outcome != policy.NOT_COMMITTED
    assert outcome != policy.COMMITTED_SYNCED
