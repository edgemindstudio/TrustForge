"""Candidate VAE publication transaction outcome policy.

Classifies publisher outcomes. Does not perform filesystem operations,
authenticate origin, establish crash durability, or authorize training.
"""

from __future__ import annotations

NOT_COMMITTED = "not_committed"
COMMITTED_SYNCED = "committed_synced"
INDETERMINATE = "indeterminate"

OUTCOMES = frozenset({
    NOT_COMMITTED,
    COMMITTED_SYNCED,
    INDETERMINATE,
})


def classify_publication_outcome(
    *,
    commit_link_created,
    commit_directory_synced,
):
    """Classify the transaction from confirmed publication events."""
    if type(commit_link_created) is not bool:
        raise TypeError("commit_link_created must be bool")
    if type(commit_directory_synced) is not bool:
        raise TypeError("commit_directory_synced must be bool")

    if commit_directory_synced and not commit_link_created:
        raise ValueError("commit synchronization without commit link")

    if not commit_link_created:
        return NOT_COMMITTED

    if not commit_directory_synced:
        return INDETERMINATE

    return COMMITTED_SYNCED


__all__ = [
    "NOT_COMMITTED",
    "COMMITTED_SYNCED",
    "INDETERMINATE",
    "OUTCOMES",
    "classify_publication_outcome",
]
