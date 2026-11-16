"""Shared Paper 1 experimental-seed validation.

The accepted experimental-seed universe is owned by the existing governed
constant in ``gaussian_mixture_source_binding``.  This module provides a
strict, family-neutral validator without defining a second seed registry.

It performs no training, data access, checkpoint work or runtime activation.
"""

from __future__ import annotations

from .gaussian_mixture_source_binding import (
    GOVERNED_EXPERIMENTAL_SEEDS,
)


def require_governed_experimental_seed(value):
    """Return one approved Paper 1 experimental seed or fail closed.

    Only Python integers 42-48 are accepted.  Booleans, floats, numeric
    strings and other coercible values are rejected without substitution.
    """
    if type(value) is not int:
        raise TypeError("experimental_seed must be an integer")
    if value not in GOVERNED_EXPERIMENTAL_SEEDS:
        raise ValueError("experimental_seed must be one of 42-48")
    return value


__all__ = ["require_governed_experimental_seed"]
