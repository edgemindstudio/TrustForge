"""Focused fitted-state integrity and transform checks; no real data."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from trustforge.data.paper01_revision_2026.cic_fitted_state import (
    capture_fitted_state,
    restore_fitted_state,
)


def fixture():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(8117, 139)).astype(np.float32)
    values[:, 0] = 0
    scaler = StandardScaler().fit(values)
    indices = np.arange(8117, dtype=np.int64)
    record = capture_fitted_state(
        scaler=scaler,
        training_indices=indices,
        context={"scope": "synthetic component fixture"},
    )
    return values, scaler, indices, record


def resign(record):
    raw = json.dumps(
        record["payload"], sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()
    record["sha256"] = hashlib.sha256(raw).hexdigest()


def test_independent_restore_and_transform():
    values, scaler, indices, record = fixture()
    restored = restore_fitted_state(record)
    np.testing.assert_array_equal(
        scaler.transform(values[:19]), restored.transform(values[:19]),
    )
    assert restored.transform(values[:19]).dtype == np.float32
    assert not np.shares_memory(scaler.mean_, restored.mean_)
    assert capture_fitted_state(
        scaler=scaler, training_indices=indices,
        context={"scope": "synthetic component fixture"},
    ) == record


def test_digest_detects_context_change():
    _, _, _, record = fixture()
    record["payload"]["declared_context"]["scope"] = "changed"
    with pytest.raises(ValueError, match="digest"):
        restore_fitted_state(record)


def test_membership_changes_identity():
    _, scaler, indices, record = fixture()
    changed = indices + 1
    other = capture_fitted_state(
        scaler=scaler, training_indices=changed,
        context=record["payload"]["declared_context"],
    )
    assert other["sha256"] != record["sha256"]


@pytest.mark.parametrize("fault", ["negative_variance", "zero_scale", "nonfinite"])
def test_invalid_live_state(fault):
    _, scaler, indices, _ = fixture()
    if fault == "negative_variance":
        scaler.var_[0] = -1
    elif fault == "zero_scale":
        scaler.scale_[0] = 0
    else:
        scaler.mean_[0] = np.nan
    with pytest.raises(ValueError):
        capture_fitted_state(
            scaler=scaler, training_indices=indices, context={"fixture": True},
        )


def test_resigned_malformed_record_rejected():
    _, _, _, record = fixture()
    damaged = deepcopy(record)
    damaged["payload"]["scaler"]["arrays"]["mean_"]["shape"] = [138]
    resign(damaged)
    with pytest.raises(ValueError):
        restore_fitted_state(damaged)


def test_wrong_fitted_count_rejected():
    _, scaler, indices, _ = fixture()
    scaler.n_samples_seen_ = np.int64(8116)
    with pytest.raises(ValueError, match="sample count"):
        capture_fitted_state(
            scaler=scaler, training_indices=indices, context={"fixture": True},
        )
