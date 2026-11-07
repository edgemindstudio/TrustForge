import hashlib
import json
import numpy as np
import pytest

from trustforge.evaluation.paper01_ms_ssim import SETTINGS
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_ms_ssim_evaluation import evaluate_ms_ssim_pairs


LABELS = np.repeat(np.arange(5, dtype=np.int64), 2000)


class View:
    def __init__(self, fault=None):
        self.identity = {
            "dataset_id": "cicmaldroid2020",
            "manifest_sha256": "a" * 64,
        }
        self.size = 10000
        self.fault = fault
        self.ready = False
        self.maximum_request = 0
        self.requests = 0

    def preflight(self):
        if self.fault == "preflight":
            raise ValueError("fixture pool failure")
        self.ready = True

    def batch(self, positions):
        assert self.ready
        self.requests += 1
        self.maximum_request = max(self.maximum_request, len(positions))
        images = np.zeros((len(positions), 40, 40, 3), dtype=np.float32)
        images[:, 0, 0, 0] = positions.astype(np.float32) / np.float32(10000)
        classes = LABELS[positions].copy()
        if self.fault == "labels":
            classes = (classes + 1) % 5
        if self.fault == "identity":
            self.identity["manifest_sha256"] = "b" * 64
        observed = positions[::-1] if self.fault == "order" else positions
        return observed, images, np.eye(5, dtype=np.float32)[classes]


class Scorer:
    def __init__(self, fault=None):
        self.identity = {**SETTINGS, "tensorflow": "fixture"}
        self.fault = fault
        self.calls = []

    def score_pairs(self, first, second):
        self.calls.append((
            np.rint(first[:, 0, 0, 0] * 10000).astype(np.int64),
            np.rint(second[:, 0, 0, 0] * 10000).astype(np.int64),
        ))
        values = first[:, 0, 0, 0].copy()
        if self.fault == "nan":
            values[0] = np.nan
        if self.fault == "missing":
            values = values[:-1]
        if self.fault == "dtype":
            values = values.astype(np.float64)
        if self.fault == "exception":
            raise RuntimeError("fixture scoring failure")
        if self.fault == "identity":
            self.identity["tensorflow"] = "changed"
        return values


def selection():
    return select_ms_ssim_pairs(
        labels=LABELS, class_count=5, manifest_sha256="a" * 64,
    )


def run(view=None, scorer=None, record=None, digest=None):
    raw = json.dumps(
        selection() if record is None else record, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()
    return evaluate_ms_ssim_pairs(
        selection_raw=raw,
        selection_sha256=hashlib.sha256(raw).hexdigest() if digest is None else digest,
        synthetic_view=View() if view is None else view,
        labels=LABELS,
        scorer=Scorer() if scorer is None else scorer,
    )


def test_complete_coverage_pair_order_and_equal_class_macro():
    view, scorer = View(), Scorer()
    report = run(view=view, scorer=scorer)
    assert report["class_coverage_complete"] is True
    assert report["complete_evaluation"] is False
    assert report["pair_count"] == 1000
    assert set(report["per_class"]) == {str(c) for c in range(5)}
    assert view.requests == 5 * 13
    assert view.maximum_request <= 32

    actual = np.concatenate([
        np.column_stack(pair) for pair in scorer.calls
    ])
    selected = selection()
    expected = np.concatenate([
        selected["classes"][str(c)]["manifest_position_pairs"]
        for c in range(5)
    ])
    np.testing.assert_array_equal(actual, expected)
    means = []
    for c in range(5):
        entry = report["per_class"][str(c)]
        expected_scores = (
            expected[c * 200:(c + 1) * 200, 0].astype(np.float32)
            / np.float32(10000)
        )
        assert entry["pair_count"] == 200
        np.testing.assert_array_equal(entry["scores"], expected_scores)
        assert entry["mean"] == pytest.approx(
            expected_scores.mean(dtype=np.float64)
        )
        means.append(entry["mean"])
    assert report["macro_mean"] == pytest.approx(np.mean(means))


def test_wrong_file_digest_rejected():
    with pytest.raises(ValueError):
        run(digest="0" * 64)


def test_rehashed_changed_pair_selection_rejected():
    record = selection()
    record["classes"]["0"]["manifest_position_pairs"][0][0] += 1
    with pytest.raises(ValueError):
        run(record=record)


@pytest.mark.parametrize("fault", ["preflight", "labels", "identity", "order"])
def test_pool_failures_propagate(fault):
    with pytest.raises((ValueError, RuntimeError)):
        run(view=View(fault))


@pytest.mark.parametrize("fault", ["nan", "missing", "dtype", "exception", "identity"])
def test_scoring_failures_prevent_complete_result(fault):
    with pytest.raises((ValueError, RuntimeError)):
        run(scorer=Scorer(fault))


def test_wrong_scorer_settings_rejected():
    scorer = Scorer()
    scorer.identity["filter_size"] = 11
    with pytest.raises(ValueError):
        run(scorer=scorer)
