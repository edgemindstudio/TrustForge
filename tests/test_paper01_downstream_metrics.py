from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve,
)

from trustforge.evaluation.paper01_downstream_metrics import (
    downstream_metrics, _binary_ranking, _top_label_ece,
)

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
).read_bytes()


def calculate(labels, probabilities, classes=3, recipe=RECIPE):
    return downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=classes, recipe_raw=recipe,
    )


def test_agreement_with_sklearn_including_tied_scores():
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(3, dtype=np.int64), 40)
    weights = rng.integers(1, 5, size=(len(labels), 3)).astype(np.float64)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    result = calculate(labels, probabilities)
    prediction = probabilities.argmax(axis=1)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        confusion_matrix(labels, prediction, labels=np.arange(3)),
    )
    assert result["macro_f1"] == pytest.approx(
        f1_score(labels, prediction, average="macro", zero_division=0)
    )
    assert result["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(labels, prediction)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=np.arange(3), zero_division=0,
    )
    aps, rocs = [], []
    for c in range(3):
        row = result["per_class"][str(c)]
        for key, expected in (
            ("precision", precision[c]), ("recall", recall[c]),
            ("f1", f1[c]), ("support", support[c]),
        ):
            assert row[key] == pytest.approx(expected)
        ap = average_precision_score(labels == c, probabilities[:, c])
        fpr, tpr, _ = roc_curve(
            labels == c, probabilities[:, c], drop_intermediate=False,
        )
        roc = tpr[fpr <= 0.01].max()
        assert row["average_precision"] == pytest.approx(ap)
        assert row["recall_at_1pct_fpr"] == pytest.approx(roc)
        aps.append(ap)
        rocs.append(roc)
    assert result["macro_auprc"] == pytest.approx(np.mean(aps))
    assert result["macro_recall_at_1pct_fpr"] == pytest.approx(np.mean(rocs))
    target = np.eye(3)[labels]
    assert result["brier"] == pytest.approx(
        np.mean(np.sum((probabilities - target) ** 2, axis=1))
    )
    assert result["complete_evaluation"] is False


def test_exact_fpr_boundary_and_no_interpolation():
    truth = np.array([True, True] + [False] * 100)
    scores = np.array([0.9, 0.7, 0.8] + [0.1] * 99)
    assert _binary_ranking(truth, scores)[1] == 1.0
    # With only 99 negatives, one false positive exceeds 1%.
    assert _binary_ranking(truth[:-1], scores[:-1])[1] == 0.5


def test_score_ties_cannot_be_split_to_improve_roc():
    truth = np.array([True, False, True, False])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    ap, recall, _, _ = _binary_ranking(truth, scores)
    assert ap == pytest.approx(0.5)
    assert recall == 0.0


def test_ece_exact_edges_empty_bins_and_final_endpoint():
    edge = np.linspace(0, 1, 16)[8]
    confidence = np.array([np.nextafter(edge, 0), edge, 1.0])
    probabilities = np.column_stack((confidence, 1 - confidence))
    labels = np.array([0, 1, 0])
    value, bins = _top_label_ece(
        probabilities, probabilities.argmax(axis=1), labels,
    )
    assert bins[7]["count"] == 1
    assert bins[8]["count"] == 1
    assert bins[14]["count"] == 1
    assert bins[14]["upper_inclusive"] is True
    assert bins[0]["accuracy"] is None
    assert bins[0]["mean_confidence"] is None
    assert bins[0]["contribution"] == 0
    assert value == pytest.approx(((1 - confidence[0]) + confidence[1]) / 3)
    assert sum(row["count"] for row in bins) == 3


def test_perfect_predictions_and_binary_brier_scaling():
    labels = np.array([0, 1], dtype=np.int64)
    perfect = calculate(labels, np.eye(2), classes=2)
    assert perfect["macro_f1"] == perfect["macro_auprc"] == 1
    assert perfect["ece"] == perfect["brier"] == 0
    result = calculate(
        labels, np.array([[0.75, 0.25], [0.25, 0.75]]), classes=2,
    )
    assert result["brier"] == pytest.approx(0.125)


def test_argmax_tie_uses_lowest_class_and_zero_precision_is_valid():
    labels = np.array([0, 1], dtype=np.int64)
    result = calculate(labels, np.full((2, 2), 0.5), classes=2)
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]
    assert result["per_class"]["1"]["precision"] == 0
    assert result["per_class"]["1"]["f1"] == 0


@pytest.mark.parametrize("probabilities", [
    np.array([[0.5, 0.5], [0.5, np.nan]]),
    np.array([[0.5, 0.5], [0.5, np.inf]]),
    np.array([[1.1, -0.1], [0.5, 0.5]]),
    np.full((2, 2), 0.4),
    np.ones((2, 2), dtype=np.int64),
    np.ones((2, 3)),
])
def test_invalid_probabilities_rejected(probabilities):
    with pytest.raises(ValueError):
        calculate(np.array([0, 1]), probabilities, classes=2)


@pytest.mark.parametrize("labels", [
    np.array([0, 0]), np.array([0, 2]),
    np.array([0.0, 1.0]), np.array([[0, 1]]),
])
def test_invalid_or_missing_class_labels_rejected(labels):
    with pytest.raises(ValueError):
        calculate(labels, np.full((2, 2), 0.5), classes=2)


def test_recipe_changes_rejected():
    with pytest.raises(ValueError):
        calculate(
            np.array([0, 1]), np.eye(2), classes=2,
            recipe=RECIPE + b" ",
        )


def test_probability_inputs_remain_unchanged_without_renormalization():
    labels = np.array([0, 1])
    probabilities = np.array([[0.6, 0.4000001], [0.3, 0.7]])
    before = probabilities.copy()
    calculate(labels, probabilities, classes=2)
    np.testing.assert_array_equal(probabilities, before)
