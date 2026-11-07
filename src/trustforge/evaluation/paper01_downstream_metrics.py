"""Recipe-bound downstream metric mathematics.

No model selection, probability calibration, provenance authentication or
complete-evaluation authorization. Failed inputs raise; they are not scores.
"""

import hashlib
import json
import numpy as np


RECIPE_SHA256 = (
    "d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85"
)


def selected_metric_recipe(raw):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != RECIPE_SHA256:
        raise ValueError("downstream metric recipe identity mismatch")
    return json.loads(raw)


def _binary_ranking(truth, scores):
    """Group score ties; compute stepwise AP and conservative empirical ROC."""
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("positive and negative examples required")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
                 len(scores) - 1]
    tp = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    recall = tp.astype(np.float64) / positives
    precision = tp.astype(np.float64) / (ends + 1)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    eligible = fp.astype(np.float64) / negatives <= 0.01
    roc_recall = float(np.max(np.r_[0.0, recall[eligible]]))
    return ap, roc_recall, positives, negatives


def _top_label_ece(probabilities, prediction, labels):
    confidence = probabilities.max(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, 16)
    bins = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, 14)
    records = []
    ece = 0.0
    for index in range(15):
        mask = bins == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        contribution = (
            count / len(labels) * abs(accuracy - mean_confidence)
            if count else 0.0
        )
        ece += contribution
        records.append({
            "bin_index": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "upper_inclusive": index == 14,
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "contribution": contribution,
        })
    return float(ece), records


def downstream_metrics(*, labels, probabilities, class_count, recipe_raw):
    selected_metric_recipe(recipe_raw)
    if type(class_count) is not int or class_count < 2:
        raise ValueError("authoritative class count >=2 required")
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1 or labels.size == 0
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0) or np.any(labels >= class_count)
    ):
        raise ValueError("ordered authoritative integer labels required")
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.shape != (len(labels), class_count)
        or probabilities.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0) or np.any(probabilities > 1)
    ):
        raise ValueError("finite N-by-K probabilities in [0,1] required")
    labels = labels.astype(np.int64, copy=False)
    probabilities = probabilities.astype(np.float64, copy=False)
    if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("probability row sums differ from one")
    support = np.bincount(labels, minlength=class_count)
    if np.any(support == 0) or np.any(support == len(labels)):
        raise ValueError("every class requires positive and negative examples")

    prediction = probabilities.argmax(axis=1)
    confusion = np.bincount(
        labels * class_count + prediction,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    per_class = {}
    for c in range(class_count):
        tp = int(confusion[c, c])
        predicted = int(confusion[:, c].sum())
        actual = int(support[c])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual
        f1 = 2 * tp / (predicted + actual) if predicted + actual else 0.0
        ap, roc_recall, positives, negatives = _binary_ranking(
            labels == c, probabilities[:, c],
        )
        per_class[str(c)] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": actual, "predicted_count": predicted,
            "average_precision": ap,
            "recall_at_1pct_fpr": roc_recall,
            "positive_count": positives, "negative_count": negatives,
        }

    ece, bins = _top_label_ece(probabilities, prediction, labels)
    squared_error = probabilities.copy()
    squared_error[np.arange(len(labels)), labels] -= 1.0
    brier = float(np.mean(np.sum(squared_error ** 2, axis=1)))
    result = {
        "schema": "trustforge.paper01.downstream-metrics.v1",
        "recipe_sha256": RECIPE_SHA256,
        "sample_count": len(labels),
        "class_count": class_count,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean([r["f1"] for r in per_class.values()])),
        "balanced_accuracy": float(np.mean([
            r["recall"] for r in per_class.values()
        ])),
        "macro_auprc": float(np.mean([
            r["average_precision"] for r in per_class.values()
        ])),
        "macro_recall_at_1pct_fpr": float(np.mean([
            r["recall_at_1pct_fpr"] for r in per_class.values()
        ])),
        "ece": ece,
        "ece_bins": bins,
        "brier": brier,
        "class_coverage_complete": True,
        "complete_evaluation": False,
    }
    # Reject nonfinite numerical output while allowing explicit empty-bin nulls.
    json.dumps(result, allow_nan=False)
    return result
