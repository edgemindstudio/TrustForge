"""Seven-seed paired arithmetic and explicit missing-result reporting.

Input success statuses are declarations, not authenticated evaluation evidence.
Attempt selection, provenance and scientific acceptance remain outer checks.
"""

from copy import deepcopy
import math
import statistics


EXPECTED_SEEDS = tuple(range(42, 49))
T975_DF6 = 2.4469118511449692
STATUSES = frozenset({"PENDING", "FAILED", "INCOMPLETE", "SUCCEEDED"})


def _score(value):
    if type(value) not in (int, float):
        raise ValueError("numeric macro-F1 required")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite macro-F1 in [0,1] required")
    return value


def summarize_declared_paired_seeds(*, records):
    """Summarize exactly seeds 42-48 without selecting execution attempts."""
    if type(records) is not list:
        raise TypeError("seed records must be a list")
    observed = {}
    differences = []
    for record in records:
        if type(record) is not dict or set(record) != {
            "experimental_seed", "status", "reason",
            "real_only_macro_f1", "real_plus_synthetic_macro_f1",
        }:
            raise ValueError("exact seed record fields required")
        seed = record["experimental_seed"]
        status = record["status"]
        if (
            type(seed) is not int or seed not in EXPECTED_SEEDS
            or seed in observed
        ):
            raise ValueError("distinct expected seed required")
        if type(status) is not str or status not in STATUSES:
            raise ValueError("unsupported seed status")
        retained = deepcopy(record)
        if status == "SUCCEEDED":
            if record["reason"] is not None:
                raise ValueError("successful pair requires null reason")
            real = _score(record["real_only_macro_f1"])
            augmented = _score(record["real_plus_synthetic_macro_f1"])
            difference = augmented - real
            retained["macro_f1_difference"] = difference
            differences.append((seed, difference))
        else:
            if (
                type(record["reason"]) is not str
                or not record["reason"].strip()
                or record["real_only_macro_f1"] is not None
                or record["real_plus_synthetic_macro_f1"] is not None
            ):
                raise ValueError("non-success requires reason and null scores")
            retained["macro_f1_difference"] = None
        observed[seed] = retained
    if set(observed) != set(EXPECTED_SEEDS):
        raise ValueError("all seven expected seeds must be explicitly represented")

    differences.sort()
    values = [value for _, value in differences]
    count = len(values)
    mean = statistics.mean(values) if values else None
    sd = statistics.stdev(values) if count >= 2 else None
    seven_pair_summary = None
    if count == 7:
        standard_error = sd / math.sqrt(7)
        margin = T975_DF6 * standard_error
        seven_pair_summary = {
            "mean_difference": mean,
            "sample_standard_deviation": sd,
            "standard_error": standard_error,
            "confidence_level": 0.95,
            "degrees_of_freedom": 6,
            "critical_value": T975_DF6,
            "interval": [mean - margin, mean + margin],
            "interval_scope": "pointwise_conditional_on_input_pair_validity",
        }

    counts = {
        status: sum(r["status"] == status for r in observed.values())
        for status in sorted(STATUSES)
    }
    return {
        "schema": "trustforge.paper01.declared-seed-summary.v1",
        "expected_seeds": list(EXPECTED_SEEDS),
        "expected_pair_count": 7,
        "declared_successful_pair_count": count,
        "status_counts": counts,
        "declared_success_fraction": count / 7,
        "declared_failed_fraction": counts["FAILED"] / 7,
        "records": [observed[seed] for seed in EXPECTED_SEEDS],
        "paired_differences": [
            {"experimental_seed": seed, "macro_f1_difference": value}
            for seed, value in differences
        ],
        "available_case_descriptive": {
            "count": count,
            "mean_difference": mean,
            "sample_standard_deviation": sd,
            "conditional_on_successful_declared_pairs": True,
        },
        "seven_pair_summary": seven_pair_summary,
        "seven_pair_summary_status": (
            "ALL_SEVEN_DECLARED_PAIRS_PRESENT" if count == 7 else "INCOMPLETE"
        ),
        "interval_withheld_reason": (
            None if count == 7 else "fewer_than_seven_successful_declared_pairs"
        ),
        "limitations": [
            "seven repetitions cannot establish normality or independence",
            "fixed-dataset pointwise estimation only",
            "no confirmatory significance or benchmark-wide superiority claim",
            "an interval containing zero does not establish equivalence",
        ],
        "attempt_selection_validation": "NOT_PERFORMED",
        "evaluation_evidence_validation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "primary_inference_authorized": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
