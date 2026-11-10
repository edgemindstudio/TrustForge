from copy import deepcopy
import json

import numpy as np
import pytest
from scipy import stats

from trustforge.evaluation.paper01_seed_reporting import (
    summarize_declared_paired_seeds,
)


def records():
    augmented = [0.35, 0.48, 0.50, 0.51, 0.55, 0.62, 0.70]
    return [
        {
            "experimental_seed": seed,
            "status": "SUCCEEDED",
            "reason": None,
            "real_only_macro_f1": 0.50,
            "real_plus_synthetic_macro_f1": value,
        }
        for seed, value in zip(range(42, 49), augmented)
    ]


def run(items=None):
    return summarize_declared_paired_seeds(
        records=records() if items is None else items,
    )


def unavailable(item, status="PENDING"):
    item.update(
        status=status, reason="fixture pair unavailable",
        real_only_macro_f1=None, real_plus_synthetic_macro_f1=None,
    )


def test_seven_pair_statistics_against_scipy():
    items = records()
    differences = np.array([
        r["real_plus_synthetic_macro_f1"] - r["real_only_macro_f1"]
        for r in items
    ])
    result = run(items)
    summary = result["seven_pair_summary"]
    expected_interval = stats.t.interval(
        0.95, df=6, loc=np.mean(differences), scale=stats.sem(differences),
    )
    assert summary["mean_difference"] == pytest.approx(np.mean(differences))
    assert summary["sample_standard_deviation"] == pytest.approx(
        np.std(differences, ddof=1)
    )
    assert summary["standard_error"] == pytest.approx(stats.sem(differences))
    assert summary["critical_value"] == pytest.approx(stats.t.ppf(0.975, 6))
    assert summary["interval"] == pytest.approx(expected_interval)
    assert result["declared_success_fraction"] == 1
    assert result["declared_failed_fraction"] == 0


def test_output_is_seed_ordered_and_inputs_unchanged():
    items = list(reversed(records()))
    before = deepcopy(items)
    result = run(items)
    assert items == before
    assert [r["experimental_seed"] for r in result["records"]] == list(range(42, 49))
    assert [r["experimental_seed"] for r in result["paired_differences"]] == list(
        range(42, 49)
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("count", range(7))
def test_fewer_than_seven_withholds_interval(count):
    items = records()
    for item in items[count:]:
        unavailable(item)
    result = run(items)
    assert result["declared_successful_pair_count"] == count
    assert result["expected_pair_count"] == 7
    assert result["seven_pair_summary"] is None
    assert result["seven_pair_summary_status"] == "INCOMPLETE"
    assert result["interval_withheld_reason"] is not None
    assert result["available_case_descriptive"]["count"] == count
    assert result["declared_success_fraction"] == count / 7
    assert len(result["records"]) == 7
    if count == 0:
        assert result["available_case_descriptive"]["mean_difference"] is None
    if count < 2:
        assert result["available_case_descriptive"]["sample_standard_deviation"] is None


def test_failure_and_incomplete_counts_keep_fixed_denominator():
    items = records()
    unavailable(items[0], "FAILED")
    unavailable(items[1], "INCOMPLETE")
    unavailable(items[2], "PENDING")
    result = run(items)
    assert result["status_counts"] == {
        "FAILED": 1, "INCOMPLETE": 1, "PENDING": 1, "SUCCEEDED": 4,
    }
    assert result["declared_failed_fraction"] == 1 / 7
    assert result["declared_success_fraction"] == 4 / 7


def test_constant_differences_produce_zero_width_interval():
    items = records()
    for item in items:
        item["real_only_macro_f1"] = 0.25
        item["real_plus_synthetic_macro_f1"] = 0.75
    summary = run(items)["seven_pair_summary"]
    assert summary["sample_standard_deviation"] == 0
    assert summary["interval"] == [0.5, 0.5]


def test_zero_is_a_valid_score_and_negative_differences_are_retained():
    items = records()
    items[0]["real_plus_synthetic_macro_f1"] = 0
    result = run(items)
    assert result["paired_differences"][0]["macro_f1_difference"] == -0.5
    assert result["declared_successful_pair_count"] == 7


@pytest.mark.parametrize("fault", [
    "missing", "duplicate", "extra_seed", "bool_seed", "float_seed",
])
def test_exact_seed_inventory_required(fault):
    items = records()
    if fault == "missing":
        items.pop()
    elif fault == "duplicate":
        items.append(deepcopy(items[0]))
    else:
        items[0]["experimental_seed"] = {
            "extra_seed": 49, "bool_seed": True, "float_seed": 42.0,
        }[fault]
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("value", [
    True, None, "0.5", float("nan"), float("inf"), -0.01, 1.01,
])
def test_invalid_successful_score_rejected(value):
    items = records()
    items[0]["real_only_macro_f1"] = value
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("fault", [
    "unknown_status", "successful_reason", "missing_reason",
    "retained_score", "extra_field",
])
def test_contradictory_records_rejected(fault):
    items = records()
    if fault == "unknown_status":
        items[0]["status"] = "COMPLETE"
    elif fault == "successful_reason":
        items[0]["reason"] = "unexpected"
    elif fault == "extra_field":
        items[0]["complete_evaluation"] = True
    else:
        unavailable(items[0])
        if fault == "missing_reason":
            items[0]["reason"] = " "
        else:
            items[0]["real_only_macro_f1"] = 0.5
    with pytest.raises(ValueError):
        run(items)


def test_non_list_rejected():
    with pytest.raises(TypeError):
        run(tuple(records()))


def test_declared_success_never_grants_scientific_acceptance():
    result = run()
    assert result["evaluation_evidence_validation"] == "NOT_PERFORMED"
    assert result["attempt_selection_validation"] == "NOT_PERFORMED"
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["primary_inference_authorized"] is False
    assert result["complete_evaluation"] is False
    assert result["full_benchmark_go"] is False
