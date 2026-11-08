from copy import deepcopy
import pytest

from trustforge.evaluation.paper01_status_inventory import (
    REQUIRED_RESULTS, validate_status_inventory,
)


def success(name):
    return {
        "result_id": name,
        "status": "SUCCEEDED",
        "reason": None,
        "result_file_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "coverage": {"expected": 5, "computed": 5},
    }


def records():
    return [success(name) for name in REQUIRED_RESULTS]


def run(items=None, **overrides):
    args = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "records": records() if items is None else items,
    }
    args.update(overrides)
    return validate_status_inventory(**args)


def test_all_declared_successes_do_not_grant_complete_evaluation():
    result = run()
    assert result["required_result_count"] == 7
    assert result["declared_required_success_count"] == 7
    assert result["all_required_declared_succeeded"] is True
    assert result["required_status_blockers"] == []
    assert result["artifact_and_semantic_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("status", ["PENDING", "FAILED"])
def test_required_non_success_remains_visible(status):
    items = records()
    items[0].update(
        status=status, reason="fixture blocker",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    result = run(items)
    assert result["declared_required_success_count"] == 6
    assert result["all_required_declared_succeeded"] is False
    assert result["required_status_blockers"] == [{
        "result_id": items[0]["result_id"],
        "status": status,
        "reason": "fixture blocker",
    }]
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("status", ["NOT_REQUESTED", "UNAVAILABLE"])
def test_optional_status_requires_declared_optional_result(status):
    items = records()
    optional = {
        "result_id": "optional_diagnostic", "status": status,
        "reason": "prospectively optional fixture diagnostic",
        "result_file_sha256": None, "configuration_sha256": None,
        "coverage": None,
    }
    items.append(optional)
    result = run(items, optional_result_ids=("optional_diagnostic",))
    assert result["all_required_declared_succeeded"] is True
    assert result["complete_evaluation"] is False
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("status", ["NOT_REQUESTED", "UNAVAILABLE"])
def test_required_result_cannot_be_waived(status):
    items = records()
    items[0].update(
        status=status, reason="cannot waive",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unknown"])
def test_result_inventory_rejected(fault):
    items = records()
    if fault == "missing":
        items.pop()
    elif fault == "duplicate":
        items.append(deepcopy(items[0]))
    else:
        items.append(success("undeclared"))
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("coverage", [
    {"expected": 5, "computed": 4},
    {"expected": 0, "computed": 0},
    {"expected": True, "computed": True},
    {"expected": 5.0, "computed": 5.0},
    {"expected": 5, "computed": 5, "extra": 1},
    None,
])
def test_invalid_success_coverage_rejected(coverage):
    items = records()
    items[0]["coverage"] = coverage
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("field,value", [
    ("reason", "contradictory success"),
    ("result_file_sha256", "not-a-digest"),
    ("configuration_sha256", None),
    ("status", "COMPLETE"),
])
def test_invalid_success_record_rejected(field, value):
    items = records()
    items[0][field] = value
    with pytest.raises(ValueError):
        run(items)


def test_completion_flag_cannot_be_injected():
    items = records()
    items[0]["complete_evaluation"] = True
    with pytest.raises(ValueError):
        run(items)


def test_failure_requires_reason_and_null_completed_result_fields():
    items = records()
    items[0]["status"] = "FAILED"
    with pytest.raises(ValueError):
        run(items)
    items[0].update(
        reason="failure", result_file_sha256=None,
        configuration_sha256=None, coverage=None,
    )
    assert run(items)["all_required_declared_succeeded"] is False
    items[0]["reason"] = " "
    with pytest.raises(ValueError):
        run(items)


@pytest.mark.parametrize("kwargs", [
    {"experimental_seed": True},
    {"experimental_seed": 49},
    {"dataset_id": "unknown"},
    {"optional_result_ids": ("duplicate", "duplicate")},
    {"optional_result_ids": (REQUIRED_RESULTS[0],)},
    {"optional_result_ids": ("missing_optional_record",)},
])
def test_invalid_context_or_optional_declarations_rejected(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


def test_deterministic_order_and_output_isolation():
    items = records()
    result = run(items)
    assert result == run(list(reversed(items)))
    result["records"][0]["coverage"]["computed"] = 99
    assert all(item["coverage"]["computed"] == 5 for item in items)
