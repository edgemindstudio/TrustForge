import hashlib
import pytest

from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS
from trustforge.evaluation.paper01_result_artifacts import (
    bind_result_artifacts, MAX_JSON_BYTES,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    records, artifacts = [], {}
    for name in REQUIRED_RESULTS:
        result = b'{"value":0}'
        configuration = b'{"estimator":"fixture"}'
        records.append({
            "result_id": name,
            "status": "SUCCEEDED",
            "reason": None,
            "result_file_sha256": digest(result),
            "configuration_sha256": digest(configuration),
            "coverage": {"expected": 5, "computed": 5},
        })
        artifacts[name] = {
            "result_raw": result, "configuration_raw": configuration,
        }
    return records, artifacts


def run(records=None, artifacts=None):
    default_records, default_artifacts = fixture()
    return bind_result_artifacts(
        dataset_id="cicmaldroid2020", experimental_seed=42,
        records=default_records if records is None else records,
        artifact_bytes=default_artifacts if artifacts is None else artifacts,
    )


def replace_raw(records, artifacts, raw, field="result_raw"):
    name = records[0]["result_id"]
    artifacts[name][field] = raw
    hash_field = (
        "result_file_sha256" if field == "result_raw"
        else "configuration_sha256"
    )
    records[0][hash_field] = digest(raw)


def test_binds_exact_bytes_without_claiming_semantics_or_completion():
    result = run()
    assert result["bound_result_count"] == 7
    assert result["artifact_bytes_verified"] is True
    assert result["metric_semantic_validation"] == "NOT_PERFORMED"
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False
    for bound in result["bound_results"].values():
        assert bound["result"] == {"value": 0}
        assert bound["configuration"] == {"estimator": "fixture"}


@pytest.mark.parametrize("field", ["result_raw", "configuration_raw"])
def test_changed_bytes_rejected_even_when_json_meaning_is_unchanged(field):
    records, artifacts = fixture()
    artifacts[records[0]["result_id"]][field] += b" "
    with pytest.raises(ValueError, match="digest"):
        run(records, artifacts)


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_exact_artifact_inventory_required(fault):
    records, artifacts = fixture()
    if fault == "missing":
        del artifacts[records[0]["result_id"]]
    else:
        artifacts["undeclared"] = next(iter(artifacts.values()))
    with pytest.raises(ValueError):
        run(records, artifacts)


@pytest.mark.parametrize("raw", [
    b'{"value":1,"value":2}',
    b'{"nested":{"value":1,"value":2}}',
    b'{"value":NaN}',
    b'{"value":Infinity}',
    b'{"value":-Infinity}',
    b'{"value":1e999}',
    b'{"value":',
    b'[]',
    b'null',
    b'{"value":"\xff"}',
    b'',
])
@pytest.mark.parametrize("field", ["result_raw", "configuration_raw"])
def test_rehashed_invalid_json_still_rejected(raw, field):
    records, artifacts = fixture()
    replace_raw(records, artifacts, raw, field)
    with pytest.raises(ValueError):
        run(records, artifacts)


def test_oversized_json_rejected():
    records, artifacts = fixture()
    raw = b'{"padding":"' + b"x" * MAX_JSON_BYTES + b'"}'
    replace_raw(records, artifacts, raw)
    with pytest.raises(ValueError):
        run(records, artifacts)


def test_exact_bytes_type_required():
    records, artifacts = fixture()
    name = records[0]["result_id"]
    artifacts[name]["result_raw"] = bytearray(b'{"value":0}')
    with pytest.raises(ValueError):
        run(records, artifacts)


def test_failed_status_retained_without_completed_artifact():
    records, artifacts = fixture()
    name = records[0]["result_id"]
    records[0].update(
        status="FAILED", reason="fixture failure",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    del artifacts[name]
    result = run(records, artifacts)
    assert result["bound_result_count"] == 6
    assert result["status_inventory"]["all_required_declared_succeeded"] is False
    assert result["complete_evaluation"] is False


def test_artifact_for_failed_result_rejected():
    records, artifacts = fixture()
    records[0].update(
        status="FAILED", reason="fixture failure",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    with pytest.raises(ValueError):
        run(records, artifacts)


def test_extra_artifact_fields_rejected():
    records, artifacts = fixture()
    artifacts[records[0]["result_id"]]["complete_evaluation"] = True
    with pytest.raises(ValueError):
        run(records, artifacts)


def test_output_does_not_alias_input_coverage():
    records, artifacts = fixture()
    result = run(records, artifacts)
    name = records[0]["result_id"]
    result["bound_results"][name]["declared_coverage"]["computed"] = 99
    assert records[0]["coverage"]["computed"] == 5
