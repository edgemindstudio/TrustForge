"""Verify exact result/configuration bytes against a status inventory.

No filesystem discovery, metric recomputation, provenance authentication or
completion authorization. Callers supply independently requested artifact bytes.
"""

import hashlib
import json

from .paper01_status_inventory import validate_status_inventory


MAX_JSON_BYTES = 16 * 1024 * 1024


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _verified_json(raw, expected_sha256):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_JSON_BYTES:
        raise ValueError("bounded exact JSON bytes required")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("artifact digest mismatch")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_object,
        parse_constant=_invalid_constant,
    )
    if type(value) is not dict:
        raise ValueError("JSON object artifact required")
    # Also rejects numeric overflow such as 1e999 parsed as infinity.
    json.dumps(value, allow_nan=False)
    return value


def bind_result_artifacts(
    *, dataset_id, experimental_seed, records,
    artifact_bytes, optional_result_ids=(),
):
    inventory = validate_status_inventory(
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        records=records,
        optional_result_ids=optional_result_ids,
    )
    successful = {
        record["result_id"]: record
        for record in inventory["records"]
        if record["status"] == "SUCCEEDED"
    }
    if type(artifact_bytes) is not dict or set(artifact_bytes) != set(successful):
        raise ValueError("exact artifact inventory for successful results required")

    bound = {}
    for name in sorted(successful):
        supplied = artifact_bytes[name]
        if type(supplied) is not dict or set(supplied) != {
            "result_raw", "configuration_raw",
        }:
            raise ValueError("exact result and configuration bytes required")
        record = successful[name]
        result = _verified_json(
            supplied["result_raw"], record["result_file_sha256"],
        )
        configuration = _verified_json(
            supplied["configuration_raw"], record["configuration_sha256"],
        )
        bound[name] = {
            "result_file_sha256": record["result_file_sha256"],
            "configuration_sha256": record["configuration_sha256"],
            "declared_coverage": record["coverage"],
            "result": result,
            "configuration": configuration,
        }

    return {
        "schema": "trustforge.paper01.result-artifact-binding.v1",
        "status_inventory": inventory,
        "bound_results": bound,
        "bound_result_count": len(bound),
        "artifact_bytes_verified": True,
        "metric_semantic_validation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
