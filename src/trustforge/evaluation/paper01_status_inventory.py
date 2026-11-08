"""Structural validation of revision evaluation status inventories.

SUCCEEDED remains a producer declaration here. Artifact bytes, metric semantics,
provenance and publication require separate validation. This component never
grants complete-evaluation status.
"""

from copy import deepcopy
import re

from .paper01_inputs import CLASS_COUNTS


REQUIRED_RESULTS = (
    "real_only_downstream",
    "real_plus_synthetic_downstream",
    "paired_downstream",
    "pooled_balanced_kid",
    "pooled_balanced_fid",
    "class_fid",
    "intra_class_ms_ssim",
)
STATUSES = {"PENDING", "SUCCEEDED", "FAILED", "NOT_REQUESTED", "UNAVAILABLE"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _text(value):
    return type(value) is str and bool(value.strip())


def validate_status_inventory(
    *, dataset_id, experimental_seed, records, optional_result_ids=(),
):
    """Validate declarations, not the truth of their underlying evidence."""
    if dataset_id not in CLASS_COUNTS:
        raise ValueError("authoritative dataset required")
    if type(experimental_seed) is not int or experimental_seed not in range(42, 49):
        raise ValueError("experimental seed 42-48 required")
    if (
        type(optional_result_ids) not in (list, tuple)
        or not all(_text(value) for value in optional_result_ids)
        or len(set(optional_result_ids)) != len(optional_result_ids)
        or set(optional_result_ids) & set(REQUIRED_RESULTS)
    ):
        raise ValueError("distinct prospectively declared optional IDs required")
    if type(records) is not list:
        raise TypeError("status records must be a list")

    allowed = set(REQUIRED_RESULTS) | set(optional_result_ids)
    observed = {}
    for record in records:
        if type(record) is not dict or set(record) != {
            "result_id", "status", "reason",
            "result_file_sha256", "configuration_sha256", "coverage",
        }:
            raise ValueError("unexpected status record fields")
        name, status = record["result_id"], record["status"]
        if type(name) is not str or name not in allowed or name in observed:
            raise ValueError("duplicate or undeclared result ID")
        if type(status) is not str or status not in STATUSES:
            raise ValueError("unsupported result status")
        if name in REQUIRED_RESULTS and status in {"NOT_REQUESTED", "UNAVAILABLE"}:
            raise ValueError("required result cannot be optional")
        if status == "SUCCEEDED":
            if record["reason"] is not None:
                raise ValueError("successful declaration requires null reason")
            for field in ("result_file_sha256", "configuration_sha256"):
                value = record[field]
                if type(value) is not str or HEX64.fullmatch(value) is None:
                    raise ValueError("successful declaration requires identities")
            coverage = record["coverage"]
            if (
                type(coverage) is not dict
                or set(coverage) != {"expected", "computed"}
                or type(coverage["expected"]) is not int
                or type(coverage["computed"]) is not int
                or coverage["expected"] <= 0
                or coverage["computed"] != coverage["expected"]
            ):
                raise ValueError("successful declaration requires equal coverage")
        else:
            if not _text(record["reason"]):
                raise ValueError("non-success status requires explicit reason")
            # Partial artifacts belong in preserved execution evidence, not
            # fields that would imply a validated completed result.
            if any(record[field] is not None for field in (
                "result_file_sha256", "configuration_sha256", "coverage",
            )):
                raise ValueError("non-success result fields must be null")
        observed[name] = deepcopy(record)

    if set(observed) != allowed:
        raise ValueError("every required and declared optional result needs a status")
    blockers = [
        {"result_id": name, "status": observed[name]["status"],
         "reason": observed[name]["reason"]}
        for name in REQUIRED_RESULTS
        if observed[name]["status"] != "SUCCEEDED"
    ]
    return {
        "schema": "trustforge.paper01.evaluation-status-inventory.v1",
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "records": [observed[name] for name in sorted(observed)],
        "required_result_count": len(REQUIRED_RESULTS),
        "declared_required_success_count": sum(
            observed[name]["status"] == "SUCCEEDED"
            for name in REQUIRED_RESULTS
        ),
        "required_status_blockers": blockers,
        "all_required_declared_succeeded": not blockers,
        "artifact_and_semantic_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
