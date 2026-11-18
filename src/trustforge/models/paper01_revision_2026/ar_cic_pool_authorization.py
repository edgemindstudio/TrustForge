"""Default-deny authorization evidence for the bounded AR CIC seed-42 pool.

This module only validates an externally supplied authorization record. It
does not create approvals, submit jobs, load checkpoints or publish pools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat


SCHEMA = "trustforge.paper01.ar-cic-seed42-pool-authorization.v1"
PLACEHOLDER_ID = "RESEARCHER_DECISION_PENDING"
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_AUTHORIZATION_BYTES = 65536

TOP_LEVEL_FIELDS = frozenset({
    "schema", "paper", "revision", "family", "dataset_id",
    "experimental_seed", "attempt", "checkpoint_sha256",
    "dataset_config_sha256", "recipe_sha256", "synthesis_commit",
    "run_root", "destination", "expected_task_count",
    "expected_batch_count", "resources", "authorization_status",
    "authorization_id",
})
RESOURCE_FIELDS = frozenset({
    "partition", "tasks", "concurrency", "cpus_per_task", "memory_gb",
    "walltime_seconds", "batches_per_task", "automatic_retries",
    "requeue", "gpu_enabled",
})


def _fail(message):
    raise PermissionError(message)


def _require_regular(path, name):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail(name + " must be a regular file")
    info = path.stat()
    if info.st_uid != os.getuid():
        _fail(name + " has a different owner")
    if stat.S_IMODE(info.st_mode) & 0o022:
        _fail(name + " is group/world writable")
    if info.st_size > MAX_AUTHORIZATION_BYTES:
        _fail(name + " is too large")
    return path


def _reject_duplicates(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            _fail("duplicate authorization JSON field: " + key)
        record[key] = value
    return record


def _reject_constant(value):
    _fail("non-finite authorization JSON constant: " + str(value))


def _require_digest(value, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        _fail(name + " requires a lowercase SHA-256")
    return value


def _require_commit(value, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        _fail(name + " requires a 40-character commit")
    return value


def validate_ar_cic_pool_authorization(path, *, plan, run_root, destination):
    """Return the validated record or raise PermissionError.

    The record must be owner-controlled and bind the exact plan, source,
    checkpoint, configuration, recipe, namespace and publication destination.
    """
    authorization = _require_regular(path, "AR CIC pool authorization")
    try:
        record = json.loads(
            authorization.read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeError, ValueError):
        _fail("AR CIC pool authorization is unreadable")
    if type(record) is not dict or set(record) != TOP_LEVEL_FIELDS:
        _fail("AR CIC pool authorization fields are invalid")

    expected_text = {
        "schema": SCHEMA,
        "paper": "paper01",
        "revision": "revision_2026",
        "family": "ar",
        "dataset_id": "cicmaldroid2020",
        "authorization_status": "APPROVED",
    }
    for key, value in expected_text.items():
        if record.get(key) != value:
            _fail("AR CIC pool authorization field differs: " + key)
    if type(record.get("experimental_seed")) is not int or (
        record["experimental_seed"] != 42
    ):
        _fail("AR CIC pool authorization seed is not 42")
    if type(record.get("attempt")) is not int or record["attempt"] != 1:
        _fail("AR CIC pool authorization attempt is not 1")
    authorization_id = record.get("authorization_id")
    if (
        type(authorization_id) is not str
        or authorization_id == PLACEHOLDER_ID
        or AUTHORIZATION_ID.fullmatch(authorization_id) is None
    ):
        _fail("AR CIC pool authorization is not approved")

    for key in ("checkpoint_sha256", "dataset_config_sha256", "recipe_sha256"):
        _require_digest(record.get(key), key)
    _require_commit(record.get("synthesis_commit"), "synthesis_commit")

    run_root = Path(run_root).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if record["run_root"] != str(run_root) or record["run_root"] != plan["run_root"]:
        _fail("AR CIC pool authorization run root differs")
    if record["destination"] != str(destination) or record["destination"] != plan["destination"]:
        _fail("AR CIC pool authorization destination differs")
    if record["synthesis_commit"] != plan["synthesis_commit"]:
        _fail("AR CIC pool authorization synthesis source differs")
    plan_keys = {
        "checkpoint_sha256": "checkpoint_sha256",
        "dataset_config_sha256": "config_sha256",
        "recipe_sha256": "recipe_sha256",
    }
    for key, plan_key in plan_keys.items():
        if record[key] != plan[plan_key]:
            _fail("AR CIC pool authorization identity differs: " + key)

    if type(record.get("expected_task_count")) is not int or (
        record["expected_task_count"] != len(plan["tasks"])
    ):
        _fail("AR CIC pool authorization task count differs")
    if type(record.get("expected_batch_count")) is not int or (
        record["expected_batch_count"] != plan["total_batches"]
    ):
        _fail("AR CIC pool authorization batch count differs")

    resources = record.get("resources")
    if type(resources) is not dict or set(resources) != RESOURCE_FIELDS:
        _fail("AR CIC pool authorization resource fields are invalid")
    for key in RESOURCE_FIELDS:
        expected = plan["resources"][key]
        if type(resources[key]) is not type(expected) or resources[key] != expected:
            _fail("AR CIC pool authorization resource differs: " + key)

    return record


__all__ = [
    "SCHEMA",
    "TOP_LEVEL_FIELDS",
    "RESOURCE_FIELDS",
    "validate_ar_cic_pool_authorization",
]
