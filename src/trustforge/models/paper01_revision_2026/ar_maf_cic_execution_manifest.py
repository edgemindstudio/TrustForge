"""CIC execution envelope binding an unchanged v1 checkpoint to fitted state.

No publication, source authentication, training or operational authorization.
Actual execution must establish the supplied evidence before calling here.
"""

import hashlib
import json
import re

from .ar_maf_checkpoint_identity import canonical_json_bytes
from .ar_maf_checkpoint_bundle import verify_checkpoint_bundle_bytes
from .gan_v3_provenance_adapter import build_gan_v3_provenance


SCHEMA = "trustforge.paper01.ar-maf.cic-execution-manifest.v1"
MAX_EXECUTION_MANIFEST_BYTES = 4 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _validate_binding(checkpoint, fitted_state, expected_fitted_state_sha256):
    if (
        type(expected_fitted_state_sha256) is not str
        or HEX64.fullmatch(expected_fitted_state_sha256) is None
    ):
        raise ValueError("independently expected fitted-state identity required")

    from trustforge.data.paper01_revision_2026.cic_fitted_state import (
        restore_fitted_state,
    )
    restore_fitted_state(fitted_state)

    if fitted_state["sha256"] != expected_fitted_state_sha256:
        raise ValueError("requested fitted-state identity mismatch")

    identity = checkpoint["identity"]
    if identity["dataset_id"] != "cicmaldroid2020":
        raise ValueError("CIC execution envelope requires CIC checkpoint")
    if identity["family"] not in ("ar", "maf"):
        raise ValueError("unsupported execution family")

    context = fitted_state["payload"]["declared_context"]
    implementation = context["implementation"]
    if (
        implementation["commit"] != checkpoint["source_commit"]
        or canonical_json_bytes(implementation["tracked_source_files"])
        != canonical_json_bytes(checkpoint["source_files"])
    ):
        raise ValueError("fitted-state and checkpoint source identities differ")

    if context["partition"] != {
        "seed": 42, "ratios": [0.7, 0.1, 0.2],
    }:
        raise ValueError("unexpected fitted-state partition policy")

    features = context["feature_columns"]
    if (
        type(features) is not list
        or len(features) != 139
        or any(type(name) is not str or not name for name in features)
        or len(set(features)) != 139
    ):
        raise ValueError("invalid ordered feature declarations")

    configuration_digest = context["configuration_sha256"]
    if (
        type(configuration_digest) is not str
        or HEX64.fullmatch(configuration_digest) is None
    ):
        raise ValueError("invalid configuration digest")

    environment = context["environment"]
    expected_fields = {
        "python", "numpy", "pandas", "scikit_learn",
        "tensorflow", "byteorder",
    }
    if (
        type(environment) is not dict
        or set(environment) != expected_fields
        or any(type(value) is not str or not value for value in environment.values())
        or environment["byteorder"] not in ("little", "big")
    ):
        raise ValueError("incomplete fitted-state environment")

    provenance = build_gan_v3_provenance(
        dataset_id="cicmaldroid2020",
        verified_preflight=context["source_preflight"],
    )
    if canonical_json_bytes(provenance) != canonical_json_bytes(
        identity["dataset_provenance"]
    ):
        raise ValueError("fitted-state and checkpoint dataset provenance differ")


def prepare_cic_execution_manifest(
    *, checkpoint_manifest_raw, fitted_state_raw,
    best_raw, last_raw, recipe_raw,
    expected_identity, expected_source_commit, expected_source_files,
    expected_fitted_state_sha256,
):
    """Validate both records and return canonical execution-envelope bytes."""
    verified = verify_checkpoint_bundle_bytes(
        manifest_raw=checkpoint_manifest_raw,
        best_raw=best_raw,
        last_raw=last_raw,
        recipe_raw=recipe_raw,
        expected_identity=expected_identity,
        expected_source_commit=expected_source_commit,
        expected_source_files=expected_source_files,
    )
    if type(fitted_state_raw) is not bytes or not (
        0 < len(fitted_state_raw) <= MAX_EXECUTION_MANIFEST_BYTES
    ):
        raise ValueError("invalid fitted-state record size")

    fitted_state = json.loads(fitted_state_raw)
    _validate_binding(
        verified["manifest"], fitted_state, expected_fitted_state_sha256,
    )
    envelope = {
        "schema": SCHEMA,
        "checkpoint_manifest": verified["manifest"],
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_raw),
        "fitted_state": fitted_state,
        "fitted_state_file_sha256": _sha256(fitted_state_raw),
    }
    # Preserve the exact captured file bytes in canonical JSON via ASCII hex.
    envelope["fitted_state_file_hex"] = fitted_state_raw.hex()
    raw = canonical_json_bytes(envelope) + b"\n"
    if len(raw) > MAX_EXECUTION_MANIFEST_BYTES:
        raise ValueError("execution manifest exceeds size limit")
    return raw


def verify_cic_execution_manifest(
    *, execution_manifest_raw, best_raw, last_raw, recipe_raw,
    expected_identity, expected_source_commit, expected_source_files,
    expected_fitted_state_sha256,
):
    """Reject missing, changed or mismatched execution evidence."""
    if type(execution_manifest_raw) is not bytes or not (
        0 < len(execution_manifest_raw) <= MAX_EXECUTION_MANIFEST_BYTES
    ):
        raise ValueError("invalid execution manifest size")
    envelope = json.loads(execution_manifest_raw)
    if type(envelope) is not dict or set(envelope) != {
        "schema", "checkpoint_manifest", "checkpoint_manifest_sha256",
        "fitted_state", "fitted_state_file_sha256", "fitted_state_file_hex",
    }:
        raise ValueError("invalid execution manifest inventory")
    if envelope["schema"] != SCHEMA:
        raise ValueError("unsupported execution manifest schema")

    checkpoint_raw = canonical_json_bytes(envelope["checkpoint_manifest"]) + b"\n"
    fitted_raw = bytes.fromhex(envelope["fitted_state_file_hex"])
    rebuilt = prepare_cic_execution_manifest(
        checkpoint_manifest_raw=checkpoint_raw,
        fitted_state_raw=fitted_raw,
        best_raw=best_raw,
        last_raw=last_raw,
        recipe_raw=recipe_raw,
        expected_identity=expected_identity,
        expected_source_commit=expected_source_commit,
        expected_source_files=expected_source_files,
        expected_fitted_state_sha256=expected_fitted_state_sha256,
    )
    if rebuilt != execution_manifest_raw:
        raise ValueError("execution manifest differs from verified evidence")
    return {
        "manifest": envelope["checkpoint_manifest"],
        "execution_manifest": envelope,
        "best_raw": best_raw,
        "last_raw": last_raw,
        "fitted_state_raw": fitted_raw,
    }
