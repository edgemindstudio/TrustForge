"""Verify fidelity/diversity inventory bytes, semantics and shared identities.

Coverage units are KID draws, pooled FID results, classes and MS-SSIM pairs.
Manifest identities remain declared links; provenance is not authenticated.
"""

import json
import re

import numpy as np

from .paper01_inputs import CLASS_COUNTS
from .paper01_result_artifacts import bind_result_artifacts
from .paper01_kid_result_validation import verify_kid_result
from .paper01_ms_ssim_result_validation import verify_ms_ssim_result
from .paper01_fid import feature_fid, complete_class_fid


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def verify_fidelity_inventory(
    *, dataset_id, experimental_seed, records, artifact_bytes,
    expected_synthetic_manifest_sha256,
    kid_inputs=None, ms_ssim_inputs=None, optional_result_ids=(),
):
    bound = bind_result_artifacts(
        dataset_id=dataset_id, experimental_seed=experimental_seed,
        records=records, artifact_bytes=artifact_bytes,
        optional_result_ids=optional_result_ids,
    )
    manifest = expected_synthetic_manifest_sha256
    if type(manifest) is not str or re.fullmatch(r"[0-9a-f]{64}", manifest) is None:
        raise ValueError("explicit shared synthetic-manifest identity required")
    classes = CLASS_COUNTS[dataset_id]
    results = bound["bound_results"]
    receipts = {}

    def check(name, configuration, count):
        item = results[name]
        if (
            _canonical(item["configuration"]) != _canonical(configuration)
            or item["declared_coverage"]
            != {"expected": count, "computed": count}
        ):
            raise ValueError("configuration or coverage mismatch: " + name)

    kid_name = "pooled_balanced_kid"
    fid_names = {"pooled_balanced_fid", "class_fid"}
    if fid_names & set(results) and kid_name not in results:
        raise ValueError("FID verification requires successful KID evidence")

    if kid_name in results:
        fields = {
            "selection_raw", "expected_selection_file_sha256",
            "expected_real_validation_identity",
            "expected_synthetic_manifest_sha256",
            "real_validation_labels", "synthetic_labels",
            "real_features", "synthetic_features", "expected_feature_extractor",
        }
        if type(kid_inputs) is not dict or set(kid_inputs) != fields:
            raise ValueError("exact KID replay inputs required")
        identity = kid_inputs["expected_real_validation_identity"]
        if (
            type(identity) is not dict
            or identity.get("dataset_id") != dataset_id
            or kid_inputs["expected_synthetic_manifest_sha256"] != manifest
        ):
            raise ValueError("KID dataset or shared manifest mismatch")
        kid = verify_kid_result(
            result_raw=artifact_bytes[kid_name]["result_raw"],
            expected_result_file_sha256=results[kid_name]["result_file_sha256"],
            **kid_inputs,
        )
        configuration = {
            "selection_file_sha256": kid["selection_file_sha256"],
            "feature_extractor": kid_inputs["expected_feature_extractor"],
        }
        draws = results[kid_name]["result"]["kid"]["draws"]
        check(kid_name, configuration, len(draws))
        receipts[kid_name] = kid

        fid_configuration = {
            **configuration,
            "kid_result_file_sha256": kid["result_file_sha256"],
        }
        real = kid_inputs["real_features"]
        synthetic = kid_inputs["synthetic_features"]
        for name in sorted(fid_names & set(results)):
            if name == "pooled_balanced_fid":
                replayed = feature_fid(real, synthetic)
                count = 1
            else:
                labels = np.repeat(
                    np.arange(classes, dtype=np.int64), kid["samples_per_class"],
                )
                replayed = complete_class_fid(
                    real, synthetic, real_labels=labels,
                    synthetic_labels=labels, class_count=classes,
                )
                count = classes
            check(name, fid_configuration, count)
            if _canonical(results[name]["result"]) != _canonical(replayed):
                raise ValueError("FID semantic replay mismatch: " + name)
            receipts[name] = {
                "result_file_sha256": results[name]["result_file_sha256"],
                "kid_result_file_sha256": kid["result_file_sha256"],
                "selection_file_sha256": kid["selection_file_sha256"],
                "real_features_sha256": kid["real_features_sha256"],
                "synthetic_features_sha256": kid["synthetic_features_sha256"],
                "feature_sets_shared_with_kid": True,
                "semantic_replay_verified": True,
            }
    elif kid_inputs is not None:
        raise ValueError("KID inputs supplied without successful KID result")

    ms_name = "intra_class_ms_ssim"
    if ms_name in results:
        fields = {
            "selection_raw", "expected_selection_file_sha256", "labels",
            "expected_manifest_sha256", "expected_scorer_identity",
            "pair_scores", "expected_pair_scores_state_sha256",
        }
        if type(ms_ssim_inputs) is not dict or set(ms_ssim_inputs) != fields:
            raise ValueError("exact MS-SSIM replay inputs required")
        if ms_ssim_inputs["expected_manifest_sha256"] != manifest:
            raise ValueError("MS-SSIM shared manifest mismatch")
        ms = verify_ms_ssim_result(
            dataset_id=dataset_id,
            result_raw=artifact_bytes[ms_name]["result_raw"],
            expected_result_file_sha256=results[ms_name]["result_file_sha256"],
            **ms_ssim_inputs,
        )
        if kid_name in receipts and not np.array_equal(
            kid_inputs["synthetic_labels"], ms_ssim_inputs["labels"],
        ):
            raise ValueError("fidelity/diversity synthetic label order differs")
        check(ms_name, {
            "selection_file_sha256": ms["selection_file_sha256"],
            "scorer": ms_ssim_inputs["expected_scorer_identity"],
        }, classes * 200)
        receipts[ms_name] = ms
    elif ms_ssim_inputs is not None:
        raise ValueError("MS-SSIM inputs supplied without successful result")

    return {
        "schema": "trustforge.paper01.fidelity-inventory-verification.v1",
        "artifact_binding": bound,
        "synthetic_manifest_sha256": manifest,
        "fidelity_diversity_verifications": receipts,
        "semantic_validation_not_performed": sorted(set(results) - set(receipts)),
        "required_status_blockers": bound["status_inventory"][
            "required_status_blockers"
        ],
        "feature_extraction_authentication": "NOT_PERFORMED",
        "image_score_recomputation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "publication_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
