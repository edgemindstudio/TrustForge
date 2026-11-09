"""Actual Inception/FID/MS-SSIM execution on one synthetic fixture pool."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)


def read(path):
    require(path.is_file() and not path.is_symlink(), "unavailable: " + str(path))
    return path.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    from trustforge.evaluation.paper01_classifier_tensorflow import require_compute_node
    from trustforge.models.paper01_revision_2026.ar_maf_training_runner import (
        capture_source_identity,
    )
    require_compute_node()
    root = Path(args.repository_root).resolve()
    require(root == Path.cwd().resolve(), "checkout mismatch")
    source = capture_source_identity(root)
    require(source[0] == args.expected_head, "revision mismatch")
    run = args.run_dir.resolve(strict=True)
    child = run / "inception"
    child.mkdir(mode=0o700)

    subprocess.run([
        sys.executable, "-B", "-u",
        str(root / "scripts/check_paper01_fidelity_integration.py"),
        "--repository-root", str(root),
        "--expected-head", args.expected_head,
        "--weights", args.weights,
        "--run-dir", str(child),
    ], check=True)

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_cic_pool import CICPoolView
    from trustforge.evaluation.paper01_classifier_control import state_digest
    from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid
    from trustforge.evaluation.paper01_ms_ssim import ThreeScaleMSSSIM
    from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
    from trustforge.evaluation.paper01_ms_ssim_evaluation import evaluate_ms_ssim_pairs
    from trustforge.evaluation.paper01_fidelity_inventory import verify_fidelity_inventory
    from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS

    child_report_raw = read(child / "report.json")
    child_report = json.loads(child_report_raw)
    require(child_report["source_commit"] == args.expected_head, "child source")
    require(child_report["job_id"] == os.environ["SLURM_JOB_ID"], "child job")
    require(child_report["actual_inception_used"] is True, "actual extractor")

    features = {}
    for name in ("real_features", "synthetic_features"):
        encoded = read(child / (name + ".npy"))
        require(
            sha(encoded) == child_report["feature_file_sha256"][name + ".npy"],
            "feature file changed",
        )
        features[name] = np.load(io.BytesIO(encoded), allow_pickle=False)

    manifest_path = child / "fixture_pool/manifest.json"
    manifest_raw = read(manifest_path)
    manifest_sha = sha(manifest_raw)
    require(manifest_sha == child_report["pool_manifest_sha256"], "pool identity")
    manifest = json.loads(manifest_raw)
    pool = CICPoolView(
        manifest_path, manifest_sha256=manifest_sha,
        family=manifest["family"], experimental_seed=42,
        recipe_sha256=manifest["recipe_sha256"],
        checkpoint_sha256=manifest["checkpoint_sha256"],
    )
    real_labels = np.repeat(np.arange(5, dtype=np.int32), [3, 2, 2, 2, 2])
    synthetic_labels = np.repeat(np.arange(5, dtype=np.int32), 2000)
    selection_raw = read(child / "selection.json")
    require(
        sha(selection_raw) == child_report["selection_file_sha256"],
        "reference selection changed",
    )

    pair_selection_raw = canonical(select_ms_ssim_pairs(
        labels=synthetic_labels, class_count=5, manifest_sha256=manifest_sha,
    ))
    write(run / "pair_selection.json", pair_selection_raw)
    # Persist selection before constructing or invoking the actual scorer.
    scorer = ThreeScaleMSSSIM()
    ms = evaluate_ms_ssim_pairs(
        selection_raw=pair_selection_raw,
        selection_sha256=sha(pair_selection_raw),
        synthetic_view=pool, labels=synthetic_labels, scorer=scorer,
    )
    scores = np.asarray([
        ms["per_class"][str(c)]["scores"] for c in range(5)
    ], dtype=np.float32)
    with (run / "pair_scores.npy").open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)

    real, synthetic = features["real_features"], features["synthetic_features"]
    selected_labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    kid = child_report["fidelity"]
    kid_raw = canonical(kid)
    configuration = {
        "selection_file_sha256": sha(selection_raw),
        "feature_extractor": kid["feature_extractor"],
    }
    fid_configuration = {
        **configuration, "kid_result_file_sha256": sha(kid_raw),
    }
    groups = {
        "pooled_balanced_kid": (kid, configuration, 10),
        "pooled_balanced_fid": (
            feature_fid(real, synthetic), fid_configuration, 1,
        ),
        "class_fid": (
            complete_class_fid(
                real, synthetic, real_labels=selected_labels,
                synthetic_labels=selected_labels, class_count=5,
            ), fid_configuration, 5,
        ),
        "intra_class_ms_ssim": (
            ms, {
                "selection_file_sha256": sha(pair_selection_raw),
                "scorer": scorer.identity,
            }, 1000,
        ),
    }
    records, artifacts = [], {}
    for name in REQUIRED_RESULTS:
        if name not in groups:
            records.append({
                "result_id": name, "status": "PENDING",
                "reason": "classifier evaluation outside this fixture",
                "result_file_sha256": None,
                "configuration_sha256": None, "coverage": None,
            })
            continue
        value, settings, count = groups[name]
        result_raw, configuration_raw = canonical(value), canonical(settings)
        write(run / (name + ".json"), result_raw)
        write(run / (name + ".configuration.json"), configuration_raw)
        records.append({
            "result_id": name, "status": "SUCCEEDED", "reason": None,
            "result_file_sha256": sha(result_raw),
            "configuration_sha256": sha(configuration_raw),
            "coverage": {"expected": count, "computed": count},
        })
        artifacts[name] = {
            "result_raw": result_raw, "configuration_raw": configuration_raw,
        }
    write(run / "status_inventory.json", canonical(records))

    verified = verify_fidelity_inventory(
        dataset_id="cicmaldroid2020", experimental_seed=42,
        records=records, artifact_bytes=artifacts,
        expected_synthetic_manifest_sha256=manifest_sha,
        kid_inputs={
            "selection_raw": selection_raw,
            "expected_selection_file_sha256": sha(selection_raw),
            "expected_real_validation_identity": {
                "dataset_id": "cicmaldroid2020", "split": "validation",
                "size": 11, "class_count": 5,
                "declared_provenance_sha256": sha(
                    b"ENGINEERING_FIXTURE_NO_REAL_DATA_PROVENANCE"
                ),
            },
            "expected_synthetic_manifest_sha256": manifest_sha,
            "real_validation_labels": real_labels,
            "synthetic_labels": synthetic_labels,
            "real_features": real, "synthetic_features": synthetic,
            "expected_feature_extractor": kid["feature_extractor"],
        },
        ms_ssim_inputs={
            "selection_raw": pair_selection_raw,
            "expected_selection_file_sha256": sha(pair_selection_raw),
            "labels": synthetic_labels,
            "expected_manifest_sha256": manifest_sha,
            "expected_scorer_identity": scorer.identity,
            "pair_scores": scores,
            "expected_pair_scores_state_sha256": state_digest([scores]),
        },
    )
    require(len(verified["fidelity_diversity_verifications"]) == 4, "group coverage")
    require(len(verified["required_status_blockers"]) == 3, "downstream blockers")
    require(verified["complete_evaluation"] is False, "completion boundary")
    require(capture_source_identity(root) == source, "source changed")
    write(run / "report.json", canonical({
        "scope": "SHARED_POOL_FIDELITY_DIVERSITY_SYNTHETIC_FIXTURE",
        "source_commit": source[0],
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "child_report_sha256": sha(child_report_raw),
        "pool_manifest_sha256": manifest_sha,
        "pair_scores_file_sha256": sha(read(run / "pair_scores.npy")),
        "pair_selection_persisted_before_scoring": True,
        "actual_inception_used": True,
        "actual_tensorflow_ms_ssim_used": True,
        "verification": verified,
        "real_data_used": False,
        "generator_lineage_authenticated": False,
        "primary_scientific_result": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }))
    print("SHARED_POOL_FIDELITY_DIVERSITY_FIXTURE=PASS", flush=True)


if __name__ == "__main__":
    main()
