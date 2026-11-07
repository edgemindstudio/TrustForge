#!/usr/bin/env bash
#SBATCH --job-name=p1r1-ms-ssim-link
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08
set -euo pipefail
: "${TF_MS_SSIM_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"
cd "$HOME/TrustForge"
revision=0c882596cc37e1180938f0d9f13ad18565f8e2b8
check_source() {
    [[ "$(git rev-parse HEAD)" == "$revision" ]]
    git diff --quiet
    git diff --cached --quiet
}
check_source
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u <<'PY'
import hashlib
import json
import os
from pathlib import Path
import socket

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

from trustforge.evaluation.paper01_classifier_tensorflow import require_compute_node
require_compute_node()

import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(
    int(os.environ["SLURM_CPUS_PER_TASK"])
)
tf.config.threading.set_inter_op_parallelism_threads(1)

from trustforge.evaluation.paper01_cic_pool import CICPoolView
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_ms_ssim import ThreeScaleMSSSIM
from trustforge.evaluation.paper01_ms_ssim_evaluation import evaluate_ms_ssim_pairs

run = Path(os.environ["TF_MS_SSIM_RUN_DIR"])
manifest_path = (
    Path.home()
    / "trustforge_revision_2026_artifacts/paper01/diagnostics"
    / "paired-classifier-ff6bbb4-6j3j8sX7/fixture_pool/manifest.json"
)
manifest_raw = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
require(
    manifest_sha
    == "53896ef041c03eaa99ea385fb00068efbb189e994e89ea5a0464edfd79951c38",
    "fixture manifest identity",
)
manifest = json.loads(manifest_raw)
view = CICPoolView(
    manifest_path,
    manifest_sha256=manifest_sha,
    family="maf",
    experimental_seed=42,
    recipe_sha256=manifest["recipe_sha256"],
    checkpoint_sha256=manifest["checkpoint_sha256"],
)
labels = np.repeat(np.arange(5, dtype=np.int64), 2000)
selection = select_ms_ssim_pairs(
    labels=labels, class_count=5, manifest_sha256=manifest_sha,
)
selection_raw = json.dumps(
    selection, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode() + bytes([10])
selection_path = run / "selection.json"
with selection_path.open("xb") as handle:
    handle.write(selection_raw)
selection_sha = hashlib.sha256(selection_raw).hexdigest()

scorer = ThreeScaleMSSSIM()
result = evaluate_ms_ssim_pairs(
    selection_raw=selection_path.read_bytes(),
    selection_sha256=selection_sha,
    synthetic_view=view,
    labels=labels,
    scorer=scorer,
)
require(result["class_coverage_complete"] is True, "class coverage")
require(result["pair_count"] == 1000, "total pair count")
require(set(result["per_class"]) == {str(c) for c in range(5)},
        "class inventory")
for record in result["per_class"].values():
    require(record["pair_count"] == len(record["scores"]) == 200,
            "class pair coverage")
    require(np.allclose(record["scores"], 1, rtol=0, atol=1e-6),
            "identical fixture image scores")
require(np.isclose(result["macro_mean"], 1, rtol=0, atol=1e-6),
        "fixture macro mean")
require(hashlib.sha256(selection_path.read_bytes()).hexdigest() == selection_sha,
        "persisted selection changed")

report = {
    "scope": "PERSISTED_PAIR_MS_SSIM_SYNTHETIC_FIXTURE",
    "source_commit": "0c882596cc37e1180938f0d9f13ad18565f8e2b8",
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "selection_file_sha256": selection_sha,
    "fixture_manifest_sha256": manifest_sha,
    "selection_persisted_before_scoring": True,
    "actual_pool_view_used": True,
    "actual_tensorflow_scorer_used": True,
    "identical_fixture_scores_passed": True,
    "evaluation": result,
    "generator_lineage_authenticated": False,
    "primary_scientific_result": False,
    "complete_evaluation": False,
    "full_benchmark_go": False,
}
with (run / "report.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, allow_nan=False)
    handle.write("\n")
print("MS_SSIM_PERSISTED_PAIR_FIXTURE=PASS")
PY

check_source
printf 'MS_SSIM_INTEGRATION_JOB=PASS\n'
