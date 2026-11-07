#!/usr/bin/env bash
#SBATCH --job-name=p1r1-fid-features
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
: "${TF_FID_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Allocation required}"
cd "$HOME/TrustForge"
revision=483b28940572ce183215d5ad986a1211dd333b20
[[ "$(git rev-parse HEAD)" == "$revision" ]]
git diff --quiet
git diff --cached --quiet
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u <<'PY'
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import numpy as np
from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(not socket.gethostname().split(".")[0].startswith("login"),
        "compute node required")
source = (
    Path.home()
    / "trustforge_revision_2026_artifacts/paper01/diagnostics"
    / "fidelity-integration-a9946a6-dbjKE8oO"
)
expected = {
    "real_features.npy":
        "99c55839f6c30fb158606df158d9c751f43ffd8dfb42ab6aad9dd4fb019142b0",
    "synthetic_features.npy":
        "514aa71bceaf708f362a4e92e9d78ea96191642a187ae7dd3f06dd1d7eb7dc75",
    "selection.json":
        "9774a6ff1160a805560d22b35983ceb300934a0ad765913ee0ba6ff2574d7927",
}
raw = {}
for name, digest in expected.items():
    path = source / name
    require(path.is_file() and not path.is_symlink(), "unavailable: " + name)
    raw[name] = path.read_bytes()
    require(hashlib.sha256(raw[name]).hexdigest() == digest,
            "digest mismatch: " + name)

real = np.load(io.BytesIO(raw["real_features.npy"]), allow_pickle=False)
synthetic = np.load(io.BytesIO(raw["synthetic_features.npy"]), allow_pickle=False)
for array in (real, synthetic):
    require(array.shape == (10, 2048) and array.dtype == np.float32
            and np.isfinite(array).all(), "feature contract")
labels = np.repeat(np.arange(5, dtype=np.int64), 2)

def analytic_constant_synthetic(r, s):
    r, s = r.astype(np.float64), s.astype(np.float64)
    require(np.array_equal(s, np.repeat(s[:1], len(s), axis=0)),
            "constant synthetic fixture changed")
    return float(
        np.sum((r.mean(axis=0) - s[0]) ** 2)
        + np.sum((r - r.mean(axis=0)) ** 2) / (len(r) - 1)
    )

pooled = feature_fid(real, synthetic)
classes = complete_class_fid(
    real, synthetic, real_labels=labels,
    synthetic_labels=labels, class_count=5,
)
analytic = {"pooled": analytic_constant_synthetic(real, synthetic)}
require(math.isclose(pooled["fid"], analytic["pooled"],
                     rel_tol=1e-10, abs_tol=1e-10), "pooled analytic agreement")
for c in range(5):
    value = analytic_constant_synthetic(real[labels == c], synthetic[labels == c])
    analytic[str(c)] = value
    result = classes["per_class"][str(c)]
    require(math.isclose(result["fid"], value, rel_tol=1e-10, abs_tol=1e-10),
            "class analytic agreement")
    require(result["diagonal_regularization"] == 0, "unexpected regularization")

require(math.isclose(
    classes["macro_cfid"], float(np.mean([analytic[str(c)] for c in range(5)])),
    rel_tol=1e-10, abs_tol=1e-10,
), "macro agreement")
try:
    complete_class_fid(
        real, synthetic, real_labels=labels,
        synthetic_labels=np.zeros(10, dtype=np.int64), class_count=5,
    )
except ValueError:
    missing_rejected = True
else:
    raise RuntimeError("missing class accepted")

report = {
    "scope": "SAVED_INCEPTION_FEATURE_FID_FIXTURE",
    "source_commit": "483b28940572ce183215d5ad986a1211dd333b20",
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "numpy": np.__version__,
    "input_file_sha256": expected,
    "pooled": pooled,
    "class_fid": classes,
    "analytic_reference": analytic,
    "analytic_agreement_passed": True,
    "missing_class_rejected": missing_rejected,
    "real_data_used": False,
    "primary_scientific_result": False,
    "complete_evaluation": False,
    "full_benchmark_go": False,
}
output = Path(os.environ["TF_FID_RUN_DIR"]) / "report.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, allow_nan=False)
    handle.write("\n")
print(json.dumps(report, sort_keys=True, allow_nan=False))
print("SAVED_FEATURE_FID_FIXTURE=PASS")
PY

[[ "$(git rev-parse HEAD)" == "$revision" ]]
git diff --quiet
git diff --cached --quiet
printf 'FID_FEATURE_JOB=PASS\n'
