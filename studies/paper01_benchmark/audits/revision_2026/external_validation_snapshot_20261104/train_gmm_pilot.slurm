#!/bin/bash
#SBATCH --job-name=p1r1-gmm-pilot
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=p1r1-gmm-pilot.%j.out
#SBATCH --error=p1r1-gmm-pilot.%j.err

set -euo pipefail

cd "$HOME/TrustForge"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gcs_py310

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTEROP_THREADS=2

echo "===== TRUSTFORGE GMM PILOT ====="
echo "JOB_ID: ${SLURM_JOB_ID:?}"
echo "HOST: $(hostname)"
echo "HEAD: $(git rev-parse HEAD)"

python - <<'VERIFY'
from pathlib import Path
import hashlib
import subprocess

home = Path.home()
runner = home / "trustforge_pretraining_validation/train_gmm_pilot.py"
checkpoint = (
    home / "trustforge_revision_2026_artifacts"
    / "paper01/gmm/malex/seed42/class0"
)

expected_runner = "fbd38ec2d97d9cc5ffa178d96a777f494f380244a4089e0a31f476431125a0ed"
expected_head = "511ec813d3e119ded1ac859f31b301039b067017"

if hashlib.sha256(runner.read_bytes()).hexdigest() != expected_runner:
    raise SystemExit("FAIL: runner fingerprint mismatch")

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

if head != expected_head:
    raise SystemExit("FAIL: repository HEAD mismatch")

if checkpoint.exists() or checkpoint.is_symlink():
    raise SystemExit("FAIL: checkpoint destination occupied")

if not checkpoint.parent.is_dir():
    raise SystemExit("FAIL: checkpoint parent missing")

print("GMM_COMPUTE_NODE_PREFLIGHT: PASS", flush=True)
VERIFY

python -u "$HOME/trustforge_pretraining_validation/train_gmm_pilot.py"     --dataset malex     --class-id 0     --seed 42     --checkpoint-directory "$HOME/trustforge_revision_2026_artifacts/paper01/gmm/malex/seed42/class0"

echo "===== GMM PILOT PROCESS COMPLETE ====="
