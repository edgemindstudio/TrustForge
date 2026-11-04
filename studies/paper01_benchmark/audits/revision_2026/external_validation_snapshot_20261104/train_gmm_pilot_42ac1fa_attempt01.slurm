#!/bin/bash
#SBATCH --job-name=p1r1-gmm-a01
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=p1r1-gmm-a01.%j.out
#SBATCH --error=p1r1-gmm-a01.%j.err

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
runner = home / "trustforge_pretraining_validation/train_gmm_pilot_42ac1fa_attempt01.py"
checkpoint = (
    home / "trustforge_revision_2026_artifacts"
    / "paper01/gmm/malex/seed42/pilots/class0_seed42_commit42ac1fa_attempt01"
)

expected_runner = "f8eac15b9ff0297fd4b81c6fbe31f61664874237772977427559bb64dfbc0caf"
expected_head = "42ac1fa794b6a2aeb7894881cc0cdb61ff77fa38"

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

python -u "$HOME/trustforge_pretraining_validation/train_gmm_pilot_42ac1fa_attempt01.py"     --dataset malex     --class-id 0     --seed 42     --checkpoint-directory "$HOME/trustforge_revision_2026_artifacts/paper01/gmm/malex/seed42/pilots/class0_seed42_commit42ac1fa_attempt01"

echo "===== GMM PILOT PROCESS COMPLETE ====="
