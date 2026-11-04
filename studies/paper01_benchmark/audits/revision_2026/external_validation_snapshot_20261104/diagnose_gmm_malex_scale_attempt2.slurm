#!/bin/bash
#SBATCH --job-name=p1r1-gmm-scale-a2
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=p1r1-gmm-scale-a2.%j.out
#SBATCH --error=p1r1-gmm-scale-a2.%j.err

set -euo pipefail

cd "$HOME/TrustForge"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gcs_py310

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="-1"
export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2


python - <<'VERIFY'
from pathlib import Path
import hashlib
import subprocess

repo = Path.home() / "TrustForge"
source = Path.home() / "trustforge_pretraining_validation/diagnose_gmm_malex_scale.py"

expected_head = "511ec813d3e119ded1ac859f31b301039b067017"
expected_source = "51af82fe203fc02f8cdb304bce7001f87b084a233a62ce8ceb869e32e632d51b"

observed_head = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    text=True
).strip()

if observed_head != expected_head:
    raise SystemExit("FAIL: compute-node repository revision changed")

if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source:
    raise SystemExit("FAIL: compute-node diagnostic fingerprint changed")

print("GMM_DIAGNOSTIC_ATTEMPT2_PREFLIGHT: PASS", flush=True)
VERIFY

python -u "$HOME/trustforge_pretraining_validation/diagnose_gmm_malex_scale.py"
