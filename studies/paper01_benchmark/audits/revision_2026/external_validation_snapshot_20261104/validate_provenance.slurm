#!/bin/bash
#SBATCH --job-name=p1r1-provenance
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=p1r1-provenance.%j.out
#SBATCH --error=p1r1-provenance.%j.err

set -euo pipefail

cd "$HOME/TrustForge"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gcs_py310

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "===== TRUSTFORGE GOVERNED PROVENANCE PREFLIGHT ====="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-UNKNOWN}"
echo "HOST: $(hostname)"
echo "HEAD: $(git rev-parse HEAD)"

python -u "$HOME/trustforge_pretraining_validation/validate_provenance.py"

echo "===== PROVENANCE PREFLIGHT COMPLETE ====="
