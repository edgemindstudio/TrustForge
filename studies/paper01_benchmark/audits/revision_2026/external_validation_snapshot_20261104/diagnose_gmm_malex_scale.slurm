#!/bin/bash
#SBATCH --job-name=p1r1-gmm-scale
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=p1r1-gmm-scale.%j.out
#SBATCH --error=p1r1-gmm-scale.%j.err

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

python -u "$HOME/trustforge_pretraining_validation/diagnose_gmm_malex_scale.py"
