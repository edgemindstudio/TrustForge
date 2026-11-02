#!/usr/bin/env bash
#SBATCH -J p1r1-rbm-gradient
#SBATCH -o p1r1-rbm-gradient.%j.out
#SBATCH -e p1r1-rbm-gradient.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=talon06,talon12,talon14

set -euo pipefail

module purge
module load slurm/slurm/25.05.6

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gcs_py310

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"

if [[ ! -d .git ]]; then
    echo "[error] submission directory is not TrustForge root" >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[error] Slurm job identity is required" >&2
    exit 2
fi

OUTPUT_JSON="$PWD/studies/paper01_benchmark/audits/revision_2026/rbm_gradient_diagnostic_cicmaldroid2020_seed42.json"

if [[ -e "$OUTPUT_JSON" || -L "$OUTPUT_JSON" ]]; then
    echo "[error] refusing existing diagnostic artifact" >&2
    exit 3
fi

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[P1 RBM GRADIENT] job_id=${SLURM_JOB_ID}"
echo "[P1 RBM GRADIENT] node=${SLURM_JOB_NODELIST:-unknown}"
echo "[P1 RBM GRADIENT] dataset=cicmaldroid2020"
echo "[P1 RBM GRADIENT] seed=42"
echo "[P1 RBM GRADIENT] budget=64_UNMODIFIED"
echo "[P1 RBM GRADIENT] authoritative_checkpoint=READ_ONLY"
echo "[P1 RBM GRADIENT] output=${OUTPUT_JSON}"

python -V

python scripts/run_paper01_rbm_gradient_diagnostic.py

echo "[P1 RBM GRADIENT] slurm_status=COMPLETE"
