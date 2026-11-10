#!/usr/bin/env bash
#SBATCH --job-name=p1r1-shared-fidelity
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08
set -euo pipefail
: "${TF_SHARED_FIDELITY_RUN_DIR:?Run directory required}"
: "${TF_INCEPTION_WEIGHTS:?Weight path required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=804686c23eac27f111f6a2182d58b4d563601932
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

"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u \
    scripts/check_paper01_shared_fidelity.py \
    --repository-root "$PWD" \
    --expected-head "$revision" \
    --weights "$TF_INCEPTION_WEIGHTS" \
    --run-dir "$TF_SHARED_FIDELITY_RUN_DIR"

check_source
printf 'SHARED_FIDELITY_INTEGRATION_JOB=PASS\n'
