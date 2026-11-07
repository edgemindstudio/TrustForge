#!/usr/bin/env bash
#SBATCH --job-name=p1r1-inception-feat
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08

set -euo pipefail
: "${TF_INCEPTION_FEATURE_RUN_DIR:?Run directory required}"
: "${TF_INCEPTION_WEIGHTS:?Weight path required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=c4d4d8f24c8dde9f98918af0582080a48a3690e8

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
    scripts/check_paper01_inception_features.py \
    --repository-root "$PWD" \
    --expected-head "$revision" \
    --weights "$TF_INCEPTION_WEIGHTS" \
    --output "$TF_INCEPTION_FEATURE_RUN_DIR/report.json"

check_source
printf 'INCEPTION_FEATURE_JOB=PASS\n'
