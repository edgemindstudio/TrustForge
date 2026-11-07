#!/usr/bin/env bash
#SBATCH --job-name=p1r1-paired-classifier
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --exclude=talon08

set -euo pipefail
: "${TF_PAIRED_CLASSIFIER_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=ff6bbb416c281cedfbd7d4a8ca26a5b274058dcf

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
    scripts/check_paper01_paired_classifier.py \
    --repository-root "$PWD" \
    --expected-head "$revision" \
    --run-dir "$TF_PAIRED_CLASSIFIER_RUN_DIR"

check_source
printf 'PAIRED_CLASSIFIER_JOB=PASS\n'
