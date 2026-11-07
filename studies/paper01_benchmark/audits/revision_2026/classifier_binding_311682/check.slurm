#!/usr/bin/env bash
#SBATCH --job-name=p1r1-classifier-bind
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08

set -euo pipefail
: "${TF_CLASSIFIER_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=69a6bc0beacd9e459eb39d6cd4a7312616907b55

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
    scripts/check_paper01_classifier_tensorflow.py \
    --repository-root "$PWD" \
    --expected-head "$revision" \
    --output "$TF_CLASSIFIER_RUN_DIR/report.json"

check_source
printf 'CLASSIFIER_BINDING_JOB=PASS\n'
