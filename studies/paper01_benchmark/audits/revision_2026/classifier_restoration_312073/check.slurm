#!/usr/bin/env bash
#SBATCH --job-name=p1r1-classifier-restore
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08
set -euo pipefail
: "${TF_CLASSIFIER_RESTORE_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=5fd21c9a50a232fe6db67d84743bdf8e406ce710
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

python_bin="$HOME/miniforge3/envs/gcs_py310/bin/python"
common_args=(
    --repository-root "$PWD"
    --expected-head "$revision"
    --run-dir "$TF_CLASSIFIER_RESTORE_RUN_DIR"
)

"$python_bin" -B -u scripts/check_paper01_classifier_restoration.py \
    --phase train "${common_args[@]}"
check_source

receipt="$TF_CLASSIFIER_RESTORE_RUN_DIR/expectations.json"
[[ -f "$receipt" && ! -L "$receipt" ]]
receipt_sha=$(sha256sum -- "$receipt" | awk '{print $1}')
printf 'CLASSIFIER_RESTORE_EXPECTATIONS_SHA256=%s\n' "$receipt_sha"

"$python_bin" -B -u scripts/check_paper01_classifier_restoration.py \
    --phase restore \
    --receipt-sha256 "$receipt_sha" \
    "${common_args[@]}"
check_source

printf 'CLASSIFIER_TWO_PROCESS_CHECK=PASS\n'
