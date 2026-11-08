#!/usr/bin/env bash
#SBATCH --job-name=p1r1-paired-restore
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --exclude=talon08
set -euo pipefail
: "${TF_PAIRED_RESTORE_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=9ba6c26d8ad93d6fc4dc592c12d306c4970d3751
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
    --run-dir "$TF_PAIRED_RESTORE_RUN_DIR"
)

"$python_bin" -B -u scripts/check_paper01_paired_restoration.py \
    --phase train "${common_args[@]}"
check_source

receipt="$TF_PAIRED_RESTORE_RUN_DIR/expectations.json"
[[ -f "$receipt" && ! -L "$receipt" ]]
receipt_sha=$(sha256sum -- "$receipt" | awk '{print $1}')
printf 'PAIRED_RESTORE_EXPECTATIONS_SHA256=%s\n' "$receipt_sha"

"$python_bin" -B -u scripts/check_paper01_paired_restoration.py \
    --phase restore \
    --receipt-sha256 "$receipt_sha" \
    "${common_args[@]}"
check_source

printf 'PAIRED_CLASSIFIER_TWO_PROCESS_CHECK=PASS\n'
