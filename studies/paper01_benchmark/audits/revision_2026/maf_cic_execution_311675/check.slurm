#!/usr/bin/env bash
#SBATCH --job-name=p1r1-maf-cic-exec
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=talon08

set -euo pipefail
: "${TF_MAF_EXECUTION_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=913d86f9c64e6a3a1a0d009f112bcfa889a02ec8
config_sha=ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b

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
    --dataset-config "$TF_MAF_EXECUTION_RUN_DIR/dataset_config.json"
    --config-sha256 "$config_sha"
    --run-dir "$TF_MAF_EXECUTION_RUN_DIR"
)

"$python_bin" -B -u scripts/check_paper01_maf_cic_execution.py \
    --phase train "${common_args[@]}"

check_source
receipt="$TF_MAF_EXECUTION_RUN_DIR/expectations.json"
[[ -f "$receipt" && ! -L "$receipt" ]]
receipt_sha=$(sha256sum -- "$receipt" | awk '{print $1}')
printf 'RESTORE_EXPECTATIONS_SHA256=%s\n' "$receipt_sha"

"$python_bin" -B -u scripts/check_paper01_maf_cic_execution.py \
    --phase restore \
    --receipt-sha256 "$receipt_sha" \
    "${common_args[@]}"

check_source
printf 'MAF_CIC_EXECUTION_TWO_PROCESS_CHECK=PASS\n'
