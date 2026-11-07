#!/usr/bin/env bash
#SBATCH --job-name=p1r1-cic-scaler
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail
: "${TF_SCALER_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"

cd "$HOME/TrustForge"
revision=e81323282a2662d775f1cee7ea7bd2730ab1184b
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

python_bin="$HOME/miniforge3/envs/gcs_py310/bin/python"

for phase in capture restore; do
    check_source
    "$python_bin" -B -u scripts/check_paper01_cic_fitted_state.py \
        --phase "$phase" \
        --repository-root "$PWD" \
        --expected-head "$revision" \
        --dataset-config "$TF_SCALER_RUN_DIR/dataset_config.json" \
        --config-sha256 "$config_sha" \
        --evidence-dir "$TF_SCALER_RUN_DIR/evidence"
done

check_source
printf 'CIC_FITTED_STATE_TWO_PROCESS_CHECK=PASS\n'
