#!/usr/bin/env bash
#SBATCH --job-name=p1r1-maf-env-synth
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --exclude=talon08

set -euo pipefail
: "${TF_MAF_SYNTH_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$HOME/TrustForge"
revision=e48016b8339cf04a0039484e84649471c7da8476

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

accepted="$HOME/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-execution-913d86f-VPyYjqz5"

"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u \
    scripts/check_paper01_maf_cic_execution_synthesis.py \
    --family maf \
    --output "$TF_MAF_SYNTH_RUN_DIR/output" \
    --repository-root "$PWD" \
    --expected-head "$revision" \
    --dataset-config "$accepted/dataset_config.json" \
    --config-sha256 ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b \
    --publication-parent "$accepted/checkpoints" \
    --generation-name maf_cic_seed42_execution \
    --expectations "$accepted/expectations.json" \
    --expectations-sha256 2e8f64a58b8900c9d6c6fd5580794c8374353abc14e51fab49e8155d675526ec

check_source
printf 'MAF_EXECUTION_ENVELOPE_SYNTHESIS=PASS\n'
