#!/usr/bin/env bash
#SBATCH --job-name=p1r1-ar-cic-pool-finalize
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/ar/cicmaldroid2020/seed42/logs/ar-pool-finalize-%j.out
#SBATCH --error=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/ar/cicmaldroid2020/seed42/logs/ar-pool-finalize-%j.err

set -euo pipefail
umask 077

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_SYNTH_EXPECTED_HEAD:?Synthesis revision required}"
: "${TF_SYNTH_DATASET_CONFIG:?Configuration required}"
: "${TF_SYNTH_CONFIG_SHA256:?Configuration digest required}"
: "${TF_SYNTH_CHECKPOINT_PARENT:?Checkpoint parent required}"
: "${AR_POOL_RUN_ROOT:?Run root required}"
: "${AR_POOL_DESTINATION:?Publication destination required}"
: "${AR_POOL_AUTHORIZATION:?Authorization record required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$TRUSTFORGE_REPO_ROOT"
check_source() {
    [[ "$(git rev-parse HEAD)" == "$TF_SYNTH_EXPECTED_HEAD" ]]
    git diff --quiet
    git diff --cached --quiet
}
check_source

[[ -f "$TF_SYNTH_DATASET_CONFIG" && ! -L "$TF_SYNTH_DATASET_CONFIG" ]]
[[ -d "$TF_SYNTH_CHECKPOINT_PARENT" && ! -L "$TF_SYNTH_CHECKPOINT_PARENT" ]]
[[ -d "$AR_POOL_RUN_ROOT" && ! -L "$AR_POOL_RUN_ROOT" ]]
[[ -d "$AR_POOL_RUN_ROOT/workers" && ! -L "$AR_POOL_RUN_ROOT/workers" ]]
[[ ! -e "$AR_POOL_DESTINATION" && ! -L "$AR_POOL_DESTINATION" ]]
printf '%s  %s\n' "$TF_SYNTH_CONFIG_SHA256" "$TF_SYNTH_DATASET_CONFIG" |
    sha256sum --check -

python_executable="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$python_executable" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'PURPOSE=CIC_SEED42_AR_FULL_POOL_FINALIZE\n'
printf 'SYNTHESIS_HEAD=%s\nSLURM_JOB_ID=%s\n' \
    "$TF_SYNTH_EXPECTED_HEAD" "$SLURM_JOB_ID"
hostname

"$python_executable" -B -u scripts/run_paper01_ar_cic_seed42_pool.py \
    --finalize \
    --repository-root "$PWD" \
    --expected-head "$TF_SYNTH_EXPECTED_HEAD" \
    --run-root "$AR_POOL_RUN_ROOT" \
    --checkpoint-parent "$TF_SYNTH_CHECKPOINT_PARENT" \
    --dataset-config "$TF_SYNTH_DATASET_CONFIG" \
    --config-sha256 "$TF_SYNTH_CONFIG_SHA256" \
    --destination "$AR_POOL_DESTINATION" \
    --authorization "$AR_POOL_AUTHORIZATION"

"$python_executable" -B -u scripts/run_paper01_ar_cic_seed42_pool.py \
    --verify \
    --manifest "$AR_POOL_DESTINATION/manifest.json"

check_source
