#!/usr/bin/env bash
#SBATCH --job-name=p1r1-ar-cic-pool
#SBATCH --array=0-49%2
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/ar/cicmaldroid2020/seed42/logs/ar-pool-%A_%a.out
#SBATCH --error=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/ar/cicmaldroid2020/seed42/logs/ar-pool-%A_%a.err

set -euo pipefail
umask 077

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_SYNTH_EXPECTED_HEAD:?Synthesis revision required}"
: "${TF_SYNTH_DATASET_CONFIG:?Configuration required}"
: "${TF_SYNTH_CONFIG_SHA256:?Configuration digest required}"
: "${TF_SYNTH_CHECKPOINT_PARENT:?Checkpoint parent required}"
: "${AR_POOL_RUN_ROOT:?Fresh run root required}"
: "${SLURM_ARRAY_TASK_ID:?Slurm array task required}"
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
[[ ! -e "$AR_POOL_RUN_ROOT/pool" && ! -L "$AR_POOL_RUN_ROOT/pool" ]]
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

task_record="$(
    "$python_executable" -B - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys

from scripts import run_paper01_ar_cic_seed42_pool as coordinator

index = int(sys.argv[1])
tasks = coordinator.partition_ranges(10)
if not 0 <= index < len(tasks):
    raise SystemExit("array task index outside the plan")
task = tasks[index]
print(
    task["class_id"],
    task["batch_start"],
    task["batch_stop"],
    task["worker_name"],
)
PY
)"
read -r class_id batch_start batch_stop worker_name <<< "$task_record"
[[ "$class_id" =~ ^[0-4]$ ]]
[[ "$batch_start" =~ ^[0-9]+$ ]]
[[ "$batch_stop" =~ ^[0-9]+$ ]]
[[ "$batch_stop" -gt "$batch_start" ]]
[[ -n "$worker_name" ]]

export TF_SYNTH_FAMILY=ar
export TF_SYNTH_EXPERIMENTAL_SEED=42
export TF_SYNTH_CLASS_ID="$class_id"
export TF_SYNTH_BATCH_START="$batch_start"
export TF_SYNTH_BATCH_STOP="$batch_stop"
export TF_SYNTH_OUTPUT="$AR_POOL_RUN_ROOT/workers/$worker_name"

printf 'PURPOSE=CIC_SEED42_AR_FULL_POOL_WORKER\n'
printf 'SYNTHESIS_HEAD=%s\nSLURM_JOB_ID=%s\nARRAY_TASK_ID=%s\n' \
    "$TF_SYNTH_EXPECTED_HEAD" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID"
printf 'CLASS_ID=%s\nBATCH_START=%s\nBATCH_STOP=%s\nOUTPUT=%s\n' \
    "$class_id" "$batch_start" "$batch_stop" "$TF_SYNTH_OUTPUT"
hostname

check_source
exec bash "$TRUSTFORGE_REPO_ROOT/slurm/paper01_revision_2026/ar_cic_range_worker.slurm"
