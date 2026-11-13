#!/usr/bin/env bash
#SBATCH --job-name=p1r1-ar-cic-range
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
umask 077

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_SYNTH_EXPECTED_HEAD:?Synthesis revision required}"
: "${TF_SYNTH_FAMILY:?Family required}"
: "${TF_SYNTH_DATASET_CONFIG:?Configuration required}"
: "${TF_SYNTH_CONFIG_SHA256:?Configuration digest required}"
: "${TF_SYNTH_CHECKPOINT_PARENT:?Checkpoint parent required}"
: "${TF_SYNTH_OUTPUT:?Fresh output directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

: "${TF_SYNTH_CLASS_ID:?Class ID required}"
: "${TF_SYNTH_BATCH_START:?Batch start required}"
: "${TF_SYNTH_BATCH_STOP:?Exclusive batch stop required}"

case "$TF_SYNTH_FAMILY" in
    ar) generation=ar_cic_seed42 ;;
    maf) generation=maf_cic_seed42_execution ;;
    *) echo "Unsupported synthesis family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
check_source() {
    [[ "$(git rev-parse HEAD)" == "$TF_SYNTH_EXPECTED_HEAD" ]]
    git diff --quiet
    git diff --cached --quiet
}
check_source

driver=scripts/run_paper01_ar_cic_range_worker.py
launcher=slurm/paper01_revision_2026/ar_cic_range_worker.slurm
for file in "$driver" "$launcher"; do
    [[ -f "$file" && ! -L "$file" ]]
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

[[ -f "$TF_SYNTH_DATASET_CONFIG" && ! -L "$TF_SYNTH_DATASET_CONFIG" ]]
[[ -d "$TF_SYNTH_CHECKPOINT_PARENT" && ! -L "$TF_SYNTH_CHECKPOINT_PARENT" ]]
[[ ! -e "$TF_SYNTH_OUTPUT" && ! -L "$TF_SYNTH_OUTPUT" ]]
printf '%s  %s\n' "$TF_SYNTH_CONFIG_SHA256" "$TF_SYNTH_DATASET_CONFIG" |
    sha256sum --check -

extra_args=()
if [[ "$TF_SYNTH_FAMILY" == maf ]]; then
    : "${TF_SYNTH_EXPECTATIONS:?Accepted execution expectations required}"
    : "${TF_SYNTH_EXPECTATIONS_SHA256:?Expectation digest required}"
    [[ -f "$TF_SYNTH_EXPECTATIONS" && ! -L "$TF_SYNTH_EXPECTATIONS" ]]
    extra_args+=(
        --expectations "$TF_SYNTH_EXPECTATIONS"
        --expectations-sha256 "$TF_SYNTH_EXPECTATIONS_SHA256"
    )
fi

python_executable="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$python_executable" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'PURPOSE=CIC_SEED42_CHECKPOINT_BACKED_AR_RANGE_WORKER\n'
printf 'SYNTHESIS_HEAD=%s\nFAMILY=%s\nSLURM_JOB_ID=%s\n' \
    "$TF_SYNTH_EXPECTED_HEAD" "$TF_SYNTH_FAMILY" "$SLURM_JOB_ID"
hostname

"$python_executable" -B -u "$driver" \
    --family "$TF_SYNTH_FAMILY" \
    --repository-root "$PWD" \
    --expected-head "$TF_SYNTH_EXPECTED_HEAD" \
    --dataset-config "$TF_SYNTH_DATASET_CONFIG" \
    --config-sha256 "$TF_SYNTH_CONFIG_SHA256" \
    --publication-parent "$TF_SYNTH_CHECKPOINT_PARENT" \
    --generation-name "$generation" \
    --class-id "$TF_SYNTH_CLASS_ID" \
    --batch-start "$TF_SYNTH_BATCH_START" \
    --batch-stop "$TF_SYNTH_BATCH_STOP" \
    --output "$TF_SYNTH_OUTPUT" \
    "${extra_args[@]}"

check_source
