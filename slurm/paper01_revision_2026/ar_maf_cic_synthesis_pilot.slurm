#!/usr/bin/env bash
#SBATCH --job-name=p1r1-cic-synthesis
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

case "$TF_SYNTH_FAMILY" in
    ar|maf) generation="${TF_SYNTH_FAMILY}_cic_seed42" ;;
    *) echo "Unsupported synthesis family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
check_source() {
    [[ "$(git rev-parse HEAD)" == "$TF_SYNTH_EXPECTED_HEAD" ]]
    git diff --quiet
    git diff --cached --quiet
}
check_source

driver=scripts/check_paper01_ar_maf_cic_synthesis_pilot.py
launcher=slurm/paper01_revision_2026/ar_maf_cic_synthesis_pilot.slurm
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

python_executable="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$python_executable" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'PURPOSE=CIC_SEED42_CHECKPOINT_BACKED_SYNTHESIS_PILOT\n'
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
    --output "$TF_SYNTH_OUTPUT"

check_source
