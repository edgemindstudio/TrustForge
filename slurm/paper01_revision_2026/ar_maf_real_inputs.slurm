#!/usr/bin/env bash
#SBATCH --job-name=p1r1-real-inputs
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_REAL_EXPECTED_HEAD:?Expected revision required}"
: "${TF_REAL_FAMILY:?Family required}"
: "${TF_REAL_DATASET_CONFIG:?Dataset configuration required}"
: "${TF_REAL_CONFIG_SHA256:?Configuration digest required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

case "$TF_REAL_FAMILY" in
    ar|maf) ;;
    *) echo "FAIL: unsupported family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$TF_REAL_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet

for file in \
    scripts/check_paper01_ar_maf_real_inputs.py \
    slurm/paper01_revision_2026/ar_maf_real_inputs.slurm
do
    [[ -f "$file" && ! -L "$file" ]]
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

[[ -f "$TF_REAL_DATASET_CONFIG" && ! -L "$TF_REAL_DATASET_CONFIG" ]]
printf '%s  %s\n' "$TF_REAL_CONFIG_SHA256" "$TF_REAL_DATASET_CONFIG" |
    sha256sum --check -

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "PURPOSE=BOUNDED_ACTUAL_SOURCE_WIRING_ONLY"
echo "SOURCE_HEAD=$TF_REAL_EXPECTED_HEAD"
echo "FAMILY=$TF_REAL_FAMILY"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname

"$PYTHON" -B -u scripts/check_paper01_ar_maf_real_inputs.py \
    --family "$TF_REAL_FAMILY" \
    --dataset-config "$TF_REAL_DATASET_CONFIG" \
    --repository-root "$TRUSTFORGE_REPO_ROOT" \
    --expected-head "$TF_REAL_EXPECTED_HEAD"

[[ "$(git rev-parse HEAD)" == "$TF_REAL_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet
