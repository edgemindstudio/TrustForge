#!/usr/bin/env bash
#SBATCH --job-name=p1r1-maf-boundary
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_BOUNDARY_EXPECTED_HEAD:?Expected revision required}"
: "${TF_BOUNDARY_DATASET_CONFIG:?Configuration required}"
: "${TF_BOUNDARY_CONFIG_SHA256:?Configuration digest required}"
: "${TF_BOUNDARY_REPORT:?Fresh report destination required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$TRUSTFORGE_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$TF_BOUNDARY_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet

for file in \
    scripts/diagnose_paper01_maf_cic_boundary.py \
    slurm/paper01_revision_2026/maf_cic_boundary_diagnostic.slurm
do
    [[ -f "$file" && ! -L "$file" ]]
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

[[ -f "$TF_BOUNDARY_DATASET_CONFIG" && ! -L "$TF_BOUNDARY_DATASET_CONFIG" ]]
printf '%s  %s\n' "$TF_BOUNDARY_CONFIG_SHA256" "$TF_BOUNDARY_DATASET_CONFIG" |
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

echo "PURPOSE=UNCHANGED_MAF_BOUNDARY_DIAGNOSTIC"
echo "SOURCE_HEAD=$TF_BOUNDARY_EXPECTED_HEAD"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "REPORT=$TF_BOUNDARY_REPORT"
hostname

"$PYTHON" -B -u scripts/diagnose_paper01_maf_cic_boundary.py \
    --repository-root "$TRUSTFORGE_REPO_ROOT" \
    --expected-head "$TF_BOUNDARY_EXPECTED_HEAD" \
    --dataset-config "$TF_BOUNDARY_DATASET_CONFIG" \
    --config-sha256 "$TF_BOUNDARY_CONFIG_SHA256" \
    --report "$TF_BOUNDARY_REPORT"

[[ "$(git rev-parse HEAD)" == "$TF_BOUNDARY_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet
