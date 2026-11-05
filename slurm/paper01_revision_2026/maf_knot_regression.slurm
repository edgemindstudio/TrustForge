#!/usr/bin/env bash
#SBATCH --job-name=p1r1-maf-knot-test
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_KNOT_EXPECTED_HEAD:?Expected revision required}"
: "${TF_KNOT_JUNIT:?Fresh JUnit destination required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

cd "$TRUSTFORGE_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$TF_KNOT_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet
[[ ! -e "$TF_KNOT_JUNIT" && ! -L "$TF_KNOT_JUNIT" ]]

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=1

echo "PURPOSE=MAF_KNOT_CORRECTION_REGRESSION"
echo "SOURCE_HEAD=$TF_KNOT_EXPECTED_HEAD"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname
"$PYTHON" -B -m pytest -q \
    --junitxml="$TF_KNOT_JUNIT" \
    tests/test_maf_knot_interpolation_revision_2026.py \
    tests/test_masked_autoflow_spline_revision_2026.py \
    tests/test_masked_autoflow_blocks_revision_2026.py \
    tests/test_masked_autoflow_conditioner_revision_2026.py \
    tests/test_masked_autoflow_flow_revision_2026.py \
    tests/test_masked_autoflow_model_revision_2026.py \
    tests/test_masked_autoflow_training_revision_2026.py

[[ "$(git rev-parse HEAD)" == "$TF_KNOT_EXPECTED_HEAD" ]]
git diff --quiet
git diff --cached --quiet
