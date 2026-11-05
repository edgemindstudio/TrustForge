#!/usr/bin/env bash
#SBATCH --job-name=p1r1-integration
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_INTEGRATION_EXPECTED_HEAD:?Expected revision required}"
: "${TF_INTEGRATION_FAMILY:?Family required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

case "$TF_INTEGRATION_FAMILY" in
    ar|maf) ;;
    *) echo "FAIL: unsupported family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "$TF_INTEGRATION_EXPECTED_HEAD" ]] || {
    echo "FAIL: source revision changed" >&2
    exit 1
}
git diff --quiet
git diff --cached --quiet

files=(
    scripts/check_paper01_ar_maf_training_integration.py
    slurm/paper01_revision_2026/ar_maf_training_integration.slurm
    studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json
    src/trustforge/models/paper01_revision_2026/ar_maf_training_core.py
    src/trustforge/models/paper01_revision_2026/ar_maf_family_steps.py
)
for file in "${files[@]}"; do
    [[ -f "$file" && ! -L "$file" ]] || {
        echo "FAIL: missing or symlinked source: $file" >&2
        exit 1
    }
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]] || {
    echo "FAIL: Python environment unavailable" >&2
    exit 1
}

export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "PURPOSE=SYNTHETIC_TRAINING_INTEGRATION_ONLY"
echo "SOURCE_HEAD=$TF_INTEGRATION_EXPECTED_HEAD"
echo "FAMILY=$TF_INTEGRATION_FAMILY"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname

"$PYTHON" -B -u scripts/check_paper01_ar_maf_training_integration.py \
    --family "$TF_INTEGRATION_FAMILY"
