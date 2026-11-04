#!/usr/bin/env bash
#SBATCH --job-name=p1r1-ar-maf-feas
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_FEAS_EXPECTED_HEAD:?Expected committed revision required}"
: "${TF_FEAS_FAMILY:?Select ar or maf}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

case "$TF_FEAS_FAMILY" in
    ar|maf) ;;
    *) echo "FAIL: unsupported family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"

if [[ "$(git rev-parse HEAD)" != "$TF_FEAS_EXPECTED_HEAD" ]]; then
    echo "FAIL: source revision changed" >&2
    exit 1
fi

git diff --quiet
git diff --cached --quiet

files=(
    scripts/check_paper01_ar_maf_feasibility.py
    studies/paper01_benchmark/revision_2026/ar_maf_execution_candidate.json
    slurm/paper01_revision_2026/ar_maf_synthetic_feasibility.slurm
)

for file in "${files[@]}"; do
    if [[ ! -f "$file" || -L "$file" ]]; then
        echo "FAIL: source missing or symlinked: $file" >&2
        exit 1
    fi
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]] || {
    echo "FAIL: expected Python environment unavailable" >&2
    exit 1
}

export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "PURPOSE=SYNTHETIC_FEASIBILITY_ONLY"
echo "SOURCE_HEAD=$TF_FEAS_EXPECTED_HEAD"
echo "FAMILY=$TF_FEAS_FAMILY"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
hostname

"$PYTHON" -B -u scripts/check_paper01_ar_maf_feasibility.py \
    --family "$TF_FEAS_FAMILY"
