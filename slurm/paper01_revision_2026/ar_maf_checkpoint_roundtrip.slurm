#!/usr/bin/env bash
#SBATCH --job-name=p1r1-state-restore
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_ROUNDTRIP_EXPECTED_HEAD:?Expected revision required}"
: "${TF_ROUNDTRIP_FAMILY:?Family required}"
: "${TF_ROUNDTRIP_DIRECTORY:?Fresh diagnostic directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

case "$TF_ROUNDTRIP_FAMILY" in
    ar|maf) ;;
    *) echo "FAIL: unsupported family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
check_source() {
    [[ "$(git rev-parse HEAD)" == "$TF_ROUNDTRIP_EXPECTED_HEAD" ]] || {
        echo "FAIL: source revision changed" >&2
        exit 1
    }
    git diff --quiet
    git diff --cached --quiet
}
check_source

files=(
    scripts/check_paper01_ar_maf_checkpoint_roundtrip.py
    slurm/paper01_revision_2026/ar_maf_checkpoint_roundtrip.slurm
    studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json
    src/trustforge/models/paper01_revision_2026/ar_maf_checkpoint_npz.py
    src/trustforge/models/paper01_revision_2026/ar_maf_checkpoint_state.py
)
for file in "${files[@]}"; do
    [[ -f "$file" && ! -L "$file" ]]
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "PURPOSE=SYNTHETIC_CHECKPOINT_ROUNDTRIP_ONLY"
echo "SOURCE_HEAD=$TF_ROUNDTRIP_EXPECTED_HEAD"
echo "FAMILY=$TF_ROUNDTRIP_FAMILY"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "DIAGNOSTIC_DIRECTORY=$TF_ROUNDTRIP_DIRECTORY"
hostname

for phase in write restore; do
    check_source
    "$PYTHON" -B -u scripts/check_paper01_ar_maf_checkpoint_roundtrip.py \
        --family "$TF_ROUNDTRIP_FAMILY" \
        --phase "$phase" \
        --directory "$TF_ROUNDTRIP_DIRECTORY"
done
check_source
