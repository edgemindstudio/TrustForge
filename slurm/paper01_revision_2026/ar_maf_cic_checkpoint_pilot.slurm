#!/usr/bin/env bash
#SBATCH --job-name=p1r1-cic-checkpoint
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

: "${TRUSTFORGE_REPO_ROOT:?Repository root required}"
: "${TF_PILOT_EXPECTED_HEAD:?Expected revision required}"
: "${TF_PILOT_FAMILY:?Family required}"
: "${TF_PILOT_DATASET_CONFIG:?Configuration required}"
: "${TF_PILOT_CONFIG_SHA256:?Configuration digest required}"
: "${TF_PILOT_PUBLICATION_PARENT:?Private publication parent required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_CPUS_PER_TASK:?CPU allocation required}"

case "$TF_PILOT_FAMILY" in
    ar|maf) ;;
    *) echo "FAIL: unsupported family" >&2; exit 1 ;;
esac

cd "$TRUSTFORGE_REPO_ROOT"
check_source() {
    [[ "$(git rev-parse HEAD)" == "$TF_PILOT_EXPECTED_HEAD" ]]
    git diff --quiet
    git diff --cached --quiet
}
check_source

for file in \
    scripts/check_paper01_ar_maf_cic_checkpoint_pilot.py \
    slurm/paper01_revision_2026/ar_maf_cic_checkpoint_pilot.slurm
do
    [[ -f "$file" && ! -L "$file" ]]
    git ls-files --error-unmatch -- "$file" >/dev/null
    sha256sum -- "$file"
done

[[ -f "$TF_PILOT_DATASET_CONFIG" && ! -L "$TF_PILOT_DATASET_CONFIG" ]]
[[ -d "$TF_PILOT_PUBLICATION_PARENT" && ! -L "$TF_PILOT_PUBLICATION_PARENT" ]]
printf '%s  %s\n' "$TF_PILOT_CONFIG_SHA256" "$TF_PILOT_DATASET_CONFIG" |
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

echo "PURPOSE=CIC_SEED42_TRAIN_PUBLISH_RESTORE_PILOT"
echo "SOURCE_HEAD=$TF_PILOT_EXPECTED_HEAD"
echo "FAMILY=$TF_PILOT_FAMILY"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PUBLICATION_PARENT=$TF_PILOT_PUBLICATION_PARENT"
hostname

for phase in train restore; do
    check_source
    "$PYTHON" -B -u scripts/check_paper01_ar_maf_cic_checkpoint_pilot.py \
        --family "$TF_PILOT_FAMILY" \
        --phase "$phase" \
        --repository-root "$TRUSTFORGE_REPO_ROOT" \
        --expected-head "$TF_PILOT_EXPECTED_HEAD" \
        --dataset-config "$TF_PILOT_DATASET_CONFIG" \
        --config-sha256 "$TF_PILOT_CONFIG_SHA256" \
        --publication-parent "$TF_PILOT_PUBLICATION_PARENT" \
        --generation-name "${TF_PILOT_FAMILY}_cic_seed42"
done
check_source
