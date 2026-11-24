#!/usr/bin/env bash
#SBATCH --job-name=p1r1-rbm-sample
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=p1r1-rbm-sample.%j.out
#SBATCH --error=p1r1-rbm-sample.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 RBM production-sampling launcher.
# Submission requires a separate execution-specific researcher authorization
# record and explicit operator authorization. The source activation boundary
# remains disabled in the entry module.
# One job invokes exactly one dataset/seed/attempt/class/sample scope.
# No synthetic pool is published and no retry, resume or requeue is performed.
#
# The resource directives above are PROVISIONAL implementation settings.
# They are not a Stage 8 resource freeze or measured production feasibility.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${RBM_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${RBM_EXPECTED_SOURCE_COMMIT:?Explicit accepted source revision required}"
: "${RBM_DATASET_ID:?Dataset ID required}"
: "${RBM_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${RBM_ATTEMPT:?Explicit numeric attempt required}"
: "${RBM_CLASS_ID:?Requested class identity required}"
: "${RBM_SAMPLE_START:?Requested sample start required}"
: "${RBM_SAMPLE_STOP:?Requested sample stop required}"
: "${RBM_CHECKPOINT_DIR:?Checkpoint directory required}"
: "${RBM_CHECKPOINT_STATE_SHA256:?Expected checkpoint state hash required}"

[[ -d "$RBM_EXECUTION_CHECKOUT" && ! -L "$RBM_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$RBM_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

[[ "$RBM_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RBM_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]
[[ "$RBM_ATTEMPT" =~ ^[0-9]+$ ]]
[[ "$RBM_CLASS_ID" =~ ^[0-9]+$ ]]
[[ "$RBM_SAMPLE_START" =~ ^[0-9]+$ ]]
[[ "$RBM_SAMPLE_STOP" =~ ^[0-9]+$ ]]
[[ "$RBM_CHECKPOINT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]]

[[ "$(git -C "$checkout" rev-parse HEAD)" == "$RBM_EXPECTED_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]]

case "$RBM_DATASET_ID" in
    ustc_tfc2016|cicmaldroid2020|malnet_image|malex)
        ;;
    *)
        echo "FAIL: unsupported dataset ID" >&2
        exit 1
        ;;
esac

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

export PYTHONPATH="$checkout/src:$checkout${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "RBM_REVISION_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$RBM_DATASET_ID"
echo "EXPERIMENTAL_SEED=$RBM_EXPERIMENTAL_SEED"
echo "ATTEMPT=$RBM_ATTEMPT"
echo "CLASS_ID=$RBM_CLASS_ID"
echo "SAMPLE_START=$RBM_SAMPLE_START"
echo "SAMPLE_STOP=$RBM_SAMPLE_STOP"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$RBM_EXPECTED_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "RESOURCE_ENVELOPE=PROVISIONAL"
echo "RBM_PRODUCTION_SAMPLING_ENABLED=SOURCE_DEFAULT_FALSE"
echo "POOL_PUBLICATION=DISABLED"
echo "AUTOMATIC_RETRY=DISABLED"

exec "$PYTHON" -B -u \
    scripts/run_paper01_rbm_production_sampling.py \
    --dataset-id "$RBM_DATASET_ID" \
    --experimental-seed "$RBM_EXPERIMENTAL_SEED" \
    --attempt "$RBM_ATTEMPT" \
    --class-id "$RBM_CLASS_ID" \
    --sample-start "$RBM_SAMPLE_START" \
    --sample-stop "$RBM_SAMPLE_STOP" \
    --expected-source-commit "$RBM_EXPECTED_SOURCE_COMMIT" \
    --repository-root "$checkout" \
    --checkpoint-dir "$RBM_CHECKPOINT_DIR" \
    --expected-checkpoint-state-sha256 "$RBM_CHECKPOINT_STATE_SHA256"
