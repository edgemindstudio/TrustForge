#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gan-prod
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=p1r1-gan-prod.%j.out
#SBATCH --error=p1r1-gan-prod.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 GAN production launcher.
# Submission requires separate operational authorization.
# One job trains exactly one dataset/experimental-seed checkpoint.
# The checkpoint destination is derived by the Python entry point; operators
# cannot supply an arbitrary path.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${GAN_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${GAN_EXPECTED_SOURCE_COMMIT:?Explicit accepted source revision required}"
: "${GAN_DATASET_ID:?Dataset ID required}"
: "${GAN_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${GAN_ATTEMPT:?Explicit numeric attempt required}"

[[ -d "$GAN_EXECUTION_CHECKOUT" && ! -L "$GAN_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$GAN_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

[[ "$GAN_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$GAN_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]
[[ "$GAN_ATTEMPT" =~ ^[0-9]{2,}$ ]]

case "$GAN_DATASET_ID" in
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
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ARGS=(
    "$PYTHON"
    -u
    scripts/train_paper01_gan_revision_production.py
    --dataset-id "$GAN_DATASET_ID"
    --experimental-seed "$GAN_EXPERIMENTAL_SEED"
    --attempt "$GAN_ATTEMPT"
    --expected-source-commit "$GAN_EXPECTED_SOURCE_COMMIT"
    --repository-root "$checkout"
)

case "$GAN_DATASET_ID" in
    ustc_tfc2016)
        : "${GAN_USTC_SOURCE_DIR:?USTC source required}"
        ARGS+=(
            --ustc-source-dir "$GAN_USTC_SOURCE_DIR"
        )
        ;;
    cicmaldroid2020)
        : "${GAN_CIC_SOURCE_CSV:?CIC source CSV required}"
        : "${GAN_CIC_FEATURE_COLUMNS:?CIC columns required}"
        ARGS+=(
            --cic-source-csv "$GAN_CIC_SOURCE_CSV"
            --cic-feature-columns "$GAN_CIC_FEATURE_COLUMNS"
        )
        ;;
    malnet_image)
        : "${GAN_MALNET_IMAGE_ROOT:?MalNet root required}"
        : "${GAN_MALNET_PREPARATION_DIR:?MalNet preparation required}"
        : "${GAN_MALNET_MAPPING_PATH:?MalNet mapping required}"
        ARGS+=(
            --malnet-image-root "$GAN_MALNET_IMAGE_ROOT"
            --malnet-preparation-dir "$GAN_MALNET_PREPARATION_DIR"
            --malnet-mapping-path "$GAN_MALNET_MAPPING_PATH"
        )
        ;;
    malex)
        : "${GAN_MALEX_LABEL_ARCHIVE:?MaleX label archive required}"
        : "${GAN_MALEX_BENIGN_ROOT:?MaleX benign root required}"
        : "${GAN_MALEX_MALWARE_ROOT:?MaleX malware root required}"
        ARGS+=(
            --malex-label-archive "$GAN_MALEX_LABEL_ARCHIVE"
            --malex-benign-root "$GAN_MALEX_BENIGN_ROOT"
            --malex-malware-root "$GAN_MALEX_MALWARE_ROOT"
        )
        ;;
esac

echo "GAN_REVISION_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$GAN_DATASET_ID"
echo "EXPERIMENTAL_SEED=$GAN_EXPERIMENTAL_SEED"
echo "ATTEMPT=$GAN_ATTEMPT"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$GAN_EXPECTED_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

"${ARGS[@]}"
