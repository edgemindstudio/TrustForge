#!/usr/bin/env bash
#SBATCH --job-name=p1r1-diff-prod
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=p1r1-diff-prod.%j.out
#SBATCH --error=p1r1-diff-prod.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 Diffusion production launcher.
# Submission requires separate operational authorization.
# One job invokes exactly one dataset/experimental-seed/attempt identity.
# The real-data gate remains disabled in source.
# This launcher publishes no checkpoint or synthetic artifact.
#
# The resource directives above are PROVISIONAL implementation settings.
# They are not a researcher-accepted scientific or production resource
# envelope, and this launcher has not been operationally verified on Talon.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${DIFFUSION_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${DIFFUSION_EXPECTED_SOURCE_COMMIT:?Explicit accepted source revision required}"
: "${DIFFUSION_DATASET_ID:?Dataset ID required}"
: "${DIFFUSION_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${DIFFUSION_ATTEMPT:?Explicit numeric attempt required}"

[[ -d "$DIFFUSION_EXECUTION_CHECKOUT" && ! -L "$DIFFUSION_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$DIFFUSION_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

[[ "$DIFFUSION_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$DIFFUSION_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]
[[ "$DIFFUSION_ATTEMPT" =~ ^[0-9]+$ ]]

[[ "$(git -C "$checkout" rev-parse HEAD)" == "$DIFFUSION_EXPECTED_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]]

case "$DIFFUSION_DATASET_ID" in
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
    scripts/train_paper01_diffusion_revision_production.py
    --dataset-id "$DIFFUSION_DATASET_ID"
    --experimental-seed "$DIFFUSION_EXPERIMENTAL_SEED"
    --attempt "$DIFFUSION_ATTEMPT"
    --expected-source-commit "$DIFFUSION_EXPECTED_SOURCE_COMMIT"
    --repository-root "$checkout"
)

case "$DIFFUSION_DATASET_ID" in
    ustc_tfc2016)
        : "${DIFFUSION_USTC_SOURCE_DIR:?USTC source required}"
        ARGS+=(
            --ustc-source-dir "$DIFFUSION_USTC_SOURCE_DIR"
        )
        ;;
    cicmaldroid2020)
        : "${DIFFUSION_CIC_SOURCE_CSV:?CIC source CSV required}"
        : "${DIFFUSION_CIC_FEATURE_COLUMNS:?CIC columns required}"
        ARGS+=(
            --cic-source-csv "$DIFFUSION_CIC_SOURCE_CSV"
            --cic-feature-columns "$DIFFUSION_CIC_FEATURE_COLUMNS"
        )
        ;;
    malnet_image)
        : "${DIFFUSION_MALNET_IMAGE_ROOT:?MalNet root required}"
        : "${DIFFUSION_MALNET_PREPARATION_DIR:?MalNet preparation required}"
        : "${DIFFUSION_MALNET_MAPPING_PATH:?MalNet mapping required}"
        ARGS+=(
            --malnet-image-root "$DIFFUSION_MALNET_IMAGE_ROOT"
            --malnet-preparation-dir "$DIFFUSION_MALNET_PREPARATION_DIR"
            --malnet-mapping-path "$DIFFUSION_MALNET_MAPPING_PATH"
        )
        ;;
    malex)
        : "${DIFFUSION_MALEX_LABEL_ARCHIVE:?MaleX label archive required}"
        : "${DIFFUSION_MALEX_BENIGN_ROOT:?MaleX benign root required}"
        : "${DIFFUSION_MALEX_MALWARE_ROOT:?MaleX malware root required}"
        ARGS+=(
            --malex-label-archive "$DIFFUSION_MALEX_LABEL_ARCHIVE"
            --malex-benign-root "$DIFFUSION_MALEX_BENIGN_ROOT"
            --malex-malware-root "$DIFFUSION_MALEX_MALWARE_ROOT"
        )
        ;;
esac

echo "DIFFUSION_REVISION_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$DIFFUSION_DATASET_ID"
echo "EXPERIMENTAL_SEED=$DIFFUSION_EXPERIMENTAL_SEED"
echo "ATTEMPT=$DIFFUSION_ATTEMPT"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$DIFFUSION_EXPECTED_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CHECKPOINT_PUBLICATION=DISABLED"
echo "SYNTHESIS_PUBLICATION=DISABLED"
echo "RESOURCE_ENVELOPE=PROVISIONAL"

"${ARGS[@]}"
