#!/usr/bin/env bash
#SBATCH --job-name=p1r1-vae-prod
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=p1r1-vae-prod.%j.out
#SBATCH --error=p1r1-vae-prod.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 VAE production launcher.
# Submission requires separate operational authorization.
# One job invokes exactly one dataset/experimental-seed/attempt identity.
# The real-data gate remains disabled in source.
# This launcher publishes no checkpoint or synthetic artifact.
#
# The resource values above are provisional implementation settings, not a
# researcher-accepted production resource envelope.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${VAE_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${VAE_EXPECTED_SOURCE_COMMIT:?Explicit accepted source revision required}"
: "${VAE_DATASET_ID:?Dataset ID required}"
: "${VAE_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${VAE_ATTEMPT:?Explicit numeric attempt required}"

[[ -d "$VAE_EXECUTION_CHECKOUT" && ! -L "$VAE_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$VAE_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

[[ "$VAE_EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$VAE_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]
[[ "$VAE_ATTEMPT" =~ ^[0-9]+$ ]]

[[ "$(git -C "$checkout" rev-parse HEAD)" == "$VAE_EXPECTED_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]]

case "$VAE_DATASET_ID" in
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
    scripts/train_paper01_vae_revision_production.py
    --dataset-id "$VAE_DATASET_ID"
    --experimental-seed "$VAE_EXPERIMENTAL_SEED"
    --attempt "$VAE_ATTEMPT"
    --expected-source-commit "$VAE_EXPECTED_SOURCE_COMMIT"
    --repository-root "$checkout"
)

case "$VAE_DATASET_ID" in
    ustc_tfc2016)
        : "${VAE_USTC_SOURCE_DIR:?USTC source required}"
        ARGS+=(
            --ustc-source-dir "$VAE_USTC_SOURCE_DIR"
        )
        ;;
    cicmaldroid2020)
        : "${VAE_CIC_SOURCE_CSV:?CIC source CSV required}"
        : "${VAE_CIC_FEATURE_COLUMNS:?CIC columns required}"
        ARGS+=(
            --cic-source-csv "$VAE_CIC_SOURCE_CSV"
            --cic-feature-columns "$VAE_CIC_FEATURE_COLUMNS"
        )
        ;;
    malnet_image)
        : "${VAE_MALNET_IMAGE_ROOT:?MalNet root required}"
        : "${VAE_MALNET_PREPARATION_DIR:?MalNet preparation required}"
        : "${VAE_MALNET_MAPPING_PATH:?MalNet mapping required}"
        ARGS+=(
            --malnet-image-root "$VAE_MALNET_IMAGE_ROOT"
            --malnet-preparation-dir "$VAE_MALNET_PREPARATION_DIR"
            --malnet-mapping-path "$VAE_MALNET_MAPPING_PATH"
        )
        ;;
    malex)
        : "${VAE_MALEX_LABEL_ARCHIVE:?MaleX label archive required}"
        : "${VAE_MALEX_BENIGN_ROOT:?MaleX benign root required}"
        : "${VAE_MALEX_MALWARE_ROOT:?MaleX malware root required}"
        ARGS+=(
            --malex-label-archive "$VAE_MALEX_LABEL_ARCHIVE"
            --malex-benign-root "$VAE_MALEX_BENIGN_ROOT"
            --malex-malware-root "$VAE_MALEX_MALWARE_ROOT"
        )
        ;;
esac

echo "VAE_REVISION_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$VAE_DATASET_ID"
echo "EXPERIMENTAL_SEED=$VAE_EXPERIMENTAL_SEED"
echo "ATTEMPT=$VAE_ATTEMPT"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$VAE_EXPECTED_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CHECKPOINT_PUBLICATION=DISABLED"
echo "SYNTHESIS_PUBLICATION=DISABLED"

"${ARGS[@]}"
