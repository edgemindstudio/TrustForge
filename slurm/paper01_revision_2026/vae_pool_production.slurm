#!/usr/bin/env bash
#SBATCH --job-name=p1r1-vae-pool
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=p1r1-vae-pool.%j.out
#SBATCH --error=p1r1-vae-pool.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 VAE production synthesis launcher.
# Submission requires separate operational authorization.
# The source-level production synthesis gate remains default-deny; this
# launcher only requests the already-gated candidate path.
# This launcher does not authorize synthesis, publication or evaluation.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${VAE_SYNTHESIS_EXECUTION_CHECKOUT:?Explicit clean synthesis checkout required}"
: "${VAE_SYNTHESIS_SOURCE_COMMIT:?Explicit synthesis source revision required}"
: "${VAE_CHECKPOINT_PUBLICATION_ROOT:?Checkpoint publication root required}"
: "${VAE_GENERATION_ID:?Checkpoint generation ID required}"
: "${VAE_EXPECTED_MANIFEST_SHA256:?Checkpoint manifest SHA-256 required}"
: "${VAE_DATASET_ID:?Dataset ID required}"
: "${VAE_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${VAE_CLASS_MAPPING:?Ordered class mapping required}"
: "${VAE_POOL_DESTINATION:?Synthetic pool destination required}"

[[ -d "$VAE_SYNTHESIS_EXECUTION_CHECKOUT" && ! -L "$VAE_SYNTHESIS_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$VAE_SYNTHESIS_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

[[ "$VAE_SYNTHESIS_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$VAE_EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$VAE_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]

[[ "$(git -C "$checkout" rev-parse HEAD)" == "$VAE_SYNTHESIS_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]]

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

export PYTHONPATH="$checkout/src:$checkout${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONDONTWRITEBYTECODE=1

echo "VAE_POOL_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$VAE_DATASET_ID"
echo "EXPERIMENTAL_SEED=$VAE_EXPERIMENTAL_SEED"
echo "GENERATION_ID=$VAE_GENERATION_ID"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SYNTHESIS_SOURCE_HEAD=$VAE_SYNTHESIS_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "PRODUCTION_SYNTHESIS_GATE=SOURCE_LEVEL_DEFAULT_DENY"
echo "SYNTHESIS_PUBLICATION_REQUESTED=1"

exec "$PYTHON" -u \
    scripts/publish_paper01_vae_pool.py \
    --repository-root "$checkout" \
    --expected-synthesis-source-commit "$VAE_SYNTHESIS_SOURCE_COMMIT" \
    --checkpoint-publication-root "$VAE_CHECKPOINT_PUBLICATION_ROOT" \
    --generation-id "$VAE_GENERATION_ID" \
    --expected-manifest-sha256 "$VAE_EXPECTED_MANIFEST_SHA256" \
    --dataset-id "$VAE_DATASET_ID" \
    --experimental-seed "$VAE_EXPERIMENTAL_SEED" \
    --class-mapping "$VAE_CLASS_MAPPING" \
    --destination "$VAE_POOL_DESTINATION"
