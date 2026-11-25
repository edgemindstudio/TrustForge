#!/usr/bin/env bash
#SBATCH --job-name=p1r1-diff-synth
#SBATCH --no-requeue
#SBATCH --output=p1r1-diff-synth.%j.out
#SBATCH --error=p1r1-diff-synth.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 Diffusion production synthesis launcher.
# Submission requires separate operational authorization.
# One job invokes exactly one authorized dataset/seed/attempt/class/range.
# The production synthesis gate remains disabled in committed source; the
# external execution authorization record and actual Slurm allocation are
# validated by the entry point before any checkpoint is loaded.
#
# This launcher performs no training. It never retries or requeues.
# Partition, CPU, memory, walltime and GPU resources are supplied at
# submission time from the future researcher-approved resource envelope; no
# provisional resource directives are embedded here.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${DIFFUSION_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${DIFFUSION_PYTHON:?Explicit Python executable required}"
: "${DIFFUSION_EXPECTED_SOURCE_COMMIT:?Explicit accepted synthesis source revision required}"
: "${DIFFUSION_DATASET_ID:?Dataset ID required}"
: "${DIFFUSION_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${DIFFUSION_ATTEMPT:?Explicit numeric attempt required}"
: "${DIFFUSION_AUTHORIZATION_ID:?Explicit execution authorization identity required}"
: "${DIFFUSION_MODE:?Explicit mode (synthesis|restoration) required}"
: "${DIFFUSION_CHECKPOINT_DIR:?Published checkpoint directory required}"
: "${DIFFUSION_CHECKPOINT_MANIFEST_SHA256:?Checkpoint manifest SHA-256 required}"
: "${DIFFUSION_CHECKPOINT_STATE_SHA256:?Best-state SHA-256 required}"
: "${DIFFUSION_CHECKPOINT_TRAINING_SOURCE_COMMIT:?Checkpoint training source commit required}"
: "${DIFFUSION_RECIPE_SHA256:?Synthesis recipe SHA-256 required}"

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
[[ "$DIFFUSION_CHECKPOINT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$DIFFUSION_CHECKPOINT_STATE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$DIFFUSION_CHECKPOINT_TRAINING_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$DIFFUSION_RECIPE_SHA256" =~ ^[0-9a-f]{64}$ ]]

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

case "$DIFFUSION_MODE" in
    synthesis)
        : "${DIFFUSION_CLASS_ID:?Class ID required for synthesis}"
        : "${DIFFUSION_SAMPLE_START:?Sample start required for synthesis}"
        : "${DIFFUSION_SAMPLE_STOP:?Sample stop required for synthesis}"
        : "${DIFFUSION_OUTPUT_DESTINATION:?Output destination required for synthesis}"
        [[ "$DIFFUSION_CLASS_ID" =~ ^[0-9]+$ ]]
        [[ "$DIFFUSION_SAMPLE_START" =~ ^[0-9]+$ ]]
        [[ "$DIFFUSION_SAMPLE_STOP" =~ ^[0-9]+$ ]]
        ;;
    restoration)
        ;;
    *)
        echo "FAIL: unsupported execution mode" >&2
        exit 1
        ;;
esac

[[ -x "$DIFFUSION_PYTHON" ]]

export PYTHONPATH="$checkout/src:$checkout${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
    "$DIFFUSION_PYTHON"
    -B -u
    scripts/run_paper01_diffusion_production_synthesis.py
    --mode "$DIFFUSION_MODE"
    --dataset-id "$DIFFUSION_DATASET_ID"
    --experimental-seed "$DIFFUSION_EXPERIMENTAL_SEED"
    --attempt "$DIFFUSION_ATTEMPT"
    --expected-authorization-id "$DIFFUSION_AUTHORIZATION_ID"
    --expected-synthesis-source-commit "$DIFFUSION_EXPECTED_SOURCE_COMMIT"
    --repository-root "$checkout"
    --checkpoint-dir "$DIFFUSION_CHECKPOINT_DIR"
    --expected-checkpoint-manifest-sha256 "$DIFFUSION_CHECKPOINT_MANIFEST_SHA256"
    --expected-checkpoint-state-sha256 "$DIFFUSION_CHECKPOINT_STATE_SHA256"
    --expected-checkpoint-training-source-commit "$DIFFUSION_CHECKPOINT_TRAINING_SOURCE_COMMIT"
    --expected-recipe-sha256 "$DIFFUSION_RECIPE_SHA256"
)

if [[ "$DIFFUSION_MODE" == "synthesis" ]]; then
    ARGS+=(
        --class-id "$DIFFUSION_CLASS_ID"
        --sample-start "$DIFFUSION_SAMPLE_START"
        --sample-stop "$DIFFUSION_SAMPLE_STOP"
        --output-destination "$DIFFUSION_OUTPUT_DESTINATION"
    )
fi

echo "DIFFUSION_SYNTHESIS_LAUNCHER_PREFLIGHT"
echo "MODE=$DIFFUSION_MODE"
echo "DATASET_ID=$DIFFUSION_DATASET_ID"
echo "EXPERIMENTAL_SEED=$DIFFUSION_EXPERIMENTAL_SEED"
echo "ATTEMPT=$DIFFUSION_ATTEMPT"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$DIFFUSION_EXPECTED_SOURCE_COMMIT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "TRAINING_INVOCATION=NONE"
echo "CHECKPOINT_SELECTION=NONE"
echo "AUTOMATIC_RETRY=DISABLED"
echo "AUTOMATIC_REQUEUE=DISABLED"
echo "RESOURCE_ENVELOPE=FROM_EXTERNAL_AUTHORIZATION"

"${ARGS[@]}"
