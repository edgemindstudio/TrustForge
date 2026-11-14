#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gmm-prod
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=p1r1-gmm-prod.%j.out
#SBATCH --error=p1r1-gmm-prod.%j.err

set -euo pipefail

# Controlled Paper 1 Revision-2026 launcher.
# Submission requires separate operational authorization.
# One job trains exactly one dataset/class/seed checkpoint.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${GMM_EXECUTION_CHECKOUT:?Explicit clean execution checkout required}"
: "${GMM_EXPECTED_SOURCE_COMMIT:?Explicit accepted source revision required}"
: "${GMM_DATASET_ID:?Dataset ID required}"
: "${GMM_CLASS_ID:?Class ID required}"
: "${GMM_EXPERIMENTAL_SEED:?Experimental seed required}"
: "${GMM_CHECKPOINT_DIR:?Checkpoint directory required}"

[[ -d "$GMM_EXECUTION_CHECKOUT" && ! -L "$GMM_EXECUTION_CHECKOUT" ]]
checkout="$(
    cd "$GMM_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

GMM_SOURCE_FREEZE="$checkout/studies/paper01_benchmark/revision_2026/gmm_production_source_freeze_v4.json"
[[ -f "$GMM_SOURCE_FREEZE" && ! -L "$GMM_SOURCE_FREEZE" ]]

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

"$PYTHON" -B - "$GMM_SOURCE_FREEZE" "$GMM_EXPECTED_SOURCE_COMMIT" <<'PYATTEST'
from pathlib import Path
import sys

from trustforge.models.paper01_revision_2026.gaussian_mixture_source_binding import (
    verify_gmm_source_binding,
)

evidence = verify_gmm_source_binding(
    repository_root=Path.cwd(),
    source_freeze_path=sys.argv[1],
    expected_source_commit=sys.argv[2],
)
print(
    "GMM_RUNTIME_SOURCE_ATTESTATION: PASS",
    evidence["expected_source_commit"],
    evidence["source_freeze_sha256"],
    flush=True,
)
PYATTEST

case "$GMM_DATASET_ID" in
    ustc_tfc2016|cicmaldroid2020|malnet_image|malex)
        ;;
    *)
        echo "FAIL: unsupported dataset ID" >&2
        exit 1
        ;;
esac

if ! [[ "$GMM_CLASS_ID" =~ ^[0-9]+$ ]]; then
    echo "FAIL: class ID must be numeric" >&2
    exit 1
fi

if ! [[ "$GMM_EXPERIMENTAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "FAIL: experimental seed must be numeric" >&2
    exit 1
fi

if [[ -e "$GMM_CHECKPOINT_DIR" || -L "$GMM_CHECKPOINT_DIR" ]]; then
    echo "FAIL: checkpoint destination already exists" >&2
    exit 1
fi

export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ARGS=(
    "$PYTHON"
    -u
    scripts/train_paper01_gmm_revision_production.py
    --dataset-id "$GMM_DATASET_ID"
    --class-id "$GMM_CLASS_ID"
    --experimental-seed "$GMM_EXPERIMENTAL_SEED"
    --checkpoint-dir "$GMM_CHECKPOINT_DIR"
    --source-freeze "$GMM_SOURCE_FREEZE"
    --expected-source-commit "$GMM_EXPECTED_SOURCE_COMMIT"
)

case "$GMM_DATASET_ID" in
    ustc_tfc2016)
        : "${GMM_USTC_SOURCE_DIR:?USTC source required}"
        ARGS+=(
            --ustc-source-dir "$GMM_USTC_SOURCE_DIR"
        )
        ;;
    cicmaldroid2020)
        : "${GMM_CIC_SOURCE_CSV:?CIC source CSV required}"
        : "${GMM_CIC_FEATURE_COLUMNS:?CIC columns required}"
        ARGS+=(
            --cic-source-csv "$GMM_CIC_SOURCE_CSV"
            --cic-feature-columns "$GMM_CIC_FEATURE_COLUMNS"
        )
        ;;
    malnet_image)
        : "${GMM_MALNET_IMAGE_ROOT:?MalNet root required}"
        : "${GMM_MALNET_PREPARATION_DIR:?MalNet preparation required}"
        : "${GMM_MALNET_MAPPING_PATH:?MalNet mapping required}"
        ARGS+=(
            --malnet-image-root "$GMM_MALNET_IMAGE_ROOT"
            --malnet-preparation-dir "$GMM_MALNET_PREPARATION_DIR"
            --malnet-mapping-path "$GMM_MALNET_MAPPING_PATH"
        )
        ;;
    malex)
        MALEX_ARCHIVE="$HOME/gencys/data/MaleX_raw/malex-fyp-image-dataset.zip"

        if [[ ! -f "$MALEX_ARCHIVE" || -L "$MALEX_ARCHIVE" ]]; then
            echo "FAIL: MaleX archive unavailable" >&2
            exit 4
        fi

        if [[ "$(stat -c %s "$MALEX_ARCHIVE")" != "87775065860" ]]; then
            echo "FAIL: MaleX archive size mismatch" >&2
            exit 4
        fi

        STAGE_BASE="${SLURM_TMPDIR:-/tmp}"

        if [[ ! -d "$STAGE_BASE" || -L "$STAGE_BASE" ]]; then
            echo "FAIL: local staging base unavailable" >&2
            exit 4
        fi

        MALEX_STAGE_ROOT="$(
            mktemp -d \
              "$STAGE_BASE/p1r1-gmm-malex-${SLURM_JOB_ID}-XXXXXXXX"
        )"

        cleanup_malex_stage() {
            if [[ -n "${MALEX_STAGE_ROOT:-}" ]]; then
                rm -rf -- "$MALEX_STAGE_ROOT"
            fi
        }
        trap cleanup_malex_stage EXIT

        MALEX_LOCAL_ARCHIVE="$MALEX_STAGE_ROOT/malex-fyp-image-dataset.zip"
        MALEX_EXTRACTED_ROOT="$MALEX_STAGE_ROOT/extracted"

        mkdir -m 700 -- "$MALEX_EXTRACTED_ROOT"

        echo "GMM_MALEX_STAGING: BEGIN"

        cp -- "$MALEX_ARCHIVE" "$MALEX_LOCAL_ARCHIVE"

        if [[ "$(stat -c %s "$MALEX_LOCAL_ARCHIVE")" != "87775065860" ]]; then
            echo "FAIL: staged MaleX archive size mismatch" >&2
            exit 4
        fi

        unzip -q "$MALEX_LOCAL_ARCHIVE" \
            "ben_bigram_imgs_zipped/byteplot_imgs_RxR/256/*.png" \
            "mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/*.png" \
            "labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5" \
            -d "$MALEX_EXTRACTED_ROOT"

        BEN_ROOT="$MALEX_EXTRACTED_ROOT/ben_bigram_imgs_zipped/byteplot_imgs_RxR/256"
        MAL_ROOT="$MALEX_EXTRACTED_ROOT/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256"
        LABEL_PATH="$MALEX_EXTRACTED_ROOT/labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5"

        BEN_COUNT="$(
            find "$BEN_ROOT" -maxdepth 1 -type f -name '*.png' | wc -l
        )"
        MAL_COUNT="$(
            find "$MAL_ROOT" -maxdepth 1 -type f -name '*.png' | wc -l
        )"

        if [[ "$BEN_COUNT" -ne 179725 || "$MAL_COUNT" -ne 864669 ]]; then
            echo "FAIL: staged MaleX image-count mismatch" >&2
            exit 4
        fi

        if [[ ! -f "$LABEL_PATH" || -L "$LABEL_PATH" ]]; then
            echo "FAIL: staged MaleX labels missing" >&2
            exit 4
        fi

        echo "GMM_MALEX_STAGING: PASS"

        ARGS+=(
            --malex-label-archive "$LABEL_PATH"
            --malex-benign-root "$BEN_ROOT"
            --malex-malware-root "$MAL_ROOT"
        )
        ;;
esac


# Only the validated identity namespace may receive a parent directory.
EXPECTED_PARENT="$HOME/trustforge_revision_2026_artifacts/paper01/gmm/$GMM_DATASET_ID/seed$GMM_EXPERIMENTAL_SEED/production"

EXPECTED_PREFIX="class${GMM_CLASS_ID}_seed${GMM_EXPERIMENTAL_SEED}_commit${GMM_EXPECTED_SOURCE_COMMIT:0:7}_attempt"

if [[ "$GMM_CHECKPOINT_DIR" != "$EXPECTED_PARENT/$EXPECTED_PREFIX"* ]]; then
    echo "FAIL: checkpoint namespace mismatch" >&2
    exit 1
fi

ATTEMPT="${GMM_CHECKPOINT_DIR##"$EXPECTED_PARENT/$EXPECTED_PREFIX"}"

if [[ ! "$ATTEMPT" =~ ^[0-9]{2,}$ ]]; then
    echo "FAIL: invalid checkpoint attempt identifier" >&2
    exit 1
fi

if [[ ! -d "$(dirname "$EXPECTED_PARENT")" ]]; then
    echo "FAIL: governed seed parent is unavailable" >&2
    exit 1
fi

if [[ -L "$EXPECTED_PARENT" ]]; then
    echo "FAIL: production parent is a symlink" >&2
    exit 1
fi

if [[ ! -e "$EXPECTED_PARENT" ]]; then
    mkdir -m 700 -- "$EXPECTED_PARENT"
fi

if [[ ! -d "$EXPECTED_PARENT" ]]; then
    echo "FAIL: production parent unavailable" >&2
    exit 1
fi

# The Python runner independently verifies the full destination identity.

echo "GMM_REVISION_LAUNCHER_PREFLIGHT"
echo "DATASET_ID=$GMM_DATASET_ID"
echo "CLASS_ID=$GMM_CLASS_ID"
echo "EXPERIMENTAL_SEED=$GMM_EXPERIMENTAL_SEED"
echo "CHECKPOINT_DIR=$GMM_CHECKPOINT_DIR"
echo "EXECUTION_CHECKOUT=$checkout"
echo "SOURCE_HEAD=$GMM_EXPECTED_SOURCE_COMMIT"
echo "SOURCE_FREEZE=$GMM_SOURCE_FREEZE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

"${ARGS[@]}"
