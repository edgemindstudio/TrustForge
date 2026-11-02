#!/usr/bin/env bash
#SBATCH -J p1r1-rbm-train
#SBATCH -o p1r1-rbm-train.%j.out
#SBATCH -e p1r1-rbm-train.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00

set -euo pipefail

module purge
module load slurm/slurm/25.05.6

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"

if [[ ! -d .git ]]; then
  echo "[error] SLURM_SUBMIT_DIR is not the TrustForge repository root" >&2
  exit 2
fi

: "${DATASET_ID:?DATASET_ID is required}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"

EXPERIMENTAL_SEED="${EXPERIMENTAL_SEED:-42}"

case "${DATASET_ID}" in
  ustc_tfc2016|cicmaldroid2020|malnet_image|malex)
    ;;
  *)
    echo "[error] unsupported DATASET_ID=${DATASET_ID}" >&2
    exit 2
    ;;
esac

if [[ -e "${CHECKPOINT_DIR}" ]]; then
  echo "[error] refusing existing CHECKPOINT_DIR=${CHECKPOINT_DIR}" >&2
  exit 3
fi

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[P1-R1 RBM SLURM] job_id=${SLURM_JOB_ID}"
echo "[P1-R1 RBM SLURM] node=${SLURM_JOB_NODELIST:-unknown}"
echo "[P1-R1 RBM SLURM] repo=$PWD"
echo "[P1-R1 RBM SLURM] dataset_id=${DATASET_ID}"
echo "[P1-R1 RBM SLURM] experimental_seed=${EXPERIMENTAL_SEED}"
echo "[P1-R1 RBM SLURM] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[P1-R1 RBM SLURM] training_budget=GOVERNED_UNMODIFIED"

python -V

MALEX_EXTRACTED_ROOT=""

if [[ "${DATASET_ID}" == "malex" ]]; then
  MALEX_ARCHIVE="$HOME/gencys/data/MaleX_raw/malex-fyp-image-dataset.zip"
  MALEX_STAGE_ROOT="/tmp/$USER/p1r1-rbm-malex-${SLURM_JOB_ID}"
  MALEX_LOCAL_ARCHIVE="${MALEX_STAGE_ROOT}/malex-fyp-image-dataset.zip"
  MALEX_EXTRACTED_ROOT="${MALEX_STAGE_ROOT}/extracted"

  cleanup_malex_stage() {
    rm -rf "${MALEX_STAGE_ROOT}"
  }

  trap cleanup_malex_stage EXIT

  if [[ ! -f "${MALEX_ARCHIVE}" ]]; then
    echo "[error] MaleX source archive missing: ${MALEX_ARCHIVE}" >&2
    exit 4
  fi

  mkdir -p "${MALEX_EXTRACTED_ROOT}"

  echo "[P1-R1 RBM SLURM] malex_staging=BEGIN"
  echo "[P1-R1 RBM SLURM] malex_stage_root=${MALEX_STAGE_ROOT}"

  cp "${MALEX_ARCHIVE}" "${MALEX_LOCAL_ARCHIVE}"

  if [[ "$(stat -c %s "${MALEX_LOCAL_ARCHIVE}")" != "87775065860" ]]; then
    echo "[error] staged MaleX archive byte-size mismatch" >&2
    exit 4
  fi

  unzip -q "${MALEX_LOCAL_ARCHIVE}" \
    "ben_bigram_imgs_zipped/byteplot_imgs_RxR/256/*.png" \
    "mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/*.png" \
    "labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5" \
    -d "${MALEX_EXTRACTED_ROOT}"

  BEN_COUNT="$(
    find "${MALEX_EXTRACTED_ROOT}/ben_bigram_imgs_zipped/byteplot_imgs_RxR/256" \
      -maxdepth 1 -type f -name "*.png" | wc -l
  )"

  MAL_COUNT="$(
    find "${MALEX_EXTRACTED_ROOT}/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256" \
      -maxdepth 1 -type f -name "*.png" | wc -l
  )"

  LABEL_PATH="${MALEX_EXTRACTED_ROOT}/labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5"

  if [[ "${BEN_COUNT}" -ne 179725 ]]; then
    echo "[error] staged MaleX benign count mismatch: ${BEN_COUNT}" >&2
    exit 4
  fi

  if [[ "${MAL_COUNT}" -ne 864669 ]]; then
    echo "[error] staged MaleX malware count mismatch: ${MAL_COUNT}" >&2
    exit 4
  fi

  if [[ ! -f "${LABEL_PATH}" ]]; then
    echo "[error] staged MaleX label archive missing" >&2
    exit 4
  fi

  echo "[P1-R1 RBM SLURM] malex_benign_count=${BEN_COUNT}"
  echo "[P1-R1 RBM SLURM] malex_malware_count=${MAL_COUNT}"
  echo "[P1-R1 RBM SLURM] malex_staging=PASS"
fi

TRAIN_ARGS=(
  scripts/train_paper01_rbm_revision_checkpoint.py
  --repo-root "$PWD"
  --dataset-id "${DATASET_ID}"
  --experimental-seed "${EXPERIMENTAL_SEED}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
)

if [[ "${DATASET_ID}" == "malex" ]]; then
  TRAIN_ARGS+=(
    --malex-extracted-root "${MALEX_EXTRACTED_ROOT}"
  )
fi

python "${TRAIN_ARGS[@]}"

echo "[P1-R1 RBM SLURM] status=COMPLETE"
