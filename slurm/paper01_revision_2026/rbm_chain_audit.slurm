#!/usr/bin/env bash
#SBATCH -J p1r1-rbm-audit
#SBATCH -o p1r1-rbm-audit.%j.out
#SBATCH -e p1r1-rbm-audit.%j.err
#SBATCH -p talon
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00

set -euo pipefail

module purge
module load slurm/slurm/25.05.6

source ~/miniforge3/etc/profile.d/conda.sh
conda activate gcs_py310

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"

if [[ ! -d .git ]]; then
  echo "[error] SLURM_SUBMIT_DIR is not the TrustForge repository root" >&2
  exit 2
fi

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
: "${CLASS_IDS:?CLASS_IDS is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

REPLICATES="${REPLICATES:-16}"
SAMPLE_INDEX_OFFSET="${SAMPLE_INDEX_OFFSET:-0}"

if [[ ! -f "${CHECKPOINT_DIR}/model_state.npz" ]]; then
  echo "[error] missing model_state.npz in ${CHECKPOINT_DIR}" >&2
  exit 3
fi

if [[ ! -f "${CHECKPOINT_DIR}/model_metadata.json" ]]; then
  echo "[error] missing model_metadata.json in ${CHECKPOINT_DIR}" >&2
  exit 3
fi

if [[ -e "${OUTPUT_JSON}" ]]; then
  echo "[error] refusing existing OUTPUT_JSON=${OUTPUT_JSON}" >&2
  exit 4
fi

case "${OUTPUT_JSON}" in
  "$PWD"/*)
    ;;
  *)
    echo "[error] OUTPUT_JSON must be inside repository root" >&2
    exit 5
    ;;
esac

CLASS_ARGS=()
for class_id in ${CLASS_IDS}; do
  CLASS_ARGS+=(--class-id "${class_id}")
done

export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[P1-R1 RBM AUDIT] job_id=${SLURM_JOB_ID}"
echo "[P1-R1 RBM AUDIT] node=${SLURM_JOB_NODELIST:-unknown}"
echo "[P1-R1 RBM AUDIT] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[P1-R1 RBM AUDIT] class_ids=${CLASS_IDS}"
echo "[P1-R1 RBM AUDIT] replicates=${REPLICATES}"
echo "[P1-R1 RBM AUDIT] sample_index_offset=${SAMPLE_INDEX_OFFSET}"
echo "[P1-R1 RBM AUDIT] output_json=${OUTPUT_JSON}"
echo "[P1-R1 RBM AUDIT] selection_status=UNSELECTED"

python \
  scripts/run_paper01_rbm_chain_audit.py \
  --repo-root "$PWD" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  "${CLASS_ARGS[@]}" \
  --replicates "${REPLICATES}" \
  --sample-index-offset "${SAMPLE_INDEX_OFFSET}" \
  --output-json "${OUTPUT_JSON}"

echo "[P1-R1 RBM AUDIT] status=COMPLETE"
