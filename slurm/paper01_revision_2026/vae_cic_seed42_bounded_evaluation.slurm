#!/usr/bin/env bash
#SBATCH --job-name=p1r1-vae-eval-s42
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/logs/vae-eval-s42-%j.out
#SBATCH --error=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/logs/vae-eval-s42-%j.err

set -euo pipefail
umask 077
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE

# Paper 1 Revision-2026 CICMalDroid2020 VAE seed-42 paired-evaluation launcher.
#
# Approved resource envelope: talon, 1 node, 1 task, 8 CPUs, 32 GB, 02:00:00,
# CPU-only, no requeue, no automatic retries, one bounded attempt.
#
# Resource approval is NOT execution authorization. This launcher remains
# non-operational until a separate researcher-approved evaluation
# authorization record exists at the fixed path below and the final
# evaluation-source revision and deployment paths are frozen.
#
# Prerequisites prepared outside this task (not created here):
# - log parent:
#   /home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/
#   evaluations/cicmaldroid2020/seed42/logs
# - evaluation run parent:
#   /home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/
#   evaluations/cicmaldroid2020/seed42/runs
# - evaluation authorization record:
#   $HOME/trustforge_revision_2026_artifacts/paper01/vae/evaluations/
#   authorizations/cicmaldroid2020_seed42_evaluation.json
# - final evaluation source commit and clean checkout
# - governed CIC dataset configuration and Inception feature-extractor weights
#
# The synthesis source d4031a9b39565e5382d08e2b28236162498a08ca is distinct
# from the final evaluation source. This launcher rejects the synthesis source
# for evaluation.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_JOB_NODELIST:?Slurm node list required}"
: "${VAE_EVAL_EXECUTION_CHECKOUT:?Explicit clean evaluation checkout required}"
: "${VAE_EVAL_SOURCE_COMMIT:?Final evaluation source commit required}"
: "${VAE_EVAL_DATASET_CONFIG:?Governed CIC dataset configuration required}"
: "${VAE_EVAL_WEIGHTS:?Governed Inception weights required}"

[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$VAE_EVAL_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$VAE_EVAL_EXECUTION_CHECKOUT" && ! -L "$VAE_EVAL_EXECUTION_CHECKOUT" ]]

checkout="$(
    cd "$VAE_EVAL_EXECUTION_CHECKOUT"
    pwd -P
)"
[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

home="$(
    cd "$HOME"
    pwd -P
)"

PYTHON="$home/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

AUTHORIZATION="$home/trustforge_revision_2026_artifacts/paper01/vae/evaluations/authorizations/cicmaldroid2020_seed42_evaluation.json"
RUN_DIR="$home/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/runs/attempt01"
POOL_MANIFEST="$home/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/pools/production/checkpoint35ca3cd756a88d6b93dcc291f02c4747_synthesis-d4031a9_attempt01/pool/manifest.json"
LAUNCHER="$checkout/slurm/paper01_revision_2026/vae_cic_seed42_bounded_evaluation.slurm"
DRIVER="$checkout/scripts/run_paper01_cic_vae_seed42_bounded_evaluation.py"

export PYTHONPATH="$checkout/src:$checkout"
export PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Fail-closed authorization, source, destination and allocation binding.
# This runs before any driver invocation and before classifier training.
# ---------------------------------------------------------------------------
"$PYTHON" -B -m trustforge.evaluation.vae_evaluation_launch_contract \
    "$AUTHORIZATION" \
    "$checkout" \
    "$LAUNCHER" \
    "$POOL_MANIFEST" \
    "$RUN_DIR" \
    "$VAE_EVAL_DATASET_CONFIG" \
    "$VAE_EVAL_WEIGHTS" \
    "$VAE_EVAL_SOURCE_COMMIT"

# ---------------------------------------------------------------------------
# Runtime environment: governed interpreter, checkout-bound source, CPU only.
# ---------------------------------------------------------------------------
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -B - "$checkout" <<'PY'
import importlib.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit("gcs_py310 interpreter required")
for name in ("numpy", "tensorflow", "keras", "trustforge"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit("required module unavailable: " + name)
PY

check_source() {
    [[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$VAE_EVAL_SOURCE_COMMIT" ]]
    [[ -z "$(git -C "$checkout" status --porcelain --untracked-files=all)" ]]
}

COMMON_ARGS=(
    --evaluator-seed 42
    --repository-root "$checkout"
    --expected-head "$VAE_EVAL_SOURCE_COMMIT"
    --dataset-config "$VAE_EVAL_DATASET_CONFIG"
    --config-sha256 "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
    --pool-manifest "$POOL_MANIFEST"
    --pool-manifest-sha256 "08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6"
    --weights "$VAE_EVAL_WEIGHTS"
)

# ---------------------------------------------------------------------------
# Preflight must pass before any classifier training can begin.
# ---------------------------------------------------------------------------
check_source
"$PYTHON" -B -u "$DRIVER" --preflight-only "${COMMON_ARGS[@]}"
check_source

# ---------------------------------------------------------------------------
# Paired classifier training. The driver creates exactly one fresh run dir.
# ---------------------------------------------------------------------------
[[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]]
"$PYTHON" -B -u "$DRIVER" --phase train "${COMMON_ARGS[@]}" \
    --run-dir "$RUN_DIR"
check_source

# ---------------------------------------------------------------------------
# Receipt: hash the actual published bytes, verify bounded identity, then
# restore in a separate Python process with the exact digest.
# ---------------------------------------------------------------------------
RECEIPT="$RUN_DIR/expectations.json"
[[ -f "$RECEIPT" && ! -L "$RECEIPT" ]]
RECEIPT_SHA="$(
    "$PYTHON" -B - "$RECEIPT" <<'PY'
import hashlib
import sys

print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
[[ "$RECEIPT_SHA" =~ ^[0-9a-f]{64}$ ]]

"$PYTHON" -B - "$RECEIPT" "$VAE_EVAL_SOURCE_COMMIT" "$SLURM_JOB_ID" <<'PY'
import json
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "schema": "trustforge.paper01.paired-restoration-expectations.v1",
    "evaluator_seed": 42,
    "synthetic_pool_manifest_sha256":
        "08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6",
    "source_commit": sys.argv[2],
    "job_id": sys.argv[3],
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit("training receipt identity mismatch: " + key)
if type(receipt.get("train_pid")) is not int:
    raise SystemExit("training receipt lacks a training process identity")
PY
printf 'EVALUATION_RECEIPT_SHA256=%s\n' "$RECEIPT_SHA"

check_source
"$PYTHON" -B -u "$DRIVER" --phase restore "${COMMON_ARGS[@]}" \
    --run-dir "$RUN_DIR" --receipt-sha256 "$RECEIPT_SHA"
check_source

printf 'VAE_CIC_SEED42_EVALUATION_LAUNCHER=PASS\n'
printf 'COMPLETE_EVALUATION=False\n'
printf 'FULL_BENCHMARK_GO=NO\n'
