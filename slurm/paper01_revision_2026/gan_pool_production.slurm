#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gan-pool-s42
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/cicmaldroid2020/seed42/logs/gan-pool-%j.out
#SBATCH --error=/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/gan/cicmaldroid2020/seed42/logs/gan-pool-%j.err

set -euo pipefail
umask 077

# Controlled Paper 1 Revision-2026 GAN seed-42 pool launcher.
# Submission requires a separate synthesis authorization grant.
# One job invokes exactly one authorized publisher call.

: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${SLURM_JOB_NODELIST:?Slurm node list required}"
: "${TRUSTFORGE_SYNTH_CHECKOUT:?Explicit clean synthesis checkout required}"

[[ -d "$TRUSTFORGE_SYNTH_CHECKOUT" && ! -L "$TRUSTFORGE_SYNTH_CHECKOUT" ]]

checkout="$(
    cd "$TRUSTFORGE_SYNTH_CHECKOUT"
    pwd -P
)"

[[ "$(git -C "$checkout" rev-parse --show-toplevel)" == "$checkout" ]]
cd "$checkout"

PYTHON="$HOME/miniforge3/envs/gcs_py310/bin/python"
[[ -x "$PYTHON" ]]

export PYTHONPATH="$checkout/src:$checkout${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

exec "$PYTHON" -B -u scripts/run_paper01_gan_seed42_pool.py
