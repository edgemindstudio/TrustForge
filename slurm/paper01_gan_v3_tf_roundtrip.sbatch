#!/bin/bash
#SBATCH --job-name=gan-v3-tf-smoke
#SBATCH --partition=talon-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/gan-v3-tf-smoke-%j.out
#SBATCH --error=slurm/logs/gan-v3-tf-smoke-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

/home/bruno.fonkeng/miniforge3/envs/gcs_py310/bin/python -u tests/manual_paper01_gan_v3_tf_roundtrip.py
