#!/usr/bin/env bash
set -euo pipefail
WT="$HOME/TrustForge-paper01-vae-stage4-attempt02"
EVIDENCE="$HOME/trustforge_revision_2026_artifacts/paper01/diagnostics/vae-stage4-cicmaldroid2020-seed42-attempt02-job316796-restoration-20261121"
export PYTHONPATH="$WT/src:$WT"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export RESTORATION_RESULT_PATH="$EVIDENCE/restoration_result.json"
cd "$WT"
exec srun   --partition=talon   --nodes=1   --ntasks=1   --cpus-per-task=8   --mem=32G   --time=00:30:00   --job-name=p1r1-vae-restore   --export=ALL   "$HOME/miniforge3/envs/gcs_py310/bin/python"   "$EVIDENCE/restoration_driver.py"
