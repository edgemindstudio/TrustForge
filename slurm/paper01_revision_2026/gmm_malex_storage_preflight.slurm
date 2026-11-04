#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gmm-storage
#SBATCH --partition=talon
#SBATCH --nodelist=talon16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=p1r1-gmm-storage.%j.out
#SBATCH --error=p1r1-gmm-storage.%j.err

set -euo pipefail

echo "===== GMM MALEX NODE STORAGE PREFLIGHT ====="
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:?}"
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-UNSET}"

STAGE_BASE="${SLURM_TMPDIR:-/tmp}"

if [[ ! -d "$STAGE_BASE" || -L "$STAGE_BASE" ]]; then
    echo "RESULT=FAIL_INVALID_STAGE_BASE"
    exit 1
fi

echo
echo "===== STORAGE CAPACITY IN BYTES ====="
df -B1 "$STAGE_BASE"

echo
echo "===== INODE AVAILABILITY ====="
df -i "$STAGE_BASE"

echo
echo "===== MOUNT INFORMATION ====="
findmnt -T "$STAGE_BASE"     -o TARGET,SOURCE,FSTYPE,OPTIONS

echo
echo "===== MINIMUM ARCHIVE-ONLY REQUIREMENT ====="

AVAILABLE="$(df -B1 --output=avail "$STAGE_BASE" | tail -1 | tr -d ' ')"

ARCHIVE_BYTES=87775065860

echo "AVAILABLE_BYTES=$AVAILABLE"
echo "ARCHIVE_BYTES=$ARCHIVE_BYTES"

if (( AVAILABLE <= ARCHIVE_BYTES )); then
    echo "RESULT=FAIL_INSUFFICIENT_FOR_ARCHIVE_ALONE"
    exit 2
fi

echo "RESULT=ARCHIVE_CAPACITY_ONLY_PASS"
echo "EXTRACTION_CAPACITY=NOT_YET_ESTABLISHED"
echo "TRAINING_EXECUTED=NO"
echo "ARCHIVE_COPIED=NO"
echo "CHECKPOINT_CREATED=NO"
