#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gan-eval-s42
#SBATCH --output=p1r1-gan-eval-s42.%j.out
#SBATCH --error=p1r1-gan-eval-s42.%j.err

set -euo pipefail

# GAN seed-42 paired evaluation launcher template.
#
# BLOCKED: no GAN evaluation resource envelope has been approved.
# The researcher must authorize a bounded Slurm resource envelope before
# this launcher can be made executable.
#
# After approval, the launcher must:
# - run only from a clean checkout at the approved evaluation source commit;
# - set a clean PYTHONPATH and CPU-only environment;
# - invoke scripts/run_paper01_cic_gan_seed42_bounded_evaluation.py for
#   --phase train, then a separate restore phase with the receipt SHA;
# - never retry, requeue, alter the attempt or substitute another pool.
#
# No classifier training, real-data evaluation, pool access, or Slurm work is
# authorized by this template.

echo "BLOCKED: GAN evaluation resource envelope not approved." >&2
echo "No classifier training or evaluation was executed." >&2
exit 78
