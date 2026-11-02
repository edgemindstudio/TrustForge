# VAE Training-Budget Candidate — Paper 1 Revision-2026

Status: PROPOSED — FEASIBILITY ONLY

## Candidate controls

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Training duration: 2 complete epochs
- Early stopping: disabled
- Validation: after every completed epoch
- Checkpoint selection: minimum validation total loss
- Tie-breaking: earliest epoch
- Best and last model states: retained separately

These values are candidates, not an approved production
training schedule.

## Dataset-specific exposure

Counts are taken from existing RBM training-budget
governance documents, not independently reloaded.

| Dataset | Train N | Updates/epoch | Total updates | Total example exposures |
|---|---:|---:|---:|---:|
| USTC-TFC2016 | 72900 | 1140 | 2280 | 145800 |
| CICMalDroid2020 | 8117 | 127 | 254 | 16234 |
| MalNet-Image | 883416 | 13804 | 27608 | 1766832 |
| MaleX | 731049 | 11423 | 22846 | 1462098 |

## Scientific rationale

Two complete epochs provide exactly two training
exposures per example under the current deterministic
VAE traversal policy.

This is a controlled feasibility comparison, not proof
that the VAE has converged or reached adequate quality.

The generator families retain model-specific optimization
policies; equal epochs do not imply equal computational
cost or equivalent optimization opportunity.

## Acceptance requirements

Before production training approval:

1. Verify authoritative adapter-reported train sizes.
2. Establish CPU execution feasibility for the full scale.
3. Evaluate whether this duration is scientifically adequate
   without inspecting held-out test results.
4. Confirm validation seed and checkpoint-selection policies.
5. Approve the representation and scientific contracts.
6. Resolve or explicitly adjudicate checkpoint security and
   durability requirements.
7. Record the final seven-seed experimental configuration.

No automatic extension, hidden tuning, or test-informed
selection is authorized.

## Authorization

VAE_TRAINING_BUDGET_STATUS=FEASIBILITY_CANDIDATE
VAE_PRODUCTION_DURATION_APPROVED=NO
VAE_REAL_DATASET_TRAINING_AUTHORIZED=NO
VAE_CHECKPOINT_PUBLICATION_AUTHORIZED=NO
