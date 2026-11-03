# Paper 1 Revision 2026 — GAN Training and Checkpoint Policy

## Status

SCIENTIFIC TRAINING POLICY APPROVED — OPERATIONAL EXECUTION DISABLED

## 1. Scope

This document governs the initial revision-2026 conditional DCGAN
training-budget and checkpoint implementation.

It does not authorize production training or final benchmark evaluation.

## 2. Training iteration identity

One generator iteration contains:

- one discriminator optimizer update on a real batch;
- one discriminator optimizer update on a generated batch;
- one generator optimizer update through the combined model.

Report generator iterations and discriminator updates separately.

The historical update arrangement is preserved for the initial
implementation, subject to regression validation.

## 3. Approved production training budget

For training-set size N and batch size B = 256:

    steps_per_pass = ceil(N / B)
    generator_iterations = min(
        10000,
        max(500, 2 * steps_per_pass)
    )

The resulting predeclared limits are:

- USTC-TFC2016: 570 generator iterations;
- CICMalDroid2020: 500 generator iterations;
- MalNet-Image: 6902 generator iterations;
- MaleX: 5712 generator iterations.

These are approved prospective benchmark training limits,
not validated convergence guarantees.

## 4. Class exposure

The approved real-data class sampler uses the separately
documented power-law exponent alpha = 0.85.

Class counts are derived from the governed training split only.

Training exposure, duplicate sampling, and effective nominal
passes must be reported.

A minimum-update floor does not establish model convergence.

## 5. Checkpoint selection

For the approved production protocol, the authoritative generator
checkpoint is the terminal checkpoint at the predeclared
completed training limit.

Optional historical FID-based G_best selection is not authoritative.

No validation or test evaluation result may retrospectively
change the checkpoint-selection rule.

The selected checkpoint must record its authoritative generator
iteration, model identity, dataset, class mapping, and seed.

## 6. Validation and failure behavior

Validation diagnostics must use the governed validation split only.

Held-out test data must remain isolated from training,
checkpoint selection, and methodological tuning.

Missing, corrupt, incomplete, or incompatible checkpoints
must fail closed.

Sampling from randomly initialized weights after a failed
checkpoint load is forbidden.

Nonfinite losses, incompatible class mappings, and invalid
image ranges must produce explicit failures.

## 7. Feasibility evidence requirements

Before considering a production budget, establish:

- architecture and loss correctness;
- real and fake batch semantics;
- deterministic sampling and update accounting;
- finite adversarial training behavior;
- validation diagnostic availability and consistency;
- checkpoint save/load fidelity;
- feasible accelerator memory and runtime;
- evidence of training progress and potential failure modes.

Feasibility pilots must use Slurm compute nodes,
not Talon login nodes.

Performance evidence must not be selected or tuned using
held-out test results or final benchmark outcomes.

## 8. Production decision boundary

Scenario A is the approved prospective GAN training protocol.

It must not be presented as a proven sufficient GAN
training budget.

The existing engineering and compute-node smoke evidence
supports prospective policy approval. Final execution
integration and acceptance remain mandatory.

## 9. Authorization state

GAN_TRAINING_BUDGET=SCENARIO_A_APPROVED
GAN_FEASIBILITY_BATCH_SIZE=256
GAN_FEASIBILITY_MIN_GENERATOR_ITERATIONS=500
GAN_FEASIBILITY_MAX_GENERATOR_ITERATIONS=10000
GAN_FEASIBILITY_NOMINAL_PASSES=2
GAN_FEASIBILITY_CHECKPOINT=TERMINAL
GAN_PRODUCTION_BUDGET_GOVERNED=YES
GAN_PRODUCTION_CHECKPOINT_POLICY_GOVERNED=YES
GAN_IMPLEMENTATION_ACCEPTED=YES
GAN_FEASIBILITY_TRAINING_AUTHORIZED=NO
GAN_METHODOLOGICAL_TRAINING_AUTHORIZED=YES
GAN_OPERATIONAL_EXECUTION_ENABLED=NO
