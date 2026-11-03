# Paper 1 Revision-2026 — GAN Scientific Policy Decision

Date: 2026-11-02

## Decision status

GAN_SCIENTIFIC_POLICY=APPROVED
GAN_CLASS_SAMPLING_ALPHA=0.85
GAN_PRODUCTION_BUDGET_POLICY=SCENARIO_A_APPROVED
GAN_CHECKPOINT_SELECTION=TERMINAL_PREDECLARED
GAN_VALIDATION_ROLE=DESCRIPTIVE_NONSELECTING

GAN_IMPLEMENTATION_ACCEPTANCE=ACCEPTED
GAN_PRODUCTION_EXECUTION_READY=NO
GAN_PRODUCTION_TRAINING_AUTHORIZED=YES

## 1. Methodological basis

The original AR, MAF, and GMM acceptance precedent requires
a scientifically governed, reproducible and validated training
procedure before benchmark execution. It does not require proof
of successful production convergence before training authorization.

GAN quality, collapse behavior and downstream utility are
benchmark outcomes, not prerequisites to beginning the benchmark.

## 2. Class-sampling decision

Approve the existing power-law policy:

    p(c) = n_c**0.85 / sum_j(n_j**0.85)

Select each example uniformly within the sampled class,
with replacement, using governed training data only.

The exponent is justified by the completed analytical
exposure audit. It is not claimed to be empirically optimal.

## 3. Training-budget decision

Approve batch size 256 and Scenario A:

    B = 256
    T = min(10000, max(500, 2*ceil(N_train/B)))

One generator iteration means two discriminator optimizer
updates followed by one generator optimizer update.

The two-pass term scales exposure with dataset size.
The 500-update floor ensures a minimum adversarial update
opportunity; it is not a convergence guarantee.
The 10000-update ceiling bounds computational expenditure.

Sampling is with replacement. Actual exposures and nominal
passes must be reported without claiming complete unique
example coverage.

No dataset-specific retrospective tuning using benchmark
outcomes is permitted.

## 4. Checkpoint-selection decision

Select the terminal checkpoint at the predeclared completed
iteration limit. Do not switch to a better-looking historical
checkpoint based on FID, KID, or other observed outcomes.

Incomplete, incompatible, corrupt or nonfinite states fail
closed. Historical and Paper 2 artifacts remain isolated.

## 5. Validation-monitoring decision

Validation is descriptive and must not change checkpoint
selection or training hyperparameters.

Record finite generator/discriminator losses, update counts,
class exposures, numerical validity and completion status.

Predeclare deterministic class-conditional generation checks
and descriptive validation-only diagnostics for the executed
benchmark. Their exact execution interface and recording
mechanism must be verified before production authorization.

The existing run already records adversarial losses and
optimizer/exposure accounting. No claim is made here that
all descriptive validation diagnostics are implemented.

The held-out test split must not influence training,
selection, monitoring-policy changes, or model tuning.

## 6. Feasibility adjudication

The earlier 509-test GAN engineering regression and
compute-node TensorFlow smoke results provide substantial
pre-training implementation evidence.

A new sustained real-data convergence pilot is not a
mandatory pre-authorization gate.

Actual runtime feasibility and failures remain reportable
during governed benchmark execution. No favorable image
quality or downstream score is guaranteed or required.

## 7. Relationship to earlier documents

This decision resolves the provisional scientific choices
in the existing GAN class-sampling and training/checkpoint
policies. Those documents require subsequent consistent
status reconciliation.

Existing authorization guards, publication constraints,
identity checks and fail-closed requirements remain active.

## 8. Remaining mandatory closure

1. Reconcile implementation acceptance and older policy flags.
2. Provide and verify an authorized production execution path.
3. Verify descriptive diagnostic recording under that path.
4. Complete the targeted final regression/contract review.
5. Issue an explicit production training-authorization record.

Do not add convergence, quality-optimization, checkpoint-search
or unrelated hardening gates.

## 9. Stop rule

Once the above finite mandatory conditions pass, declare:

    GAN = COMPLETE / TRAINING AUTHORIZED

Then stop GAN hardening and proceed to the next family.

## Final acceptance disposition

The original methodological authorization standard is met.

The production entry point has been implemented, guarded,
and regression-tested. Fixed-seed class-conditional
generation checks and bounded validation-only diagnostics
have been implemented and tested with controlled fixtures.
The terminal runner records those diagnostics before
checkpoint publication.

The production launcher has not been verified through
an authorized end-to-end Slurm execution. Accordingly,
operational execution remains disabled.

This distinction does not reopen the scientific training
authorization decision.

GAN = COMPLETE / TRAINING AUTHORIZED
