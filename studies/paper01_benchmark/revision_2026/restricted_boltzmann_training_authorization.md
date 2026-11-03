# TrustForge Paper 1 Revision-2026
# Formal RBM Training Authorization

Date: 2026-11-02

## Decision

RBM_TRAINING_IMPLEMENTATION_ACCEPTED=YES
RBM_METHODOLOGICAL_TRAINING_AUTHORIZED=YES
RBM_PRODUCTION_SAMPLING_AUTHORIZED=NO
RBM_PRODUCTION_SYNTHESIS_AUTHORIZED=NO
RBM_PRODUCTION_GIBBS_DEPTH_SELECTED=NO

Decision status:
RBM TRAINING AUTHORIZED / PRODUCTION SAMPLING RESTRICTED

## Scope of authorization

This decision authorizes the governed training procedure
for the revision-2026 shared class-conditional RBM.

The approved procedure includes:

- Governed continuous-visible conditional RBM.
- Governed CD-1 training updates.
- Fixed batch size of 256.
- Maximum updates equal to 2 * ceil(N_train / 256).
- Validation at steps 0, B and 2B.
- No validation-driven checkpoint selection.
- Final governed update as authoritative checkpoint.
- Dataset, seed and class-identity validation.
- Integrity-checked checkpoint persistence and restoration.
- Fail-closed numerical and publication safeguards.

This decision does not alter the approved training
hyperparameters or the checkpoint-selection policy.

## Engineering evidence

The dedicated RBM regression suite passed:

457 tests across 17 dedicated test modules.

The final acceptance review also confirmed:

- Training entry point exists.
- Training budget is implemented.
- Governed checkpoint persistence is implemented.
- Governed checkpoint loading is implemented.
- Existing checkpoint launcher integrates the training
  and persistence functions.
- The training module does not import the production
  sampling module.

No new training or sampling job was executed as part
of the authorization decision.

## Training and production sampling distinction

The governed training method uses CD-1 updates.

Production synthetic-data generation requires a
separately governed Gibbs-chain depth and potentially
other production sampling controls.

The latter are not prerequisites for the correctness
of the independently specified training update budget.

This authorization does not select or authorize:

- Production Gibbs-chain depth.
- Production burn-in.
- New production synthetic sampling.
- Production synthesis publication.
- A claim of mixing, convergence or distributional
  equivalence.

## Historical assessment preservation

The completed RBM confirmatory assessment remains:

CONFIRMATORY_ASSESSMENT=INDETERMINATE
PRODUCTION_GIBBS_DEPTH_SELECTED=NO

Its historical evidence, hashes, findings and recorded
decision are not modified.

Earlier general RBM authorization denials are superseded
only for the narrowly defined training operation by this
later explicit decision.

They remain effective for production sampling and
synthesis wherever those activities lack independent
authorization.

## Scientific limitations

Training authorization is not proof of:

- Successful real-data convergence.
- Adequate Gibbs mixing.
- Semantic class fidelity.
- Favorable FID, KID or downstream utility.
- Successful production benchmark completion.

Those are experimental outcomes or separate sampling
authorization questions.

## Operational boundary

This record does not submit Slurm jobs, create
checkpoints, activate production sampling, or bypass
existing runtime safeguards.

Any actual training execution must use the governed
dataset, seed, checkpoint destination and compute-node
workflow and satisfy the applicable execution controls.

Operational readiness must be checked independently
before a training job is submitted.

## Final disposition

RBM_TRAINING_HARDENING_STATUS=CLOSED
RBM_TRAINING_AUTHORIZATION=GRANTED
RBM_PRODUCTION_SAMPLING_AUTHORIZATION=WITHHELD
RBM_FULL_FAMILY_PRODUCTION_READINESS=INCOMPLETE

Next work:
Complete VAE methodological acceptance while preserving
the RBM production sampling restriction.
