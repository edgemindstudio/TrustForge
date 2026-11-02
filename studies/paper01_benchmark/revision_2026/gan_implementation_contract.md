# Paper 1 Revision 2026 — Conditional GAN Implementation Contract

Status: DRAFT — IMPLEMENTATION NOT AUTHORIZED
Scope: Revised Paper 1 benchmark only

## 1. Scientific identity

The revised GAN family preserves the historical conditional DCGAN
architecture defined in `gan/models.py`.

The generator receives latent noise and explicit class conditioning.
The discriminator receives images and explicit class conditioning.

The existing adversarial formulation is preserved unless a separately
governed correctness finding establishes that a correction is necessary.

The Paper 2 ACGAN is not substituted for this baseline.

## 2. Canonical representation

All four datasets use the governed revision-2026 data interfaces.

Canonical inputs:
- shape: NHWC, 40 x 40 x 3;
- dtype: float32;
- numerical domain: [0,1].

GAN training applies the deterministic mapping:

    x_internal = 2 * x_canonical - 1

The generator uses tanh outputs in [-1,1].
Production synthesis returns finite canonical outputs in [0,1].

Invalid input shapes, dtypes, values, and labels must fail explicitly.

## 3. Governed data access

Dataset classes:
- USTCRevision2026;
- CICMalDroidRevision2026;
- MalNetImageRevision2026;
- MaleXRevision2026.

Training uses the approved `train` split through `canonical_batch`.
Validation uses only the approved validation split.

The held-out test split must not influence training, model selection,
architecture decisions, or checkpoint selection.

No independent dataset splitting or recanonicalization is permitted.

The implementation must validate dataset identity, class mapping,
split membership, sample counts, and returned batch semantics.

## 4. Class conditioning

Class labels are active inputs to both generator and discriminator.

Synthetic labels must correspond to the actual conditioning passed
to the generator, not post-hoc assignment.

The implementation must support all governed dataset class counts,
including the maximum 47-class setting.

## 5. Training semantics

Preserve the conditional DCGAN architecture and its adversarial
training objective.

Runtime controls must be consumed and recorded, including:
- seed;
- training budget;
- batch size;
- latent dimension;
- optimizer learning rate and beta_1;
- discriminator and generator update schedule;
- any smoothing or noise controls.

The effective training budget, checkpoint-selection rule,
validation cadence, and any optional stabilization controls
require explicit governance before production training.

Historical defaults are not automatically authoritative.

Training must use bounded data batches and avoid requiring
complete high-volume datasets in accelerator memory.

Numerical failure must stop the governed run; no silent
changes to architecture, objective, or optimization are allowed.

## 6. Checkpoint semantics

Production checkpoints must bind:
- dataset and split identity;
- seed and class mapping;
- architecture and effective configuration;
- authoritative training step;
- selected checkpoint rule;
- integrity and artifact provenance.

Missing, corrupt, incompatible, or identity-mismatched checkpoints
must fail explicitly.

Production synthesis must never proceed with random weights after
checkpoint loading fails.

Checkpoint publication must prevent incomplete state from being
accepted as a completed production checkpoint.

## 7. Sampling semantics

The primary budget is 2,000 samples per class.

Sampling must use the requested class conditioning and a
documented reproducible random-stream identity.

Outputs must have valid shape, range, dtype, and class labels.

Publication must protect against accidental overwrites and
record sufficient provenance to link every sample to its
source checkpoint, dataset, seed, and class.

## 8. Historical isolation

Do not rewrite historical Paper 1 evidence or outputs.

Do not alter Paper 2 ACGAN implementation or experimental artifacts.

Revision-specific adapters and modules should reuse the historical
conditional DCGAN architecture without changing historical behavior.

## 9. Required regression and acceptance gates

Before training authorization, verify:

1. Correct generator and discriminator shapes and conditioning.
2. Canonical input and output transformations.
3. All four dataset interfaces and train/validation isolation.
4. Active runtime hyperparameters and training-update semantics.
5. Deterministic seed and sampling identities.
6. Strict checkpoint preflight and rejection of invalid weights.
7. Non-overwriting outputs and complete provenance.
8. Finite numerical behavior and explicit failure handling.
9. Repository regression and historical/Paper 2 isolation.
10. Feasibility under the governed benchmark scale.

Unresolved training-budget and checkpoint-selection decisions
must be recorded and closed before production authorization.

## 10. Authorization state

GAN_ARCHITECTURE_REDESIGN_REQUIRED=NO
GAN_IMPLEMENTATION_CONTRACT_STATUS=DRAFT
GAN_TRAINING_BUDGET_GOVERNED=NO
GAN_CHECKPOINT_SELECTION_GOVERNED=NO
GAN_IMPLEMENTATION_AUTHORIZED=NO
GAN_PRODUCTION_TRAINING_AUTHORIZED=NO
