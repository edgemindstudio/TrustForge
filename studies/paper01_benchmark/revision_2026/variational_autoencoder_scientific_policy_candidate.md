# Conditional VAE — Scientific Training Policy Candidate

## Paper 1 Revision-2026

Status: PROPOSED — NOT APPROVED

This document records a candidate scientific protocol.
It does not authorize real-data training, checkpoint acceptance,
or production synthesis.

## 1. Experimental scope

- Datasets: USTC-TFC2016, CICMalDroid2020, MalNet-Image, MaleX.
- Generator family: class-conditional VAE.
- Experimental seeds: seven, under the final approved seed registry.
- Primary synthetic generation budget: 2000 samples per class.
- Execution environment: UND Talon, CPU-only.

The generation budget is not a training-update budget.

## 2. Architecture and observation model

Existing Revision-2026 candidate implementation:

- Canonical image shape: 40 x 40 x 3.
- Internal image range: [-1, 1].
- Latent dimension: 100.
- Encoder: conditional convolutional Gaussian posterior.
- Decoder: conditional convolutional network with tanh mean.
- Observation model: fixed-variance Gaussian.
- Observation variance: 1.0.
- KL coefficient: 1.0.
- KL annealing: none.
- Objective: Gaussian NLL plus KL, mean over examples.
- Production generation rule: requires separate approval.

These settings are not newly approved by this document.

## 3. Candidate optimization configuration

- Optimizer: Adam.
- Initial learning-rate candidate: 0.001.
- Initial batch-size candidate: 64.
- Training unit: complete deterministic epoch.
- Training duration: UNDECIDED.
- Validation frequency: once per completed epoch.
- Early stopping: candidate disabled for controlled pilot.

The learning rate and batch size require a bounded CPU pilot
before final selection.

No arbitrary pilot duration may become the production
training duration without a recorded scientific decision.

## 4. Training exposure

For dataset-specific training size N:

updates_per_epoch = ceil(N / batch_size)

total_updates = completed_epochs * updates_per_epoch

All completed optimizer updates and training examples must
be recorded.

Dataset size, update exposure, and computational cost
must be reported separately for each dataset.

## 5. Validation and checkpoint selection

Candidate primary criterion:

- Metric: validation total loss.
- Direction: minimize.
- Validation traversal: complete validation partition.
- Aggregation: sample-weighted.
- Tie-breaking: earliest completed epoch.
- Best checkpoint: selected by the approved criterion.
- Last checkpoint: final completed epoch.

Required decisions before acceptance:

- Final validation posterior seed schedule.
- Final training duration and stopping policy.
- Whether best and last states require separate persistence.
- Exact effective-configuration evidence.

The current synthetic runner does not implement persistent
best-checkpoint selection.

## 6. Canonicalization and representation

All four dataset adapters use common.canonicalization.

Current canonicalizer behavior:

- Explicit source numeric normalization.
- NHWC conversion.
- Bilinear resize with antialiasing to 40 x 40 if needed.
- Grayscale channel replication.
- Final shape and range enforcement.

The scientific consequences of reducing high-resolution malware
images to 40 x 40 require documented justification.

MalNet-Image RGB source handling must be reconciled with any
previously proposed grayscale byteplot representation.

No source representation may be silently changed.

## 7. Engineering evidence

Verified synthetic milestones:

- VAE training and validation smoke test: PASS.
- Synthetic training-to-commit integration: PASS.
- Checkpoint regression baseline: 713 PASS, 2 XFAIL.

Open limitations:

- Two directory-substitution findings.
- GPFS crash durability not established.
- Checkpoint origin authentication not established.
- Production entrypoint integration not verified.
- Final scientific protocol not approved.

## 8. Authorization boundary

VAE_SCIENTIFIC_POLICY_STATUS=PROPOSED
VAE_IMPLEMENTATION_ACCEPTANCE=PENDING
VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO
VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO

A subsequent explicit approval record must identify the
final scientific controls and supporting evidence.
