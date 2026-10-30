# Paper 1 Revision 2026 — GAN Checkpoint v3 Provenance Contract

## Status

PROPOSED — IMPLEMENTATION NOT AUTHORIZED BY THIS DOCUMENT

GAN feasibility training: DISABLED.
GAN production training: DISABLED.

## 1. Version separation

Checkpoint schema v2.0 remains unchanged and independently valid.

Schema v3.0 introduces mandatory checkpoint-bound dataset provenance.

No silent conversion of v2 checkpoints into v3 is permitted.

Historical v2 artifacts must not be represented as having v3 provenance.

## 2. Required v3 artifacts

- model_metadata.json
- TensorFlow model_state.ckpt state files
- gan_exposure_evidence.json
- gan_dataset_provenance.json

The authoritative metadata must identify and hash every required
evidence artifact and every checkpoint state file.

## 3. Dataset provenance evidence

The provenance record must contain:

- Explicit schema identity and version
- Governed dataset ID
- Dataset source-contract identity
- Dataset-specific source or preparation fingerprints
- Dataset partition-policy identity and parameters
- Evidence scope and verification method
- Verification limitations
- Relevant dataset contract identifiers

The record must use canonical JSON.

Unknown, missing, incompatible, or contradictory fields fail closed.

## 4. Dataset-specific requirements

### USTC-TFC2016

Require the four established NCHW source-file SHA-256 fingerprints.

Require partition seed 42 and validation fraction 0.1.

The historical NCHW source and NHWC convenience representation
must not be confused.

### CICMalDroid2020

Require the governed source and feature-column contracts,
hash validation enabled, seed 42, and fractions 0.7/0.1/0.2.

### MalNet-Image

Require governed preparation and mapping contracts, with hash
validation and production-contract validation enabled.

Do not claim full per-image hashing unless separately established.

### MaleX

Require governed label-archive, exclusion, membership, and split
contracts, with hash validation and production-contract validation
enabled.

Do not claim full per-image hashing unless separately established.

## 5. Publication rules

Validate the provenance evidence before publication.

Bind its canonical JSON SHA-256 to the checkpoint metadata.

Preserve fail-closed publication and no-overwrite semantics.

Publish metadata last, after the state and evidence artifacts
are durably written and verified.

## 6. Loading rules

Validate the declared checkpoint schema version explicitly.

For v3, verify all state and evidence hashes before constructing
models or restoring state.

Verify provenance evidence against checkpoint dataset identity.

Reject missing, malformed, conflicting, or unsupported evidence.

A self-consistent checkpoint does not independently establish
trusted origin or continuous immutability of the training data.

## 7. Compatibility and tests

Preserve the existing v2 regression and round-trip tests.

Add independent v3 schema and provenance-evidence tests.

Test missing artifacts, hash mismatch, identity mismatch,
noncanonical JSON, schema confusion, and interrupted publication.

Run real TensorFlow checkpoint tests on Slurm compute nodes only.

## 8. Authorization

Provenance verification does not authorize training.

GAN_FEASIBILITY_TRAINING_ENABLED = False
GAN_PRODUCTION_TRAINING_ENABLED = False

No feasibility or production GAN training is permitted by this
contract.
