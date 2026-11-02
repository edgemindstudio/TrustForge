# VAE Checkpoint v1 — Engineering Contract

## 1. Status and authority

`VAE_CHECKPOINT_SCHEMA_VERSION=1.0`

`VAE_CHECKPOINT_DESIGN_STATUS=PROPOSED`

`VAE_CHECKPOINT_IMPLEMENTATION_ACCEPTED=NO`

`VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO`

This contract describes a proposed artifact-backed checkpoint
format for the revised Paper 1 conditional VAE.

It does not authorize real-data training, checkpoint publication,
or production synthesis.

## 2. Model-state boundary

One checkpoint bundle must contain a complete, mutually
consistent encoder and decoder state.

Encoder state includes trainable and nontrainable variables,
including BatchNormalization moving statistics.

Decoder state includes all trainable and nontrainable variables.

Missing, duplicated, malformed, nonfinite, or incompatible
state tensors must cause explicit rejection.

Weights-only restoration is distinct from exact training
resumption. A resumable checkpoint requires separately
governed optimizer, random-state, traversal-progress, and
runtime-state semantics.

## 3. Proposed artifact inventory

Required authoritative files:

- `vae_encoder_state.npz`
- `vae_decoder_state.npz`
- `vae_checkpoint_manifest.json`

The two state artifacts must be integrity-bound to one
manifest and must never be accepted independently.

The use of NPZ is an engineering candidate, not yet an
accepted serialization decision. Loading must avoid pickle
and reject unexpected archive members.

## 4. Scientific identity

The manifest must bind:

- Schema version and model family
- Paper revision and dataset identifier
- Training and validation partition provenance
- Class mapping and class count
- Experimental seed
- Canonical image shape `(40,40,3)`
- Latent dimension `100`
- Conditional architecture version
- Observation-model version
- Fixed Gaussian variance `1.0`
- KL coefficient `1.0`
- No KL annealing
- Effective runtime configuration
- Selected epoch and completed training updates
- Explicit checkpoint role: `best` or `last`
- Recorded validation criterion and selection evidence
- Encoder and decoder state-file SHA-256 digests

Manifest fields must be exact and strictly typed.
Unknown or missing fields must cause rejection.

## 5. Checkpoint roles

A `best` checkpoint represents a selection decision made
under an explicitly governed validation criterion.

A `last` checkpoint represents the most recent completed
checkpoint under the governed execution policy.

Best and last may refer to the same completed epoch.

Checkpoint persistence must not invent validation criteria
or silently alter deterministic tie-breaking rules.

## 6. Persistence protocol

The proposed persistence lifecycle is:

1. Validate identity, provenance, role, and source state.
2. Reject an invalid or preexisting authoritative destination.
3. Acquire an exclusive publication lock.
4. Write state artifacts into a staging directory.
5. Synchronize state files and calculate SHA-256 hashes.
6. Construct and strictly validate the checkpoint manifest.
7. Verify the complete staged bundle.
8. Publish state artifacts without overwriting existing files.
9. Synchronize the authoritative directory.
10. Verify published state artifacts.
11. Publish the manifest last without overwriting.
12. Synchronize the authoritative directory and release lock.

Interruption before completed publication must not make a
checkpoint loadable.

Unexpected partial artifacts, stale publication locks,
corruption, or missing files must fail closed.

Cleanup and interrupted-publication recovery require
explicit engineering tests.

## 7. Secure restoration

Restoration must verify the complete artifact inventory,
manifest, file hashes, model identity, tensor inventory,
shapes, dtypes, and finiteness before modifying any
caller-provided model variables.

No untrusted pickle loading is permitted.

Unexpected symlinks, nonregular files, oversized artifacts,
and unsupported schema versions must be rejected.

The restoration path must not bypass provenance validation
or infer authorization from the existence of a manifest.

## 8. Acceptance tests

Required synthetic-only engineering tests include:

1. Complete paired-state round-trip.
2. Encoder and decoder restoration fidelity.
3. BatchNormalization-state restoration.
4. Missing encoder-state rejection.
5. Missing decoder-state rejection.
6. Corrupt state-file rejection.
7. Manifest corruption and schema mismatch rejection.
8. Dataset, seed, and class-mapping mismatch rejection.
9. Architecture and tensor-shape mismatch rejection.
10. Nonfinite tensor rejection.
11. No-overwrite persistence.
12. Partial-publication rejection.
13. Publication-lock rejection.
14. Unexpected artifact and symlink rejection.
15. Best/last role validation.
16. Restoration failure without partial model mutation.

These tests establish engineering behavior only.

## 9. Unresolved decisions

The following remain pending:

- Final serialization format and archive limits
- Exact architecture and observation version identifiers
- Complete partition-provenance schema
- Effective runtime-configuration schema
- Validation criterion and deterministic tie-breaking
- Validation posterior seed schedule
- Optimizer-state and exact-resumption policy
- Filesystem interruption and recovery acceptance

No production checkpoint format is accepted until these
decisions and the relevant tests are completed.

## 10. Nonauthorization

`VAE_CHECKPOINT_DESIGN_STATUS=PROPOSED`

`VAE_CHECKPOINT_IMPLEMENTATION_ACCEPTED=NO`

`VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO`
