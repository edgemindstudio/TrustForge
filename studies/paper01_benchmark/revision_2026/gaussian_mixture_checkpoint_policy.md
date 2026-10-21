# Paper 1 Revision 2026 — Gaussian Mixture Checkpoint and Model-Identity Policy

`STATUS=IMPLEMENTED_AND_INTEGRATED_VALIDATED`

`GMM_CHECKPOINT_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

The governed checkpoint implementation is now present in:

`src/trustforge/models/paper01_revision_2026/gaussian_mixture_checkpoint.py`

The checkpoint-focused validation suite passes 12 tests.

The broader integrated revised-GMM stack including checkpoint persistence
passes 171 tests.

The implementation validates:

- authoritative `model_state.npz` persistence using uncompressed `numpy.savez`;
- exact state keys `logits`, `means`, and `raw_scales`;
- exact governed float64 dtypes and shapes;
- `allow_pickle=False` checkpoint loading;
- canonical compact UTF-8 JSON metadata with one trailing newline;
- SHA-256 binding between metadata and exact state-file bytes;
- dataset, class, and experimental-seed scientific identity;
- selected-best-validation checkpoint semantics;
- earliest-step tie-break consistency;
- complete ordered validation history;
- fail-closed state and metadata schema validation;
- no silent authoritative checkpoint overwrite;
- state-first and metadata-last publication;
- file and directory synchronization during publication;
- post-publication round-trip loading and integrity validation;
- rejection of malformed, tampered, incomplete, or identity-mismatched
  checkpoint pairs.

This status advancement applies to checkpoint persistence and load validation.

Model-consistent sampling remains unimplemented and unaccepted.

Overall Gaussian-mixture benchmark training remains unauthorized until
sampling and its required checkpoint preflight are implemented and validated.

## 1. Scope

This policy governs persistence and identity for each revised Paper 1
class-conditional censored diagonal Gaussian mixture model.

It defines:

- the authoritative numerical state artifact;
- the class-model metadata artifact;
- the relationship to the TrustForge native execution manifest;
- integrity validation;
- class/seed identity validation;
- best-checkpoint persistence;
- fail-closed sampling behavior.

It does not yet authorize production training.

Production implementation and implementation acceptance remain separately
governed.

## 2. Three-layer artifact contract

The revised GMM uses three distinct persistence layers.

### Layer 1 — numerical model state

The authoritative numerical state is:

`model_state.npz`

It contains only the trainable state required to reconstruct the selected
class model.

### Layer 2 — class-model metadata

The authoritative class-model metadata is:

`model_metadata.json`

It records the scientific identity and model-selection facts required to
interpret and validate the numerical state.

### Layer 3 — TrustForge execution manifest

Execution-wide provenance remains in the native TrustForge execution
manifest.

The class-model artifacts must not duplicate execution-wide provenance
without a model-specific reason.

`GMM_CHECKPOINT_LAYERING=STATE_PLUS_CLASS_METADATA_PLUS_NATIVE_MANIFEST`

## 3. Numerical state format

The selected numerical state format is uncompressed NumPy NPZ.

`GMM_STATE_FILENAME=model_state.npz`

`GMM_STATE_FORMAT=NUMPY_NPZ_UNCOMPRESSED`

The file is written using `numpy.savez`.

It must be loaded with:

`allow_pickle=False`

`GMM_STATE_ALLOW_PICKLE=NO`

The state contains exactly these keys:

- `logits`
- `means`
- `raw_scales`

No additional object, Python pickle, optimizer instance, callback object, or
arbitrary metadata may be embedded in the NPZ state.

`GMM_STATE_KEYS=logits,means,raw_scales`

## 4. Required state shapes and dtypes

The authoritative state must satisfy:

### logits

Shape:

`(5,)`

dtype:

`float64`

### means

Shape:

`(5,4800)`

dtype:

`float64`

### raw_scales

Shape:

`(5,4800)`

dtype:

`float64`

Therefore:

`GMM_STATE_COMPONENTS=5`

`GMM_STATE_DIMENSION=4800`

`GMM_STATE_DTYPE=FLOAT64`

Loading must fail closed on any unexpected key, missing key, wrong shape, or
wrong dtype.

## 5. State semantic validation

After loading, the implementation must validate:

- all logits are finite;
- all means are finite;
- all raw scales are finite;
- transformed scales are finite;
- transformed scales satisfy the governed minimum-scale contract;
- softmax mixture weights are finite;
- mixture weights are positive;
- mixture weights sum to one within the production numerical tolerance.

Latent means are not required to lie in `[0,1]`.

No invalid parameter may be repaired by clipping, replacement, deletion, or
fallback.

## 6. Serialization evidence

A revision-scale serialization prototype used:

- `K=5`;
- `D=4800`;
- float64 logits;
- float64 means;
- float64 raw scales.

Two independently written uncompressed NPZ files from identical in-memory
arrays were byte-identical in the audited Talon environment and had the same
SHA-256 digest.

The uncompressed state size was:

`384800 bytes`

Exact round-trip validation passed for:

- keys;
- shapes;
- dtypes;
- logits values;
- mean values;
- raw-scale values.

Compressed NPZ was also byte-identical in the audited environment, but
reduced the synthetic state only from approximately 384800 bytes to 369584
bytes.

The revision therefore selects the simpler uncompressed NPZ representation.

This audit demonstrates determinism in the tested execution environment.

It does not assert that every future NumPy release or operating environment
must independently regenerate byte-identical ZIP container bytes.

Artifact SHA-256 records the integrity of the concrete persisted state.

## 7. Class-model metadata format

The class-model metadata filename is:

`model_metadata.json`

`GMM_METADATA_FILENAME=model_metadata.json`

The metadata must be canonical JSON written as:

- UTF-8;
- sorted keys;
- compact deterministic separators;
- no NaN or infinity values;
- exactly one trailing newline.

The writer must refuse non-finite numeric metadata.

`GMM_METADATA_FORMAT=CANONICAL_JSON`

## 8. Required metadata identity

The class-model metadata must contain at least:

- `schema_version`;
- `schema_type`;
- `dataset_id`;
- `class_id`;
- `experimental_seed`;
- `state_file`;
- `state_sha256`;
- `components`;
- `dimension`;
- `parameter_dtype`;
- `observation_model`;
- `selected_step`;
- `selected_validation_nll`;
- `train_class_count`;
- `validation_class_count`;
- `maximum_updates`;
- `validation_interval_updates`;
- `validation_history`.

The schema type is:

`paper01_revision2026_gmm_class_model`

The initial schema version is:

`1.0`

## 9. Structural metadata

The following structural values must agree with the governed model:

`components=5`

`dimension=4800`

`parameter_dtype=float64`

`observation_model=censored_diagonal_gaussian_mixture`

A mismatch between metadata and loaded state is a hard error.

## 10. Dataset, class, and seed identity

Every persisted class model is uniquely bound to:

`(dataset_id, class_id, experimental_seed)`

Sampling must be requested against an explicit identity tuple.

The implementation must compare the requested tuple with
`model_metadata.json`.

Sampling must fail closed if any element differs.

`GMM_MODEL_IDENTITY=DATASET_CLASS_EXPERIMENTAL_SEED`

There is no implicit class substitution.

There is no implicit seed substitution.

There is no nearest available model selection.

## 11. No global fallback

The revised GMM does not persist or consume a global fallback model.

`GMM_GLOBAL_FALLBACK_MODEL=NO`

If the exact requested class checkpoint is unavailable, malformed, or
invalid, sampling fails.

The implementation must not:

- load another class model;
- load a global model;
- train a temporary substitute;
- relabel samples from another model.

`GMM_MISSING_CLASS_CHECKPOINT_BEHAVIOR=FAIL_CLOSED`

## 12. Selected-checkpoint semantics

The training-budget policy governs checkpoint selection by lowest finite
validation censored NLL.

Only the selected authoritative class state must be persisted as:

`model_state.npz`

The metadata records:

- every governed validation checkpoint step;
- every finite validation NLL used for model selection;
- the selected step;
- the selected validation NLL.

The selected metadata values must exactly correspond to one entry in
`validation_history`.

`GMM_PERSISTED_STATE=SELECTED_BEST_VALIDATION_CHECKPOINT`

The final optimizer state is not automatically persisted as the
authoritative model.

## 13. Validation history

`validation_history` is an ordered JSON list.

Each entry must contain:

- `step`;
- `validation_nll`.

Step 0 must be present because the initialized model is a governed
checkpoint candidate.

Steps must be strictly increasing.

Every validation NLL must be finite.

The selected checkpoint must be the minimum finite validation NLL under the
governed earliest-step tie break.

This record makes checkpoint selection independently auditable without
requiring persistence of every intermediate parameter snapshot.

## 14. Intermediate training snapshots

Intermediate parameter snapshots may exist temporarily for execution
recovery or implementation debugging.

They are not scientific model artifacts unless explicitly promoted by a
future governed decision.

The authoritative post-training class model is the selected validation
checkpoint represented by:

- `model_state.npz`;
- `model_metadata.json`.

`GMM_INTERMEDIATE_SNAPSHOTS_AUTHORITATIVE=NO`

## 15. State integrity binding

`model_metadata.json` must contain the SHA-256 digest of
`model_state.npz`.

The digest field is:

`state_sha256`

The algorithm is:

`sha256`

Sampling must recompute the state-file digest before loading the model.

Digest mismatch is a hard failure.

`GMM_STATE_INTEGRITY_ALGORITHM=SHA256`

`GMM_VERIFY_STATE_SHA256_BEFORE_LOAD=YES`

## 16. Relationship to native TrustForge manifest

The native TrustForge execution manifest remains the authoritative record
for execution-wide provenance, including:

- study identity;
- experiment identity;
- concrete execution identity;
- repository identity;
- Git commit;
- Git dirty/clean state;
- resolved configuration digest;
- runtime environment;
- dataset path;
- artifact root;
- execution seed;
- scheduler information where available.

The GMM class metadata must not independently invent alternative values for
those execution-wide fields.

The native manifest should link the produced class-model artifacts through
its artifact/model/checksum facilities.

`GMM_NATIVE_MANIFEST_REQUIRED_FOR_EXECUTION_PROVENANCE=YES`

## 17. Scientific identity versus execution identity

The class-model identity:

`(dataset_id, class_id, experimental_seed)`

is scientific model identity within the revised GMM benchmark.

The native TrustForge:

`execution_id`

identifies a concrete execution attempt.

These identities must not be conflated.

A retry may have a new execution identity while targeting the same
scientific class-model identity.

Whether a concrete retry becomes accepted evidence is governed separately
by TrustForge provenance and acceptance policy.

`GMM_SCIENTIFIC_IDENTITY_IS_EXECUTION_IDENTITY=NO`

## 18. Atomic publication requirement

The authoritative class model must not appear complete while either of its
required files is incomplete.

Production implementation must use a safe publication procedure such as:

1. write state to a temporary path;
2. fsync/close as appropriate;
3. compute and verify state SHA-256;
4. write metadata to a temporary path;
5. validate metadata against the state;
6. atomically rename the state into its final filename;
7. atomically rename metadata into its final filename only after state
   publication succeeds.

A partially written artifact pair must not be treated as a valid class
model.

`GMM_CHECKPOINT_PUBLICATION=ATOMIC_PAIR_VALIDATION_REQUIRED`

## 19. Overwrite policy

An existing authoritative class-model artifact pair must not be silently
overwritten.

A rerun must use its own execution artifact root unless an explicit,
separately governed replacement action is authorized.

`GMM_AUTHORITATIVE_CHECKPOINT_OVERWRITE=NO`

This preserves execution evidence and prevents accidental mutation of an
already materialized result.

## 20. Sampling preflight

Before generating any synthetic sample, the implementation must validate:

1. metadata file exists;
2. state file exists;
3. metadata schema type/version are supported;
4. dataset identity matches the requested dataset;
5. class identity matches the requested class;
6. experimental seed matches the requested seed;
7. state filename matches metadata;
8. state SHA-256 matches metadata;
9. NPZ contains exactly the governed keys;
10. all shapes and dtypes match;
11. all parameters are finite;
12. transformed scales satisfy the floor;
13. mixture weights are finite and normalized;
14. selected checkpoint metadata is internally consistent.

Only after all checks pass may sampling proceed.

`GMM_SAMPLING_PREFLIGHT_REQUIRED=YES`

## 21. Governed checkpoint constants

`GMM_STATE_FILENAME=model_state.npz`

`GMM_STATE_FORMAT=NUMPY_NPZ_UNCOMPRESSED`

`GMM_STATE_ALLOW_PICKLE=NO`

`GMM_STATE_KEYS=logits,means,raw_scales`

`GMM_STATE_COMPONENTS=5`

`GMM_STATE_DIMENSION=4800`

`GMM_STATE_DTYPE=FLOAT64`

`GMM_METADATA_FILENAME=model_metadata.json`

`GMM_METADATA_FORMAT=CANONICAL_JSON`

`GMM_MODEL_IDENTITY=DATASET_CLASS_EXPERIMENTAL_SEED`

`GMM_GLOBAL_FALLBACK_MODEL=NO`

`GMM_MISSING_CLASS_CHECKPOINT_BEHAVIOR=FAIL_CLOSED`

`GMM_PERSISTED_STATE=SELECTED_BEST_VALIDATION_CHECKPOINT`

`GMM_INTERMEDIATE_SNAPSHOTS_AUTHORITATIVE=NO`

`GMM_STATE_INTEGRITY_ALGORITHM=SHA256`

`GMM_VERIFY_STATE_SHA256_BEFORE_LOAD=YES`

`GMM_NATIVE_MANIFEST_REQUIRED_FOR_EXECUTION_PROVENANCE=YES`

`GMM_SCIENTIFIC_IDENTITY_IS_EXECUTION_IDENTITY=NO`

`GMM_CHECKPOINT_PUBLICATION=ATOMIC_PAIR_VALIDATION_REQUIRED`

`GMM_AUTHORITATIVE_CHECKPOINT_OVERWRITE=NO`

`GMM_SAMPLING_PREFLIGHT_REQUIRED=YES`

`GMM_CHECKPOINT_POLICY_SELECTED=YES`

## 22. Authorization state

The checkpoint and model-identity policy is selected.

Checkpoint persistence and load validation are now implemented and
integrated-validated.

Sampling remains separately unresolved and unaccepted.

Therefore:

`GMM_CHECKPOINT_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_CHECKPOINT_FOCUSED_VALIDATION=PASSED`

`GMM_CHECKPOINT_FOCUSED_TEST_COUNT=12`

`GMM_CHECKPOINT_INTEGRATED_VALIDATION=PASSED`

`GMM_CHECKPOINT_INTEGRATED_GMM_TEST_COUNT=171`

`GMM_IMPLEMENTATION_AUTHORIZED=PARTIAL_TRAINING_STACK_AND_CHECKPOINT_ONLY`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
