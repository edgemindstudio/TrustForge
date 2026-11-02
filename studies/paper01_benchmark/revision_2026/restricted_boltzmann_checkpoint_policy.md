# Paper 1 Revision 2026 — Restricted Boltzmann Checkpoint and Model-Identity Policy

`STATUS=SELECTED`

`RBM_CHECKPOINT_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs persistence and model identity for each revised Paper 1
shared class-conditional Restricted Boltzmann Machine.

It defines:

- authoritative numerical state;
- model metadata;
- scientific identity;
- class-mapping identity;
- state integrity binding;
- checkpoint publication;
- checkpoint loading;
- fail-closed behavior.

It does not yet authorize production training.

## 2. Shared model identity

The revised RBM uses one shared class-conditional model per:

`(dataset_id, experimental_seed)`

Therefore:

`RBM_CHECKPOINT_SCIENTIFIC_IDENTITY=DATASET_ID_PLUS_EXPERIMENTAL_SEED`

There is no independent class-specific model checkpoint.

`RBM_CLASS_SPECIFIC_CHECKPOINTS=FORBIDDEN`

A synthesis request may include:

`(dataset_id, class_id, experimental_seed)`

but `class_id` selects a row of the checkpoint's governed class-interaction
matrix rather than selecting a different checkpoint.

## 3. Three-layer artifact contract

The revised RBM uses three persistence layers.

### Layer 1 — numerical model state

The authoritative numerical state is:

`model_state.npz`

### Layer 2 — model metadata

The authoritative model metadata is:

`model_metadata.json`

### Layer 3 — TrustForge execution manifest

Execution-wide provenance remains in the native TrustForge execution manifest.

Therefore:

`RBM_CHECKPOINT_LAYERING=STATE_PLUS_MODEL_METADATA_PLUS_NATIVE_MANIFEST`

## 4. Numerical state format

The selected numerical state format is uncompressed NumPy NPZ.

`RBM_STATE_FILENAME=model_state.npz`

`RBM_STATE_FORMAT=NUMPY_NPZ_UNCOMPRESSED`

The file must be written with:

`numpy.savez`

and loaded with:

`allow_pickle=False`

Therefore:

`RBM_STATE_ALLOW_PICKLE=NO`

## 5. Exact state keys

The authoritative state contains exactly:

- `W`
- `a`
- `b`
- `U`

Therefore:

`RBM_STATE_KEYS=W,a,b,U`

No optimizer instance, callback object, arbitrary Python object, or pickle
payload may be embedded in the numerical state.

## 6. Required state shapes

The architecture policy fixes:

- visible dimension `D=4800`;
- hidden width `H=128`.

For a dataset containing `K` classes:

### W

Shape:

`(4800,128)`

### a

Shape:

`(4800,)`

### b

Shape:

`(128,)`

### U

Shape:

`(K,128)`

Therefore:

`RBM_STATE_VISIBLE_DIMENSION=4800`

`RBM_STATE_HIDDEN_UNITS=128`

`RBM_STATE_CLASS_INTERACTION_ROWS=DATASET_CLASS_COUNT`

## 7. State dtype

The precision policy requires authoritative internal and checkpoint state to be
float64.

Therefore:

`RBM_STATE_DTYPE=FLOAT64`

Every state array must have exact dtype:

`float64`

Loading must fail closed on any dtype mismatch.

## 8. State semantic validation

After loading, the implementation must validate:

- exact key set;
- exact governed shapes;
- exact float64 dtype;
- every element of `W` is finite;
- every element of `a` is finite;
- every element of `b` is finite;
- every element of `U` is finite.

No invalid parameter may be repaired by:

- clipping;
- replacement;
- deletion;
- dtype coercion;
- fallback initialization.

Therefore:

`RBM_INVALID_CHECKPOINT_REPAIR=FORBIDDEN`

## 9. Required metadata

`model_metadata.json` must record at least:

- schema version;
- model family;
- Paper 1 revision identity;
- `dataset_id`;
- `experimental_seed`;
- class count;
- ordered class mapping;
- visible dimension;
- hidden width;
- internal dtype;
- state filename;
- state SHA-256;
- batch size;
- class-balance exponent;
- CD depth;
- optimizer identity;
- learning rate;
- maximum update count;
- selected authoritative step;
- validation schedule;
- ordered validation diagnostic history;
- checkpoint-selection rule.

## 10. Ordered class mapping

The class-to-hidden interaction matrix `U` is meaningful only relative to the
exact ordered class mapping.

Therefore the metadata must bind the exact ordered mapping between:

`class_id`

and:

`class_label`

The number of mapping entries must equal:

`K`

and must equal the first dimension of:

`U`

Therefore:

`RBM_CLASS_MAPPING_IDENTITY_BOUND=YES`

Loading must fail closed if:

- mapping length differs from `U.shape[0]`;
- class IDs are duplicated;
- class labels are duplicated where the dataset contract requires unique
  labels;
- ordering differs from the requested governed dataset mapping.

## 11. Dataset and seed identity

Checkpoint loading must validate exact equality of:

- requested dataset identity;
- persisted dataset identity;
- requested experimental seed;
- persisted experimental seed.

There is no implicit dataset substitution.

There is no implicit seed substitution.

Therefore:

`RBM_CHECKPOINT_IDENTITY_MISMATCH_ACTION=FAIL_CLOSED`

## 12. Scientific identity versus execution identity

The scientific checkpoint identity is:

`(dataset_id, experimental_seed)`

Execution identity may additionally include:

- git commit;
- configuration digest;
- Slurm job ID;
- execution timestamp;
- artifact root;
- host information.

These execution facts remain provenance but do not redefine the scientific
model identity.

Therefore:

`RBM_SCIENTIFIC_IDENTITY_IS_EXECUTION_IDENTITY=NO`

## 13. Authoritative step

The training-budget policy selects:

`RBM_CHECKPOINT_SELECTION=FINAL_GOVERNED_UPDATE`

Therefore persisted authoritative metadata must record:

`selected_step = max_updates`

The authoritative state must be the exact state after the final governed
optimizer update.

There is no validation-selected alternative model.

Therefore:

`RBM_PERSISTED_STATE=FINAL_GOVERNED_UPDATE`

## 14. Validation history

The metadata must preserve the complete ordered governed validation diagnostic
history.

The training-budget policy requires observations at:

- `step=0`;
- `step=B`;
- `step=2B`.

Each record must contain at least:

- step;
- reconstruction diagnostic value;
- finite-state status.

The history is diagnostic provenance only.

It must not imply best-checkpoint selection.

Therefore:

`RBM_VALIDATION_HISTORY_ROLE=DIAGNOSTIC_PROVENANCE_ONLY`

## 15. No historical best-checkpoint semantics

Historical files named:

`RBM_best.weights.h5`

must remain historical provenance.

They must not be relabeled or silently reused as revision checkpoints.

Therefore:

`RBM_HISTORICAL_BEST_CHECKPOINT_REUSE=FORBIDDEN`

Revision checkpoint metadata must explicitly identify final-update authority.

## 16. Integrity binding

The exact bytes of:

`model_state.npz`

must be bound from metadata using SHA-256.

Therefore:

`RBM_STATE_INTEGRITY_HASH=SHA256`

The persisted metadata field:

`state_sha256`

must equal the SHA-256 digest of the exact authoritative state-file bytes.

Loading must fail closed on digest mismatch.

`RBM_STATE_HASH_MISMATCH_ACTION=FAIL_CLOSED`

## 17. Metadata serialization

Model metadata must use:

- UTF-8 JSON;
- deterministic canonical key ordering;
- compact deterministic serialization;
- exactly one trailing newline.

The metadata must not contain NaN or Infinity tokens.

Therefore:

`RBM_METADATA_FORMAT=CANONICAL_UTF8_JSON`

## 18. Publication order

Checkpoint publication must be fail-safe.

The required publication sequence is:

1. write state to a temporary file;
2. fsync the temporary state file;
3. validate the temporary state;
4. compute exact state SHA-256;
5. construct metadata binding that digest;
6. write metadata to a temporary file;
7. fsync the temporary metadata file;
8. atomically rename state into final filename;
9. atomically rename metadata into final filename only after state publication;
10. fsync the containing directory;
11. perform post-publication round-trip validation.

Therefore:

`RBM_CHECKPOINT_PUBLICATION=STATE_FIRST_METADATA_LAST_ATOMIC`

## 19. Incomplete pair behavior

A checkpoint is authoritative only when both required files exist and validate:

- `model_state.npz`;
- `model_metadata.json`.

A state file without valid metadata is not authoritative.

Metadata without the matching state file is not authoritative.

Therefore:

`RBM_INCOMPLETE_CHECKPOINT_PAIR_ACTION=FAIL_CLOSED`

## 20. Overwrite semantics

An already-published authoritative checkpoint must not be silently replaced.

Therefore:

`RBM_AUTHORITATIVE_CHECKPOINT_OVERWRITE=NO`

A repeated scientific identity must either:

- resolve to the already-valid authoritative checkpoint; or
- use an explicitly distinct governed execution/publication path.

## 21. Revision-specific artifact path

Revision checkpoints must be stored only beneath revision-specific Paper 1
artifact paths.

Historical RBM checkpoint directories must not be modified.

Therefore:

`RBM_HISTORICAL_CHECKPOINT_MUTATION=FORBIDDEN`

## 22. Checkpoint loading preflight

Before a checkpoint may be used for synthesis, loading must verify at least:

1. both checkpoint files exist;
2. metadata parses successfully;
3. metadata schema is supported;
4. model family is Restricted Boltzmann;
5. Paper 1 revision identity matches;
6. dataset identity matches;
7. experimental seed matches;
8. class mapping matches;
9. class count matches `U`;
10. visible dimension is `4800`;
11. hidden width is `128`;
12. state key set is exact;
13. state shapes are exact;
14. state dtypes are float64;
15. every state value is finite;
16. state SHA-256 matches metadata;
17. selected step equals governed `max_updates`;
18. checkpoint-selection rule is final governed update;
19. validation history has the governed schedule;
20. no historical class-specific checkpoint fallback occurs.

Therefore:

`RBM_SAMPLING_CHECKPOINT_PREFLIGHT=REQUIRED`

## 23. Missing checkpoint behavior

If the exact requested dataset/seed checkpoint is missing, malformed, or
identity-inconsistent, synthesis must fail closed.

There is no fallback to:

- another seed;
- another dataset;
- a class-specific historical checkpoint;
- the latest arbitrary epoch file;
- random reinitialization.

Therefore:

`RBM_MISSING_CHECKPOINT_BEHAVIOR=FAIL_CLOSED`

## 24. Implementation gate

Before checkpoint persistence may be accepted, focused tests must demonstrate
at least:

1. exact state keys `W,a,b,U`;
2. exact float64 dtypes;
3. exact governed shapes;
4. class mapping is identity-bound;
5. dataset identity is enforced;
6. experimental seed identity is enforced;
7. state SHA-256 is verified;
8. `allow_pickle=False` is used;
9. malformed state fails closed;
10. malformed metadata fails closed;
11. state/metadata hash mismatch fails closed;
12. incomplete checkpoint pairs fail closed;
13. authoritative overwrite is refused;
14. final governed update is the only authoritative state;
15. reconstruction MSE cannot select a checkpoint;
16. validation history is diagnostic only;
17. historical `RBM_best` files cannot be consumed as revision checkpoints;
18. publication is state-first and metadata-last;
19. post-publication round-trip validation succeeds;
20. synthesis checkpoint preflight rejects identity mismatch.

## 25. Current authorization state

The checkpoint and model-identity policy is selected.

Persistence implementation and focused validation remain pending.

Therefore:

`RBM_CHECKPOINT_IMPLEMENTATION_STATUS=PENDING`

`RBM_CHECKPOINT_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
