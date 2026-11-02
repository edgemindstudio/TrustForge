# Paper 1 Revision 2026 — Restricted Boltzmann Implementation Contract

`STATUS=SELECTED`

`RBM_IMPLEMENTATION_CONTRACT_SELECTED=YES`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Purpose

This contract defines the revision-specific implementation boundary for the
Paper 1 Restricted Boltzmann Machine.

Historical code beneath:

`restrictedboltzmann/`

remains historical provenance and must not be modified to implement the
revision-2026 RBM.

Therefore:

`RBM_HISTORICAL_IMPLEMENTATION_MUTATION=FORBIDDEN`

## 2. Revision package location

All revised RBM implementation modules must live beneath:

`src/trustforge/models/paper01_revision_2026/`

All focused revised RBM tests must live beneath:

`tests/`

with revision-specific filenames.

Therefore:

`RBM_REVISION_IMPLEMENTATION_ROOT=src/trustforge/models/paper01_revision_2026`

## 3. Required module decomposition

The selected implementation is divided into these modules:

### Observation model

`restricted_boltzmann_observation.py`

Owns:

- bounded-exponential normalization;
- analytic visible mean;
- numerically stable inverse-CDF sampling;
- range validation;
- float64 observation arithmetic.

### Core model

`restricted_boltzmann_model.py`

Owns:

- trainable `W`;
- trainable `a`;
- trainable `b`;
- trainable `U`;
- hidden conditional probabilities;
- class-conditioned hidden sampling;
- visible natural parameters;
- exact model-state validation.

### Initialization

`restricted_boltzmann_initialization.py`

Owns:

- governed `W ~ N(0,0.01^2)`;
- full-training global-mean visible-bias moment matching;
- `b=0`;
- `U=0`;
- governed initialization diagnostics.

### Traversal

`restricted_boltzmann_traversal.py`

Owns:

- `p(c) proportional to n_c^0.85`;
- governed class draw;
- governed within-class draw;
- batch size `256`;
- nominal dataset-level exposure geometry;
- reproducible training traversal identity.

### Contrastive divergence

`restricted_boltzmann_cd.py`

Owns:

- CD-1 positive phase;
- fixed-class negative Gibbs transition;
- terminal hidden expectation;
- sufficient-statistic differences;
- no persistent chain.

### Optimizer

`restricted_boltzmann_optimizer.py`

Owns:

- exact manual SGD;
- learning rate `5e-4`;
- zero momentum;
- zero weight decay;
- no gradient clipping;
- one governed parameter update.

### Training budget

`restricted_boltzmann_training_budget.py`

Owns:

- `B = ceil(N_train / 256)`;
- `max_updates = 2 * B`;
- validation steps `0`, `B`, `2B`;
- no minimum-update floor;
- no early stopping.

### Validation

`restricted_boltzmann_validation.py`

Owns:

- deterministic mean-field reconstruction diagnostic;
- finite-state validation;
- visible-mean range validation;
- diagnostic-only reconstruction MSE;
- no checkpoint selection.

### In-memory training orchestration

`restricted_boltzmann_training.py`

Owns composition of:

- initialization;
- traversal;
- CD-1;
- SGD;
- budget;
- validation.

It must return the exact final governed model state.

It must not persist checkpoints directly.

Therefore:

`RBM_TRAINING_PERSISTS_CHECKPOINTS=NO`

### Checkpoint persistence

`restricted_boltzmann_checkpoint.py`

Owns:

- `model_state.npz`;
- `model_metadata.json`;
- exact `W,a,b,U` persistence;
- float64 dtype enforcement;
- dataset/seed identity;
- ordered class mapping;
- SHA-256 state binding;
- atomic state-first metadata-last publication;
- fail-closed loading;
- final-update authority.

### Sampling

`restricted_boltzmann_sampling.py`

Owns:

- checkpoint preflight;
- governed sampling identity;
- model-based chain initialization;
- fixed-class Gibbs transitions;
- bounded-exponential visible draws;
- canonical float32 `(N,40,40,3)` output;
- primary budget `2000` samples/class/seed.

Production chain length remains unresolved until the governed chain audit is
completed.

Therefore:

`RBM_SAMPLING_CHAIN_LENGTH_IMPLEMENTATION_BLOCKED=YES`

## 4. Module-boundary principle

No module may silently duplicate another module's policy-defining logic.

In particular:

- sampling must call the governed observation sampler rather than reimplement it;
- training must call the governed CD primitive rather than embed an alternate
  Gibbs implementation;
- checkpoint loading must reconstruct the governed core model;
- traversal probabilities must come from the traversal module;
- budget arithmetic must come from the budget module.

Therefore:

`RBM_POLICY_LOGIC_DUPLICATION=FORBIDDEN`

## 5. Canonical input boundary

The revised RBM consumes canonical benchmark images:

- shape `(B,40,40,3)`;
- dtype `float32`;
- values in `[0,1]`.

The model-facing implementation converts them batchwise to:

- flattened shape `(B,4800)`;
- dtype `float64`.

Therefore:

`RBM_CANONICAL_INPUT_SHAPE=40x40x3`

`RBM_CANONICAL_INPUT_DTYPE=FLOAT32`

`RBM_INTERNAL_VECTOR_DTYPE=FLOAT64`

`RBM_INTERNAL_VISIBLE_DIMENSION=4800`

## 6. Core state

The shared class-conditional model contains exactly:

- `W` shape `(4800,128)`;
- `a` shape `(4800,)`;
- `b` shape `(128,)`;
- `U` shape `(K,128)`.

No class-specific `W`, `a`, or `b` is permitted.

Therefore:

`RBM_CORE_STATE=W,a,b,U`

`RBM_SHARED_MODEL_PER_DATASET_SEED=YES`

## 7. Historical code isolation

Revision modules must not import:

- historical `restrictedboltzmann.models.BernoulliRBM`;
- historical `restrictedboltzmann.sample.sample_gibbs`;
- historical class-checkpoint discovery;
- historical threshold-binarization helpers.

Therefore:

`RBM_REVISION_IMPORTS_HISTORICAL_BERNOULLI_MODEL=FORBIDDEN`

`RBM_REVISION_IMPORTS_HISTORICAL_SAMPLER=FORBIDDEN`

`RBM_REVISION_THRESHOLD_BINARIZATION_PATH=FORBIDDEN`

## 8. Test decomposition

Focused tests must be introduced separately for at least:

- `test_restricted_boltzmann_observation_revision_2026.py`;
- `test_restricted_boltzmann_model_revision_2026.py`;
- `test_restricted_boltzmann_initialization_revision_2026.py`;
- `test_restricted_boltzmann_traversal_revision_2026.py`;
- `test_restricted_boltzmann_cd_revision_2026.py`;
- `test_restricted_boltzmann_optimizer_revision_2026.py`;
- `test_restricted_boltzmann_training_budget_revision_2026.py`;
- `test_restricted_boltzmann_validation_revision_2026.py`;
- `test_restricted_boltzmann_training_revision_2026.py`;
- `test_restricted_boltzmann_checkpoint_revision_2026.py`;
- `test_restricted_boltzmann_sampling_revision_2026.py`.

Sampling tests may begin with initialization and transition primitives, but
production chain-length acceptance must remain blocked until the compute-node
chain audit selects the depth.

## 9. Implementation order

The governed implementation order is:

1. observation;
2. core model;
3. initialization;
4. traversal;
5. CD-1;
6. optimizer;
7. training budget;
8. validation;
9. in-memory training;
10. checkpoint persistence;
11. sampling primitives;
12. compute-node sampling-chain audit;
13. production chain-length policy;
14. final sampling integration;
15. focused RBM stack validation;
16. full repository test suite;
17. training authorization review.

Therefore:

`RBM_IMPLEMENTATION_ORDER_GOVERNED=YES`

## 10. Authorization discipline

Passing one module's tests does not authorize training.

Training authorization requires:

- all selected policy modules implemented;
- checkpoint persistence validated;
- production chain length selected from governed evidence;
- sampling fully validated;
- focused RBM stack passing;
- full repository suite passing;
- no regression against unrelated generator families.

Therefore:

`RBM_PARTIAL_IMPLEMENTATION_IMPLIES_TRAINING_AUTHORIZATION=NO`

## 11. Current state

The implementation architecture is selected.

No implementation module is yet accepted by this contract.

Therefore:

`RBM_IMPLEMENTATION_STATUS=PENDING`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
