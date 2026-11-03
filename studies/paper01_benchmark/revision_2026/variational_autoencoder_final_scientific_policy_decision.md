# TrustForge Paper 1 Revision-2026
# VAE Final Scientific Policy Decision

Date: 2026-11-02

## Decision status

VAE_SCIENTIFIC_POLICY_STATUS=APPROVED
VAE_TRAINING_BUDGET_POLICY=FIXED_TWO_EPOCHS
VAE_CHECKPOINT_SELECTION_POLICY=MIN_VALIDATION_TOTAL_LOSS
VAE_IMPLEMENTATION_ACCEPTANCE=PENDING
VAE_REAL_DATASET_TRAINING_AUTHORIZED=NO
VAE_CHECKPOINT_PUBLICATION_AUTHORIZED=NO
VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO

This record approves the specified VAE scientific methodology.
It does not authorize execution, publication or synthesis.

## 1. Model and objective

Use the existing Revision-2026 class-conditional
convolutional VAE, with:

- Canonical image shape: 40 x 40 x 3.
- Canonical input domain: [0,1].
- Internal model domain: [-1,1].
- Latent dimension: 100.
- Conditional Gaussian approximate posterior.
- Conditional tanh decoder mean.
- Fixed Gaussian observation variance: 1.0.
- KL coefficient: 1.0.
- No KL annealing.
- No posterior log-variance clipping.
- No gradient clipping.

Train using the existing Gaussian reconstruction
negative log-likelihood plus KL objective.

Do not interpret decoder-mean generation as drawing
from the full Gaussian observation distribution.

## 2. Optimization budget

VAE_OPTIMIZER=ADAM
VAE_LEARNING_RATE=0.001
VAE_BATCH_SIZE=64
VAE_COMPLETED_EPOCHS=2
VAE_EARLY_STOPPING=DISABLED

For dataset-specific training size N:

UPDATES_PER_EPOCH=ceil(N/64)
TOTAL_UPDATES=2*ceil(N/64)

Complete deterministic traversals of the governed
training split are required.

This budget is a fixed comparative protocol.

It is not evidence of convergence or an assertion
that two epochs are optimal.

No automatic epoch extension or hidden tuning
is permitted.

## 3. Validation

Validate after each completed training epoch.

Use the full approved validation partition.

Validation must:

- Use inference-mode network behavior.
- Consume no optimizer updates.
- Use explicitly seeded posterior sampling.
- Aggregate losses by sample count.
- Reject incomplete traversal and nonfinite metrics.

The existing runner's validation seed convention is:

[experimental_seed, -(validation_batch_index + 1)]

where validation batch indices start at zero.

The same batch-identity seed schedule is reused
across completed epochs. This is a predeclared
validation random-stream policy, not adaptive tuning.

VAE_VALIDATION_SEED_POLICY=FIXED_PER_BATCH_ACROSS_EPOCHS

No held-out test information may influence
training or checkpoint selection.

## 4. Checkpoint selection

VAE_SELECTION_METRIC=VALIDATION_TOTAL_LOSS
VAE_SELECTION_DIRECTION=MINIMIZE
VAE_SELECTION_TIE_BREAK=EARLIEST_EPOCH

The best checkpoint is the complete encoder
and decoder state from the earliest completed epoch
attaining minimum validation total loss.

The last checkpoint is the complete state from
the second completed epoch.

Best and last states must be distinguishable
by checkpoint role and complete metadata.

Production synthesis, once separately authorized,
must use the explicitly selected best checkpoint.

The existence of best/last weight snapshots
does not itself authorize their publication.

## 5. Dataset and representation policy

The benchmark uses the four governed Revision-2026
malware-image adapters and their existing canonical
representation.

The canonical representation is a shared benchmark
interface, not a claim that reducing higher-resolution
images to 40 x 40 retains all source information.

Source representation and provenance must be
reported accurately. MalNet source RGB handling
must not be misrepresented as grayscale byteplot
source data.

Training and validation must use the governed
splits without independent repartitioning.

The held-out test split remains inaccessible
to training and checkpoint selection.

## 6. Evidence and acceptance boundary

Existing synthetic VAE tests demonstrate training,
validation, epoch selection, weight capture and
a disposable checkpoint-commit lifecycle.

They do not prove real-dataset origin binding,
actual production execution, model convergence,
or deployment-specific filesystem crash durability.

The existing checkpoint integrity and fail-closed
restoration requirements remain applicable.

Before methodological training authorization:

1. Verify governed real-dataset integration.
2. Verify the approved controls are consumed at runtime.
3. Bind genuine dataset provenance and identity.
4. Verify best/last checkpoint state and manifest roles.
5. Complete focused implementation regression.
6. Issue an explicit implementation and training
   authorization decision.

Do not introduce mandatory favorable-loss,
convergence, hyperparameter-search or sustained
pilot gates.

## 7. Operational boundary

The synthetic-only runner retains its restriction.

No real-data training job is enabled by this record.

No checkpoint publishing, Slurm submission,
or production synthesis is authorized.

## Final disposition

VAE scientific methodology: APPROVED
VAE implementation acceptance: PENDING
VAE training authorization: NOT YET GRANTED
VAE production synthesis: NOT AUTHORIZED
