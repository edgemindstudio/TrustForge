# Variational Autoencoder Implementation Contract

## Paper 1 Revision 2026

## 1. Status and authorization

Status: PROPOSED

Scope: Revised Paper 1 benchmark only.

`VAE_ARCHITECTURE_COMPATIBILITY=PASS`

`VAE_IMPLEMENTATION_ACCEPTANCE=PENDING`

`VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

Architecture compatibility does not constitute implementation
acceptance or training authorization.

## 2. Historical preservation

The historical GenCyberSynth conditional VAE provides a
reference architecture, not an automatically accepted
revision-2026 implementation.

Historical source files, checkpoints, and experiment evidence
must remain unchanged.

The historical training objective used dimension-averaged MSE
plus a beta-weighted KL term.

The revised observation model and loss scaling must be
explicitly governed rather than silently inherited.

## 3. Canonical observation contract

All model-facing datasets must use the governed revision-2026
canonical representation:

- Shape: `(B, 40, 40, 3)`
- Layout: NHWC
- Dtype: float32
- Range: `[0,1]`
- Labels: governed one-hot class labels

The internal VAE representation is:

`x_internal = 2 * x_canonical - 1`

The output conversion is:

`x_canonical = (x_internal + 1) / 2`

The transformation must not alter partition membership,
class identity, or sample provenance.

## 4. Conditional probabilistic model

The revised family is a class-conditional convolutional VAE.

Prior:

`p(z) = Normal(0, I)`

Approximate posterior:

`q_phi(z | x, y) = Normal(mu_phi(x,y), diag(var_phi(x,y)))`

Decoder observation model, proposed:

`p_theta(x | z,y) = Normal(mu_theta(z,y), sigma_x^2 I)`

The decoder mean is bounded by tanh in the internal
`[-1,1]` representation.

Class conditioning must influence the actual decoder
distribution, not merely post-generation labels.

## 5. Proposed training objective

The candidate per-example objective is:

`L = Gaussian_NLL_sum + beta_KL * KL_sum`

Both terms are aggregated as sample-weighted means over
examples.

The Gaussian variance, beta_KL, latent dimensionality,
and any KL schedule must be fixed and recorded before
implementation acceptance and real-data training.

The objective must not be described as an exact ELBO
unless its likelihood, scaling, and KL weighting
justify that designation.

`VAE_OBSERVATION_MODEL=FIXED_VARIANCE_GAUSSIAN`

`VAE_OBSERVATION_VARIANCE=1.0`

`VAE_KL_WEIGHT=1.0`

`VAE_KL_ANNEALING=NONE`

`VAE_RECONSTRUCTION_REDUCTION=SUM_OVER_DIMENSIONS`

`VAE_OBJECTIVE_AGGREGATION=MEAN_OVER_EXAMPLES`

`VAE_PRIMARY_GENERATION=DECODER_MEAN`

`VAE_NUMERICAL_DESIGN_STATUS=PROPOSED_PRIMARY_BASELINE`

### 5.1 Mathematical objective

Let D = 4800 and let L denote the configured latent dimension.

The internal observation x lies in [-1,1]^D.

The decoder produces mu_theta(z,y) using tanh.

The observation likelihood is:

p_theta(x|z,y) = Normal(mu_theta(z,y), I).

The per-example Gaussian negative log likelihood is:

NLL(x,z,y) =
    0.5 * sum_j((x_j - mu_theta_j(z,y))^2)
    + 0.5 * D * log(2*pi).

The posterior is diagonal Gaussian with parameters
mu_phi(x,y) and logvar_phi(x,y).

The exact per-example KL against the standard Gaussian is:

KL =
    -0.5 * sum_k(
        1 + logvar_k - mu_k^2 - exp(logvar_k)
    ).

The training objective is the example mean of:

E_q[NLL(x,z,y)] + KL.

The reconstruction expectation may be estimated using a
reparameterized latent draw. Such an estimate must not
be described as an exact evaluation of the expectation.

The fixed Gaussian normalization constant may be omitted
from optimization only if the reported objective explicitly
distinguishes this constant-free loss from the full NLL.

With KL weight one and the complete likelihood terms,
the expected objective is the negative ELBO.

A tanh-bounded decoder mean does not make the Gaussian
observation distribution itself bounded.

The Gaussian likelihood and decoder-mean production
generation are distinct and must be reported separately.

### 5.2 Numerical-design boundary

The numerical values above are a proposed primary baseline.

They are not evidence of implementation correctness,
posterior quality, or benchmark performance.

No real-data training or production synthesis is authorized.

The baseline must be reviewed and locked before the
implementation acceptance decision.


## 6. Governed dataset integration

The revised runner must use the existing dataset adapters:

- USTCRevision2026
- CICMalDroidRevision2026
- MalNetImageRevision2026
- MaleXRevision2026

Training batches must come from:

`canonical_batch("train", logical_positions)`

Validation batches must come from:

`canonical_batch("validation", logical_positions)`

The held-out test partition must not be accessed for
training, early stopping, checkpoint selection, or tuning.

The runner must not silently repartition datasets or
disable required source and partition integrity checks.

Canonical batches must be processed incrementally rather
than constructing a full canonical dataset in memory.

## 7. Validation and selection

Validation must use inference-mode network behavior.

Stochastic validation must have governed random-state
semantics.

Validation metrics must be weighted by the number of
examples in each batch.

Best checkpoint selection must use a recorded validation
criterion and deterministic tie-breaking rule.

Best and last checkpoints must be distinctly identified.

Non-finite validation values must fail explicitly.

## 8. Checkpoint integrity

Encoder and decoder weights must form one logically
complete checkpoint bundle.

A published checkpoint must bind:

- Dataset and partition identity
- Class mapping and model architecture
- Experiment seed
- Effective runtime configuration
- Selected epoch and selection criterion
- Encoder and decoder state integrity
- Model and observation-model version

Incomplete, inconsistent, missing, or corrupt checkpoint
bundles must not be accepted.

Checkpoint restoration must fail closed.

A success marker cannot substitute for verified weights.

## 9. Sampling

Production synthesis must use the governed class condition
and the explicitly selected checkpoint.

Candidate primary generation semantics:

1. Draw latent vectors from the standard Gaussian prior.
2. Decode with the requested class condition.
3. Use the decoder mean as the generated internal image.
4. Convert deterministically to canonical `[0,1]`.

This is decoder-mean generation, not a full stochastic
Gaussian observation draw.

The distinction must be explicit in benchmark reporting.

The final generation rule requires approval before
production synthesis.

All output samples must have correct shape, dtype,
class identity, finite values, and canonical range.

## 10. Runtime-control integrity

Every reported experimental control must affect runtime.

Declared configuration, consumed configuration, derived
values, and recorded provenance must be distinguishable.

Unrecognized or incompatible scientific controls must
not be silently ignored.

Numerical failure must not trigger unapproved changes
to architecture, likelihood, sampling, or conditioning.

## 11. Regression requirements

The dedicated VAE implementation regression suite must
demonstrate:

1. Canonical/internal transformation correctness.
2. Encoder and decoder tensor compatibility.
3. Correct conditional-label participation.
4. Finite posterior mean and log-variance.
5. Correct diagonal-Gaussian KL mathematics.
6. Correct Gaussian reconstruction mathematics.
7. Governed reconstruction/KL scaling.
8. Finite loss and gradients.
9. Real optimizer-step parameter updates.
10. Validation inference-mode behavior.
11. Sample-weighted validation aggregation.
12. Governed train/validation-only access.
13. Absence of test-partition training access.
14. Reproducible traversal and sampling.
15. Correct selected-best checkpoint identity.
16. Complete paired-checkpoint restoration.
17. Missing/corrupt checkpoint rejection.
18. Correct per-class sample counts.
19. Canonical synthetic output integrity.
20. No unapproved scientific fallback.
21. All four dataset-adapter interfaces.
22. Bounded-memory batch traversal.

Tests must operate on small controlled fixtures before
any real-data training authorization.

## 12. Acceptance gate

Implementation acceptance requires:

- Locked observation variance and KL controls
- Approved sampling semantics
- Complete revised implementation
- Passing dedicated regression tests
- Passing relevant broader regression tests
- Verified checkpoint persistence/restoration
- Reviewed runtime-control integrity
- Recorded final authorization decision

Until those conditions are satisfied:

`VAE_IMPLEMENTATION_ACCEPTANCE=PENDING`

`VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

Acceptance must not be inferred from historical evidence
or another generator family's test results.

## 13. Historical and execution boundaries

This proposal does not authorize training or synthesis.

Historical GenCyberSynth source code and artifacts remain
provenance-only.

Any later execution must separately satisfy the governed
dataset, seed, computational, artifact, and reporting
requirements of Paper 1 revision 2026.

## 14. Proposed conditional architecture baseline

Status: PROPOSED — implementation validation pending.

`VAE_LATENT_DIMENSION=100`

`VAE_ENCODER_CONV_CHANNELS=32,64,128`

`VAE_ENCODER_DENSE_UNITS=256`

`VAE_DECODER_FEATURE_CHANNELS=64`

`VAE_DECODER_TRANSPOSED_CONV_CHANNELS=64,32`

`VAE_DECODER_OUTPUT_ACTIVATION=TANH`

`VAE_CANONICAL_IMAGE_SHAPE=40,40,3`

`VAE_POSTERIOR=DIAGONAL_GAUSSIAN`

`VAE_REPARAMETERIZATION=SEED_GOVERNED`

The encoder receives an internal [-1,1] image and its
governed one-hot class label.

The class condition is broadcast spatially and
concatenated with the image before convolution.

Encoder backbone:

1. Conv2D(32, kernel=3, stride=2, same, relu).
2. Conv2D(64, kernel=3, stride=2, same, relu).
3. BatchNormalization.
4. Conv2D(128, kernel=3, stride=1, same, relu).
5. Flatten.
6. Dense(256, relu).
7. Separate posterior mean and log-variance heads.

The decoder receives a latent vector and the
requested one-hot class label.

Decoder backbone:

1. Concatenate latent vector and class condition.
2. Dense(10 * 10 * 64, relu).
3. Reshape(10, 10, 64).
4. Conv2DTranspose(64, kernel=3, stride=2, same, relu).
5. Conv2DTranspose(32, kernel=3, stride=2, same, relu).
6. Conv2D(3, kernel=3, same, tanh).

At canonical 40x40 resolution, the architecture
does not require a resizing operation.

Posterior sampling uses:

z = mean + exp(0.5 * log_variance) * epsilon

where epsilon is generated by an explicitly governed
random-number stream.

Posterior log-variance must not be silently clipped.

Invalid or non-finite posterior parameters and
sampling results must fail explicitly.

The architecture module must not load datasets,
create experiment directories, train models,
publish checkpoints, or synthesize production data.

Historical experiment YAML files specify a
2000-step KL warmup but do not explicitly record
the VAE latent dimensionality.

The proposed revised latent dimension of 100 is
therefore not represented as a historically verified
experimental control.

The proposed revised no-annealing objective also
differs from the historical declared KL-warmup
configuration and must be disclosed.

No revised VAE training or production synthesis
is authorized by this architecture amendment.
