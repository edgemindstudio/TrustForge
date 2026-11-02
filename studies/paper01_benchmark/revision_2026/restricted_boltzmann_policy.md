# Paper 1 Revision 2026 — Restricted Boltzmann Machine Policy

`STATUS=SELECTED`

`RBM_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs the Restricted Boltzmann Machine family for the revised
Paper 1 TrustForge benchmark.

Historical RBM artifacts and implementations remain preserved for provenance.

The revision implementation may differ from the historical implementation where
required for scientific consistency, canonical-domain correctness,
reproducibility, or scalability.

This policy selects the revised RBM observation/model family.

It does not yet authorize production implementation acceptance or real benchmark
training.

## 2. Historical Paper 1 model semantics

The historical Paper 1 RBM implementation was a Bernoulli-visible,
Bernoulli-hidden RBM.

Historical training:

- thresholded model-facing values at `0.5`;
- converted visible training observations to binary `{0,1}`;
- used Bernoulli hidden sampling;
- used Bernoulli visible sampling during Gibbs transitions.

Historical synthesis likewise generated binary visible states.

Therefore the historical RBM did not natively model intermediate canonical
intensities.

## 3. Historical configuration-resolution defect

The historical Paper 1 configuration contained an intended nested RBM block
with:

- learning rate `5e-4`;
- contrastive-divergence depth `CD_K=5`;
- hidden units `512`;
- batch size `256`;
- epochs `50`;
- training-per-class cap `2000`.

The same configuration also contained generic top-level compatibility settings,
including:

- `EPOCHS=5000`;
- `BATCH_SIZE=256`;
- `LR=2e-4`.

The historical RBM trainer did not read the nested
`restrictedboltzmann:` block.

Instead it resolved flat keys and generic fallbacks such as:

- `RBM_HIDDEN`, otherwise default `256`;
- `RBM_EPOCHS`, otherwise top-level `EPOCHS`;
- `RBM_BATCH`, otherwise top-level `BATCH_SIZE`;
- `CD_K`, otherwise default `1`;
- `RBM_LR`, otherwise top-level `LR`.

The historical Paper 1 SLURM wrapper did not flatten or translate the nested
RBM block before dispatch.

The application dispatch layer likewise passed the merged configuration to the
RBM trainer without an RBM-specific nested-to-flat translation.

For the recovered USTC seed-42 artifact-producing execution at historical
commit:

`37c60929b3545e5ff68d9107280a63ece0b62f45`

the runtime log reported:

- hidden units `256`;
- epochs `5000`;
- batch size `256`;
- `CD_K=1`.

The same execution trained class 0 using `9000` examples rather than the
intended `2000`-example cap.

Therefore the recovered execution did not instantiate the intended frozen RBM
scientific configuration.

This evidence is specific to the recovered execution and must not be silently
generalized to every historical RBM run without independent evidence.

`RBM_HISTORICAL_CONFIG_RUNTIME_MISMATCH_CONFIRMED=YES`

## 4. Revised canonical-domain incompatibility of binary visibles

The governed Paper 1 canonical representation is:

`40 x 40 x 3`

`float32`

`[0,1]`

`D=4800`

A representative audit of the first 512 revised training examples from each
dataset applied the historical threshold rule:

`x_binary = 1 if x > 0.5 else 0`

Observed scalar fractions changed by this binarization were approximately:

- USTC: `0.15588745`;
- CICMalDroid2020: `0.99999756`;
- MalNet-Image: `0.35085937`;
- MaleX: `0.96775146`.

Observed mean absolute binarization errors were approximately:

- USTC: `0.03892686`;
- CICMalDroid2020: `0.42789236`;
- MalNet-Image: `0.07110288`;
- MaleX: `0.38067129`.

These are representative 512-example audits, not full-dataset censuses.

The results establish that threshold binarization would materially alter the
revised canonical observation representation, especially for CICMalDroid2020
and MaleX.

Therefore:

`RBM_REVISION_THRESHOLD_BINARIZATION=FORBIDDEN`

## 5. Required revision properties

The revised RBM must:

1. consume the governed canonical `[0,1]` representation directly;
2. preserve intermediate continuous intensity values as legitimate observations;
3. preserve Restricted Boltzmann Machine bipartite energy-model identity;
4. retain conditionally independent visible units given hidden state;
5. retain conditionally independent hidden units given visible state;
6. support tractable block Gibbs transitions;
7. produce values in `[0,1]` by model construction;
8. require no arbitrary endpoint epsilon;
9. require no universal dequantization;
10. require no post-generation clipping;
11. use seed-governed stochastic sampling;
12. fail closed on non-finite model state or generated values.

## 6. Rejected revision alternatives

### 6.1 Historical Bernoulli-visible RBM

Rejected.

Reason:

The revised benchmark contains legitimate continuous canonical values.
Deterministic threshold binarization materially changes those observations.

### 6.2 Bernoulli-visible RBM with thresholded training and visible probabilities as output

Rejected.

Reason:

Training would still optimize a binary observation model after destroying
intermediate canonical intensities, while generation would report probabilities
rather than samples from the trained visible-state model.

Training and generation semantics would therefore be mismatched.

### 6.3 Ordinary Gaussian-visible Bernoulli-hidden RBM

Not selected as the primary revision model.

Reason:

An unconstrained Gaussian visible conditional has support outside `[0,1]`.

Using it directly would require either accepting non-canonical generated values
or applying a post-generation repair such as clipping.

Post-hoc canonical clipping is not accepted.

### 6.4 Truncated-Gaussian visible RBM

Technically feasible but not selected as the primary revision model.

A TensorFlow 2.20 stateless parameterized-truncated-normal audit passed 77
tested `(mean, scale)` tail regimes with zero invalid cases and demonstrated
same-seed bitwise reproducibility.

However, the Gaussian-visible formulation introduces quadratic visible-energy
terms and an additional governed visible-scale structure.

No current Paper 1 evidence establishes that this added parameterization is
required.

The simpler bounded linear-energy RBM therefore remains preferable for the
benchmark objective.

## 7. Selected revision family

The selected revision model is a:

**bounded continuous-visible / Bernoulli-hidden Restricted Boltzmann Machine**

using:

**truncated-exponential visible conditional distributions on `[0,1]`**.

The visible state is:

`v in [0,1]^D`

with:

`D=4800`

The hidden state is:

`h in {0,1}^H`

The governed energy form is:

`E(v,h) = -a^T v - b^T h - v^T W h`

subject to the bounded visible domain.

This preserves the ordinary bipartite RBM interaction structure while replacing
the historical binary visible state space with the governed continuous bounded
state space.

`RBM_VISIBLE_DOMAIN=ZERO_ONE_CONTINUOUS`

`RBM_HIDDEN_DOMAIN=BERNOULLI`

`RBM_VISIBLE_CONDITIONAL=TRUNCATED_EXPONENTIAL`

## 8. Hidden conditional

For hidden unit `j`:

`P(h_j=1 | v) = sigmoid(b_j + W[:,j]^T v)`

Hidden units remain conditionally independent given the visible vector.

No thresholding of `v` is permitted before evaluating this conditional.

## 9. Visible conditional

For visible coordinate `i`, define:

`theta_i(h) = a_i + W[i,:] h`

Then:

`p(v_i | h) ∝ exp(theta_i v_i)`

for:

`0 <= v_i <= 1`.

The normalized scalar conditional is:

`p(v_i | h) = exp(theta_i v_i) / Z(theta_i)`

where:

`Z(theta) = (exp(theta) - 1) / theta`

for nonzero `theta`, with the continuous limit:

`Z(0) = 1`.

At:

`theta = 0`

the visible conditional is exactly:

`Uniform(0,1)`.

Visible coordinates remain conditionally independent given the hidden state.

## 10. Endpoint semantics

The bounded-exponential visible conditional is a continuous density on the
closed interval `[0,1]`.

Exact `0` and exact `1` are valid observations and require:

- no displacement;
- no endpoint epsilon;
- no dequantization;
- no threshold conversion.

The model does not assign positive discrete atomic probability to either
endpoint.

Endpoint observations are represented through finite boundary density.

This is an explicit family limitation, not an implementation accident.

The revised benchmark already permits family-specific endpoint semantics when
they are explicit and scientifically governed.

In particular, the accepted bounded Masked AutoFlow represents endpoint
observations through continuous boundary density rather than positive discrete
point mass.

Therefore positive endpoint atoms are not imposed as a universal requirement on
all seven generator families.

The manuscript must not claim that the revised RBM reproduces empirical endpoint
point masses exactly.

`RBM_ENDPOINT_EPSILON=NO`

`RBM_ENDPOINT_ATOMIC_MASS=NO`

`RBM_ENDPOINT_BOUNDARY_DENSITY=YES`

## 11. Model-consistent visible sampling

For one visible scalar with natural parameter `theta`, sampling must draw from
the governed bounded-exponential conditional itself.

The implementation may use a numerically stable inverse-CDF formulation.

A revision prototype tested natural parameters spanning:

`-1e6` through `+1e6`

including the near-zero regime.

Across 21 tested natural-parameter values:

- all generated values were finite;
- all values remained inside `[0,1]`;
- no posthoc clipping was used;
- no invalid case occurred;
- identical stateless seeds produced bitwise-identical audited samples.

A separate distributional validation compared empirical means from `200000`
samples per tested natural parameter with the analytic conditional mean.

Across 15 tested values from `-20` through `+20`:

- all cases passed;
- no absolute empirical/analytic mean error exceeded `0.005`.

`RBM_BOUNDED_EXPONENTIAL_NUMERICAL_AUDIT=PASS`

`RBM_BOUNDED_EXPONENTIAL_DISTRIBUTIONAL_AUDIT=PASS`

## 12. Analytic conditional mean

For nonzero `theta`, the conditional mean may be written stably using the
appropriate sign-specific formulation equivalent to:

`E[V | theta] = 1/(1-exp(-theta)) - 1/theta`

with continuous limit:

`E[V | 0] = 0.5`.

The implementation must use numerically stable arithmetic around zero and in
large-magnitude tails.

The analytic mean may be used for validation or diagnostics.

It must not silently replace stochastic Gibbs visible sampling where the
governed training or generation procedure requires an actual visible draw.

## 13. No output repair

Generated visible values must already satisfy:

`0 <= v <= 1`

as a consequence of the selected conditional distribution.

The implementation must not apply canonical-space clipping as a repair stage.

Numerical range validation after sampling is required.

A violation must fail closed.

`RBM_OUTPUT_POSTHOC_CLIPPING=NO`

## 14. Canonical input and output contract

Model-facing revised observations remain:

- shape `(B,40,40,3)`;
- dtype `float32`;
- finite;
- range `[0,1]`.

Internal model precision remains to be selected separately.

Generated samples must ultimately return the same canonical representation.

No RBM-specific resizing, requantization, thresholding, or endpoint displacement
may replace the benchmark-wide canonical representation.

## 15. Class-conditioning policy boundary

The revised Paper 1 benchmark requires class-aware synthesis.

The exact revised RBM class-conditioning architecture remains to be governed
before implementation authorization.

Permitted design directions include separate class-specific RBMs or an explicit
conditional RBM construction, but no choice is made by this policy alone.

The implementation must not generate an unconditional sample pool and assign
class labels afterward.

`RBM_POSTHOC_SYNTHETIC_LABEL_ASSIGNMENT=FORBIDDEN`

## 16. Training-policy boundary

This policy does not yet select:

- hidden-unit count;
- class-conditioning architecture;
- contrastive-divergence depth or alternative negative-phase estimator;
- optimizer;
- learning rate;
- batch geometry;
- training traversal;
- maximum update budget;
- validation objective;
- checkpoint-selection rule;
- checkpoint persistence format;
- model initialization;
- internal numerical precision;
- production sampling-chain length.

These decisions require separate evidence and governance.

Historical values must not become revision defaults merely because they appear
in historical configuration files.

## 17. Historical preservation

Historical Bernoulli RBM code, configurations, checkpoints, logs, and synthetic
artifacts remain provenance evidence.

They must not be rewritten to make the historical implementation appear to have
used the revised bounded continuous-visible model.

The revised implementation must use revision-specific code and artifact paths.

`HISTORICAL_RBM_ARTIFACT_MUTATION=FORBIDDEN`

## 18. Implementation gate

Before revised RBM implementation may be accepted, focused regression evidence
must demonstrate at least:

1. finite hidden probabilities for canonical inputs;
2. no input threshold-binarization path;
3. bounded-exponential visible sampling is finite;
4. visible samples remain in `[0,1]` without clipping;
5. the `theta=0` conditional behaves as Uniform(0,1);
6. extreme positive and negative natural parameters remain numerically stable;
7. fixed scientific identity and seed reproduce stochastic draws;
8. changed seed can change stochastic draws;
9. finite model energy on valid states;
10. finite training objective;
11. finite gradients or finite governed update statistics;
12. genuine class-aware training and synthesis;
13. canonical `(B,40,40,3)` compatibility;
14. no posthoc label assignment;
15. no endpoint epsilon;
16. no post-generation clipping;
17. fail-closed non-finite-state handling.

## 19. Current authorization state

The revised RBM observation/model family is selected.

Production implementation is not yet accepted.

Real benchmark training is not authorized.

Therefore:

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
