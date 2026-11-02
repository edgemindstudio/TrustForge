# Paper 1 Revision 2026 — Restricted Boltzmann Initialization Policy

`STATUS=SELECTED`

`RBM_INITIALIZATION_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs initialization of the revised Paper 1 Restricted Boltzmann
Machine.

It selects initial values for:

- visible-hidden weight matrix `W`;
- visible bias vector `a`;
- hidden bias vector `b`;
- class-to-hidden interaction matrix `U`.

It does not yet select optimizer, contrastive-divergence depth, training
traversal, update budget, checkpoint rule, or production sampling-chain length.

## 2. Governed architecture context

The selected revised architecture is:

`D=4800`

`H=128`

with shared class-to-hidden conditioning.

State shapes are:

`W: (4800,128)`

`a: (4800,)`

`b: (128,)`

`U: (K,128)`

All authoritative model state uses float64.

## 3. Weight-scale candidates audited

The revision evaluated visible-hidden Gaussian initialization scales:

- `0.005`;
- `0.010`;
- `0.020`;
- approximately `0.0201455741`, corresponding to
  `sqrt(2/(D+H))`.

The audit used canonical revised batches from all four datasets with:

- `b=0`;
- `U=0`;

so that visible-to-hidden initialization scale was isolated.

Across all tested scales and all four datasets:

- hidden logits remained finite;
- no tested hidden probability fell below `0.01`;
- no tested hidden probability exceeded `0.99`;
- no immediate hidden saturation was observed.

## 4. Selected visible-hidden weight initialization

The selected initialization is:

`W_ij ~ Normal(0, 0.01^2)`

with float64 storage.

Therefore:

`RBM_INITIAL_WEIGHT_STD=0.01`

This value is selected because it:

1. avoids immediate hidden saturation;
2. provides greater initial signal than `0.005`;
3. produces narrower initial hidden-logit spread than `0.020`;
4. avoids importing a generic Xavier-style scale without revision-specific need;
5. remains uniform across all four datasets.

## 5. Zero visible-bias baseline rejected

For the selected bounded-exponential visible conditional:

`a_i=0`

implies:

`E[V_i]=0.5`

when hidden contribution is absent.

Representative 256-example audits found observed scalar means approximately:

- USTC: `0.07224477`;
- CICMalDroid2020: `0.47147728`;
- MalNet-Image: `0.07041047`;
- MaleX: `0.39905805`.

Therefore zero visible bias creates especially large initialization mismatch for
USTC and MalNet-Image.

`RBM_ZERO_VISIBLE_BIAS_INITIALIZATION=REJECTED`

## 6. Per-coordinate empirical initialization rejected

A representative 4096-position audit evaluated per-coordinate empirical mean
matching with symmetric pseudocount smoothing.

The method reconstructed the smoothed coordinate means accurately.

However, coordinates observed entirely as zero produced extremely negative
initial natural parameters.

Observed examples included:

- USTC minimum candidate visible bias near `-8194`;
- MalNet-Image minimum candidate visible bias near `-8194`.

This is numerically legal in float64 but unnecessarily extreme for model
initialization.

The magnitude would depend strongly on the audit population and smoothing
convention.

Therefore the revision does not use per-coordinate empirical inversion as the
primary initialization.

`RBM_PER_COORDINATE_VISIBLE_BIAS_INITIALIZATION=REJECTED`

## 7. Selected visible-bias initialization

The selected visible-bias initialization is based on the dataset-wide governed
training scalar mean.

For one dataset, define:

`m_train`

as the arithmetic mean of all canonical scalar values in the governed training
split.

The initial visible natural parameter `a0` is the unique scalar satisfying:

`E[V | theta=a0] = m_train`

under the selected bounded-exponential visible conditional.

All visible-bias coordinates are initialized to:

`a_i = a0`

for:

`i=1,...,4800`.

Therefore:

`RBM_VISIBLE_BIAS_INITIALIZATION=GLOBAL_TRAINING_MEAN_MOMENT_MATCH`

## 8. Representative scalar-bias audit

A 4096-position representative audit produced:

### USTC

- empirical scalar mean: approximately `0.063764118469`;
- candidate `a0`: approximately `-15.682763288746`.

### CICMalDroid2020

- empirical scalar mean: approximately `0.447801859279`;
- candidate `a0`: approximately `-0.630516236862`.

### MalNet-Image

- empirical scalar mean: approximately `0.071689729239`;
- candidate `a0`: approximately `-13.948829250475`.

### MaleX

- empirical scalar mean: approximately `0.399391633345`;
- candidate `a0`: approximately `-1.237797943578`.

For each audited dataset, the bounded-exponential mean implied by the candidate
bias matched the empirical scalar mean to numerical precision.

These values are evidence only.

They are not frozen production constants because the audit used only the first
4096 governed training positions.

## 9. Production visible-bias census

The authoritative production initialization must compute:

`m_train`

over the complete governed training split for the current dataset.

The implementation must not hard-code the representative audit values from
Section 8.

The full training census must:

- use only the governed training partition;
- use canonical float32 observations;
- accumulate in float64;
- include all 4800 canonical scalar values from every training example;
- exclude validation and test observations;
- be deterministic.

`RBM_VISIBLE_BIAS_PRODUCTION_CENSUS=FULL_TRAIN_SPLIT`

## 10. Mean inversion

The scalar mean inversion must solve the bounded-exponential mean equation in
float64.

For natural parameter `theta`:

`E[V|theta]`

is monotonic from 0 toward 1.

The inversion must therefore use a deterministic numerically stable method such
as bracketed bisection.

It must not use an unconstrained optimizer whose result depends on optimizer
state or random initialization.

The recovered scalar bias must be finite.

`RBM_VISIBLE_BIAS_INVERSION=DETERMINISTIC_BRACKETED_ROOT`

## 11. Hidden bias initialization

The selected hidden bias initialization is:

`b=0`

for all hidden units.

Therefore initial hidden activation probabilities are governed by:

- canonical visible input;
- initialized `W`;
- class-conditioning term `U`.

No empirical hidden-frequency target is imposed before training.

`RBM_INITIAL_HIDDEN_BIAS=ZERO`

## 12. Class-conditioning initialization

The selected class-to-hidden interaction initialization is:

`U=0`

for all classes and hidden units.

Therefore the initial model begins with shared representation behavior across
classes.

Class-specific behavior must be learned from governed class-labeled training
data.

This avoids imposing arbitrary random class asymmetry before any training
evidence is observed.

`RBM_INITIAL_CLASS_INTERACTION=ZERO`

## 13. Initialization randomness

Only `W` requires stochastic initialization.

Its randomness must be derived from the governed scientific identity and seed.

Initialization must be exactly reproducible for identical:

- dataset identity;
- architecture identity;
- experimental seed;
- initialization policy version.

Changing the experimental seed must be capable of changing `W`.

Visible bias, hidden bias, and class-interaction initialization remain
deterministic for a fixed governed dataset and training partition.

## 14. No test-set influence

Initialization must not use:

- validation metrics;
- test observations;
- test labels;
- downstream classifier performance;
- synthetic-image evaluation results.

The visible-bias training mean is training-partition-only evidence.

`RBM_INITIALIZATION_TEST_SET_INFLUENCE=FORBIDDEN`

## 15. Fail-closed requirements

Initialization must fail closed if:

- the governed training split is empty;
- canonical values are non-finite;
- canonical values lie outside `[0,1]`;
- the full-training scalar mean is non-finite;
- the recovered visible bias is non-finite;
- any initialized model parameter is non-finite.

No default zero-bias fallback is permitted after a failed training-mean census.

`RBM_INITIALIZATION_FAILURE_ACTION=FAIL_CLOSED`

## 16. Implementation gate

Before initialization implementation may be accepted, focused evidence must
demonstrate at least:

1. `W` has shape `(4800,128)`;
2. `W` uses float64;
3. `W` empirical initialization scale is consistent with `0.01`;
4. identical governed identity and seed reproduce identical `W`;
5. changed seed changes `W`;
6. full-training scalar mean excludes validation and test data;
7. full-training scalar mean accumulates in float64;
8. visible-bias inversion is deterministic;
9. recovered `a0` is finite;
10. every coordinate of `a` initially equals `a0`;
11. `b` initializes exactly to zero;
12. `U` initializes exactly to zero;
13. initial hidden probabilities are finite;
14. no hidden saturation blocker appears in representative canonical batches;
15. no per-coordinate pseudocount path exists in the authoritative initializer;
16. initialization fails closed on invalid state.

## 17. Current authorization state

The revised RBM initialization is selected as:

- `W ~ Normal(0,0.01^2)`;
- dataset-wide full-training-mean visible-bias moment matching;
- `b=0`;
- `U=0`.

Initialization implementation remains unaccepted.

Real benchmark training remains unauthorized.

Therefore:

`RBM_INITIALIZATION_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
