# Paper 1 Revision 2026 — Restricted Boltzmann Sampling Initialization Policy

`STATUS=SELECTED`

`RBM_SAMPLING_INITIALIZATION_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs only the initial state of revised RBM synthesis chains.

It does not yet select:

- production burn-in length;
- total Gibbs transition count;
- thinning;
- multi-chain versus continuation semantics.

Those remain separately governed.

## 2. Historical initialization is rejected

Historical RBM synthesis used Bernoulli-visible initialization, including
Bernoulli(0.5)-style random starts.

That initialization matched the historical Bernoulli-visible model but does
not match the revised bounded continuous-visible observation family.

Therefore:

`RBM_HISTORICAL_BERNOULLI_INITIALIZATION=REJECTED`

## 3. Real-data initialization is rejected

Production synthesis must not initialize a chain from:

- a training example;
- a validation example;
- a test example;
- a nearest real example;
- a class prototype computed directly from held-out observations.

This avoids real-example dependence and prevents synthesis from becoming a
short perturbation of a selected empirical record.

Therefore:

`RBM_REAL_DATA_CHAIN_INITIALIZATION=FORBIDDEN`

`RBM_VALIDATION_DATA_CHAIN_INITIALIZATION=FORBIDDEN`

`RBM_TEST_DATA_CHAIN_INITIALIZATION=FORBIDDEN`

## 4. Selected model-based initialization

The revised RBM visible conditional is:

`p(v_i | h) proportional to exp(theta_i * v_i)`

for:

`0 <= v_i <= 1`

with:

`theta = a + W h`.

For chain initialization, set the hidden state to the neutral all-zero vector:

`h_0 = 0`

which gives:

`theta_0 = a`.

The initial visible state is sampled independently by coordinate from:

`v_0 ~ p(v | h=0)`

using the governed bounded-exponential inverse-CDF sampler.

Therefore:

`RBM_SAMPLING_INITIAL_HIDDEN_STATE=ZERO_VECTOR`

`RBM_SAMPLING_INITIAL_VISIBLE_PARAMETER=a`

`RBM_SAMPLING_INITIAL_VISIBLE_DISTRIBUTION=BOUNDED_EXPONENTIAL_P_V_GIVEN_H_ZERO`

## 5. Why the all-zero hidden state is used

The zero hidden vector provides a deterministic model-state reference that:

- requires no external data;
- introduces no arbitrary visible-space constant;
- uses the learned visible bias `a`;
- remains valid for every governed dataset;
- does not privilege a particular hidden feature pattern.

It is an initialization device only.

It must not be interpreted as a claim that the equilibrium hidden state is
zero.

## 6. Class semantics

The requested class identity is validated and clamped before the first Gibbs
transition.

Class identity does not alter the initial `p(v|h=0)` distribution directly
because the governed visible conditional is class-independent given hidden
state.

Class conditioning enters on the first hidden draw through:

`p(h | v_0, y)`.

Therefore:

`RBM_SAMPLING_CLASS_CLAMP_BEFORE_FIRST_GIBBS_TRANSITION=YES`

`RBM_SAMPLING_INITIAL_VISIBLE_CLASS_DEPENDENCE=NO`

`RBM_SAMPLING_FIRST_HIDDEN_DRAW_CLASS_CONDITIONED=YES`

## 7. First governed transition

After initialization, the first production Gibbs transition is:

1. hold requested class `y` fixed;
2. sample
   `h ~ Bernoulli(sigmoid(b + W^T v + U_y))`;
3. compute
   `theta = a + W h`;
4. sample
   `v ~ bounded_exponential(theta)`.

Class identity remains fixed for all subsequent transitions.

Therefore:

`RBM_SAMPLING_FIXED_CLASS_GIBBS=REQUIRED`

`RBM_SAMPLING_CLASS_RESAMPLING=FORBIDDEN`

## 8. Numerical precision

Initialization uses the authoritative internal RBM precision:

`float64`.

The bounded-exponential visible sampler must use the same stable float64
implementation selected by the precision policy.

Therefore:

`RBM_SAMPLING_INITIALIZATION_DTYPE=FLOAT64`

## 9. Range semantics

Every sampled initialization coordinate must satisfy:

`0 <= v_i <= 1`

without post-hoc clipping.

Any non-finite or out-of-range internal sample is a fail-closed error.

Therefore:

`RBM_SAMPLING_INITIALIZATION_POSTHOC_CLIPPING=FORBIDDEN`

`RBM_SAMPLING_INITIALIZATION_RANGE_FAILURE_ACTION=FAIL_CLOSED`

## 10. Endpoint semantics

The initialization distribution is continuous on `[0,1]`.

It does not create positive point mass at exactly `0` or `1`.

This is consistent with the selected revised RBM observation model.

Therefore:

`RBM_SAMPLING_INITIALIZATION_ENDPOINT_ATOMIC_MASS=NO`

## 11. Randomness

All stochastic initialization draws must be derived from governed sampling
identity.

Ambient process-global randomness is not authoritative.

The same:

- checkpoint identity;
- requested class;
- experimental seed;
- synthetic sample index;

must reproduce the same initialization when all governed sampling parameters
are unchanged.

Therefore:

`RBM_SAMPLING_INITIALIZATION_RANDOMNESS=GOVERNED`

## 12. No fallback initialization

If model-state preflight fails, synthesis must fail closed.

The sampler must not fall back to:

- Bernoulli(0.5);
- Uniform(0,1);
- zero visible vectors;
- empirical training examples;
- random normal vectors;
- historical initialization behavior.

Therefore:

`RBM_SAMPLING_INITIALIZATION_FALLBACK=FORBIDDEN`

## 13. Chain-length separation

The selected initialization policy does not establish that any particular
number of Gibbs transitions is sufficient for equilibration.

In particular:

- historical `CD_K` is not a synthesis-chain-length authority;
- training CD depth does not determine synthesis burn-in;
- one Gibbs transition is not automatically accepted for production synthesis.

Therefore:

`RBM_TRAINING_CD_K_IS_SAMPLING_CHAIN_LENGTH=NO`

`RBM_HISTORICAL_GIBBS_K_AUTOMATICALLY_REUSED=NO`

`RBM_PRODUCTION_BURN_IN_SELECTED=NO`

`RBM_PRODUCTION_CHAIN_LENGTH_SELECTED=NO`

## 14. Implementation gate

Before sampling initialization may be accepted, focused tests must demonstrate
at least:

1. initialization uses `h=0`;
2. initial visible natural parameter equals `a`;
3. initial visibles use the governed bounded-exponential sampler;
4. all initialization arithmetic is float64;
5. samples are finite;
6. samples remain in `[0,1]` without clipping;
7. no training example is used for initialization;
8. no validation example is used for initialization;
9. no test example is used for initialization;
10. requested class is clamped before the first Gibbs transition;
11. first hidden draw uses the requested class row of `U`;
12. class identity cannot change inside the chain;
13. same governed sampling identity reproduces initialization;
14. changed governed sampling identity can change initialization;
15. Bernoulli-visible historical initialization cannot override this policy;
16. missing or invalid checkpoint state fails closed.

## 15. Current authorization state

The revised RBM sampling initialization is selected as:

- `h_0 = 0`;
- `v_0 ~ p(v|h=0)`;
- visible natural parameter `theta=a`;
- bounded-exponential float64 sampling;
- requested class clamped before the first Gibbs transition;
- no empirical-data initialization;
- no clipping or fallback initialization.

Production burn-in and total chain length remain unresolved.

Therefore:

`RBM_SAMPLING_INITIALIZATION_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
