# Paper 1 Revision 2026 — Gaussian Mixture Initialization Policy

`STATUS=IMPLEMENTED_AND_FOCUSED_VALIDATED`

`GMM_INITIALIZATION_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

The governed class-model initialization implementation is now present in:

`src/trustforge/models/paper01_revision_2026/gaussian_mixture_initialization.py`

and its focused validation is included in the combined revision-2026 GMM
observation/model/initialization suite.

The initialization-only focused suite now passes 24 tests.

The broader observation/model/initialization regression suite passes 49 tests.

Current validation also enforces:

- exact governed initialization-position selection;
- explicit greedy k-means++ local-trial count;
- population-variance (`ddof=0`) semantics;
- minimum five distinct canonical vectors;
- fail-closed degenerate initialization;
- one-ULP scale-floor representation;
- maximum one-ULP transformed-scale round-trip tolerance;
- exact-integer validation for class counts and experimental seeds;
- rejection of float/string coercion for governed integer identities;
- arbitrary-precision Python integer evaluation of initialization positions;
- explicit rejection of class counts outside the int64 logical-position domain.

`GMM_INITIALIZATION_COMBINED_REGRESSION_TEST_COUNT=49`

`GMM_INITIALIZATION_INTEGER_INPUT_VALIDATION=EXACT_INTEGER_ONLY`

`GMM_INITIALIZATION_POSITION_ARITHMETIC=PYTHON_ARBITRARY_PRECISION_INTEGER`

`GMM_INITIALIZATION_LOGICAL_POSITION_DOMAIN=INT64`

This status advancement applies only to initialization.

Overall Gaussian-mixture training remains unauthorized until the remaining
training, traversal, checkpoint, and sampling implementation contracts are
completed and validated.

## 1. Scope

This policy governs initialization for the revised Paper 1
class-conditional censored diagonal Gaussian mixture model.

It does not authorize real dataset training.

Optimizer configuration, minibatch training procedure, convergence policy,
checkpointing, and final implementation acceptance remain separately
governed.

## 2. Fixed mixture cardinality

The revised GMM uses:

`GMM_COMPONENTS=5`

for every governed dataset and class.

The revised training-class-count audit established:

- USTC minimum class count: 8100;
- CICMalDroid minimum class count: 877;
- MalNet-Image minimum class count: 79;
- MaleX minimum class count: 125794.

Every governed class therefore supports five mixture components.

The implementation must fail closed if a future class contains fewer than
five revised training examples.

It must not silently reduce the component count.

`GMM_MIN_CLASS_COUNT_FOR_K5=5`

`GMM_SMALL_CLASS_COMPONENT_REDUCTION=NO`

## 3. Historical initialization evidence

The historical Paper 1 checkpoints used:

`init_params=kmeans`

`n_init=1`

and the experimental run seed as the sklearn random state.

Historical component-health auditing showed:

### USTC

- 5 of 135 fitted components had weight below `1e-3`;
- 8 of 135 fitted components had weight below `1e-2`;
- minimum fitted component weight was approximately `0.000111`.

### CICMalDroid2020

- no fitted component had weight below `1e-2`;
- minimum fitted component weight was approximately `0.02195`.

The historical initialization was therefore generally functional, but the
revision does not retain unbounded full-class k-means initialization.

## 4. Scalability requirement

The revised canonical dimensionality is:

`GMM_DIMENSION=4800`

Class sizes vary substantially.

The largest revised MalNet-Image training class contains 619118 examples.

Initialization must therefore have bounded memory and bounded clustering
cost independent of full class size.

## 5. Initialization sample

The selected maximum initialization sample size is:

`GMM_INITIALIZATION_SAMPLE_CAP=4096`

For a class with:

`N <= 4096`

all revised training examples from that class are used for initialization.

For a class with:

`N > 4096`

exactly 4096 class examples are selected using deterministic evenly spaced
positions across that class's ordered revised training positions.

The exact governed logical-position rule is:

`position_i = floor(i * (N - 1) / (4096 - 1))`

for:

`i = 0,1,...,4095`

The production implementation must use integer arithmetic for this formula.

It must not use floating-point `linspace`, floating-point rounding, or another
numerically equivalent-looking position generator as the authoritative
identity rule.

This construction:

- includes logical position `0`;
- includes logical position `N - 1`;
- returns exactly 4096 positions;
- produces strictly increasing positions for every `N > 4096`;
- produces no duplicate logical positions;
- produces adjacent gaps differing by at most one.

For:

`N <= 4096`

the selected logical positions remain:

`0,1,...,N-1`

in the class's existing ordered revised training position order.

The bounded sample-selection procedure is independent of image values.

The sampling procedure must be reproducible and must not require loading the
entire canonical class into memory.

The audited rule was validated for class counts including 4097, 8100, 12221,
605255, and 619118, with exact endpoint inclusion, uniqueness, reproducibility,
strict ordering, and maximum adjacent-gap spread of one.

`GMM_INITIALIZATION_SAMPLE_SELECTION=DETERMINISTIC_EVENLY_SPACED_INTEGER`

`GMM_INITIALIZATION_POSITION_FORMULA=FLOOR_I_TIMES_N_MINUS_1_OVER_4095`

`GMM_INITIALIZATION_POSITION_ENDPOINTS_INCLUDED=YES`

`GMM_INITIALIZATION_POSITION_FLOATING_POINT_ARITHMETIC=NO`

## 6. Mean initialization

Initial component means are selected using seeded greedy k-means++ on the
bounded class initialization sample.

The operation is used only to select the five initial centers.

It is not a full k-means fitting stage.

No Lloyd iterations over the full class are required.

For the governed component count:

`K = 5`

the production implementation must use:

`n_local_trials = 3`

explicitly.

This matches the audited scikit-learn 1.7.2 greedy k-means++ behavior for
`n_local_trials=None`, where:

`2 + int(log(5)) = 3`

The explicit value removes dependence on a library default while preserving
the audited greedy-center-selection semantics.

The production implementation must not substitute:

`n_local_trials = 1`

because that invokes vanilla k-means++ and was observed to select a different
set of centers under the same seed and input.

`GMM_MEAN_INITIALIZATION=KMEANS_PLUS_PLUS_GREEDY`

`GMM_KMEANS_PLUS_PLUS_LOCAL_TRIALS=3`

`GMM_KMEANS_PLUS_PLUS_LIBRARY_DEFAULT_DEPENDENCY=NO`

`GMM_FULL_CLASS_KMEANS_INITIALIZATION=NO`

The k-means++ random state must be derived from the governed experimental
seed for that model run.

Therefore initialization stochasticity remains part of the experimental run
identity rather than an unrelated fixed architecture seed.

`GMM_INITIALIZATION_SEED_SOURCE=EXPERIMENTAL_SEED`

## 7. Mixture-weight initialization

All five mixture components begin with equal prior weight.

For K=5:

`pi_k = 0.2`

for every component at initialization.

The implementation may represent these weights through equal mixture logits.

`GMM_INITIAL_WEIGHT_POLICY=UNIFORM`

## 8. Scale initialization

Initial diagonal variances are estimated from the same bounded
class-specific initialization sample used for mean initialization.

For each canonical dimension, the production initialization variance is the
population variance of the complete governed initialization sample:

`sample_variance_d = (1 / M) * sum_i (x_id - mean_d)^2`

where:

`M`

is the number of examples in the governed initialization sample for that
class.

Equivalently, the variance convention is:

`ddof = 0`

The implementation must therefore use population-variance semantics
(`ddof=0`), not the unbiased sample-variance convention (`ddof=1`).

This choice treats the deterministic bounded initialization matrix as the
complete data object from which the initial GMM state is constructed. The
variance is an initialization parameter for maximum-likelihood optimization;
it is not being reported as an unbiased statistical estimate of an unknown
population variance.

The previously recorded real-data initialization audits did not preserve
executable evidence sufficient to reconstruct their `ddof` convention.
Those historical audit percentages therefore must not be cited as evidence
that they established either `ddof=0` or `ddof=1`.

This production rule is an explicit prospective clarification of the revised
benchmark initialization contract.

The governed minimum variance is then applied:

`initial_variance_d = max(sample_variance_d, 1e-3)`

`GMM_INITIALIZATION_VARIANCE_ESTIMATOR=POPULATION`

`GMM_INITIALIZATION_VARIANCE_DDOF=0`

and the mathematical desired initialization scale is:

`desired_initial_scale_d = sqrt(initial_variance_d)`

The production trainable scale representation remains:

`scale = GMM_MIN_SCALE + softplus(raw_scale)`

with:

`GMM_MIN_VARIANCE=0.001`

`GMM_MIN_SCALE=0.03162277660168379`

For finite `raw_scale`, this representation is strictly greater than
`GMM_MIN_SCALE`.

Therefore, when:

`desired_initial_scale_d > GMM_MIN_SCALE`

the desired scale is converted directly to the finite raw-scale
representation.

When:

`desired_initial_scale_d == GMM_MIN_SCALE`

the production initialization scale is deterministically promoted to:

`nextafter(GMM_MIN_SCALE, +infinity)`

in float64 before inverse-softplus conversion.

This is the smallest representable float64 scale strictly above the governed
floor.

It is not a tunable epsilon and does not change the governed optimization
floor.

The audited one-ULP promotion produced:

- governed floor scale:
  `0.031622776601683791`;
- next representable float64 scale:
  `0.031622776601683798`;
- absolute scale increment:
  `6.9388939039072284e-18`;
- relative scale increment:
  approximately `2.19427e-16`;
- relative variance increment:
  approximately `4.33681e-16`;
- finite corresponding raw scale:
  approximately `-39.509389291916882`.

The transformed finite raw scale recovered the next representable scale
exactly in the audited environment.

The same class-wide diagonal production initialization scale vector is used
to initialize all five components before optimization separates
component-specific scales.

`GMM_INITIAL_SCALE_FLOOR_REPRESENTATION=NEXTAFTER_ONE_ULP`

`GMM_INITIAL_SCALE_ARBITRARY_EPSILON=NO`

## 9. Real-data validation evidence

The candidate initialization procedure was audited against revised canonical
data from representative classes spanning small, large, endpoint-heavy, and
continuous regimes.

### USTC class 0

- class count: 8100;
- initialization sample: 4096;
- canonical initialization matrix: 4096 x 4800;
- memory: approximately 75 MiB;
- exact-zero fraction: approximately 88.32%;
- deterministic k-means++ centers: PASS;
- initial variance-floor fraction: approximately 12.69%.

### CICMalDroid class 0

- class count: 877;
- all 877 examples used;
- deterministic k-means++ centers: PASS;
- initial variance-floor fraction: approximately 27.75%.

### MalNet-Image class 12

This was the smallest revised MalNet training class.

- class count: 79;
- all 79 examples used;
- deterministic k-means++ centers: PASS;
- initial variance-floor fraction: approximately 91.17%.

This result confirms that explicit variance-floor protection is necessary
for sparse or near-constant canonical dimensions.

### MalNet-Image class 4

This was the largest revised MalNet training class.

- class count: 619118;
- initialization sample bounded to 4096;
- initialization matrix memory: approximately 75 MiB;
- deterministic k-means++ centers: PASS;
- initial variance-floor fraction: approximately 36.65%.

The bounded initialization design therefore remains practical for the
largest revised class.

### MaleX classes 0 and 1

Both governed MaleX classes were audited with the 4096-example cap.

Both produced:

- deterministic k-means++ centers;
- bounded canonical means;
- finite initial scales;
- no dimensions at the variance floor in the audited samples.

## 10. Required implementation behavior

The production implementation must:

1. validate that the class contains at least five training examples;
2. derive the bounded deterministic initialization sample;
3. canonicalize only the selected initialization examples;
4. validate that the canonical initialization sample contains at least five
   distinct canonical vectors;
5. fail closed if fewer than five distinct canonical vectors are present;
6. select five numerically distinct k-means++ centers using the experimental
   seed;
7. initialize mixture weights uniformly;
8. derive the class-wide initial diagonal scale vector from the same sample;
9. apply the governed minimum variance;
10. derive the mathematical desired initial scale;
11. if the desired scale equals `GMM_MIN_SCALE`, promote it by exactly one
    float64 ULP using `nextafter(GMM_MIN_SCALE, +infinity)`;
12. convert the resulting production initialization scale into the governed
    trainable raw-scale representation;
13. validate the recovered transformed scale against the intended production
    scale using a maximum tolerance of one float64 ULP per dimension;
14. reject non-finite means, weights, variances, scales, or raw-scale values;
15. preserve canonical `[0,1]`, 4800-dimensional semantics.

The distinctness requirement applies to numerical canonical vectors, not
merely to row indices. Different row indices containing identical canonical
vectors do not satisfy the five-distinct-center requirement.

The all-dataset initialization audit covered all 63 revised training classes.
Every governed initialization sample contained at least five distinct
canonical vectors. The minimum observed distinct-vector count was 52
(MalNet-Image class 12, 79 selected examples).

The scale-parameterization round-trip audit showed that finite float64
inverse-softplus followed by the governed transformed-scale mapping recovered
the intended scale either exactly or within one float64 ULP in the audited
range. Bitwise equality is therefore not required for non-floor scales.

`GMM_INITIALIZATION_MIN_DISTINCT_CANONICAL_VECTORS=5`

`GMM_INITIALIZATION_CENTER_DISTINCTNESS=NUMERICAL_CANONICAL_VECTOR`

`GMM_INITIALIZATION_INSUFFICIENT_UNIQUE_VECTORS=FAIL_CLOSED`

`GMM_INITIALIZATION_SCALE_ROUNDTRIP_MAX_ULP=1`

`GMM_INITIALIZATION_SCALE_ROUNDTRIP_BITWISE_EQUALITY_REQUIRED=NO`

The implementation must not silently fall back to a global model.

The implementation must not silently reduce K.

## 11. Current governed constants

`GMM_COMPONENTS=5`

`GMM_DIMENSION=4800`

`GMM_INITIALIZATION_SAMPLE_CAP=4096`

`GMM_INITIALIZATION_SAMPLE_SELECTION=DETERMINISTIC_EVENLY_SPACED_INTEGER`

`GMM_INITIALIZATION_POSITION_FORMULA=FLOOR_I_TIMES_N_MINUS_1_OVER_4095`

`GMM_INITIALIZATION_POSITION_ENDPOINTS_INCLUDED=YES`

`GMM_INITIALIZATION_POSITION_FLOATING_POINT_ARITHMETIC=NO`

`GMM_MEAN_INITIALIZATION=KMEANS_PLUS_PLUS_GREEDY`

`GMM_KMEANS_PLUS_PLUS_LOCAL_TRIALS=3`

`GMM_KMEANS_PLUS_PLUS_LIBRARY_DEFAULT_DEPENDENCY=NO`

`GMM_FULL_CLASS_KMEANS_INITIALIZATION=NO`

`GMM_INITIALIZATION_SEED_SOURCE=EXPERIMENTAL_SEED`

`GMM_INITIAL_WEIGHT_POLICY=UNIFORM`

`GMM_INITIALIZATION_VARIANCE_ESTIMATOR=POPULATION`

`GMM_INITIALIZATION_VARIANCE_DDOF=0`

`GMM_INITIALIZATION_MIN_DISTINCT_CANONICAL_VECTORS=5`

`GMM_INITIALIZATION_CENTER_DISTINCTNESS=NUMERICAL_CANONICAL_VECTOR`

`GMM_INITIALIZATION_INSUFFICIENT_UNIQUE_VECTORS=FAIL_CLOSED`

`GMM_INITIALIZATION_SCALE_ROUNDTRIP_MAX_ULP=1`

`GMM_INITIALIZATION_SCALE_ROUNDTRIP_BITWISE_EQUALITY_REQUIRED=NO`

`GMM_INITIAL_SCALE_SOURCE=CLASS_SAMPLE_DIAGONAL_VARIANCE`

`GMM_MIN_VARIANCE=0.001`

`GMM_MIN_SCALE=0.03162277660168379`

`GMM_SMALL_CLASS_COMPONENT_REDUCTION=NO`

`GMM_INITIALIZATION_POLICY_SELECTED=YES`

## 12. Authorization state

The initialization policy is selected.

The complete training procedure and production implementation are not yet
accepted.

Therefore:

`GMM_INITIALIZATION_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_INITIALIZATION_FOCUSED_VALIDATION=PASSED`

`GMM_INITIALIZATION_FOCUSED_TEST_COUNT=24`

`GMM_IMPLEMENTATION_AUTHORIZED=PARTIAL_INITIALIZATION_ONLY`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
