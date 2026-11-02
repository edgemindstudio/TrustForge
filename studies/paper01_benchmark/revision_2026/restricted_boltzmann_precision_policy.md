# Paper 1 Revision 2026 — Restricted Boltzmann Precision Policy

`STATUS=SELECTED`

`RBM_PRECISION_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs numerical precision for the revised Paper 1 Restricted
Boltzmann Machine.

It selects precision for:

- model parameters;
- energy calculations;
- hidden logits and probabilities;
- bounded-exponential visible normalization;
- analytic visible moments;
- visible sampling arithmetic;
- training update statistics;
- checkpoint state.

It does not change the benchmark-wide canonical representation.

## 2. Canonical boundary

The Paper 1 canonical model-facing representation remains:

- shape `(B,40,40,3)`;
- dtype `float32`;
- finite;
- values in `[0,1]`.

Therefore:

`RBM_CANONICAL_INPUT_DTYPE=float32`

Generated benchmark-facing synthetic images must also return as:

`float32`

after successful internal validation.

`RBM_CANONICAL_OUTPUT_DTYPE=float32`

Internal precision may be higher than the canonical boundary dtype.

## 3. Precision candidates audited

The revision compared:

- TensorFlow `float32`;
- TensorFlow `float64`.

The audit exercised the selected bounded-exponential visible conditional across
natural parameters spanning approximately:

`-1e6` through `+1e6`

including the numerically sensitive near-zero regime.

The audit evaluated:

- log normalizer;
- analytic conditional mean;
- inverse-CDF visible sampling;
- finite arithmetic;
- canonical support preservation.

## 4. Float32 audit result

Float32 produced finite values for:

- log normalizer;
- analytic conditional mean;
- sampled visible values.

However:

`FLOAT32_SAMPLES_BOUNDED=NO`

A focused audit used:

- 19 natural-parameter values;
- 100000 draws per value;
- 1900000 total draws.

Observed out-of-bounds samples:

`59`

All observed failures occurred near:

`theta ≈ ±1e-4`

Specifically:

### theta approximately -1e-4

- samples below zero: `0`;
- samples above one: `30`;
- maximum observed sample: approximately `1.00021600723`.

### theta approximately +1e-4

- samples below zero: `29`;
- samples above one: `0`;
- minimum observed sample: approximately `-0.000216007233`.

Maximum observed canonical-boundary violation:

approximately:

`2.16007232666e-4`

All samples remained finite.

Therefore float32 arithmetic, under the audited inverse-CDF formulation, can
violate the mathematically required `[0,1]` support.

`RBM_FLOAT32_VISIBLE_SAMPLING_SUPPORT_AUDIT=FAIL`

## 5. Float64 audit result

Float64 passed the corresponding support audit.

Observed properties included:

- finite log-normalizer values;
- finite analytic conditional means;
- finite sampled values;
- sampled values remained inside `[0,1]`;
- stable behavior across extreme negative and positive natural parameters.

Therefore:

`RBM_FLOAT64_VISIBLE_SAMPLING_SUPPORT_AUDIT=PASS`

## 6. Float32 versus float64 normalization evidence

Across the audited natural-parameter values, the maximum observed absolute
difference between float32 and float64 log-normalizer values was approximately:

`0.00301055796444`

The maximum observed absolute difference between float32 and float64 analytic
conditional means was approximately:

`8.33333251649e-06`

The principal selection concern is not only approximation error.

The decisive concern is that float32 sampling violated the selected visible
distribution's bounded support.

## 7. Selected internal precision

The authoritative RBM internal model precision is:

`float64`

Therefore the following must use float64:

- `W`;
- `a`;
- `b`;
- `U`;
- hidden logits;
- hidden probabilities;
- visible natural parameters;
- bounded-exponential normalization;
- bounded-exponential analytic moments;
- inverse-CDF visible sampling arithmetic;
- energy values;
- training update statistics;
- checkpoint model state.

`RBM_INTERNAL_DTYPE=float64`

## 8. Boundary conversion

Canonical float32 input batches must be converted to float64 before entering
RBM model arithmetic.

The conversion must be explicit.

No additional rescaling, clipping, thresholding, or quantization is permitted
during this conversion.

Conceptually:

`canonical_float32 -> exact-value-preserving float64 cast -> RBM arithmetic`

Generated float64 samples must first pass internal validation requiring:

- finite values;
- values in `[0,1]`;
- expected shape.

Only then may they be cast to canonical float32 for benchmark output.

`RBM_OUTPUT_CAST_AFTER_VALIDATION=REQUIRED`

## 9. No posthoc clipping

Float64 selection must not be used as justification for retaining a clipping
repair path.

If any generated internal visible value lies outside `[0,1]`, generation must
fail closed.

The implementation must not perform:

`clip(sample, 0, 1)`

as a repair.

`RBM_INTERNAL_OUT_OF_RANGE_ACTION=FAIL_CLOSED`

`RBM_POSTHOC_CLIPPING=FORBIDDEN`

## 10. Near-zero numerical branch

The bounded-exponential implementation requires an explicitly governed
near-zero numerical branch.

The branch exists to evaluate the continuous limit stably.

It must not change the scientific distribution.

For quantities such as the log normalizer and analytic mean, a mathematically
valid local series expansion may be used.

For visible sampling, the `theta -> 0` limit is:

`Uniform(0,1)`.

Any threshold used to choose between equivalent numerical formulas is a
numerical implementation constant, not a scientific observation-space epsilon.

It must not alter canonical observations or redefine endpoint semantics.

## 11. Hidden-state sampling

Hidden states remain Bernoulli.

Random uniform variates used for hidden-state sampling must be generated under
the governed scientific seed.

Probability comparison arithmetic must use the selected internal precision or
an explicitly validated equivalent path.

A hidden-state draw must remain exactly representable as binary:

`0` or `1`.

## 12. Checkpoint precision

Persisted authoritative RBM state must retain float64 parameter values.

Checkpoint serialization must not silently downcast model parameters to
float32.

If an external serialization format cannot preserve the selected dtype exactly,
that format is not acceptable as the authoritative checkpoint representation.

`RBM_CHECKPOINT_STATE_DTYPE=float64`

## 13. Memory consequence

At the selected architecture:

`D=4800`

`H=128`

the shared conditional model remains small enough for float64 state.

Previously audited float64 state sizes were approximately:

- USTC: `4.73 MiB`;
- CICMalDroid2020: `4.73 MiB`;
- MalNet-Image: `4.77 MiB`;
- MaleX: `4.73 MiB`.

These values exclude optimizer state, gradients, activations, and checkpoint
metadata.

The float64 choice therefore does not create a model-state scalability blocker
under the selected shared architecture.

## 14. Implementation gate

Before precision implementation may be accepted, focused evidence must
demonstrate at least:

1. canonical inputs arrive as float32;
2. internal RBM parameters are float64;
3. canonical batches are explicitly cast to float64;
4. no thresholding occurs during the cast;
5. hidden logits are finite;
6. hidden probabilities are finite and bounded;
7. visible natural parameters are finite;
8. bounded-exponential normalization is finite;
9. near-zero normalization behavior is stable;
10. inverse-CDF visible samples are finite;
11. inverse-CDF visible samples remain in `[0,1]`;
12. extreme tested natural parameters remain stable;
13. fixed-seed sampling is reproducible;
14. authoritative checkpoint state preserves float64;
15. output range is validated before canonical float32 cast;
16. no posthoc clipping path exists.

## 15. Current authorization state

The revised RBM numerical precision is selected as:

`float64 internal`

with:

`float32 canonical input/output boundary`.

Precision implementation remains unaccepted.

Real benchmark training remains unauthorized.

Therefore:

`RBM_PRECISION_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
