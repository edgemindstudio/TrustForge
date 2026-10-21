# Paper 1 Revision 2026 — MAF Observation-Domain Options

## Status

`DECISION_STATUS=COMPLETE`

`MASKED_AUTOFLOW_TRAINING_AUTHORIZED=YES`

This document compares candidate observation-domain treatments for the
revision-2026 block-conditional Masked Autoregressive Flow.

No option is authorized merely by appearing here.

## 1. Governing problem

The canonical benchmark representation is float32 in the closed interval:

`[0,1]`

The four datasets do not share identical value structure.

Observed revision evidence established that:

- USTC contains a very large exact-endpoint mass and is effectively grayscale
  replicated across RGB channels;
- CIC contains legitimate continuous canonical values and almost no exact
  endpoints;
- MalNet-Image contains substantial exact-zero mass and genuine RGB structure;
- MaleX contains legitimate continuous canonical values and a smaller but
  nontrivial endpoint mass.

Therefore one generator-wide observation rule must not assume that all revised
data are either purely discrete or purely continuous.

The chosen MAF observation model must support the governed canonical
representation without silently changing one dataset family into another.

## 2. Required properties

Any selected observation-domain design must satisfy all of the following:

1. accepts canonical float32 values in `[0,1]`;
2. handles exact zero explicitly;
3. handles exact one explicitly;
4. does not require arbitrary endpoint epsilon clipping;
5. does not apply universal discrete dequantization to legitimate continuous
   observations;
6. does not repair generated values using post-hoc clipping;
7. has a mathematically defined forward transform;
8. has a mathematically defined inverse transform;
9. has a tractable Jacobian determinant;
10. preserves class-conditional block-MAF semantics;
11. is reproducible;
12. can be regression tested at exact boundaries;
13. generates values in `[0,1]` by construction;
14. remains computationally tractable at the full benchmark scale.

## 3. Option A — epsilon-shifted logit transform

### Definition

Map canonical values away from the boundaries using a fixed epsilon and then
apply logit.

Representative form:

`x' = epsilon + (1 - 2*epsilon) * x`

followed by:

`u = log(x' / (1 - x'))`

### Advantages

- simple;
- conventional with Gaussian-base affine flows;
- easy Jacobian;
- easy implementation.

### Problems

- exact zero and exact one are moved to arbitrary interior values;
- likelihood depends on the chosen epsilon;
- epsilon becomes a generator-specific scientific hyperparameter;
- generated inverse values may require additional range handling depending on
  the precise transform construction;
- repeats the same arbitrary-boundary issue rejected for the revised
  autoregressive observation model.

### Current assessment

`OPTION_A_STATUS=REJECT_CANDIDATE`

Reason:

The benchmark already has evidence that exact endpoint mass is substantial and
dataset-dependent. An arbitrary epsilon is not an acceptable hidden repair.

## 4. Option B — universal stochastic dequantization plus logit

### Definition

Add random noise to canonical values before applying an interior transform.

This generalizes the historical MAF strategy.

### Advantages

- conventional for quantized image likelihood modeling;
- avoids exact-boundary singularities;
- compatible with ordinary continuous flows.

### Problems

- CIC and MaleX contain legitimate continuous canonical values;
- applying the same dequantization to already-continuous observations changes
  their measurement semantics;
- introduces generator-specific stochastic preprocessing;
- exact endpoint observations are no longer represented as exact endpoints;
- likelihood interpretation differs between naturally quantized and
  continuously canonicalized datasets.

### Current assessment

`OPTION_B_STATUS=REJECT_CANDIDATE`

Reason:

Universal dequantization is not scientifically uniform across the four revised
datasets.

## 5. Option C — bounded monotonic autoregressive spline flow

### Definition

Use a continuous base distribution with bounded support and a monotonic
invertible transform defined directly on `[0,1]`.

A representative implementation is a monotonic rational-quadratic spline with
fixed boundary knots at zero and one.

The spline parameters for block `i` are predicted by the masked conditional
network from:

- earlier blocks;
- class condition.

### Advantages

- input domain remains `[0,1]`;
- exact zero and exact one are valid boundary coordinates;
- no epsilon shift is required;
- no universal dequantization is required;
- no post-hoc clipping is required;
- outputs remain in `[0,1]` by construction;
- forward and inverse are both explicit;
- Jacobian is tractable;
- remains an autoregressive normalizing flow.

### Limitations

- a continuous bounded flow models endpoint observations through boundary
  density rather than through positive discrete point mass;
- exact observed quantization atoms are therefore approximated as part of a
  continuous density model rather than represented as a mixed
  discrete-continuous distribution;
- implementation and testing are substantially more complex than affine MAF;
- spline parameterization requires additional governed choices such as bin
  count and minimum bin width/height/derivative.

### Family identity

This remains a masked autoregressive normalizing flow, but the transform is
spline-based rather than affine.

If selected, the manuscript must describe the revision implementation
accurately rather than implying that it is the historical affine MAF.

### Current assessment

`OPTION_C_STATUS=PRIMARY_CANDIDATE`

## 6. Option D — bounded parametric monotonic autoregressive transform

### Definition

Use a bounded base distribution such as Uniform(0,1) and an elementwise
monotonic invertible distribution transform whose parameters are predicted
autoregressively.

Examples include two-parameter bounded monotonic CDF/inverse-CDF transforms.

### Advantages

- exact bounded domain;
- no epsilon clipping;
- no universal dequantization;
- no post-generation clipping;
- simpler implementation than a general spline;
- tractable forward, inverse, and Jacobian.

### Limitations

- substantially less flexible than spline transforms;
- less standard as a benchmark MAF implementation;
- may require more flow stages to achieve comparable expressivity;
- family description would need especially careful methodological wording.

### Current assessment

`OPTION_D_STATUS=SECONDARY_CANDIDATE`

## 7. Mixed discrete-continuous endpoint model

A theoretically exact model for the empirical support could combine:

- positive probability mass at zero;
- positive probability mass at one;
- continuous density on the open interval `(0,1)`.

However, incorporating such mixed support consistently inside every
invertible MAF dimension would substantially change the normalizing-flow
construction and likelihood semantics.

This is not selected as the primary revision direction because Paper 1 is a
comparative generator benchmark rather than a paper introducing a new
mixed-measure normalizing-flow architecture.

`MIXED_SUPPORT_FLOW_STATUS=OUT_OF_SCOPE_FOR_PRIMARY_REVISION`

## 8. Provisional recommendation

The current strongest candidate is:

`bounded block-conditional masked autoregressive spline flow`

with:

- canonical domain `[0,1]`;
- bounded base support;
- monotonic spline transforms;
- exact structural boundaries at zero and one;
- class conditioning;
- 25 autoregressive spatial blocks;
- five flow stages;
- no post-hoc clipping;
- no universal dequantization;
- no arbitrary endpoint epsilon.

This recommendation is provisional until implementation complexity and
TensorFlow-only feasibility are validated.

## 9. Bounded-spline feasibility result

A TensorFlow-only bounded monotonic spline prototype was executed in the
governed Talon environment using:

- TensorFlow 2.20.0;
- Keras 3.11.3;
- vectorized block width of 192 dimensions;
- eight monotonic bins;
- exact canonical zero and one test values.

The prototype used a monotonic piecewise-linear spline solely to test the
bounded-domain construction.

Observed results:

- forward shape: `(4, 192)`;
- log-Jacobian shape: `(4, 192)`;
- exact zero mapped to exact zero;
- exact one mapped to exact one;
- forward outputs remained in `[0,1]`;
- forward values were finite;
- log-Jacobians were finite;
- maximum forward/inverse round-trip absolute error:
  `1.6093254089355469e-06`;
- maximum forward/inverse log-Jacobian cancellation error:
  `0.0`;
- parameter gradients were present and finite;
- no endpoint epsilon displacement was used;
- no post-hoc clipping was used.

Result:

`BOUNDED_SPLINE_DOMAIN_FEASIBILITY=PASS`

This establishes that the bounded monotonic spline family is technically
feasible in the existing TensorFlow-only environment.

It does not yet select the final spline parameterization.

In particular, the feasibility prototype does not by itself establish that a
piecewise-linear spline is sufficiently expressive for the final benchmark
implementation, nor does it constitute validation of a rational-quadratic
spline implementation.

Decision state at the bounded-spline feasibility checkpoint:

`CHECKPOINT_OPTION_C_BOUNDED_SPLINE_FAMILY_FEASIBLE=YES`

`CHECKPOINT_FINAL_SPLINE_PARAMETERIZATION_SELECTED=NO`

`CHECKPOINT_MAF_OBSERVATION_DOMAIN_POLICY_COMPLETE=NO`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`

These checkpoint markers record the state before the final spline decision in
Section 10 and are not the current authoritative status.

## 10. Final spline-parameterization decision

The revision selects:

`8-bin bounded monotonic piecewise-linear spline`

as the authoritative observation-domain transform candidate for implementation.

This decision is based on both feasibility evidence and implementation-scale
comparison.

The eight-bin piecewise-linear construction requires:

- 16 transform parameters per scalar;
- 3,072 transform outputs per 192-dimensional block;
- 216,128 shared-decoder parameters per flow with the governed
  `256 -> 64 -> 3072` decoder;
- 1,080,640 shared-decoder parameters across five flows;
- approximately 4.12 MiB of float32 decoder weights across five flows.

Raw transform-parameter activation storage is approximately:

- batch 64: 18.75 MiB;
- batch 128: 37.50 MiB;
- batch 256: 75.00 MiB.

The corresponding eight-bin rational-quadratic candidate would require:

- 25 transform parameters per scalar;
- 4,800 transform outputs per block;
- 328,448 decoder parameters per flow;
- 1,642,240 decoder parameters across five flows;
- approximately 6.26 MiB of decoder weights;
- approximately 29.30 MiB, 58.59 MiB, and 117.19 MiB of raw transform
  activations at batches 64, 128, and 256 respectively.

The rational-quadratic alternative therefore introduces additional parameter,
activation, implementation, and regression complexity.

No current revision evidence demonstrates that this added complexity is
necessary for the Paper 1 benchmark.

The piecewise-linear transform already demonstrated:

- exact zero preservation;
- exact one preservation;
- closed `[0,1]` forward and inverse domains;
- finite forward values;
- finite log-Jacobians;
- finite parameter gradients;
- maximum tested round-trip error approximately `1.61e-6`;
- exact tested forward/inverse log-Jacobian cancellation;
- no endpoint epsilon;
- no universal dequantization;
- no post-hoc clipping.

Piecewise-linear derivative discontinuities at internal knots are accepted as
part of the governed transform. They occur on a measure-zero set and do not
invalidate invertibility or the piecewise-defined Jacobian.

The selected implementation must use strictly positive governed minimum bin
widths and heights so that every segment remains monotonic and invertible.

Current decision:

`FINAL_SPLINE_PARAMETERIZATION_SELECTED=YES`

`MAF_SPLINE_TYPE=PIECEWISE_LINEAR`

`MAF_SPLINE_BINS=8`

`MAF_OBSERVATION_DOMAIN_POLICY_COMPLETE=YES`

This decision does not authorize experiment execution.

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`

## 11. Numerical-conditioning correction

Five-flow composition testing revealed that independently normalized spline
widths and heights can produce numerically extreme segment slopes even when
all individual bins remain strictly positive.

Observed under the initial independent-width/height parameterization:

- maximum individual-stage round-trip error remained small;
- five-flow round-trip maximum absolute error increased to approximately
  `3.805e-4`;
- one flow exhibited an observed minimum slope of approximately `0.00791`;
- the same flow exhibited an observed maximum slope of approximately `125.21`;
- maximum inverse slope exceeded `126`;
- reverse-stage reconstruction error amplified progressively across the
  composed flow stack.

The flow structure itself remained invertible and finite, but the composition
was unnecessarily ill-conditioned in float32.

Therefore the revision replaces independently normalized heights with a
slope-conditioned parameterization.

### Governed slope-conditioned construction

Each scalar still uses exactly 16 decoder outputs:

- 8 raw width parameters;
- 8 raw relative-slope controls.

Widths are normalized exactly as before.

Each raw slope control is mapped to a bounded log-relative-slope:

`a_i = L * tanh(r_i)`

and then:

`q_i = exp(a_i)`

with governed limit:

`L = 0.75`

Heights are derived from widths and relative slopes:

`h_i = (w_i * q_i) / sum_j(w_j * q_j)`

Therefore the actual piecewise-linear slope is:

`s_i = h_i / w_i`

and remains positive and explicitly conditioned.

The theoretical slope range under this normalization is bounded by:

`exp(-2L) <= s_i <= exp(2L)`

For:

`L = 0.75`

this gives approximately:

`0.22313 <= s_i <= 4.48169`

### Prototype evidence

Using the existing five-flow model weights and the governed test input, the
slope-conditioned construction produced:

- observed minimum slope: approximately `0.27593`;
- observed maximum slope: approximately `3.57927`;
- five-flow latent minimum: approximately `0.00438`;
- five-flow latent maximum: approximately `0.99108`;
- maximum five-flow round-trip absolute error:
  approximately `1.907e-6`;
- mean five-flow round-trip absolute error:
  approximately `1.335e-7`;
- maximum forward/inverse log-Jacobian cancellation error:
  approximately `6.104e-5`;
- all reconstructed values finite.

The five-flow maximum round-trip error therefore decreased by roughly two
orders of magnitude relative to the independently normalized height
parameterization.

### Final parameter interpretation

The selected transform remains:

`8-bin bounded monotonic piecewise-linear spline`

but its 16 scalar decoder outputs are now interpreted as:

`8 raw widths + 8 raw relative-slope controls`

rather than:

`8 raw widths + 8 independently normalized raw heights`

Current authoritative parameters:

`MAF_SPLINE_TYPE=PIECEWISE_LINEAR`

`MAF_SPLINE_BINS=8`

`MAF_LOG_SLOPE_LIMIT=0.75`

`MAF_SPLINE_PARAMETERIZATION=SLOPE_CONDITIONED`

`MAF_OBSERVATION_DOMAIN_POLICY_COMPLETE=YES`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`
