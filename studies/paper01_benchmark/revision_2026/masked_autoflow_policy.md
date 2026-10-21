# Paper 1 Revision 2026 — Masked AutoFlow / MAF Policy

## Status

`STATUS=READY_FOR_TRAINING`

`MASKED_AUTOFLOW_TRAINING_AUTHORIZED=YES`

The historical Masked AutoFlow implementation is preserved for provenance but
is not scientifically authoritative for the revision-2026 benchmark.

The revision requires a governed replacement implementation before any revised
MAF training or synthesis is authorized.

## 1. Family identity

The benchmark family remains:

`maskedautoflow`

Historical experiment contracts identify the earlier implementation separately
as:

`historical_gencys_maskedautoflow`

The revision may therefore replace the historical implementation while
preserving the Masked Autoregressive Flow family identity.

The replacement must remain scientifically recognizable as a masked
autoregressive normalizing-flow model. A different flow family must not be
silently substituted under the MAF label.

## 2. Canonical model-facing representation

The revised benchmark model-facing representation is:

- shape: `40 x 40 x 3`;
- flattened dimensionality for vector-flow operations: `D = 4800`;
- dtype: float32;
- canonical range: `[0,1]`.

Generator-specific internal transforms may be introduced only when explicitly
governed and scientifically justified.

Historical preprocessing behavior is not automatically authoritative.

## 3. Historical correctness blocker — autoregressive masking

The historical MADE implementation uses the same inclusive mask relation for
hidden and output masked-dense layers.

Empirical dependency auditing demonstrated that, across multiple seeds,
selected output dimensions changed when their own corresponding input
dimension changed.

Therefore the historical implementation violates the inverse routine's stated
assumption that the parameters for dimension `i` depend only on dimensions
strictly preceding `i`.

Revision requirement:

- an output for autoregressive group or dimension `i` must not depend on the
  value being transformed in that same autoregressive step;
- the masking rule must be structurally testable;
- regression tests must demonstrate absence of forbidden self-dependence.

## 4. Historical conditioning blocker

The historical training path is unconditional.

Historical labels are accepted by some APIs but ignored by the density model.

Historical synthesis draws one unconditional sample pool and subsequently
assigns class labels by slicing that pool into class-sized blocks.

This is not acceptable for the revision benchmark.

Revision requirement:

- class identity must participate in the learned generative distribution;
- training must optimize a conditional density or an explicitly class-specific
  governed model;
- requested class identity must affect synthesis;
- synthetic labels must describe the distribution from which the sample was
  generated;
- post-hoc assignment of arbitrary class labels to unconditional samples is
  forbidden.

## 5. Historical scalar-inverse scalability blocker

For the canonical representation:

`D = 40 x 40 x 3 = 4800`

The historical scalar-order inverse evaluates each MADE once per dimension for
each flow.

With five flows:

`4800 x 5 = 24000`

sequential MADE evaluations are required per inverse batch.

The primary revision synthesis budget is:

`2000 synthetic samples per class per experimental seed`

Across the four revision datasets and seven experimental seeds:

- USTC-TFC2016: 126,000 images;
- CICMalDroid2020: 70,000 images;
- MalNet-Image: 658,000 images;
- MaleX: 28,000 images.

Total:

`882000 revised MAF synthetic images`

Observed workload estimates for the historical five-flow inverse include:

- batch 256: 83,160,000 sequential MADE evaluations;
- batch 512: 41,664,000 sequential MADE evaluations;
- batch 1024: 20,832,000 sequential MADE evaluations;
- batch 2000: 10,584,000 sequential MADE evaluations.

Revision requirement:

- the replacement must not require scalar-sequential inversion across all
  4,800 canonical dimensions;
- any grouped or block-autoregressive construction must preserve a valid
  triangular Jacobian and exact governed sampling semantics;
- the number of sequential inverse stages must be explicitly bounded and
  tested.

## 6. Historical boundary/preprocessing blocker

The historical training implementation enables dequantization by default and
adds uniform noise before flattening.

It also clips transformed training and validation values away from exact zero
and exact one.

This behavior is not automatically valid for the revision datasets because the
canonical benchmark includes both heavily quantized representations and
legitimate continuous canonical values.

Revision requirement:

- no universal historical dequantization rule may be inherited silently;
- exact canonical endpoints must not be moved merely to avoid numerical
  inconvenience without an explicit observation-model policy;
- any internal bounded-to-unbounded transform must define its treatment of
  exact zero and exact one;
- family-specific preprocessing must remain deterministic or seed-governed and
  reproducible.

## 7. Historical synthesis-distribution blocker

The historical sampler performs inverse-flow sampling and then clips generated
values into `[0,1]`.

This means the evaluated synthetic distribution is not identical to the
continuous distribution represented by the trained invertible model whenever
inverse samples fall outside the canonical interval.

Revision requirement:

- post-hoc clipping must not be used to repair an otherwise unbounded flow
  sample distribution;
- bounded output semantics must be part of the governed generative model or
  governed observation transform;
- generated outputs must satisfy the canonical range by construction.

## 8. Replacement architecture requirements

Before implementation, the revision must define a family-specific design that
simultaneously satisfies:

1. genuine Masked Autoregressive Flow family identity;
2. correct autoregressive masking;
3. class-aware generation;
4. canonical `40 x 40 x 3` compatibility;
5. bounded-output semantics without silent post-hoc clipping;
6. explicit endpoint treatment;
7. exact and reproducible sampling;
8. tractable synthesis at the full four-dataset, seven-seed benchmark scale;
9. finite training loss and gradients;
10. deterministic structural construction under a governed seed.

A grouped or block-autoregressive MAF is permitted for evaluation as a
candidate design only if its Jacobian structure, conditioning semantics, and
inverse-stage complexity are explicitly specified and regression tested.

An inverse autoregressive flow, coupling flow, RealNVP-style model, or other
different normalizing-flow family must not be relabeled as MAF merely to obtain
faster sampling.

## 9. Evaluation scope

MAF likelihood is a model-training quantity.

The revision benchmark must not assume that raw MAF likelihood values are
directly comparable across datasets with different endpoint, quantization, or
continuous-value structure.

Primary benchmark evaluation remains based on the governed synthetic-image and
downstream evaluation protocol.

## 10. Historical preservation

Historical code, experiment contracts, checkpoints, manifests, and execution
evidence remain historical artifacts.

They are not rewritten by this policy.

Historical acceptance does not imply revision-2026 methodological acceptance.

## 11. Authorization gate

Real revision-2026 MAF training and synthesis remain unauthorized until:

- the replacement architecture is governed;
- its observation/boundary semantics are governed;
- structural mask tests pass;
- class-conditioning tests pass;
- bounded-output tests pass;
- inverse-complexity tests pass;
- finite-loss and finite-gradient tests pass;
- canonical 40 x 40 x 3 compatibility passes;
- a focused Paper 1 regression suite passes.

Current state:

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`
