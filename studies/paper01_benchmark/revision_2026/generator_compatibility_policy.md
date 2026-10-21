# Paper 1 Revision 2026 — Generator Compatibility Policy

Status: GOVERNED
Scope: Revised Paper 1 benchmark only

## 1. Canonical benchmark representation

All revised Paper 1 generator families consume the same canonical benchmark representation:

- spatial shape: 40 × 40
- channels: 3
- layout: NHWC
- dtype: float32
- canonical numeric range: [0,1]

The canonical representation is produced by the governed revision-2026
canonicalization layer. Historical dataset-specific representations are
preserved as provenance but are not generator-facing inputs for revised
Paper 1 execution.

## 2. Architecture-specific transformations

A generator family may apply an architecture-appropriate internal
transformation after receiving the canonical representation, provided that:

1. the transformation is explicitly documented;
2. it is deterministic where appropriate;
3. it does not change dataset membership or class labels;
4. it is applied consistently across all revised datasets;
5. the corresponding inverse or output conversion is defined where needed;
6. synthetic outputs exposed to benchmark evaluation are returned to the
   canonical [0,1], 40 × 40 × 3 representation.

Architecture-specific preprocessing does not invalidate benchmark fairness
when it is required by the mathematical model and is governed in advance.

## 3. Class semantics

A generator evaluated as class-conditional must learn or otherwise model
class-specific distributions.

Synthetic samples MUST NOT receive class labels solely by partitioning or
relabeling samples from an unconditional generator.

Per-class models are acceptable when they constitute the intended
class-conditional formulation of that generator family.

## 4. Runtime-control integrity

Every hyperparameter reported as an experimental control must affect the
actual runtime execution.

Configuration fields that are historical, unused, stale, or metadata-only
must not be represented as active revised-2026 experimental controls.

The revised execution contract must distinguish:

- declared configuration;
- consumed runtime configuration;
- derived runtime values;
- recorded provenance.

## 5. Computational viability

A generator configuration must be computationally viable for the complete
revised benchmark:

- four datasets;
- seven experimental seeds;
- 2,000 synthetic samples per class;
- up to 47 classes in a dataset;
- canonical dimensionality D = 4,800.

A historical implementation that is mathematically valid but operationally
infeasible at revised benchmark scale may be replaced by a governed,
family-preserving configuration or implementation.

## 6. Controlled architecture correction

Historical implementations are preserved as provenance.

The revised benchmark may correct historical model design, likelihood,
conditioning, covariance structure, sampling procedure, preprocessing, or
runtime configuration when the historical choice would undermine:

- representation consistency;
- class semantics;
- fairness;
- reproducibility;
- numerical stability;
- computational viability;
- benchmark scalability.

Such corrections must preserve the generator family being evaluated and must
be documented before revised experiments are executed.

## 7. Silent fallback prohibition

A revised benchmark execution must not silently change the scientific model
definition in response to numerical failure.

Examples include silently changing:

- covariance family;
- component count;
- conditioning mode;
- likelihood family;
- hidden dimensionality;
- sampling semantics.

A failed governed configuration must fail explicitly unless a predetermined
and recorded fallback is part of the approved experimental specification.

## 8. Generator compatibility status

### GAN
Status: PASS

- canonical input: 40 × 40 × 3
- internal range: [-1,1]
- governed bridge: [0,1] -> [-1,1]
- architecture redesign required: no

### VAE
Status: PASS

- canonical input: 40 × 40 × 3
- internal range: [-1,1]
- governed bridge: [0,1] -> [-1,1]
- architecture redesign required: no

### Diffusion
Status: PASS

- canonical input: 40 × 40 × 3
- internal representation: [0,1] forward-noising process
- architecture redesign required: no

### Autoregressive / Conditional PixelCNN
Status: CORRECTION REQUIRED

Historical revised-scale blockers:

- output channel count hard-coded to one;
- Bernoulli/BCE observation model;
- binary sampling semantics;
- incompatible with continuous 40 × 40 × 3 canonical representation.

Revised specification: pending family-specific governance.

### Masked AutoFlow / MAF
Status: CORRECTION REQUIRED

Historical revised-scale blockers:

- unconditional density model;
- synthetic class labels assigned after unconditional sampling;
- sequential inverse sampling across D = 4,800 dimensions;
- approximately 24,000 sequential MADE evaluations per inverse pass under
  the historical five-flow configuration.

Revised specification: pending family-specific governance.

### Gaussian Mixture
Status: CORRECTION REQUIRED

Historical revised-scale blockers:

- flattened canonical dimensionality D = 4,800;
- historical default full covariance scales quadratically in D;
- historical retry behavior may silently change covariance family.

Class-specific modeling itself is acceptable.

Revised specification: pending family-specific governance.

### Restricted Boltzmann Machine
Status: CORRECTION REQUIRED

Historical revised-scale blockers:

- Bernoulli visible-unit model;
- revised continuous inputs are binarized at threshold 0.5;
- Gibbs synthesis emits binary visible states.

Class-specific per-class RBMs and the D = 4,800 / H = 256 parameter scale are
otherwise structurally viable.

Revised specification: pending family-specific governance.

## 9. Historical preservation

This policy does not rewrite or invalidate historical Paper 1 artifacts,
historical experiment contracts, or historical execution evidence.

Historical evidence remains historical.

The revision-2026 methodology is scientifically authoritative for the new
Paper 1 submission.
