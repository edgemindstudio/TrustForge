# Autoregressive Observation-Model Policy — Paper 1 Revision 2026

## Status

DESIGN DECISION ESTABLISHED FROM REAL-DATA AUDIT.

Implementation and regression validation remain pending.

## Scope

This policy governs the observation distribution used by the revised Paper 1
conditional autoregressive generator.

Historical autoregressive implementations and historical experiment artifacts
remain unchanged for provenance.

## Canonical input domain

All revised Paper 1 generators consume the governed canonical representation:

- shape: 40 x 40 x 3
- layout: NHWC
- dtype: float32
- numeric domain: [0,1]

The autoregressive family must operate on this same canonical representation.

No AR-specific requantization of the benchmark input is authorized.

## Rejected transformed-space Gaussian formulation

The initial revision-2026 PixelCNN implementation predicted a correlated
three-dimensional Gaussian in logit-transformed RGB space.

Canonical values were clipped to:

`[epsilon, 1 - epsilon]`

before applying the logit transform.

That formulation is rejected as the final Paper 1 observation model.

The rejection is scientific rather than merely software-related.

Exact canonical values at 0 or 1 are mapped to values determined by the chosen
epsilon. Consequently, the resulting objective is materially epsilon-sensitive
for datasets with substantial endpoint mass.

Changing epsilon does not resolve the modeling problem.

The transformed-space Gaussian objective must not be described as the final
Paper 1 image likelihood.

## Cross-dataset real-data audit

A deterministic audit was performed on 512 governed training images from each
revised dataset.

Audit sample-selection seed:

`42`

Each audited image contained:

`40 x 40 x 3 = 4,800`

canonical scalar observations.

Each dataset therefore contributed:

`2,457,600`

scalar observations.

### USTC-TFC2016

Training population:

`72,900`

Observed audit characteristics:

- exact zero fraction: 0.8471179199
- exact one fraction: 0.0015588379
- exact endpoint fraction: 0.8486767578
- near-endpoint fraction at 1e-3: 0.8486767578
- near-endpoint fraction at 1e-2: 0.8531079102
- exact 8-bit-grid fraction: 0.9943420410
- 8-bit requantization MAE: 0.0000061156
- 8-bit requantization RMSE: 0.0000872486
- maximum requantization delta: 0.0019525886
- all-three-channels-equal fraction: 1.0000000000

USTC therefore contains overwhelming exact boundary mass and is almost
entirely aligned with an 8-bit grid.

### CICMalDroid2020

Training population:

`8,117`

Observed audit characteristics:

- minimum canonical value: 0.0201472919
- exact zero fraction: 0.0000000000
- exact one fraction: 0.0000097656
- exact endpoint fraction: 0.0000097656
- near-endpoint fraction at 1e-3: 0.0001367188
- near-endpoint fraction at 1e-2: 0.0004418945
- exact 8-bit-grid fraction: 0.0000610352
- 8-bit requantization MAE: 0.0009970204
- 8-bit requantization RMSE: 0.0011500388
- maximum requantization delta: 0.0019608140
- all-three-channels-equal fraction: 1.0000000000

CIC is therefore overwhelmingly off the 8-bit grid after governed continuous
feature preprocessing and canonical resizing.

### MalNet-Image

Training population:

`883,416`

Observed audit characteristics:

- exact zero fraction: 0.6472648112
- exact one fraction: 0.0000000000
- exact endpoint fraction: 0.6472648112
- near-endpoint fraction at 1e-3: 0.6553434245
- near-endpoint fraction at 1e-2: 0.6624877930
- exact 8-bit-grid fraction: 0.6472920736
- 8-bit requantization MAE: 0.0003398842
- 8-bit requantization RMSE: 0.0006660101
- maximum requantization delta: 0.0019607916
- channel-0/channel-1 equality fraction: 0.1739660645
- channel-1/channel-2 equality fraction: 0.0119567871
- all-three-channels-equal fraction: 0.0119567871

MalNet therefore contains very substantial exact boundary mass while retaining
genuine RGB channel structure.

### MaleX

Training population:

`731,049`

Observed audit characteristics:

- exact zero fraction: 0.0220898438
- exact one fraction: 0.0000646973
- exact endpoint fraction: 0.0221545410
- near-endpoint fraction at 1e-3: 0.0236853027
- near-endpoint fraction at 1e-2: 0.0290283203
- exact 8-bit-grid fraction: 0.0228710938
- 8-bit requantization MAE: 0.0009582304
- 8-bit requantization RMSE: 0.0011196194
- maximum requantization delta: 0.0019608140
- all-three-channels-equal fraction: 1.0000000000

MaleX therefore contains some exact boundary observations but is predominantly
continuous after governed resizing.

## Consequences of the audit

The four datasets do not share a purely discrete 8-bit canonical observation
space.

They also do not share a purely interior continuous observation space.

Accordingly:

1. endpoint observations must be modeled explicitly;
2. legitimate continuous interior values must remain legitimate observations;
3. no arbitrary endpoint epsilon may define the scientific objective;
4. no AR-only requantization may replace the governed canonical
   representation;
5. genuine RGB structure must remain representable.

## Governed replacement family

The revised autoregressive generator will use a boundary-censored logistic
observation model directly on the canonical interval [0,1].

The distribution has point mass at the two censoring boundaries and continuous
density in the open interval.

For one scalar conditional logistic variable with location `mu` and positive
scale `s`:

- if x = 0:
  use the probability mass `P(Z <= 0)`;
- if 0 < x < 1:
  use the logistic density at x;
- if x = 1:
  use the probability mass `P(Z >= 1)`.

Here `Z` is the uncensored logistic latent variable and the observed canonical
value is:

`X = clip(Z, 0, 1)`

This removes the need for a logit transform and removes endpoint epsilon from
the scientific likelihood.

## RGB channel coupling

The three canonical channels must not be assumed independent.

The governed per-pixel factorization will use channel ordering:

`R -> G -> B`

with conditional locations:

`mu_R = base_R`

`mu_G = base_G + alpha_RG * centered(R)`

`mu_B = base_B + alpha_BR * centered(R) + alpha_BG * centered(G)`

where the coupling coefficients are bounded before use.

This permits genuine RGB dependence for MalNet while remaining valid for the
three grayscale-replicated datasets.

The exact centering and coefficient parameterization must be fixed in the
implementation contract and regression-tested before training.

## PixelCNN output dimensionality

The existing revision PixelCNN predicts nine scalar parameters per spatial
position.

The replacement observation family will retain nine output channels:

- 3 location parameters;
- 3 log-scale parameters;
- 3 RGB coupling parameters.

Therefore the causal PixelCNN backbone does not require an output-dimensionality
increase merely to replace the rejected Gaussian observation model.

The semantic meaning of the nine output channels changes and must be tested
explicitly.

## Sampling semantics

Sampling must occur directly in canonical space.

For each raster-order pixel:

1. sample red from its censored logistic conditional;
2. use sampled red to construct the green conditional location;
3. sample green;
4. use sampled red and green to construct the blue conditional location;
5. sample blue;
6. retain the censored values in [0,1].

No sigmoid inversion or arbitrary endpoint clipping epsilon is part of the
governed sampling definition.

## Reporting boundary

The autoregressive training objective may be reported as the negative
log-likelihood under the governed censored logistic observation model.

Likelihood values must not be compared directly with historical
logit-Gaussian values as though they were measurements from the same
observation model.

Historical values remain provenance only.

## Implementation gate

Before any revised Paper 1 autoregressive experiment is launched, the
replacement implementation must demonstrate:

- finite loss for exact zero observations;
- finite loss for exact one observations;
- finite loss for interior observations;
- no endpoint epsilon parameter;
- stable positive scale parameterization;
- bounded RGB coupling parameters;
- correct RGB conditional ordering;
- spatial causality;
- first-pixel class conditioning;
- deterministic sampling under a fixed seed;
- changed stochastic samples under a changed seed;
- output values restricted to [0,1];
- finite gradients in a training step;
- canonical 40 x 40 x 3 compatibility.

Until those conditions pass, the autoregressive generator remains:

`IMPLEMENTATION UNDER REVISION — TRAINING NOT AUTHORIZED`
