# Paper 1 Revision 2026 — Gaussian Mixture Scale Policy

`STATUS=SELECTED`

`GMM_SCALE_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs variance-collapse prevention and scale parameterization
for the revised Paper 1 class-conditional censored diagonal Gaussian mixture
model.

It does not authorize real dataset training.

Initialization, optimizer configuration, minibatch policy, stopping
criterion, and final implementation acceptance remain separately governed.

## 2. Historical evidence

The historical Paper 1 diagonal GMM configurations used:

`reg_covar=1e-3`

A direct audit of the fitted historical Paper 1 checkpoints demonstrated
that this regularization magnitude was materially active.

### USTC

Across the audited diagonal covariance parameters:

- approximately 62.87% were effectively equal to `1e-3`;
- approximately 63.33% were within 1% of `1e-3`.

The median fitted variance was `1e-3`.

### CICMalDroid2020

Across the audited diagonal covariance parameters:

- approximately 47.91% were effectively equal to `1e-3`;
- approximately 49.29% were within 1% of `1e-3`.

The median fitted variance was approximately `0.001025`.

The historical regularization therefore functioned as an active
variance-collapse control rather than merely an unused numerical safeguard.

## 3. Revised parameterization requirement

The revised censored GMM uses differentiable optimization.

Component standard deviations must therefore be represented through a
strictly positive, explicitly lower-bounded parameterization.

The selected parameterization is:

`scale = GMM_MIN_SCALE + softplus(raw_scale)`

where:

`GMM_MIN_VARIANCE=1e-3`

and:

`GMM_MIN_SCALE=sqrt(1e-3)`

Numerically:

`GMM_MIN_SCALE=0.03162277660168379`

The resulting component variance is:

`variance = scale^2`

and therefore always satisfies:

`variance > 1e-3`

for finite trainable `raw_scale`.

The historical `1e-3` magnitude is retained as the revision regularization
reference, while the differentiable softplus parameterization replaces the
historical sklearn covariance-floor mechanism.

## 4. Candidate-floor prototype evidence

The revision prototype evaluated the following candidate standard-deviation
floors:

- `0.01`;
- `sqrt(1e-3) ~= 0.0316227766`;
- `0.05`.

All three candidate parameterizations produced:

- positive scales;
- finite scales;
- finite log scales;
- finite gradients.

No numerical requirement therefore forced selection of a floor different
from the historical reference magnitude.

## 5. Selection rationale

### 5.1 Floor 0.01

Not selected.

It would permit substantially sharper components than the historical Paper 1
regularization regime without evidence that such additional concentration is
required.

### 5.2 Floor 0.05

Not selected.

It would impose stronger smoothing than the historical Paper 1
regularization regime without evidence that the larger minimum scale is
required.

### 5.3 Floor sqrt(1e-3)

Selected.

It preserves continuity with the historical regularization magnitude while
using a differentiable parameterization suitable for the revised censored
likelihood.

It also protects the 4800-dimensional revised model from arbitrarily
collapsing individual diagonal scales toward zero.

## 6. Optimization semantics

The trainable quantity is `raw_scale`.

Production code must not optimize an unconstrained standard deviation
directly.

Production code must not use:

`scale = exp(raw_scale)`

without an independent governed lower bound.

Production code must not permit:

- zero scale;
- negative scale;
- sub-policy scale;
- non-finite scale.

The implementation must fail closed if non-finite scale values occur.

## 7. Relationship to censored likelihood

The scale floor applies to the latent diagonal Gaussian component used by
the censored observation model.

For each scalar dimension:

- latent values at or below 0 map to the endpoint atom at 0;
- latent values strictly between 0 and 1 remain continuous observations;
- latent values at or above 1 map to the endpoint atom at 1.

The scale floor therefore regulates the latent Gaussian distribution.

It is not an image-space clipping threshold.

## 8. Current governed constants

`GMM_MIN_VARIANCE=0.001`

`GMM_MIN_SCALE=0.03162277660168379`

`GMM_SCALE_PARAMETERIZATION=SOFTPLUS_PLUS_FLOOR`

`GMM_SCALE_FLOOR_HISTORICAL_CONTINUITY=YES`

`GMM_SCALE_POLICY_SELECTED=YES`

## 9. Authorization state

The scale policy is selected.

The complete training procedure is not yet governed or accepted.

Therefore:

`GMM_IMPLEMENTATION_AUTHORIZED=NO`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
