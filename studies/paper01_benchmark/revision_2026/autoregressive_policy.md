# Paper 1 Revision 2026 — Autoregressive Generator Policy

Status: GOVERNED
Scope: Revised Paper 1 benchmark only
Family: Conditional PixelCNN

## 1. Family preservation

The revised autoregressive generator remains a conditional PixelCNN-style
masked-convolution autoregressive model.

The historical Bernoulli observation model is not retained because revised
Paper 1 uses a continuous 40x40x3 float32 representation in [0,1].

Historical implementations and artifacts remain preserved as provenance.

## 2. Autoregressive unit

The autoregressive factorization operates over raster-ordered spatial pixels.

Each spatial position is modeled as one three-dimensional RGB random vector.

For pixel position t:

    p(x_t | x_<t, y)

where:

- x_t is the RGB vector at the current pixel;
- x_<t contains all preceding raster-order pixels;
- y is the class label.

The three RGB channels at the same pixel are modeled jointly rather than
assigned an artificial within-pixel channel ordering.

## 3. Canonical input

Input to the revised model is the governed canonical representation:

- shape: 40x40x3
- layout: NHWC
- dtype: float32
- range: [0,1]

Before likelihood evaluation, values are deterministically clamped away from
the exact endpoints and transformed into logit space.

Let:

    x_clamped = clip(x, epsilon, 1 - epsilon)
    z = log(x_clamped / (1 - x_clamped))

The exact epsilon value is an execution parameter and must be recorded in the
revised runtime contract.

## 4. Conditional observation model

For every spatial pixel, the PixelCNN predicts a full three-dimensional
Gaussian distribution in logit space.

The output head produces nine parameters per pixel:

- 3 conditional means;
- 3 log diagonal scale parameters;
- 3 strict lower-triangular parameters.

These parameters define a lower-triangular Cholesky factor L for the
conditional RGB covariance.

The diagonal elements of L must be positive and numerically bounded through
an explicit governed parameterization.

The resulting conditional model is:

    z_t ~ Normal(mu_t, L_t L_t^T)

where z_t is the logit-transformed RGB vector for the current pixel.

## 5. Training objective

The historical BinaryCrossentropy objective is not used for revised Paper 1.

Training minimizes the mean conditional Gaussian negative log-likelihood in
logit space over all pixels and samples.

The implementation must:

- use the full three-dimensional conditional covariance;
- avoid independent Bernoulli channel losses;
- avoid binary target conversion;
- fail on non-finite likelihood values;
- record the actual consumed likelihood configuration.

Because the logit transform is deterministic and independent of model
parameters, optimization may be performed in transformed space.

If likelihood values are ever reported as densities in canonical image
space, the appropriate transformation Jacobian must also be included.

## 6. Class conditioning

Class conditioning must not be treated as another causally masked image
channel.

The class label is supplied as an unmasked conditioning signal to the hidden
PixelCNN representation.

The revised implementation must ensure that every spatial prediction,
including the first raster-order pixel, is conditioned on the requested
class.

An acceptable implementation may project the one-hot class label into hidden
feature biases or embeddings that are broadcast spatially and injected into
the masked-convolution stack.

## 7. Causality

The image pathway must preserve PixelCNN raster-order causality.

A prediction for pixel t must not depend on the observed value of pixel t or
any future pixel.

The revised implementation must include tests demonstrating that modifying a
future pixel cannot change an earlier conditional prediction.

Class-conditioning signals are exempt from the image causality mask because
the class label is known conditioning information.

## 8. Sampling

Sampling proceeds sequentially over spatial pixels and is vectorized over the
sample batch.

For each pixel:

1. evaluate the conditional distribution parameters;
2. construct the three-dimensional Cholesky factor L;
3. draw epsilon ~ Normal(0,I);
4. compute

       z_t = mu_t + temperature * L_t * epsilon

5. transform back to canonical image space:

       x_t = sigmoid(z_t)

The completed synthetic image must be returned as:

- shape: 40x40x3
- dtype: float32
- range: [0,1]

No Bernoulli thresholding is permitted.

## 9. Sampling temperature

Temperature is an active revised runtime control.

Temperature scales conditional stochastic variation during sampling:

    L_effective = temperature * L

The primary revised benchmark temperature must be fixed in advance and
recorded in the execution contract.

A temperature value of 1.0 represents sampling from the learned conditional
distribution without variance rescaling.

## 10. Removed historical controls

The revised continuous Gaussian PixelCNN does not use categorical-token
sampling controls.

The following historical controls are therefore not active revised controls:

- TOP_K
- TOP_P

They must not appear in the revised execution contract as consumed runtime
hyperparameters.

Historical records containing those fields remain unchanged.

## 11. Output channels

For canonical RGB data:

    C = 3

and the revised probabilistic output head therefore has:

    9 output channels

corresponding to:

    3 means
    3 log diagonal scales
    3 lower-triangular covariance parameters

The historical one-channel sigmoid probability head is not used.

## 12. Runtime-control integrity

The revised execution contract must record only controls actually consumed by
the implementation.

At minimum, the autoregressive runtime record must identify:

- image shape;
- number of classes;
- masked-convolution width;
- masked-convolution depth;
- optimizer;
- learning rate;
- batch size;
- training stopping rule;
- logit epsilon;
- covariance parameter bounds;
- sampling temperature;
- model seed.

Unused historical metadata must not be presented as active revised controls.

## 13. Revised compatibility status

Conditional PixelCNN:

- canonical 40x40x3 support: GOVERNED
- continuous observation model: GOVERNED
- correlated same-pixel RGB likelihood: GOVERNED
- unmasked class conditioning: GOVERNED
- Bernoulli/BCE likelihood: REJECTED FOR REVISION 2026
- binary sampling: REJECTED FOR REVISION 2026
- TOP_K / TOP_P sampling: NOT APPLICABLE
- implementation status: PENDING
- validation status: PENDING
