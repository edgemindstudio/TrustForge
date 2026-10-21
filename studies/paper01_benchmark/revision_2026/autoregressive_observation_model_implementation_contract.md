# Autoregressive Observation-Model Implementation Contract
## Paper 1 Revision 2026

## Status

GOVERNED IMPLEMENTATION CONTRACT.

This contract specializes:

`autoregressive_observation_model_policy.md`

It fixes the exact numerical parameterization that must be implemented and
regression-tested before revised Paper 1 autoregressive training is authorized.

## Observation family

For every raster-order spatial location, the PixelCNN predicts a
channel-coupled boundary-censored logistic distribution directly on the
canonical interval:

`[0,1]`

The latent scalar variable is logistic.

The observed scalar variable is:

`X = clip(Z, 0, 1)`

Therefore:

- exact `0` is an atom with probability `P(Z <= 0)`;
- exact `1` is an atom with probability `P(Z >= 1)`;
- values strictly between `0` and `1` use the logistic density.

The resulting distribution is mixed discrete/continuous and is interpreted
with respect to a reference measure containing Lebesgue measure on `(0,1)`
plus atoms at `0` and `1`.

No logit transform of observations is used.

No endpoint epsilon occurs in the likelihood definition.

## PixelCNN output parameter ordering

The existing nine PixelCNN output channels are retained.

Their governed semantic order is:

1. `base_mu_r`
2. `base_mu_g`
3. `base_mu_b`
4. `raw_log_scale_r`
5. `raw_log_scale_g`
6. `raw_log_scale_b`
7. `raw_alpha_rg`
8. `raw_alpha_br`
9. `raw_alpha_bg`

The nine channels no longer represent a Gaussian mean plus Cholesky factor.

## Scale parameterization

For each channel:

`log_scale = clip(raw_log_scale, min_log_scale, max_log_scale)`

and:

`scale = exp(log_scale)`

The initial governed limits remain:

`min_log_scale = -7.0`

`max_log_scale = 3.0`

These limits are numerical scale bounds.

They are not endpoint clipping parameters and do not alter the observed
canonical pixel value.

Every effective scale is therefore strictly positive and finite.

## RGB coupling parameterization

Raw coupling outputs are bounded using:

`alpha = tanh(raw_alpha)`

Therefore each effective coupling coefficient lies in:

`(-1,1)`

The governed channel order is:

`R -> G -> B`

The already-observed canonical channels are centered as:

`centered(x) = x - 0.5`

The effective conditional locations are:

`mu_r = base_mu_r`

`mu_g = base_mu_g + alpha_rg * (r - 0.5)`

`mu_b = base_mu_b
        + alpha_br * (r - 0.5)
        + alpha_bg * (g - 0.5)`

During likelihood evaluation, `r` and `g` are the observed canonical target
channels for the same spatial pixel.

During sampling, they are the already-generated red and green channels for
that spatial pixel.

No future spatial value or future channel may enter an earlier conditional.

## Scalar censored-logistic log probability

For an effective location `mu`, positive scale `s`, observation `x`, and:

`z = (x - mu) / s`

the interior logistic log-density is evaluated stably as:

`log_pdf(x) =
    -softplus(z)
    -softplus(-z)
    -log(s)`

For an exact lower-boundary observation:

`x = 0`

the log probability is:

`log P(X = 0) =
    log sigmoid((0 - mu) / s)`

implemented using a numerically stable log-sigmoid operation.

For an exact upper-boundary observation:

`x = 1`

the log probability is:

`log P(X = 1) =
    log sigmoid((mu - 1) / s)`

implemented using a numerically stable log-sigmoid operation.

For:

`0 < x < 1`

the interior logistic log-density is used.

Boundary classification is exact because canonicalization explicitly permits
exact `0.0` and `1.0`, and the benchmark audit measured those exact values.

No tolerance-based endpoint reassignment is authorized in the likelihood.

## RGB pixel log probability

The governed factorization for one RGB pixel is:

`p(r,g,b | spatial_context, class)
 = p(r | context, class)
   p(g | r, context, class)
   p(b | r,g, context, class)`

The per-pixel log probability is therefore the sum of the three scalar channel
log probabilities.

The training objective is the negative mean RGB-pixel log probability over
the batch and spatial dimensions.

It must remain finite for valid canonical data.

## Sampling

Sampling is performed sequentially within every raster-order pixel:

1. construct red location and scale;
2. sample latent red logistic value;
3. censor red with `clip(z_r, 0, 1)`;
4. construct green location using sampled red;
5. sample and censor green;
6. construct blue location using sampled red and sampled green;
7. sample and censor blue.

For one scalar logistic draw, inverse-CDF sampling is used:

`latent = mu + scale * (log(u) - log1p(-u))`

where `u` is drawn from a stateless uniform random generator.

The implementation may use machine-representable open-interval protection
needed to prevent evaluating `log(0)`.

Such protection is a numerical RNG safeguard only.

It must:

- not alter the training likelihood;
- not introduce an observation-space endpoint epsilon;
- not replace boundary censoring;
- be derived from floating-point machine limits rather than an experimental
  hyperparameter.

The observed sample is finally:

`clip(latent, 0, 1)`

Exact `0` and `1` samples are therefore legitimate outputs of the governed
model.

## Temperature

For stochastic generation, temperature acts on the logistic scale:

`effective_scale = temperature * scale`

For:

`temperature > 0`

sampling uses the scaled logistic distribution.

For:

`temperature = 0`

sampling is deterministic and returns:

`clip(mu, 0, 1)`

with the RGB conditional order still respected:

- deterministic red is generated first;
- deterministic green location uses deterministic red;
- deterministic blue location uses deterministic red and green.

Temperature must never modify training likelihood evaluation.

## Causality boundary

The existing spatial PixelCNN masking remains governed:

- mask A excludes the current and future spatial positions;
- mask B may consume the causal hidden representation while preserving spatial
  causality;
- class conditioning is externally available at every spatial position,
  including the first pixel.

Within a pixel, channel dependence is permitted only in the governed order:

`R -> G -> B`

This channel coupling is part of the output distribution and does not weaken
spatial causality.

## Required implementation replacements

The revision-specific implementation must retire use of these primitives from
the active AR training/sampling path:

- `canonical_to_logit`
- `gaussian_rgb_nll`
- Gaussian Cholesky parameter interpretation
- sigmoid-of-Gaussian sampling

Historical code and evidence remain untouched.

The replacement active primitives must provide the conceptual equivalents of:

- parameter splitting for censored logistic locations, scales, and couplings;
- stable scalar censored-logistic log probability;
- coupled RGB censored-logistic NLL;
- coupled RGB sampling;
- raster-order PixelCNN sampling using the new observation model.

## Regression requirements

The revised regression suite must explicitly demonstrate:

1. output tensor remains `(B,H,W,9)`;
2. scale parameters are positive and finite;
3. effective coupling coefficients are bounded by magnitude below or equal to
   one;
4. exact zero produces finite NLL;
5. exact one produces finite NLL;
6. interior values produce finite NLL;
7. mixed zero/interior/one batches produce finite NLL;
8. the loss API has no `epsilon` argument;
9. green likelihood changes when red changes while network parameters are held
   fixed and coupling is nonzero;
10. blue likelihood changes when red or green changes under nonzero coupling;
11. spatial causality remains intact;
12. first-pixel class conditioning remains intact;
13. fixed-seed stochastic sampling is exactly reproducible;
14. changed seed changes stochastic samples;
15. temperature zero is deterministic and seed-independent;
16. all generated samples are finite and lie in `[0,1]`;
17. a training step yields finite loss and finite gradients;
18. canonical 40 x 40 x 3 model-facing compatibility is preserved.

## Training authorization

Implementation and regression evidence have satisfied the governed
observation-model implementation gate.

Validated evidence includes:

- censored-logistic primitive regression;
- boundary and interior finite likelihoods;
- no endpoint-epsilon API;
- bounded RGB coupling;
- governed R -> G -> B likelihood dependence;
- finite-gradient validation;
- PixelCNN spatial-causality preservation;
- first-pixel class-conditioning preservation;
- fixed-seed stochastic reproducibility;
- changed-seed stochastic variation;
- temperature-zero determinism;
- canonical [0,1] output bounds;
- real optimizer-step validation;
- canonical 40 x 40 x 3 compatibility.

The dedicated autoregressive implementation-contract suite passed:

`17 passed`

The broader focused Paper 1 revision suite passed before final gap closure:

`68 passed`

The autoregressive implementation is therefore authorized for governed
revision-2026 Paper 1 experiments.

`AUTOREGRESSIVE_TRAINING_AUTHORIZED=YES`

This authorization applies only to the revised 2026 censored-logistic
autoregressive implementation and does not retroactively validate historical
logit-Gaussian experiments or artifacts.

Experiment execution remains subject to the separate governed controls for
dataset identity, partition identity, experimental seed, configuration,
checkpointing, artifact provenance, and reporting.
