# CICMalDroid2020 RBM class-conditioning diagnostic — seed 42

## Status and scope

This document records a diagnostic investigation of the authoritative
Paper 1 revision-2026 CICMalDroid2020 RBM seed-42 checkpoint.

The investigation is diagnostic only. It does not authorize changes to
the governed training configuration, checkpoint, sampling policy, or
production Gibbs-chain depth.

## Authoritative evidence

- Dataset: `cicmaldroid2020`
- Experimental seed: `42`
- Governed optimizer updates: `64`
- Learning rate: `0.0005`
- Authoritative checkpoint state SHA-256:
  `10f64703515290ea1493b731bb8e941f7b200c6ad18a61acfc8a1be8bd17cd72`
- Gradient evidence:
  `rbm_gradient_diagnostic_cicmaldroid2020_seed42.json`
- Gradient evidence SHA-256:
  `5f3f94df37f561cdefea332c1322c7a2c2f69e5e2bab8973626649e865616d69`
- Successful diagnostic replay: Slurm job `308178`, compute node `talon11`
- Earlier failed diagnostic attempt: job `308177`, caused by a
  diagnostic validation-serialization error; no accepted artifact was
  produced by that attempt.

The successful replay reproduced the authoritative `W`, `a`, `b`, and
`U` parameters with exact array equality, and reproduced the complete
checkpoint validation history exactly.

## Gradient observations

Across 64 updates, the observed full `delta_U` norm had:

- Mean: `0.0636997709`
- Minimum: `0.0540750427`
- Maximum: `0.0809463317`
- First update: `0.0626204967`
- Final update: `0.0682628302`

Directional retention was computed for each class as the norm of its
signed cumulative gradient divided by the sum of per-update gradient
norms.

| Class | Directional retention |
| --- | ---: |
| 0 | 0.4507494777 |
| 1 | 0.7444184728 |
| 2 | 0.8174309972 |
| 3 | 0.8247948031 |
| 4 | 0.8310994613 |

Class 0 exhibited greater directional cancellation than classes 1–4.
The evidence does not support describing all five class gradients as
severely canceled.

## Parameter-update identity

The class-conditioning matrix was initialized to zero. Under the
governed manual-SGD update rule, its class rows satisfy:

`U[64,c] = 0.0005 * sum_t(delta_U[t,c])`

The predicted and observed final row norms agreed for all five
classes to floating-point precision.

| Class | Final U-row L2 norm |
| --- | ---: |
| 0 | 0.000199104364836 |
| 1 | 0.000586335226604 |
| 2 | 0.00104340167216 |
| 3 | 0.000796954538443 |
| 4 | 0.000692514094631 |

These observations establish nonzero class-conditioning gradients and
account quantitatively for the learned class-conditioning parameters
under the actual training schedule.

## Class separation during training

Maximum pairwise class-row distances were:

| Update | Maximum pairwise U-row L2 distance |
| --- | ---: |
| 1 | 0.0000277800793 |
| 16 | 0.000370282169 |
| 32 | 0.000750373052 |
| 48 | 0.00115908547 |
| 64 | 0.00165309989 |

The sampled trajectory showed increasing class separation, without
evidence of a plateau at the sampled update boundaries. This does not
prove that extending training would improve synthetic sample quality.

## Final authoritative parameter scales

| Group | L2 norm | RMS magnitude |
| --- | ---: | ---: |
| W | 7.83916229223 | 0.010001014389 |
| a | 25.0413369663 | 0.361440565959 |
| b | 0.00136733985937 | 0.000120856910844 |
| U | 0.00160836280209 | 0.000063576121981 |

Both hidden-bias and class-conditioning parameter magnitudes are small
relative to other parameter groups. Cross-group comparisons are
descriptive because dimensions, initialization, and parameter roles
differ.

## Functional conditioning

The authoritative model computes hidden probabilities using:

`p(h=1 | v,y) = sigmoid(vW + b + U[y,:])`

Holding the visible input constant and changing only the class label
produced measurable but small differences.

| Constant visible input | Maximum pairwise probability L2 difference | Maximum single-unit difference |
| --- | ---: | ---: |
| 0.0 | 0.000413274968984 | 0.00011595879426 |
| 0.5 | 0.000396340104851 | 0.00011587044198 |
| 1.0 | 0.000360555261496 | 0.000115603373643 |

The class-logit identity was verified for the `0.5` probe. The three
controlled-input probes satisfied the sigmoid derivative bound:

`|sigmoid(z + U[c,h]) - sigmoid(z + U[d,h])|
 <= 0.25 * |U[c,h] - U[d,h]|`

This bounds the immediate hidden-probability effect of class changes
for any fixed valid visible input. It does not determine the effect
after multiple stochastic Gibbs transitions.

## Bounded interpretation

The evidence establishes that CIC class-conditioning gradients are
nonzero, that their accumulated updates explain the learned `U`
parameters, and that the immediate conditional hidden-probability
response is small for the measured inputs.

The short governed 64-update schedule and class-dependent directional
cancellation are plausible contributors to the small final `U`.
The investigation does not establish that the schedule is inadequate,
that longer training would improve generation, or that generated
samples are class-independent.

## Outstanding decision

Production Gibbs-chain depth remains unselected. Sampling behavior
and class-conditioned generative effects must be assessed under the
separate governed chain-audit protocol. This record does not revise
the authoritative checkpoint or benchmark policy.
