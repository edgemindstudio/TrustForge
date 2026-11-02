# Paper 1 Revision 2026 — Restricted Boltzmann Contrastive-Divergence Policy

`STATUS=SELECTED`

`RBM_CD_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs the revised Paper 1 Restricted Boltzmann Machine
negative-phase estimator.

It selects:

- contrastive-divergence family;
- Gibbs depth;
- positive-phase hidden treatment;
- negative-phase hidden treatment;
- class conditioning during Gibbs transitions;
- stochastic-chain semantics.

It does not yet select:

- optimizer;
- learning rate;
- epoch/update budget;
- final batch policy;
- class-balanced traversal policy;
- checkpoint selection;
- production synthesis burn-in.

## 2. Historical evidence

Historical RBM configuration intended:

`CD_K=5`

Recovered artifact-producing historical execution evidence showed:

`CD_K=1`

because historical nested RBM configuration was not correctly propagated into
the runtime configuration.

Neither historical value is automatically authoritative for the revised
benchmark.

`RBM_HISTORICAL_CD_VALUE_AUTOMATICALLY_REUSED=NO`

## 3. Revised candidates audited

The revision examined fixed-chain contrastive divergence with depths:

- CD-1;
- CD-3;
- CD-5;
- CD-10 for deeper stability comparison.

Audits used the revised:

- bounded continuous visible conditional;
- Bernoulli hidden conditional;
- shared class-conditioned architecture;
- float64 internal arithmetic;
- selected initialization family.

No training parameter updates were performed during these structural audits.

## 4. Initial depth observations

For USTC and MalNet-Image, CD-1, CD-3, and CD-5 update directions were
relatively similar.

For CICMalDroid2020 and MaleX, shallow single-chain probes initially appeared
to show large directional differences between CD depths.

This initially suggested possible depth sensitivity.

However, depth alone was not yet identifiable because each negative phase also
contains stochastic Monte Carlo variation.

## 5. Prefix-coupled audits

The revision therefore compared depths using prefix-coupled Gibbs chains.

Under prefix coupling:

- CD-3 extends the exact first step used by CD-1;
- CD-5 extends the exact first three steps used by CD-3;
- CD-10 extends the exact first five steps used by CD-5.

USTC and MalNet-Image continued to show comparatively high directional
agreement.

CICMalDroid2020 and MaleX continued to show much lower apparent agreement.

This established that the earlier result was not merely caused by using
different first-step random streams.

## 6. Stochastic-variance audit

A further audit separated depth effects from stochastic estimator variance.

For each dataset and fixed batch, independent replicate negative phases were
compared at the same depth.

At batch size 32:

- CICMalDroid2020 showed very low same-depth replicate agreement;
- MaleX showed very low same-depth replicate agreement;
- the magnitude of same-depth disagreement was comparable to the observed
  CD-5 versus CD-10 disagreement.

USTC and MalNet-Image showed much higher same-depth reproducibility.

Therefore the low CICMalDroid2020 and MaleX depth cosines could not be
interpreted as clean evidence that short chains had failed to converge.

`RBM_CD_DEPTH_ONLY_EXPLAINS_OBSERVED_DIRECTIONAL_VARIANCE=NO`

## 7. Production-scale batch audit

The revision then repeated the critical CICMalDroid2020 and MaleX comparison
using:

`B=256`

At this larger batch size, update-estimator agreement improved substantially.

Representative `W` cosine similarities were approximately:

### CICMalDroid2020

- CD-1 replicate versus replicate: `0.66`;
- CD-5 replicate versus replicate: `0.66-0.68`;
- CD-1 versus CD-5 independent streams: `0.67`.

### MaleX

- CD-1 replicate versus replicate: `0.68-0.69`;
- CD-5 replicate versus replicate: `0.68-0.70`;
- CD-1 versus CD-5 independent streams: `0.69`.

The CD-1 versus CD-5 difference was therefore comparable to ordinary
same-depth Monte Carlo variability.

No revision-specific evidence demonstrated that CD-5 produced a materially more
stable update estimator than CD-1 at the production-candidate batch scale.

## 8. Selected contrastive-divergence depth

The revised primary estimator is:

`CD-1`

Therefore:

`RBM_CD_K=1`

The selection is based on revised evidence and computational proportionality.

It is not selected merely because a historical artifact-producing execution
happened to use CD-1.

`RBM_CD1_SELECTION_BASIS=REVISION_EVIDENCE`

## 9. Rejection of deeper primary CD

CD-3 and CD-5 require approximately three and five times the Gibbs transition
work of CD-1 respectively.

The audits did not demonstrate a corresponding reduction in stochastic update
variance or a stable directional improvement sufficient to justify that
additional cost.

CD-10 likewise did not establish a distinct stable estimator for the
high-variance datasets.

Therefore:

`RBM_CD3_PRIMARY=REJECTED`

`RBM_CD5_PRIMARY=REJECTED`

`RBM_CD10_PRIMARY=REJECTED`

This does not claim that deeper chains can never improve a trained RBM.

It states only that the revision evidence does not justify making deeper CD the
primary Paper 1 benchmark estimator.

## 10. Positive phase

For observed canonical visible batch `v0` and clamped class label `y`, the
positive hidden conditional is:

`p(h_j=1 | v0,y) = sigmoid(b_j + W[:,j]^T v0 + U[y,j])`

The positive-phase statistics must use hidden probabilities rather than an
additional Bernoulli hidden sample.

This reduces avoidable Monte Carlo noise in the positive phase.

Therefore:

`RBM_POSITIVE_PHASE_HIDDEN=CONDITIONAL_EXPECTATION`

## 11. Negative Gibbs transition

The negative phase begins from the observed canonical visible batch.

For each CD-1 update:

1. sample hidden state from `p(h | v0,y)`;
2. sample reconstructed visible state from the selected bounded-exponential
   `p(v | h)`;
3. evaluate negative hidden probabilities from `p(h | v1,y)`.

The class label remains clamped throughout the chain.

Therefore:

`RBM_NEGATIVE_CHAIN_INITIALIZATION=DATA`

`RBM_NEGATIVE_CHAIN_CLASS_LABEL=CLAMPED`

`RBM_NEGATIVE_GIBBS_STEPS=1`

## 12. Negative sufficient statistics

The negative visible state is sampled.

At the terminal negative visible state, hidden sufficient statistics must use
conditional hidden probabilities rather than another Bernoulli sample.

Therefore:

`RBM_NEGATIVE_TERMINAL_HIDDEN=CONDITIONAL_EXPECTATION`

This avoids an unnecessary final hidden-sampling source of variance while
preserving the stochastic visible Gibbs transition.

## 13. Parameter update statistics

For batch size `B`, the update estimator must use positive-minus-negative
statistics for:

- `W`;
- `a`;
- `b`;
- `U`.

Conceptually:

`Delta_W ∝ <v h_prob>_positive - <v h_prob>_negative`

`Delta_a ∝ <v>_positive - <v>_negative`

`Delta_b ∝ <h_prob>_positive - <h_prob>_negative`

`Delta_U ∝ <class-row h_prob>_positive - <class-row h_prob>_negative`

Exact optimizer scaling is governed separately.

## 14. Class-conditioning semantics

The class identifier used in the negative chain must be the same governed class
identifier associated with the positive observation.

Changing the class label during the negative chain is forbidden.

Generating an unlabeled negative sample and assigning its class afterward is
forbidden.

`RBM_CD_CLASS_SWITCHING=FORBIDDEN`

`RBM_CD_POSTHOC_CLASS_ASSIGNMENT=FORBIDDEN`

## 15. Persistent CD

Persistent contrastive divergence is not selected for the primary revision.

Introducing persistent chains would add:

- chain-state lifecycle requirements;
- checkpoint-state requirements;
- class-conditioned persistent-state allocation;
- new reproducibility semantics;
- additional interaction with training traversal and class imbalance.

The current evidence does not establish that this complexity is necessary for
Paper 1.

Therefore:

`RBM_PERSISTENT_CD=REJECTED_FOR_PRIMARY_BENCHMARK`

This is a scope decision, not a claim that persistent CD is scientifically
invalid.

## 16. Stochastic variance interpretation

Single-batch CD update direction is a stochastic estimator.

Low cosine agreement between two independently sampled negative phases must not
automatically be interpreted as failure of Gibbs-depth convergence.

Observed difference must be distinguished from:

- Monte Carlo sampling variance;
- batch-composition variance;
- class-composition variance;
- actual systematic depth bias.

`RBM_CD_VARIANCE_MUST_NOT_BE_MISLABELED_AS_DEPTH_BIAS=YES`

## 17. Batch-size dependency

The audits showed materially improved replicate agreement when increasing from
batch size 32 to batch size 256 for CICMalDroid2020 and MaleX.

This policy does not by itself freeze the final training batch size.

However, the training/traversal policy must explicitly account for this
evidence when selecting the production batch construction.

`RBM_CD_BATCH256_VARIANCE_EVIDENCE=PASS`

## 18. Randomness

All Gibbs randomness must be derived from the governed scientific execution
identity.

Randomness must be stateless or otherwise reproducibly derived such that an
identical governed run reproduces the same chain transitions.

Random streams used for:

- hidden sampling;
- visible sampling;

must be distinguishable and deterministically derived.

No ambient untracked random-state dependency is permitted.

## 19. Numerical requirements

Every CD transition must enforce the selected precision policy.

Required properties include:

- float64 internal arithmetic;
- finite hidden probabilities;
- Bernoulli hidden states exactly in `{0,1}`;
- finite visible natural parameters;
- finite bounded-exponential samples;
- visible samples inside `[0,1]`;
- no posthoc clipping;
- finite update statistics.

Any violation must fail closed.

## 20. Implementation gate

Before CD implementation may be accepted, focused evidence must demonstrate at
least:

1. CD depth is exactly one;
2. the negative chain starts from the observed canonical batch;
3. class labels remain clamped;
4. positive hidden statistics use conditional expectations;
5. one Bernoulli hidden transition is sampled;
6. one bounded-exponential visible transition is sampled;
7. terminal negative hidden statistics use conditional expectations;
8. all parameter update statistics are finite;
9. visible samples remain in `[0,1]`;
10. no clipping repair path exists;
11. identical governed seed reproduces identical update statistics;
12. changed experimental seed can change the stochastic transition;
13. persistent chain state is absent from the primary implementation;
14. no historical flat/nested `CD_K` fallback can override the revision policy.

## 21. Current authorization state

The revised RBM negative-phase estimator is selected as:

- data-initialized contrastive divergence;
- one Gibbs step;
- class label clamped;
- positive hidden conditional expectation;
- sampled hidden transition;
- sampled bounded-exponential visible transition;
- terminal negative hidden conditional expectation.

Therefore:

`RBM_NEGATIVE_PHASE=CD1`

`RBM_CD_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
