# Paper 1 Revision 2026 — Conditional GAN Class-Sampling Policy

## Status

PROVISIONAL SCIENTIFIC SELECTION — IMPLEMENTATION NOT AUTHORIZED

## 1. Scope

This policy governs real-training-example selection for the revised
shared class-conditional DCGAN across four Paper 1 datasets.

It does not govern synthetic-generation class quotas, validation sampling,
or held-out test evaluation.

## 2. Authoritative training data

Use only the governed revision-2026 training splits.

Class counts must be derived independently for each dataset using its
authoritative numerical class mapping and training-split membership.

No validation or test examples may enter the GAN training sampler.

## 3. Class probability

For training class c with positive count n_c, define:

    w_c = n_c ** alpha
    p_c = w_c / sum_j(w_j)

The provisional primary exponent is:

    alpha = 0.85

The exponent applies consistently across datasets.

After sampling class c, select one governed training example uniformly
from that class, with replacement.

## 4. Analytical evidence

MalNet-Image has 47 classes and 883416 governed training examples.

Its minimum, median, and maximum class counts are:

    79, 585, 619118

For one nominal pass of 883416 sampled draws, calculated
per-example expected exposure factors are:

| Alpha | Smallest class | Median class | Largest class |
|---|---:|---:|---:|
| 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 0.85 | 3.288645 | 2.435497 | 0.856831 |
| 0.70 | 9.929768 | 5.446035 | 0.674055 |

The prepared training index contains 883417 offsets.

These are analytical expected-exposure calculations,
not observed GAN training or sample-quality results.

## 5. Selection rationale

Natural sampling preserves extreme dominant-class exposure.

Alpha 0.70 introduces substantially stronger rare-class repetition.

Alpha 0.85 provides comparatively mild rebalancing and has
supporting cross-dataset exposure analysis in the existing
revision-2026 RBM traversal investigation.

RBM optimization results are not treated as GAN performance evidence.

This selection must not be retrospectively tuned against final
GAN benchmark metrics.

## 6. Nominal exposure and budget

One nominal exposure pass is N_train independently sampled
real-example draws under the governed class distribution.

A nominal pass does not guarantee that every unique example
is visited exactly once.

The number of nominal passes and maximum GAN optimizer updates
remain separately governed decisions.

The 2000-per-class synthetic-generation budget does not
determine the real-data training-sampling distribution.

## 7. Reproducibility and isolation

Required implementation properties:

- Exact probabilities derived from positive train-only counts.
- Explicit and stable class-ID mapping.
- Reproducible class and within-class sampling.
- No physical duplication of rare-class files.
- No synthetic balancing examples in real GAN training.
- No implicit per-batch all-class quota.
- Original validation and test distributions preserved.
- No silent substitution of natural or uniform sampling.
- Effective sampling parameters recorded in run metadata.

## 8. Remaining scientific limitation

Exposure arithmetic does not establish GAN convergence,
minority-class fidelity, discriminator generalization,
or optimal synthetic-data utility.

The selected exponent is a predeclared methodological choice
whose limitations must be reported.

## 9. Authorization state

GAN_CLASS_SAMPLING_PROVISIONAL_SELECTION=POWER_LAW_0.85
GAN_CLASS_SAMPLING_ANALYTICAL_AUDIT=COMPLETE
GAN_CLASS_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO
GAN_TRAINING_BUDGET_GOVERNED=NO
GAN_CHECKPOINT_SELECTION_GOVERNED=NO
GAN_PRODUCTION_TRAINING_AUTHORIZED=NO
