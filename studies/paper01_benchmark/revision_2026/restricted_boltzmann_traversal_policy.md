# Paper 1 Revision 2026 — Restricted Boltzmann Training Traversal Policy

`STATUS=SELECTED`

`RBM_TRAVERSAL_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs training-example traversal and class exposure for the
revised Paper 1 shared class-conditional Restricted Boltzmann Machine.

It selects:

- training batch size;
- class-sampling distribution;
- within-class sampling;
- nominal epoch exposure;
- handling of class imbalance.

It does not yet select:

- total epoch/update budget;
- validation/checkpoint selection;
- stopping rule;
- production synthesis burn-in.

## 2. Shared-model imbalance problem

The revised RBM uses one shared class-conditional model per dataset and
experimental seed.

Therefore shared parameters `W`, `a`, and `b` receive updates from all classes.

Natural empirical traversal would allow large classes to dominate these shared
parameters.

The effect is especially severe for MalNet-Image.

Exact governed MalNet-Image training counts contain:

- 47 classes;
- 883416 total training examples;
- minimum class count 79;
- median class count 585;
- maximum class count 619118.

The natural maximum-to-minimum class ratio is approximately:

`7836.94`

Therefore:

`RBM_NATURAL_CLASS_TRAVERSAL=REJECTED`

## 3. Full uniform class sampling

Fully uniform class sampling was evaluated and rejected.

For MalNet-Image, uniform class exposure would assign approximately:

`18796`

expected examples per class per nominal epoch.

This would imply approximately:

- `237.9x` exposure for the smallest class;
- `32.1x` exposure for the median class;
- `0.030x` exposure for the largest class.

This is excessive rare-class repetition and excessive dominant-class
undersampling.

Therefore:

`RBM_UNIFORM_CLASS_SAMPLING=REJECTED`

## 4. Square-root balancing

Square-root class weighting:

`p(c) proportional to n_c^0.5`

was also evaluated.

For MalNet-Image this produced approximately:

- smallest-class repeat factor `35.26x`;
- median repeat factor `12.96x`;
- largest-class retention `0.398x`.

This remains too aggressive for the primary benchmark.

Therefore:

`RBM_SQRT_CLASS_SAMPLING=REJECTED`

## 5. Median-cap balancing

Median-capped natural exposure was also evaluated.

For MalNet-Image it produced approximately:

- smallest-class repeat factor `45.54x`;
- median repeat factor `45.54x`;
- largest-class retention `0.043x`.

This also over-corrects the empirical distribution.

Therefore:

`RBM_MEDIAN_CAP_CLASS_SAMPLING=REJECTED`

## 6. Power-law balancing family

The revision evaluated class probabilities proportional to:

`n_c^alpha`

for candidate exponents between:

`alpha=0.50`

and:

`alpha=1.00`

where:

- `alpha=1.00` reproduces natural empirical sampling;
- smaller alpha values progressively increase class balancing.

The selected exponent is:

`alpha=0.85`

Therefore:

`RBM_CLASS_BALANCE_EXPONENT=0.85`

## 7. MalNet-Image evidence

Under `alpha=0.85`, exact MalNet-Image arithmetic gives approximately:

- smallest-class repeat factor: `3.288645`;
- median repeat factor: `2.435497`;
- largest-class retention: `0.856831`;
- exposure maximum-to-minimum ratio: `2041.85`.

This materially reduces dominant-class control while avoiding the pathological
rare-class repetition produced by uniform or square-root sampling.

## 8. Cross-dataset evidence

The same `alpha=0.85` rule behaves conservatively across all four benchmark
datasets.

### USTC-TFC2016

Because all nine training classes contain 8100 examples:

- repeat factor remains exactly `1.0`;
- exposure remains identical to natural traversal.

### CICMalDroid2020

Observed repeat-factor range is approximately:

`0.933` to `1.107`

Thus only mild balancing is introduced.

### MalNet-Image

Observed repeat-factor range is approximately:

`0.857` to `3.289`

The extreme natural imbalance is moderated without flattening the distribution.

### MaleX

Observed repeat-factor range is approximately:

`0.956` to `1.210`

Thus only mild balancing is introduced.

Therefore:

`RBM_CLASS_BALANCE_CROSS_DATASET_AUDIT=PASS`

## 9. Selected class probability

For training class `c` with training count `n_c`, define:

`w_c = n_c^0.85`

and:

`p_c = w_c / sum_j(w_j)`

The class used for each sampled training example must be drawn according to
`p_c`.

Therefore:

`RBM_CLASS_SAMPLING=POWER_LAW`

`RBM_CLASS_SAMPLING_PROBABILITY=n_c^0.85_NORMALIZED`

## 10. Within-class example sampling

After a class is selected, the training example must be selected uniformly from
that class's governed training split.

Therefore:

`RBM_WITHIN_CLASS_SAMPLING=UNIFORM`

Sampling with replacement is permitted and required where repeated exposure is
implied by the governed class distribution.

No validation or test example may enter the training sampler.

`RBM_TRAINING_SPLIT_ONLY=YES`

`RBM_VALIDATION_IN_TRAINING_SAMPLER=FORBIDDEN`

`RBM_TEST_IN_TRAINING_SAMPLER=FORBIDDEN`

## 11. Batch size

The selected training batch size is:

`256`

Therefore:

`RBM_BATCH_SIZE=256`

This selection is consistent with the CD variance audit, where batch size 256
substantially improved negative-phase estimator reproducibility relative to
batch size 32.

The batch size is uniform across all four datasets.

`RBM_DATASET_SPECIFIC_BATCH_SIZE=FORBIDDEN`

## 12. Batch construction

Each batch contains exactly 256 independently sampled training examples under
the governed class-sampling process, except where implementation-level terminal
batch semantics are rendered unnecessary by fixed-size sampled epochs.

The implementation must not force every class to appear in every batch.

Such per-batch uniform class quotas would reintroduce the rejected full-uniform
class objective.

Therefore:

`RBM_PER_BATCH_ALL_CLASS_QUOTA=FORBIDDEN`

## 13. Nominal epoch size

For dataset training size `N_train`, one nominal RBM epoch contains exactly:

`N_train`

sampled training-example draws.

Thus expected total exposure per epoch is held equal to one natural empirical
pass even though class composition is reweighted.

Therefore:

`RBM_NOMINAL_EPOCH_EXPOSURE=N_train`

`RBM_EPOCH_SIZE_PRESERVES_TOTAL_TRAIN_EXPOSURE=YES`

This keeps epoch-scale compute comparable to a natural pass while controlling
class dominance.

## 14. Exposure semantics

An epoch does not mean that every unique training example is visited exactly
once.

Under governed rebalanced sampling:

- some examples may be selected more than once;
- some examples may not be selected during a given epoch;
- expected class exposure follows `n_c^0.85`.

Therefore:

`RBM_EPOCH_IS_UNIQUE_FULL_PASS=NO`

## 15. Rare-class protection

Rare classes must not be expanded by physically duplicating files or creating
new training artifacts.

Repeated exposure exists only as sampler behavior.

Therefore:

`RBM_PHYSICAL_DATA_DUPLICATION=FORBIDDEN`

No synthetic samples may be inserted into RBM training to balance classes.

`RBM_SYNTHETIC_TRAIN_BALANCING=FORBIDDEN`

## 16. Dominant-class protection

No single class may revert to natural empirical dominance through an
implementation shortcut.

The exact governed probability vector must be derived from the authoritative
training counts and exponent `0.85`.

Hard-coded MalNet-specific probabilities are forbidden.

Therefore:

`RBM_CLASS_PROBABILITIES_DERIVED_FROM_TRAIN_COUNTS=REQUIRED`

## 17. Reproducibility

Class selection and within-class example selection must be deterministically
derived from the governed scientific execution identity.

Identical dataset, experimental seed, and training position must reproduce the
same sampled traversal.

Ambient process-global randomness is not authoritative.

Therefore:

`RBM_TRAVERSAL_RANDOMNESS=GOVERNED`

## 18. Dataset isolation

Class counts must be computed independently for each dataset's governed
training split.

No class-frequency information may be shared across datasets.

Therefore:

`RBM_CROSS_DATASET_TRAVERSAL_SHARING=FORBIDDEN`

## 19. Validation and test distributions

Validation and test evaluation must retain their governed empirical split
distributions.

The `alpha=0.85` training rebalance must not alter validation or test sampling
semantics.

Therefore:

`RBM_VALIDATION_CLASS_REBALANCING=FORBIDDEN`

`RBM_TEST_CLASS_REBALANCING=FORBIDDEN`

## 20. Implementation gate

Before traversal implementation may be accepted, focused evidence must
demonstrate at least:

1. batch size is exactly 256;
2. class probabilities are proportional to `n_c^0.85`;
3. probabilities are derived from training counts only;
4. probabilities sum to one within numerical tolerance;
5. within-class example sampling is uniform;
6. nominal epoch draw count equals `N_train`;
7. validation examples cannot enter the training sampler;
8. test examples cannot enter the training sampler;
9. no physical data duplication occurs;
10. no synthetic balancing examples are introduced;
11. USTC remains exactly balanced;
12. CIC receives only mild rebalancing;
13. MalNet-Image rare classes are not uniformly oversampled;
14. MaleX receives only mild rebalancing;
15. identical governed seed reproduces identical traversal;
16. changed experimental seed can change traversal;
17. no historical per-class independent-model traversal path can override this
    policy.

## 21. Current authorization state

The revised RBM traversal is selected as:

- shared-model class-aware sampling;
- power-law exponent `0.85`;
- uniform sampling within selected class;
- batch size `256`;
- nominal epoch size equal to `N_train`;
- governed reproducible traversal;
- validation/test empirical distributions preserved.

Therefore:

`RBM_TRAVERSAL_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
