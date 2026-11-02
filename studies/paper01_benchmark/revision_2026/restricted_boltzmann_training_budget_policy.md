# Paper 1 Revision 2026 — Restricted Boltzmann Training-Budget, Validation, and Checkpoint-Selection Policy

`STATUS=SELECTED`

`RBM_TRAINING_BUDGET_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs:

- maximum optimizer-update budget;
- validation cadence;
- validation semantics;
- early-stopping behavior;
- authoritative checkpoint selection.

It does not yet govern:

- checkpoint file format;
- production checkpoint persistence;
- synthesis burn-in;
- synthesis initialization;
- final production implementation acceptance.

## 2. Historical behavior is not authoritative

Historical RBM execution used epoch-based training and reconstruction-loss
monitoring.

Historical code included:

- patience-based early stopping;
- validation reconstruction loss;
- `RBM_best.weights.h5`;
- `RBM_last.weights.h5`.

The historical implementation was Bernoulli-visible and is not methodologically
authoritative for the revised bounded continuous-visible RBM.

Therefore:

`RBM_HISTORICAL_MSE_CHECKPOINT_SELECTION=REJECTED`

`RBM_HISTORICAL_PATIENCE_EARLY_STOPPING=REJECTED`

## 3. Why a universal fixed update count is rejected

The four revised training splits differ substantially in size:

- USTC-TFC2016: `72900`;
- CICMalDroid2020: `8117`;
- MalNet-Image: `883416`;
- MaleX: `731049`.

With batch size `256`, one fixed optimizer-update count would correspond to
radically different effective data exposure.

For example, `5000` updates would represent approximately:

- `17.56` natural passes for USTC;
- `157.69` natural passes for CICMalDroid;
- `1.45` natural passes for MalNet-Image;
- `1.75` natural passes for MaleX.

Therefore:

`RBM_UNIVERSAL_FIXED_UPDATE_COUNT=NO`

## 4. Governed batch geometry

The traversal policy fixes:

`RBM_BATCH_SIZE=256`

For each dataset define:

`B = ceil(N_train / 256)`

where `B` is the optimizer-update count corresponding to one nominal
training-set exposure under the governed sampled traversal.

Therefore:

`RBM_TRAINING_BUDGET_BATCH_GEOMETRY=CEIL_N_TRAIN_DIV_256`

## 5. Candidate pass-multiplier audit

The revision evaluated exact nominal-pass multipliers:

- `2`;
- `3`;
- `4`;
- `5`.

No minimum-update floor was applied in the final comparison.

Under the selected `alpha=0.85` class-sampling distribution, increasing the
pass multiplier compounds rare-class repeated exposure.

For MalNet-Image, approximate maximum expected repeat factors were:

- two passes: `6.578`;
- three passes: `9.866`;
- four passes: `13.155`;
- five passes: `16.444`.

The two-pass rule provides repeated optimization exposure while minimizing
unnecessary amplification of already-governed rare-class oversampling.

## 6. Selected maximum-update rule

The selected training budget is:

`max_updates = 2 * ceil(N_train / 256)`

Therefore:

`RBM_MAXIMUM_PASS_MULTIPLIER=2`

`RBM_MAX_UPDATES_FORMULA=2_TIMES_CEIL_N_TRAIN_DIV_256`

No minimum-update floor is used.

`RBM_MINIMUM_UPDATE_FLOOR=NONE`

## 7. Resulting dataset budgets

The selected formula gives:

### USTC-TFC2016

- `N_train = 72900`;
- `B = 285`;
- `max_updates = 570`;
- effective nominal exposure approximately `2.002` passes.

### CICMalDroid2020

- `N_train = 8117`;
- `B = 32`;
- `max_updates = 64`;
- effective nominal exposure approximately `2.019` passes.

### MalNet-Image

- `N_train = 883416`;
- `B = 3451`;
- `max_updates = 6902`;
- effective nominal exposure approximately `2.000` passes.

### MaleX

- `N_train = 731049`;
- `B = 2856`;
- `max_updates = 5712`;
- effective nominal exposure approximately `2.000` passes.

Therefore:

`RBM_TWO_PASS_BUDGET_CROSS_DATASET_AUDIT=PASS`

## 8. Expected class exposure under selected traversal

The selected training sampler uses:

`p(c) proportional to n_c^0.85`

The two-pass budget therefore yields approximate expected class-repeat ranges:

### USTC-TFC2016

`2.002` to `2.002`

### CICMalDroid2020

`1.884` to `2.234`

### MalNet-Image

`1.714` to `6.578`

### MaleX

`1.913` to `2.421`

These values are expected sampling exposures, not guarantees that every
individual example is visited a specific number of times.

## 9. Fixed-budget semantics

Training runs to the exact governed maximum update count unless a fail-closed
numerical or integrity condition occurs.

There is no adaptive extension based on apparent convergence.

There is no dataset-specific budget tuning.

Therefore:

`RBM_DATASET_SPECIFIC_UPDATE_TUNING=FORBIDDEN`

`RBM_POSTHOC_BUDGET_EXTENSION=FORBIDDEN`

## 10. Early stopping

Patience-based early stopping is not used.

Therefore:

`RBM_PATIENCE_EARLY_STOPPING=NO`

The benchmark uses a predeclared training budget rather than allowing different
seeds or datasets to receive different optimization opportunity based on a
noisy surrogate metric.

## 11. Exact likelihood limitation

For the revised RBM, the exact normalized marginal likelihood of visible
observations requires the global partition function and is not tractable for
routine benchmark checkpoint selection.

Therefore the implementation must not claim exact validation likelihood or
exact validation NLL unless an exact partition-function computation is
actually available and independently validated.

`RBM_EXACT_VALIDATION_LIKELIHOOD_AVAILABLE=NO`

`RBM_FAKE_EXACT_NLL=FORBIDDEN`

## 12. Validation purpose

Validation remains required as a diagnostic integrity surface.

It may be used to detect:

- non-finite state;
- invalid visible-range behavior;
- pathological reconstruction behavior;
- gross numerical instability;
- unexpected class-conditioning failure.

Validation does not determine how many updates are performed.

`RBM_VALIDATION_CONTROLS_TRAINING_LENGTH=NO`

## 13. Reconstruction diagnostic

For validation diagnostics, the implementation may use deterministic
one-step reconstruction based on conditional expectations:

1. compute hidden probabilities from validation input and class label;
2. use hidden probabilities as hidden expectations;
3. compute visible conditional parameters;
4. compute the analytic bounded-exponential visible mean;
5. compare reconstructed mean to the input.

A mean-squared reconstruction error may be recorded as a diagnostic only.

Therefore:

`RBM_VALIDATION_RECONSTRUCTION_MSE=DIAGNOSTIC_ONLY`

It must not be described as:

- exact likelihood;
- exact NLL;
- proof of generative quality;
- authoritative checkpoint-selection objective.

## 14. Validation distribution

Validation uses the governed empirical validation split.

The training `alpha=0.85` class rebalance must not be applied to validation.

Therefore:

`RBM_VALIDATION_CLASS_REBALANCING=FORBIDDEN`

Validation examples must never contribute gradients or optimizer updates.

`RBM_VALIDATION_GRADIENTS=NO`

## 15. Validation cadence

Define:

`B = ceil(N_train / 256)`

Validation diagnostics are evaluated at:

- `step = 0`;
- `step = B`;
- `step = 2B`.

Thus every numerically successful training run has exactly three governed
validation observations.

Therefore:

`RBM_VALIDATION_SCHEDULE=STEP_0_B_2B`

`RBM_VALIDATION_TOTAL_OBSERVATIONS=3`

`RBM_FINAL_UPDATE_IS_VALIDATION_STEP=YES`

No duplicate final validation evaluation is permitted.

`RBM_DUPLICATE_FINAL_VALIDATION=NO`

## 16. Authoritative checkpoint selection

The authoritative model state is the state after the final governed optimizer
update:

`step = max_updates`

Therefore:

`RBM_CHECKPOINT_SELECTION=FINAL_GOVERNED_UPDATE`

The step-zero and midpoint validation observations are diagnostics and are not
alternative authoritative model candidates.

This prevents noisy reconstruction diagnostics from silently changing
scientific training opportunity across seeds or datasets.

## 17. Best-checkpoint semantics

The revised benchmark must not create an authoritative `best` checkpoint based
on reconstruction MSE.

Therefore:

`RBM_RECONSTRUCTION_MSE_BEST_CHECKPOINT=FORBIDDEN`

A compatibility alias named `best` must not silently imply validation-based
selection.

Revision-specific checkpoint naming must make final-update authority explicit.

## 18. Failure semantics

Training must fail closed if any governed validation observation detects:

- non-finite parameters;
- non-finite hidden probabilities;
- non-finite visible conditional parameters;
- non-finite analytic visible means;
- visible means outside `[0,1]` beyond numerical tolerance;
- non-finite reconstruction diagnostic.

A failed state must not become an authoritative checkpoint.

`RBM_NONFINITE_VALIDATION_ACTION=FAIL_CLOSED`

## 19. Test-set isolation

The test split must remain untouched during:

- training;
- validation diagnostics;
- checkpoint selection.

Therefore:

`RBM_TEST_IN_TRAINING=FORBIDDEN`

`RBM_TEST_IN_CHECKPOINT_SELECTION=FORBIDDEN`

## 20. Reproducibility

The maximum update count must be derived deterministically from:

- governed training-set size;
- fixed batch size `256`;
- fixed pass multiplier `2`.

No runtime heuristic may modify the budget.

Therefore:

`RBM_TRAINING_BUDGET_REPRODUCIBLE=YES`

## 21. Implementation gate

Before this policy may be implemented as accepted production behavior, focused
tests must demonstrate at least:

1. `B = ceil(N_train / 256)`;
2. `max_updates = 2 * B`;
3. no minimum-update floor exists;
4. USTC budget equals `570`;
5. CIC budget equals `64`;
6. MalNet-Image budget equals `6902`;
7. MaleX budget equals `5712`;
8. validation occurs exactly at `0`, `B`, and `2B`;
9. final validation is not duplicated;
10. patience-based early stopping is absent;
11. reconstruction MSE is diagnostic only;
12. reconstruction MSE cannot select the authoritative checkpoint;
13. exact likelihood is not falsely reported;
14. the authoritative checkpoint is the final governed update;
15. validation contributes no gradients;
16. test data cannot influence training or checkpoint selection;
17. non-finite validation state fails closed;
18. historical `RBM_best` behavior cannot override revision semantics.

## 22. Current authorization state

The revised RBM training-budget and validation policy is selected as:

- batch geometry `ceil(N_train / 256)`;
- exact two-pass maximum-update budget;
- no minimum-update floor;
- no patience-based early stopping;
- validation at `0`, `B`, and `2B`;
- reconstruction diagnostics only;
- final governed update as authoritative checkpoint.

Therefore:

`RBM_TRAINING_BUDGET_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
