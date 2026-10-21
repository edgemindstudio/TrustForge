# Paper 1 Revision 2026 — Gaussian Mixture Training-Budget and Checkpoint Policy

`STATUS=IMPLEMENTED_AND_FOCUSED_VALIDATED`

`GMM_TRAINING_BUDGET_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

The governed training-budget implementation is now present in:

`src/trustforge/models/paper01_revision_2026/gaussian_mixture_training_budget.py`

and its focused validation suite passed 27 tests.

The implementation validates:

- `B = ceil(N / 128)`;
- `max_updates = max(100, 5 * B)`;
- `validation_interval = max(20, B)`;
- exactly five post-update validation checkpoints;
- exactly six total checkpoint observations including step 0;
- exact inclusion of `max_updates` as the final scheduled validation step;
- absence of duplicate final validation;
- fail-closed invalid budget inputs.

This status advancement applies only to budget and validation-schedule arithmetic.

Overall Gaussian-mixture training remains unauthorized until optimizer updates,
validation evaluation, checkpoint persistence, and sampling are implemented
and validated.

## 1. Scope

This policy governs:

- the maximum optimization budget for each revised Paper 1
  class-conditional censored diagonal Gaussian mixture;
- validation-evaluation cadence;
- validation-based checkpoint selection.

It does not yet govern:

- within-class training-position traversal;
- per-pass shuffling;
- random permutation construction;
- checkpoint file format;
- production implementation acceptance.

Real benchmark training therefore remains unauthorized.

## 2. Why a fixed epoch count is rejected

The governed production traversal uses balanced batches with:

`GMM_TRAIN_BATCH_SIZE_MAX=128`

and preserves the exact per-pass batch-count formula:

`B = ceil(N_train_class / 128)`

Revised class sizes differ substantially.

Observed train batches per complete class pass range from:

`1`

to:

`4837`.

Examples include:

- MalNet-Image class 12: 79 examples, 1 batch per pass;
- USTC classes: 8100 examples, 64 batches per pass;
- MalNet-Image class 4: 619118 examples, 4837 batches per pass;
- MaleX malware class: 605255 examples, 4729 batches per pass.

A fixed number of epochs would therefore represent radically different
numbers of optimization updates across classes.

The revision consequently governs optimization in update units rather than
using a universal epoch count.

`GMM_UNIVERSAL_FIXED_EPOCH_COUNT=NO`

## 3. Maximum-update rule

For each class define:

`B = ceil(N_train_class / 128)`

where `B` is the number of balanced training batches in one complete class
pass under the governed traversal policy.

The balanced traversal correction changes within-pass batch sizes but does
not change `B`, so the selected training-budget arithmetic remains unchanged.

`GMM_TRAINING_BUDGET_BATCH_GEOMETRY_SOURCE=TRAVERSAL_POLICY`

The maximum number of optimizer updates is:

`max_updates = max(100, 5 * B)`

Therefore:

`GMM_MINIMUM_UPDATE_BUDGET=100`

`GMM_MAXIMUM_PASS_MULTIPLIER=5`

The term "maximum pass multiplier" describes the large-class branch of the
rule.

For classes where the 100-update floor dominates, the effective number of
data exposures may exceed five complete passes.

The policy must not describe five passes as a universal hard ceiling.

`GMM_MAX_UPDATES_FORMULA=MAX_100_OR_5_TIMES_TRAIN_BATCHES`

## 4. Small-class semantics

For classes with very few examples, the 100-update floor intentionally
provides more than five complete data exposures.

The most extreme audited governed case is MalNet-Image class 12:

- training examples: 79;
- training batches per pass: 1;
- maximum updates: 100;
- effective maximum passes: 100;
- validation examples: 12.

This behavior is intentional and must not be confused with accidental
oversampling.

The same governed class batch may therefore be revisited across multiple
updates.

## 5. Validation cadence

For each class define the validation interval:

`validation_interval = max(20, B)`

Therefore:

`GMM_MINIMUM_VALIDATION_INTERVAL_UPDATES=20`

`GMM_VALIDATION_INTERVAL_FORMULA=MAX_20_OR_TRAIN_BATCHES`

Under the selected maximum-update rule, every governed class run produces
exactly five post-update validation evaluations.

Together with the initialized-model evaluation at step 0, every complete
numerically successful class run therefore has exactly six validation
checkpoint observations.

The exact validation-step schedule is:

- always evaluate at `step=0`;
- after optimization, evaluate at every positive step divisible by
  `validation_interval`;
- the final governed `max_updates` step is always one of those scheduled
  interval evaluations.

This follows directly from the governed formulas.

If `B < 20`:

`max_updates = 100`

and:

`validation_interval = 20`

so validation steps are:

`0, 20, 40, 60, 80, 100`

If `B >= 20`:

`max_updates = 5 * B`

and:

`validation_interval = B`

so validation steps are:

`0, B, 2B, 3B, 4B, 5B`

The implementation must not perform a duplicate validation evaluation at
`max_updates` after already evaluating that same scheduled step.

`GMM_VALIDATION_POST_UPDATE_EVALUATIONS=5`

`GMM_VALIDATION_TOTAL_CHECKPOINT_OBSERVATIONS=6`

`GMM_MAX_UPDATE_IS_SCHEDULED_VALIDATION_STEP=YES`

`GMM_DUPLICATE_FINAL_VALIDATION=NO`

Validation is evaluated on the class-specific revised validation split.

Validation data must never contribute gradients or optimizer updates.

`GMM_VALIDATION_GRADIENTS=NO`

## 6. Full class-specific validation objective

At every validation checkpoint, the implementation must evaluate the
class-specific censored-mixture mean negative log likelihood over the
entire governed validation subset for that class.

When validation requires multiple batches, the reported class validation
NLL must be the example-count-weighted mean across those batches.

The final partial validation batch must therefore be weighted by its actual
number of examples.

`GMM_VALIDATION_OBJECTIVE=FULL_CLASS_MEAN_CENSORED_NLL`

## 7. Initial checkpoint

Validation must also be evaluated before the first optimizer update.

The initialized model is therefore checkpoint candidate:

`step=0`

This prevents training from being assumed beneficial merely because
optimization was attempted.

`GMM_INITIALIZED_MODEL_IS_CHECKPOINT_CANDIDATE=YES`

## 8. Checkpoint selection

The authoritative class model is the checkpoint with the lowest finite
class-validation mean censored NLL among:

- initialization at step 0;
- every governed positive-step validation checkpoint reached during
  optimization.

Under the governed budget formulas, `max_updates` always coincides exactly
with the fifth positive-step scheduled validation checkpoint. It is therefore
not evaluated or inserted a second time as a separate candidate.

Lower validation NLL is better.

`GMM_CHECKPOINT_SELECTION=LOWEST_VALIDATION_CENSORED_NLL`

The final optimizer state is not automatically authoritative.

`GMM_ALWAYS_USE_FINAL_OPTIMIZER_STATE=NO`

If multiple checkpoints have exactly equal validation NLL under the
production numeric representation, the earliest checkpoint wins.

`GMM_CHECKPOINT_TIE_BREAK=EARLIEST_STEP`

This deterministic tie-break avoids arbitrary later-state preference.

## 9. Early stopping

The selected revision does not use patience-based early termination.

All models may run to their governed maximum update budget unless a
fail-closed numerical or execution error occurs.

The best validation checkpoint is retained after the governed run.

This keeps the optimization opportunity deterministic across experimental
runs while still preventing the final state from automatically replacing a
better earlier state.

`GMM_PATIENCE_EARLY_STOPPING=NO`

`GMM_RUN_TO_MAXIMUM_BUDGET=YES`

## 10. Small-class validation evidence

The smallest revised MalNet training class was audited directly:

- class: MalNet-Image class 12;
- train examples: 79;
- validation examples: 12;
- SGD learning rate: 0.001;
- batch size: 128;
- maximum audited updates: 100.

Validation NLL evolved approximately as follows:

- step 0: `-1853.78`;
- step 20: `-3722.74`;
- step 40: `-3791.51`;
- step 60: `-3809.94`;
- step 80: `-3818.00`;
- step 100: `-3822.47`.

The best observed validation checkpoint was:

`step=100`

No validation increase occurred across the finer five-update audit cadence.

This provides direct evidence that the 100-update floor did not induce
observed validation deterioration in the smallest governed class examined.

## 11. Five-pass validation evidence

USTC class 0 was audited as a representative class where the five-pass
branch governs the maximum budget:

- train examples: 8100;
- validation examples: 900;
- batches per pass: 64;
- maximum updates: 320;
- exact complete passes represented: 5.

Validation NLL evolved approximately as follows:

- step 0: `2707.46`;
- step 64: `93.61`;
- step 128: `-122.57`;
- step 192: `-198.79`;
- step 256: `-238.02`;
- step 320: `-262.31`.

The best observed validation checkpoint was:

`step=320`

No validation increase occurred.

This provides direct evidence that five complete passes remained
numerically stable and validation-improving in the audited normal-sized
class.

## 12. Benchmark-wide maximum geometry

Under the selected candidate formulas, the audited maximum workload across
the 63 class models for one complete four-dataset generator execution is:

`GMM_CLASS_MODELS_PER_FULL_BENCHMARK_RUN=63`

Maximum training updates:

`69235`

Maximum validation batches:

`9565`

These are upper bounds before any execution failure.

They do not represent dataset examples or experimental seeds.

Each experimental seed constitutes a separate governed model run.

## 13. Dataset-specific maximum-update evidence

Observed maximum training-update totals per complete dataset execution were:

- USTC: 2880;
- CICMalDroid: 510;
- MalNet-Image: 37285;
- MaleX: 28560.

The large-class branch remains bounded to five complete class passes.

The small-class branch receives the governed 100-update floor.

## 14. Validation leakage prohibition

The validation split is used only for model-selection decisions specified
by this policy.

It must not be used for:

- gradient calculation;
- scale initialization;
- mean initialization;
- mixture-weight initialization;
- training-batch construction;
- synthetic sample generation;
- downstream test evaluation.

The test split remains untouched by GMM training and checkpoint selection.

`GMM_TEST_SPLIT_USED_FOR_MODEL_SELECTION=NO`

## 15. Numerical failure behavior

A class run must fail closed if any scheduled validation evaluation
produces:

- non-finite class validation NLL;
- non-finite model parameters;
- non-finite transformed scales;
- invalid mixture weights.

A failed validation state must not silently become a checkpoint candidate.

A numerical failure must not trigger:

- optimizer substitution;
- learning-rate substitution;
- gradient clipping;
- component deletion;
- global-model fallback;
- NaN replacement;
- posthoc parameter clipping.

## 16. Governed training-budget constants

`GMM_TRAIN_BATCH_SIZE_MAX=128`

`GMM_TRAINING_BUDGET_BATCH_GEOMETRY_SOURCE=TRAVERSAL_POLICY`

`GMM_MINIMUM_UPDATE_BUDGET=100`

`GMM_MAXIMUM_PASS_MULTIPLIER=5`

`GMM_MAX_UPDATES_FORMULA=MAX_100_OR_5_TIMES_TRAIN_BATCHES`

`GMM_MINIMUM_VALIDATION_INTERVAL_UPDATES=20`

`GMM_VALIDATION_INTERVAL_FORMULA=MAX_20_OR_TRAIN_BATCHES`

`GMM_VALIDATION_POST_UPDATE_EVALUATIONS=5`

`GMM_VALIDATION_TOTAL_CHECKPOINT_OBSERVATIONS=6`

`GMM_MAX_UPDATE_IS_SCHEDULED_VALIDATION_STEP=YES`

`GMM_DUPLICATE_FINAL_VALIDATION=NO`

`GMM_VALIDATION_OBJECTIVE=FULL_CLASS_MEAN_CENSORED_NLL`

`GMM_INITIALIZED_MODEL_IS_CHECKPOINT_CANDIDATE=YES`

`GMM_CHECKPOINT_SELECTION=LOWEST_VALIDATION_CENSORED_NLL`

`GMM_CHECKPOINT_TIE_BREAK=EARLIEST_STEP`

`GMM_ALWAYS_USE_FINAL_OPTIMIZER_STATE=NO`

`GMM_PATIENCE_EARLY_STOPPING=NO`

`GMM_RUN_TO_MAXIMUM_BUDGET=YES`

`GMM_TEST_SPLIT_USED_FOR_MODEL_SELECTION=NO`

`GMM_TRAINING_BUDGET_POLICY_SELECTED=YES`

## 17. Authorization state

The training-budget and checkpoint-selection policy is selected.

Training traversal is now separately implemented and focused-validated.

The production optimizer/training loop, checkpoint persistence, and sampling
implementation remain unresolved.

Therefore:

`GMM_TRAINING_BUDGET_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_TRAINING_BUDGET_FOCUSED_VALIDATION=PASSED`

`GMM_TRAINING_BUDGET_FOCUSED_TEST_COUNT=27`

`GMM_IMPLEMENTATION_AUTHORIZED=PARTIAL_INITIALIZATION_TRAVERSAL_AND_BUDGET_ONLY`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
