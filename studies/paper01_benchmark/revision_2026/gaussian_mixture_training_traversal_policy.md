# Paper 1 Revision 2026 — Gaussian Mixture Training Traversal Policy

`STATUS=IMPLEMENTED_AND_FOCUSED_VALIDATED`

`GMM_TRAINING_TRAVERSAL_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

The governed class-specific traversal implementation is now present in:

`src/trustforge/models/paper01_revision_2026/gaussian_mixture_traversal.py`

and its focused validation suite passed 38 tests.

The implementation validates:

- `B = ceil(N / 128)` batches per complete pass;
- balanced `q, r = divmod(N, B)` batch geometry;
- maximum within-pass batch-size spread of one;
- seeded per-pass permutation;
- no replacement sampling;
- exact complete-pass coverage;
- no remainder dropping or cross-pass padding;
- exact stopping at the governed update limit;
- same-seed traversal reproducibility;
- different-seed order variability;
- maximum example-exposure spread of one.

This status advancement applies only to traversal.

Overall Gaussian-mixture training remains unauthorized until the remaining
optimizer/training, checkpoint, and sampling implementation contracts are
completed and validated.

## 1. Scope

This policy governs class-specific training-position traversal for the
revised Paper 1 class-conditional censored diagonal Gaussian mixture model.

It governs:

- per-pass permutation;
- experimental-seed dependence;
- exact class coverage;
- final partial batches;
- stopping inside a later pass when the governed update budget is exhausted.

It does not yet authorize production training.

Checkpoint file format, production implementation acceptance, and final
training authorization remain separately governed.

## 2. Class-specific traversal domain

Each GMM class model trains only on revised training examples belonging to
that class.

Let the class-specific ordered training positions be:

`P = [p_0, p_1, ..., p_(N-1)]`

The traversal procedure operates on these class-specific logical positions.

It must not mix examples from other classes into the class model's
optimization batches.

`GMM_TRAINING_SCOPE=CLASS_SPECIFIC`

## 3. Randomness source

Each class model uses a deterministic pseudo-random traversal stream derived
from the governed experimental seed.

`GMM_TRAVERSAL_SEED_SOURCE=EXPERIMENTAL_SEED`

The same experimental seed and identical class-position sequence must
produce identical traversal batches.

A different experimental seed must be permitted to produce a different
within-class order.

Partition construction remains governed separately by the fixed partition
seed and must not depend on the experimental seed.

`GMM_PARTITION_SEED_IS_TRAVERSAL_SEED=NO`

## 4. Per-pass permutation

At the beginning of every complete or attempted class pass, the complete
class-specific training-position array is permuted once.

Let:

`N`

be the class training count and:

`B = ceil(N / 128)`

be the governed number of optimizer batches per complete pass.

The permutation is partitioned sequentially into exactly `B` balanced
batches.

Let:

`q, r = divmod(N, B)`

Then:

- the first `r` batches contain `q + 1` examples;
- the remaining `B - r` batches contain `q` examples.

Therefore:

- every batch contains at most 128 examples;
- every training example appears exactly once in a completed pass;
- batch sizes within a pass differ by at most one;
- the governed number of batches per pass remains `ceil(N / 128)`.

This balanced partition replaces the earlier full-128-plus-small-remainder
geometry.

The change prevents a small remainder batch from receiving the same optimizer
update weight as a much larger full batch.

For example, USTC `N=8100` retains 64 batches per pass, but uses batch sizes
126 or 127 instead of 63 batches of 128 followed by a final batch of 36.

`GMM_TRAIN_BATCH_SIZE_MAX=128`

`GMM_BATCHES_PER_PASS_FORMULA=CEIL_N_OVER_128`

`GMM_BALANCED_BATCH_PARTITION=YES`

`GMM_BALANCED_BATCH_SIZE_SPREAD_MAX=1`

`GMM_FULL_128_PLUS_REMAINDER_GEOMETRY=NO`

A new permutation is constructed when the next pass begins.

`GMM_TRAINING_PASS_ORDER=SEEDED_RANDOM_PERMUTATION`

`GMM_RESHUFFLE_EACH_PASS=YES`

The implementation must not generate a fresh random sample independently
for every batch.

`GMM_WITH_REPLACEMENT_BATCH_SAMPLING=NO`

## 5. Exact complete-pass coverage

A complete pass must contain every class-specific training example exactly
once.

No example may be:

- omitted;
- duplicated within the same complete pass;
- replaced by an example sampled from another class.

`GMM_COMPLETE_PASS_EXACT_COVERAGE=YES`

This property is independent of class size.

## 6. Balanced complete-pass batches

There is no special undersized remainder batch.

After the class permutation is generated, all `N` examples are divided across
the governed:

`B = ceil(N / 128)`

batches using the balanced `q, r = divmod(N, B)` rule.

No example is dropped.

`GMM_DROP_REMAINDER=NO`

Examples from the next pass must not be used to pad the current pass.

`GMM_CROSS_PASS_BATCH_PADDING=NO`

No duplicated examples are introduced to equalize batch sizes.

`GMM_INTRA_PASS_BATCH_PADDING=NO`

For every complete pass:

`1 <= batch_size <= 128`

and:

`max_batch_size - min_batch_size <= 1`

The exact total remains:

`sum(batch_sizes) = N`

## 7. Maximum-update interaction

The maximum update budget is governed separately as:

`max_updates = max(100, 5 * batches_per_pass)`

Training stops exactly when the governed update budget is reached.

If the update budget is exhausted partway through a newly permuted pass,
the remainder of that pass is not forced to execute.

`GMM_FORCE_COMPLETION_OF_FINAL_PASS=NO`

This keeps the optimizer budget authoritative.

## 8. Exposure-balance property

Because each attempted pass starts from a permutation without replacement,
all examples receive equal exposure after every completed pass.

If training terminates partway through a later pass, examples included in
the executed prefix of that pass may receive one additional exposure.

Therefore the selected traversal guarantees:

`max_example_exposures - min_example_exposures <= 1`

`GMM_MAX_EXPOSURE_SPREAD=1`

No class example may differ from another by more than one training exposure
under the governed maximum-update schedule.

## 9. Traversal audit evidence

The candidate traversal procedure was audited on representative governed
class sizes.

### MalNet smallest class

`N=79`

- batches per pass: 1;
- maximum updates: 100;
- exposure count for every example: 100;
- exposure spread: 0;
- same-seed reproducibility: PASS;
- different-seed order change: PASS;
- complete-pass exact coverage: PASS.

### MalNet class with 106 examples

`N=106`

- batches per pass: 1;
- maximum updates: 100;
- exposure count for every example: 100;
- exposure spread: 0;
- same-seed reproducibility: PASS;
- different-seed order change: PASS;
- complete-pass exact coverage: PASS.

### MalNet class with 206 examples

`N=206`

- batches per pass: 2;
- maximum updates: 100;
- exposure count for every example: 50;
- exposure spread: 0;
- balanced per-pass batch sizes: 103 and 103.

### CICMalDroid smallest audited class

`N=877`

- batches per pass: 7;
- maximum updates: 100;
- 14 complete passes plus 2 updates into the next pass;
- minimum example exposures: 14;
- maximum example exposures: 15;
- exposure spread: 1;
- examples at 15 exposures: 256;
- examples at 14 exposures: 621;
- balanced per-pass batch sizes: 125 or 126.

This directly demonstrates the bounded one-exposure imbalance when the
governed update budget terminates inside a later pass.

### USTC

`N=8100`

- batches per pass: 64;
- maximum updates: 320;
- exactly 5 complete passes;
- every example exposure count: 5;
- exposure spread: 0;
- balanced per-pass batch sizes: 126 or 127.

### Largest MalNet class

`N=619118`

- batches per pass: 4837;
- maximum updates: 24185;
- exactly 5 complete passes;
- every example exposure count: 5;
- exposure spread: 0;
- balanced per-pass batch sizes: 127 or 128.

### MaleX malware class

`N=605255`

- batches per pass: 4729;
- maximum updates: 23645;
- exactly 5 complete passes;
- every example exposure count: 5;
- exposure spread: 0;
- balanced per-pass batch sizes: 127 or 128.

### Balanced-batch geometry audit

The balanced candidate was audited on representative revised class sizes
including:

- `N=79`;
- `N=106`;
- `N=206`;
- `N=877`;
- `N=8100`;
- `N=12221`;
- `N=125794`;
- `N=605255`;
- `N=619118`.

Across all audited cases:

- `B = ceil(N / 128)` was unchanged;
- total examples per complete pass remained exactly `N`;
- maximum batch size remained at most 128;
- within-pass batch-size spread was at most one;
- the governed maximum-update count was unchanged;
- maximum example-exposure spread remained at most one.

Representative improvements included:

- USTC: batch-size ratio improved from `128 / 36 = 3.5556` to
  `127 / 126 = 1.00794`;
- MaleX malware: improved from `128 / 71 = 1.8028` to
  `128 / 127 = 1.00787`;
- MalNet `N=206`: improved from `128 / 78 = 1.6410` to exactly
  `103 / 103 = 1`.

The balanced geometry is therefore selected for production traversal.

`GMM_BALANCED_BATCH_GEOMETRY_AUDIT=PASSED`

## 10. Reproducibility contract

For a fixed:

- revised dataset;
- revised partition;
- class identifier;
- experimental seed;
- batch size;
- maximum-update policy;

the traversal sequence must be reproducible.

This includes:

- per-pass permutations;
- batch boundaries;
- partial-batch membership;
- the exact prefix used when training stops partway through a pass.

`GMM_TRAVERSAL_REPRODUCIBLE=YES`

## 11. Fail-closed behavior

Traversal must fail closed if:

- class positions contain duplicates before permutation;
- class-position count disagrees with the governed split count;
- a complete pass omits or duplicates positions;
- a produced batch is empty;
- a produced batch exceeds size 128;
- traversal crosses the governed maximum-update limit;
- random-state construction cannot be reproduced from the governed
  experimental seed.

The implementation must not silently switch to replacement sampling.

## 12. Governed traversal constants

`GMM_TRAIN_BATCH_SIZE_MAX=128`

`GMM_BATCHES_PER_PASS_FORMULA=CEIL_N_OVER_128`

`GMM_BALANCED_BATCH_PARTITION=YES`

`GMM_BALANCED_BATCH_SIZE_SPREAD_MAX=1`

`GMM_FULL_128_PLUS_REMAINDER_GEOMETRY=NO`

`GMM_DROP_REMAINDER=NO`

`GMM_CROSS_PASS_BATCH_PADDING=NO`

`GMM_INTRA_PASS_BATCH_PADDING=NO`

`GMM_BALANCED_BATCH_GEOMETRY_AUDIT=PASSED`

`GMM_TRAINING_SCOPE=CLASS_SPECIFIC`

`GMM_TRAVERSAL_SEED_SOURCE=EXPERIMENTAL_SEED`

`GMM_PARTITION_SEED_IS_TRAVERSAL_SEED=NO`

`GMM_TRAINING_PASS_ORDER=SEEDED_RANDOM_PERMUTATION`

`GMM_RESHUFFLE_EACH_PASS=YES`

`GMM_WITH_REPLACEMENT_BATCH_SAMPLING=NO`

`GMM_COMPLETE_PASS_EXACT_COVERAGE=YES`

`GMM_DROP_REMAINDER=NO`

`GMM_CROSS_PASS_BATCH_PADDING=NO`

`GMM_FORCE_COMPLETION_OF_FINAL_PASS=NO`

`GMM_MAX_EXPOSURE_SPREAD=1`

`GMM_TRAVERSAL_REPRODUCIBLE=YES`

`GMM_TRAINING_TRAVERSAL_POLICY_SELECTED=YES`

## 13. Authorization state

The training traversal policy is selected.

Production implementation and checkpoint persistence remain unresolved.

Therefore:

`GMM_TRAVERSAL_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_TRAVERSAL_FOCUSED_VALIDATION=PASSED`

`GMM_TRAVERSAL_FOCUSED_TEST_COUNT=38`

`GMM_IMPLEMENTATION_AUTHORIZED=PARTIAL_INITIALIZATION_AND_TRAVERSAL_ONLY`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
