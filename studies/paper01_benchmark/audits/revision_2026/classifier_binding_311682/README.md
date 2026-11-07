# Classifier TensorFlow binding bounded acceptance

Decision: PASS for synthetic-fixture binding checks.

- Job 311682: COMPLETED|0:0 on talon07; elapsed 00:00:24.
- Execution revision: 69a6bc0beacd9e459eb39d6cd4a7312616907b55.
- Actual EvalCNN architecture and checked Adam controls passed.
- Same-seed initialization replay and alternate-seed variation passed.
- Optimizer update and validation-state isolation checks passed.
- Both fixture arms used identical initial weights.
- Each completed 20 epochs/updates and selected epoch index 19.
- Selected validation losses:
  real-only fixture 1.3566337823867798;
  augmented fixture 1.3365768194198608.
- Saved histories, selection, stopping outcomes and report identity passed
  standard-library verification.

Initial weights SHA-256:
4e5a33f021164e0d148bba5f058d0dc01a0cd8ca48ad3f0b7c07f9013e7a19ba

Limits:
- Synthetic fixtures only; no real-data scientific utility conclusion.
- prepare_paired_classifiers was not exercised.
- Both arms reached the epoch limit and selected their final epoch.
- Actual patience stopping and nonterminal-best restoration were not
  demonstrated by this job; earlier control tests cover those with stand-ins.
- No held-out test evaluation, selected metric suite or complete evaluation.

FULL_BENCHMARK_GO=NO
