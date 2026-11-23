# Stage 5 — Authenticated Production Aggregation Linkage Acceptance

**Recorded:** 2026-11-23
**Status:** `STAGE5_AGGREGATION_LINKAGE=ACCEPTED_AND_CLOSED`
**Scope:** Bounded implementation and fixture-verification milestone only.

## A. Scientific objective

The accepted implementation connects persisted, authenticated paired evaluation
evidence to the existing seven-seed statistical reporting function. Its parent
requirement is Stage 5 evidence-backed aggregation and completion accounting:
produce aggregate evidence only from attempts that satisfy the predeclared
execution and integrity conditions, while retaining explicit missing, failed and
incomplete seed outcomes.

The benchmark identity remains four datasets, seven generator families and
experimental seeds 42–48, with the accepted 2,000-samples-per-class synthetic
budget and paired Real-only versus Real+Synthetic macro-F1 estimand.

## B. Accepted implementation

Accepted source paths and hashes:

- `src/trustforge/evaluation/paper01_production_aggregation.py`
  SHA-256 `1a41d421401222f32cc8945afbb7848df6f1b8f001c58d75f763572e9090c328`
- `tests/test_paper01_production_aggregation_revision_2026.py`
  SHA-256 `3ddfdec41f7db26ab296e285fe221432a2c3423be068d1c54596d41327d49e1a`

Accepted behaviors:

- Each seed carries its own synthetic-pool manifest identity; pending, failed
  and incomplete seeds are represented without fabricated pool digests.
- Cross-seed pool substitution is rejected against the evaluation record,
  generator lineage and checkpoint evidence.
- The existing augmented-checkpoint identity verifier is reused to bind the
  training receipt, context evidence, published classifier bundle, selected
  weights and expected pool manifest.
- The existing synthetic-pool consumption verifier is reused to check training
  identity, training size, selected/final optimizer updates, synthetic
  presentations, stream digest and selected weights.
- Paired prediction and metric evidence is replayed from saved test predictions
  with the accepted classifier and metric recipes; the replayed linkage must
  equal the persisted paired record exactly.
- Attempts are selected by ascending attempt number, choosing the first attempt
  that authenticates; scores never influence selection.
- Seven seed records are always constructed in order 42–48, with PENDING,
  FAILED and INCOMPLETE states and null scores preserved.
- The accepted `summarize_declared_paired_seeds` function is invoked unchanged.
- Aggregate output is published exclusively (`O_CREAT|O_EXCL`, mode 0600), and
  every output records `benchmark_complete=false`,
  `full_benchmark_go=false`, `primary_inference_authorized=false`.

Accepted classifier recipe:
`1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982`.
Accepted metric recipe:
`d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85`.

## C. Verification evidence

Nonexperimental fixture tests executed by OpenCode before this preservation:

```text
pytest -q tests/test_paper01_production_aggregation_revision_2026.py
19 passed in 4.14s

pytest -q tests/test_paper01_production_aggregation_revision_2026.py \
  tests/test_paper01_seed_reporting.py \
  tests/test_paper01_paired_results.py \
  tests/test_paper01_evaluation_validation.py \
  tests/test_paper01_pool_consumption.py \
  tests/test_paper01_classifier_bundle.py
146 passed in 5.27s
```

Independent source review verified the exact corrected source, confirmed the
accepted hashes, and confirmed that the accepted statistical reporter remained
unchanged. Tests were not rerun during this preservation operation.

## D. Acceptance decision

`STAGE5_AGGREGATION_LINKAGE=ACCEPTED_AND_CLOSED`

This closes the bounded aggregation-linkage implementation and fixture
verification only. It does not assert that production aggregation over a real
accepted run inventory has been performed, nor that Stage 5 overall is complete.

## E. Outstanding requirements

`STAGE_5_OVERALL=IN_PROGRESS`

- Production aggregation over an actual accepted run inventory has not been
  executed.
- Full benchmark aggregation and reporting remain incomplete; no seven-seed
  matrix results exist.
- Stage 6 reproduction, Stage 7 integrated acceptance and Stage 8 launch
  freeze/GO remain separately governed.

## F. Scientific preservation

`VAE_CIC_SEED42_PAIRED_EVALUATION=ACCEPTED_AND_CLOSED`

`RBM_S2_01=BLOCKED`

`BENCHMARK_DESIGN=4_DATASETS_7_FAMILIES_7_SEEDS`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

The accepted statistical methodology, experimental evidence, classifier and
metric recipes, and existing checkpoint and synthesis artifacts remain
unchanged.

## G. Limitations

The aggregation module authenticates persisted checkpoint bundles, receipts,
context evidence, pool-consumption records, unified-evaluation verification
records and paired predictions. It does not independently rerun Inception
feature extraction, KID, FID, cFID or MS-SSIM semantics; those remain recorded
unified-verification evidence, explicitly labeled
`fidelity_semantics: RECORDED_UNIFIED_VERIFICATION` and
`feature_replay: NOT_PERFORMED`.

The module performs artifact-contract authentication, not adversarial
provenance security. It does not establish full-matrix execution, statistical
significance, or publication readiness.
