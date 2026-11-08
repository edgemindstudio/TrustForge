# Revision-2026 evaluation completeness contract

Status: PROSPECTIVE IMPLEMENTATION CONTRACT / NOT EXECUTION VERIFIED.

This contract defines new revision reporting. It does not amend historical
execution-evidence linkage records, historical evaluator behavior or results.
It does not authorize production execution or full benchmark scale-out.

## Scope and identities

The locked target is four datasets, seven generator families, experimental
seeds 42-48 and 2000 synthetic images per authoritative class: 196 seed-level
evaluations. Each evaluation includes real-only and real-plus-synthetic
classifier arms.

Scientific identity and execution-attempt identity remain separate.
A seed-level evaluation is distinct from a seven-seed dataset/family aggregate.
A complete seed-level evaluation cannot establish full benchmark readiness.

Each evaluation must bind explicit identities for:

- Dataset, authoritative ordered class mapping and experimental seed.
- Governed training, validation and test memberships.
- Dataset configuration, source provenance and relevant fitted preprocessing.
- Generator checkpoint and its producing execution.
- Complete synthetic manifest, consumed image bytes and checkpoint lineage.
- Classifier and downstream metric recipes.
- Evaluator source inventory and execution environment.
- Both classifier training reports and selected-weight artifacts.
- Ordered test labels, probabilities and prediction artifacts.
- Fidelity selections, extractor weights/settings and feature artifacts.
- MS-SSIM selections, scorer settings and pair-score artifacts.

Generator and evaluator revisions may differ. Each role must retain its own
source identity and verified compatibility evidence. Equal timestamps, names,
seeds or file presence do not authenticate lineage.

## Required result groups

Every seed-level evaluation must retain:

1. Both classifier-arm metric reports: macro-F1, balanced accuracy,
   macro average precision, top-label ECE, multiclass Brier, descriptive
   macro recall at 1% FPR, and their selected classwise/bin records.
2. Paired linkage and augmented-minus-real-only macro-F1.
3. Pooled class-balanced KID with every selected subset estimate.
4. Pooled balanced FID and every authoritative class FID, plus macro cFID.
5. Three-scale intra-class MS-SSIM: every class, all 200 selected pairs per
   class, class means and equal-weight macro mean.

FID/cFID remain secondary diagnostics; requiring their execution does not
promote them to the primary scientific endpoint.

No failed class, pair, subset or required result may be omitted from coverage.
A mathematically valid zero is a result; an execution failure is not zero.
Negative unbiased KID estimates remain valid.

## Metric status records

Each expected result has exactly one status:

- PENDING: not completed; include an explicit reason.
- SUCCEEDED: validated finite result and verified artifact/configuration links.
- FAILED: attempted but invalid or interrupted; retain cause and surviving evidence.
- NOT_REQUESTED: permitted only for prospectively optional results, with reason.
- UNAVAILABLE: permitted only for optional results, with reason.

All result groups above are required. NOT_REQUESTED and UNAVAILABLE cannot
satisfy them. Additional optional metrics must be identified before execution.
Duplicate result identifiers or undeclared result groups are rejected.

A status label alone is not evidence. SUCCEEDED requires validation of the
actual referenced bytes and semantic identities. Preserve already completed
artifacts when another required component fails.

## Paired downstream acceptance

Both arms must share the governed real training, validation and test memberships,
evaluator seed, classifier recipe and initial-weight identity. Record source and
environment compatibility, including any justified baseline reuse.

Each arm must link its validation-selected training report to the actual loaded
selected weights. Test traversal must be complete and ordered, with matching
labels between arms. Inference must not update model or optimizer state.

Test data cannot select checkpoints, tune settings, filter samples or calibrate
probabilities. Report actual training samples, updates, epochs and stopping
outcomes; equal epochs do not imply equal updates or exposure.

## Fidelity and diversity acceptance

Use the selected validation-only balanced references and complete 2000/class
synthetic pool. Persist real references before family comparisons and persist
all selections before their scoring.

Verify selected sample order, labels, feature dimensions and extractor identity.
KID, pooled FID and class FID must consume the same selected feature sets.
Report class counts, effective sizes, singular covariance limitations and any
declared numerical roundoff correction.

MS-SSIM must consume the persisted manifest-bound pair selection. Require every
class and selected pair, and propagate invalid/nonfinite pair scores.

## Completion decision

A seed-level evaluation is COMPLETE only after all required results succeed,
expected/computed coverage agrees, all required artifact and provenance links
are verified, paired compatibility and test isolation pass, and durable
collision-safe publication succeeds.

Otherwise it is INCOMPLETE. Report every blocker explicitly. A malformed or
contradictory record is INVALID and cannot be promoted by a supplied boolean.

Engineering fixtures must remain explicitly identified and cannot satisfy
primary real-data evaluation acceptance.

The completion validator must verify evidence rather than trust caller-supplied
completion flags. Component reports retaining complete_evaluation=false do not
automatically become complete merely because they are collected together.

## Seven-seed and benchmark reporting

Each dataset/family aggregate requires seven valid paired seed results.
Retain every expected seed and attempt status. Missing seeds remain visible;
no favorable-result substitution or result-driven replacement seeds.

Publish all seven macro-F1 differences, their mean, sample SD and selected
pointwise 95% Student-t interval with six degrees of freedom. Fewer than seven
valid pairs means INCOMPLETE; available results may be described separately.

Across the benchmark, report expected/completed/failed counts and explicit
denominators. Seven-seed aggregates do not authorize a benchmark-wide
superiority claim or close operational readiness gates.

## Implementation acceptance still required

- Machine-readable revision report schema and semantic validator.
- Missing, failed, duplicate, contradictory and tampered-evidence tests.
- Durable selected-classifier publication and fresh-process restoration.
- Actual paired training-to-prediction linkage integration.
- Full fidelity-result integration and bounded real-data orchestration.
- Reporting/aggregation checks and preserved failure evidence.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO
