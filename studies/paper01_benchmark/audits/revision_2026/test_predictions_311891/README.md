# Trained EvalCNN test-prediction bounded acceptance

Decision: PASS for synthetic-fixture integration.

- Job 311891: COMPLETED|0:0 on talon07; elapsed 00:02:49.
- Execution revision: 1d90df8977a3a4f8ad81c125ec240bcf1a02f541.
- Actual EvalCNN trained with the selected classifier controls.
- Selected-weight and test-identity expectations were persisted before
  fixture test access.
- Complete ordered test traversal covered 257 examples in batches [128,128,1].
- Selected-weight binding and unchanged inference execution state passed.
- Saved prediction archive, ordered prediction identity and metric replay passed.
- Earlier scoped suite: 35 passed, including the 18 downstream metric checks.
  These overlapping counts are not additive.

Fixture outputs:
macro-F1 0.2796875; balanced accuracy 0.4; macro-AUPRC 0.8;
macro recall at 1% FPR 0.6; ECE 0.3772412401692876;
Brier 0.7286033305526893.

These are engineering fixture values, not scientific utility results.
This job exercised one trained classifier, not a complete paired evaluation.
Real-data provenance, generator lineage, selected-weight artifact publication,
fresh-process model restoration and complete evaluation remain open.

Prediction archive remains external:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/test-predictions-1d90df8-QL9nZt0Q/predictions.npz

hashes.json records its file identity.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO
