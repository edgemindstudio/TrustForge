# Reference-to-Inception/KID bounded acceptance

Decision: PASS for synthetic-fixture integration.

- Job 311694: COMPLETED|0:0 on talon05; elapsed 00:01:06.
- Execution revision: a9946a6700088755afa96f3894bdc1749f670f81.
- Class-balanced selection was persisted before feature extraction.
- Actual verified-weight Inception extraction produced ten finite float32
  2048-dimensional feature rows per condition: two per class.
- Maximum real-fixture image request was ten.
- Saved selection, feature files and report identities passed verification.
- Independent NumPy recalculation reproduced all ten deterministic KID draws
  and their mean: 0.24403195325683105.

Each draw used all ten selected rows. Near-identical draw estimates therefore
do not establish variability across smaller subsets or experimental seeds.

Feature arrays and fixture pool remain external:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/fidelity-integration-a9946a6-dbjKE8oO

hashes.json records the external feature-file identities.

This verifies engineering integration. No real data, authenticated generator
lineage or primary scientific fidelity result was established. FID/cFID,
MS-SSIM, downstream metrics, completeness/reporting and real-data orchestration
remain open.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO
