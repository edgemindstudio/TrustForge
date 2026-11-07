# Saved Inception feature FID/cFID bounded acceptance

Decision: PASS for saved-feature engineering fixture.

- Job 311699: COMPLETED|0:0 on talon05; elapsed 00:00:04.
- Execution revision: 483b28940572ce183215d5ad986a1211dd333b20.
- Consumed pinned Inception fixture features from job 311694.
- Pooled FID: 192.9252681277613.
- Equal-weight five-class macro cFID: 188.28267980411027.
- Pooled and every class result matched an independent analytic reference
  for the constant-synthetic-feature fixture.
- Singular empirical covariances were handled without diagonal regularization.
- Missing-class rejection passed.
- Twelve earlier mathematical tests include an independent full-rank reference.

This fixture does not exercise a nonzero covariance cross-term at 2048
dimensions, production-scale resource use or full fidelity orchestration.
The reported values are engineering fixture outputs, not scientific results.
Real-data provenance, generator lineage and complete evaluation remain open.

FULL_BENCHMARK_GO=NO
