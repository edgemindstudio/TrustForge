# Paired classifier publication/restoration bounded acceptance

Decision: PASS for integrated synthetic-fixture execution.

- Job 312074: COMPLETED|0:0 on talon06; elapsed 00:08:26.
- Execution revision: 9ba6c26d8ad93d6fc4dc592c12d306c4970d3751.
- Training/publication PID 3451847; fresh restoration PID 3452880.
- Actual paired training used shared initialization and governed fixture views.
- Both selected classifiers were published with validated training manifests.
- An external expectation receipt preceded publication.
- Both classifiers were restored and evaluated in the fresh process.
- Each arm covered 257 ordered test examples in batches [128,128,1].
- Saved bundle, context, prediction and paired-result replay checks passed.
- Fixture augmented-minus-real-only macro-F1: -0.19999999999999998.

This is an engineering acceptance result, not a scientific utility comparison.
The synthetic pool contains fixture images; generator lineage is unauthenticated.
Inference weights were restored, not optimizer continuation state.
No complete fidelity/diversity evaluation or full evaluation acceptance occurred.

External weight and prediction artifacts:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/paired-restore-9ba6c26-WJFaH3GM

hashes.json retains checked identities for external state and prediction files.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO
