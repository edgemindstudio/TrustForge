# Paired classifier entry-point bounded acceptance

Decision: PASS for synthetic-fixture execution of prepare_paired_classifiers.

- Job 311683: COMPLETED|0:0 on talon07; elapsed 00:11:32.
- Execution revision: ff6bbb416c281cedfbd7d4a8ca26a5b274058dcf.
- Actual real, synthetic and combined input views were exercised.
- Fixture counts: 9 real training, 5 validation, 10000 synthetic images.
- Complete-pool streaming preflight passed.
- Both arms used identical initial classifier weights.
- Real-only: 20 epochs, 20 updates, selected epoch index 19.
- Real-plus-synthetic: 20 epochs, 1580 updates, selected epoch index 17.
- Selected validation losses: real-only 1.3566337823867798;
  real-plus-synthetic 1.356911301612854.
- Driver checks confirmed selected-weight snapshot identities.
- Saved histories passed coverage, update, selection and stopping checks.

The augmented arm demonstrates nonterminal-best selection/restoration.
Both arms reached the epoch limit; actual patience-triggered stopping was
not exercised. Earlier stand-in tests remain separate evidence for patience.

This is engineering evidence, not a scientific utility comparison.
The pool contains fixture images; generator lineage was not authenticated.
No real data or held-out test was evaluated. No complete metric suite ran.
Bounded requests do not establish an end-to-end large-dataset memory profile.

External fixture artifacts:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/paired-classifier-ff6bbb4-6j3j8sX7

FULL_BENCHMARK_GO=NO
