# MAF CIC execution-envelope synthesis acceptance

Decision: PASS for bounded five-image checkpoint-backed synthesis.

- Job 311680: COMPLETED|0:0 on talon16; elapsed 00:00:44.
- Synthesis revision: e48016b8339cf04a0039484e84649471c7da8476.
- Training revision: 913d86f9c64e6a3a1a0d009f112bcfa889a02ec8.
- Consumed the required execution envelope from accepted job 311675.
- Live scaler state, training membership and selected preprocessing context
  matched the captured training evidence.
- Restored best-checkpoint validation NLL: 83.87659391096837,
  matching the recorded value across all 1158 validation examples.
- Archive: five finite float32 40x40x3 images in [0,1].
- Ordered conditioning IDs: [0,1,2,3,4].
- Same-seed byte replay passed; alternate-seed variation passed.
- Saved report and archive hashes were checked.

Archive remains external:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-envelope-synth-e48016b-nKnbRjZa/output/samples.npz

This establishes bounded MAF/CIC seed42 synthesis-consumer adoption.
It does not establish semantic class fidelity, full 2000/class synthesis,
production evaluator acceptance, other-family adoption or full readiness.

FULL_BENCHMARK_GO=NO
