# CIC fitted-state bounded acceptance

Decision: PASS for capture and independent-process restoration only.

- Execution revision: e81323282a2662d775f1cee7ea7bd2730ab1184b.
- Slurm job: 311667; talon07; COMPLETED|0:0; elapsed 00:00:51.
- Capture PID: 2387636; restore PID: 2387759.
- Both phases agreed on the fitted-state payload identity.
- Restored and live scaler transforms were identical on training positions
  [0, 1, 63, 64, 8116], retaining float32 output.
- Source/configuration, ordered features, training membership, implementation
  inventory, environment and imported project code were bound by the driver.
- This is bounded evidence, not an exhaustive transform comparison.
- No model training, checkpoint migration or production provenance integration
  occurred. Existing checkpoint identities remain unchanged.
- S2-03 remains open for production provenance integration.
- FULL_BENCHMARK_GO=NO.

External run directory:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/cic-fitted-state-e813232-ogfyI95a

The fitted-state file remains external. hashes.json records its file digest;
payload_sha256 records the canonical payload digest, a distinct identity.

Environment: Python 3.10.19; NumPy 2.2.6; pandas 2.3.3;
scikit-learn 1.7.2; TensorFlow 2.20.0; little-endian.
