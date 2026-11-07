# MAF CIC execution-envelope bounded acceptance

Decision: PASS for CICMalDroid2020, MAF, experimental seed 42.

- Execution revision: 913d86f9c64e6a3a1a0d009f112bcfa889a02ec8.
- Job 311675: COMPLETED|0:0 on talon16; elapsed 00:12:11.
- Allocation: 8 CPUs, 32 GB, 30-minute limit; talon08 excluded.
- Selected two-epoch recipe completed 254 optimizer updates.
- Fitted state was captured before and checked after training.
- Publication and loading required the explicit fitted-state expectation.
- Capture/training PID 2057119; fresh restoration PID 2058606.
- Best and last each replayed all 1158 validation examples.
- Expected and observed validation NLL: 83.87659391096837 for both roles.
- An incompatible fitted-state expectation was rejected.
- Saved receipt, envelope, fitted-state and checkpoint hashes were verified.

Fitted-state payload SHA-256:
c177c31e8602e94b06dc965e14dafe8ec3374fd07816b93107df3f5c1fc7f5cf

External artifacts:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-execution-913d86f-VPyYjqz5

The execution manifest and model state remain external. hashes.json records
the manifest file identity; the reports retain checkpoint artifact identities.

This accepts the bounded MAF/CIC seed42 execution path. It does not establish
AR execution through the new caller, adoption by other family consumers,
complete synthesis, evaluator acceptance or full benchmark readiness.

General training activation remains disabled.
S2-03 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO.
