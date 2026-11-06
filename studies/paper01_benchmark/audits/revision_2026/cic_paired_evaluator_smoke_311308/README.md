# CICMalDroid2020 paired evaluator smoke test

**Decision.** PASS for the bounded engineering smoke test.

Talon job `311308` completed on `talon08` in 2m 07s with exit code `0:0`
and printed `CIC_PAIRED_EVALUATOR_SMOKE=PASS`. The job used TrustForge
revision `12f8a708089cca4141cd657f8f01d4a0ff776b7a` and diagnostic directory `cic-paired-eval-12f8a70-Jgm47gYK`.

The [JSON report](report.json) preserves the three arms and their paired
macro-F1 differences. All arms started with identical weights and selected
epoch 2 on the same governed real validation split. The shared held-out test
split contained 2,323 examples.

Saved [Slurm launcher](smoke.slurm) SHA-256
`a7d753583639d9449033955c851b77f5a8eb20efb9e3ee7743547714a9c57100`.

This run exercised two epochs per arm and five synthetic images per augmented
arm. Benchmark-scale utility and semantic class fidelity remain unverified.
Full Paper 1 benchmark status remains **NO-GO**.
