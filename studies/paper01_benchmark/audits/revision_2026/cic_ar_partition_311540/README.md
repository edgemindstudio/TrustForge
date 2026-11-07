# CIC seed-42 AR partition diagnostic

**Decision.** PASS for the 32-image partition diagnostic.

Both Talon array tasks completed with exit code `0:0`.
Execution revision `21cb566d15f2759aaf62a3548ddd19217ab07a3d`. Run directory `ar-cic-partition-21cb566-VMT6NwTn`.

| Class-0 batch | Slurm array task | Sampling seed | Sampling seconds |
| ---: | --- | ---: | ---: |
| 0 | `311540_0` | 23501 | 124.07 |
| 1 | `311540_0` | 23502 | 123.06 |
| 2 | `311540_1` | 23503 | 117.79 |
| 3 | `311540_1` | 23504 | 118.75 |

The diagnostic verifier checked all four archives, assembled batches 0–3
in memory from reversed worker inputs, and rejected gap and overlap cases.
It printed `CIC_AR_PARTITION_DIAGNOSTIC=PASS`.

This is bounded assembly evidence. Production manifest publication
remains pending. Full Paper 1 benchmark status remains **NO-GO**.
