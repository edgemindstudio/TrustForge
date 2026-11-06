# CIC seed-42 checkpoint-backed batch-eight diagnostic

**Decision.** PASS for class 0, batch 0, for AR and MAF.

Execution revision `1412a64c49ce06636c5cac8ec03bb889d643f63f`. Both Talon jobs completed with exit
code `0:0`. The archive verification printed `AR_BATCH8_ARCHIVE=PASS`
and `MAF_BATCH8_ARCHIVE=PASS`.

| Family | Job | Sampling seed | Sampler and NumPy conversion (s) |
| --- | ---: | ---: | ---: |
| AR | 311471 | 23501 | 125.20 |
| MAF | 311472 | 35251 | 3.17 |

The [AR report](ar_report.json) and [MAF report](maf_report.json) preserve
checkpoint and archive digests, source revisions, and validation NLL.
Each timing represents one eight-image batch on its reported CPU node.
Checkpoint-backed full-pool publication and evaluator integration remain
pending. Full Paper 1 benchmark status remains **NO-GO**.
