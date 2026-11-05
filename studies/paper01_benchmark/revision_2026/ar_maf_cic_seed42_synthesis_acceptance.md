# CICMalDroid2020 seed-42 AR and MAF synthesis pilot

**Decision.** PASS for the bounded checkpoint-to-synthesis pilot.
Full Paper 1 benchmark status remains **NO-GO**.

Synthesis revision `cbcde605d93f8bfb03d9c794c2b6aeabdf08ad4c`. Talon diagnostic run
`ar-maf-cic-synthesis-cbcde60-Y7HczdR3`. Both jobs ended with `COMPLETED` and exit code `0:0`.

| Family | Job | Elapsed | Training revision | Best checkpoint SHA-256 | Sample archive SHA-256 | Expected validation NLL | Restored validation NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AR | `311064` | `00:06:54` | `e4473abcdbc53f0de3a01ebe48858f2f3c7b841a` | `d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0` | `4b45339fa539e3070dea7096821f6e7158a20de6dac97fec66cbe3461fc54a22` | `-5.960004059556029` | `-5.960004059556029` |
| MAF | `311065` | `00:01:44` | `d33c0e9146caab11fb3b492361873daf059ce6ce` | `3604bd4009e2199400665b28ef156d5db8e6d2e75094e5eb1d193b090d89264d` | `251e4994d1c24ae4e529621d7255bc92822556df6f9fad2d26cc14b91b5e942a` | `83.87659391096837` | `83.87659391096837` |

Each job restored the verified best checkpoint and reproduced validation NLL
within tolerance across all 1,158 CIC validation examples. Archive verification
confirmed five finite float32 images of shape 40 × 40 × 3 in [0, 1], with
ordered conditioning IDs [0, 1, 2, 3, 4]. The job reports recorded byte-exact
replay with sampling seed 42001 and a changed batch with seed 42002.
The saved archive contains the first batch.

Semantic class fidelity, downstream utility, and throughput at 2,000 samples
per class remain open.
