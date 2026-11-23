# VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-01 Paired Evaluation Evidence

**Status:** Researcher-accepted and closed individual experimental cell.
Bounded evidence; not a full benchmark result.

## Accepted identity

- Dataset: `cicmaldroid2020`
- Generator: `vae`
- Evaluator seed: 42
- Attempt: 01
- Slurm job: `316983`
- Evaluation source commit:
  `04d1a3cb929c41b24c8b4e8e8e4a25bc68fe1a4b`
- Authorization ID: `P1R2026-VAE-CIC-S42-EVAL-A01-20261123`
- Authorization SHA-256:
  `e92da8a146d5b775ab79a71cb6efad85c8de8331d8a7ee54b65768023d8435ef`
- Launcher SHA-256:
  `7feaaa72c302b198e86d677ded7c6e8192b80909f542d505f05e21f277f9b1de`
- Evaluation driver SHA-256:
  `fb0c6b56932be11d888c80c72b2b685902fdb5f23c4445137d35fab8dd7dac65`
- Production pool manifest SHA-256:
  `08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6`
- Training receipt SHA-256:
  `6b08c5f73687759e025bf921329e12d85dbc49f45a29a34836f4cc5836c77281`
- Classifier recipe SHA-256:
  `1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982`
- Downstream metric recipe SHA-256:
  `d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85`

## Execution

Slurm job `316983` (`p1r1-vae-eval-s42`) on `talon06`: `COMPLETED`,
exit `0:0`, elapsed `00:29:43`, 8 CPUs, 32G requested, 2-hour limit,
no requeue, zero restarts, batch MaxRSS `2,781,472K`. One submission; one
attempt; no retry. Phases reported `PREFLIGHT`, `TRAIN`, `RESTORE` and
`BOUNDED_EVALUATION` PASS.

Partitions: train 8,117; validation 1,158; held-out test 2,323; five classes
with governed per-class counts. `test_accessed=false`.

Both arms shared initial weights
`4e5a33f021164e0d148bba5f058d0dc01a0cd8ca48ad3f0b7c07f9013e7a19ba`,
used EvalCNN, no class weighting, Adam at the accepted rate, batch 128,
maximum 20 epochs, patience 5, real-validation selection and best-weight
restoration. Both selected epoch 19 (zero-based) and stopped at the maximum
epoch count.

| Arm | Training examples/epoch | Optimizer updates | Selected weights |
|---|---:|---:|---|
| Real-only | 8,117 | 1,280 | `33ad5a3c…` |
| Real+Synthetic | 18,117 | 2,840 | `ed4624ff…` |

The accepted pool contributes 10,000 **distinct** synthetic samples
(2,000/class). Augmented training presented those 10,000 samples once per
epoch across 20 epochs, i.e. **200,000 synthetic presentations** and 2,840
augmented optimizer updates. Distinct samples and repeated presentations are
not the same quantity.

## Accepted results

Metrics were independently recomputed from the persisted prediction arrays
using the accepted metric implementation and recipe; every value agreed
exactly with the stored reports (difference 0.0).

| Metric | Real-only | Real+Synthetic | Difference |
|---|---:|---:|---:|
| Macro F1 | 0.6465410253 | 0.6844622754 | +0.0379212501 |
| Balanced accuracy | 0.6362720813 | 0.6669344713 | +0.0306623900 |
| Macro AUPRC | 0.7401845206 | 0.7718072623 | +0.0316227417 |
| Brier score | 0.4233299249 | 0.3730527112 | −0.0502772137 |
| ECE | 0.1416111567 | 0.0433544592 | −0.0982566975 |

Per-class F1:

| Class | Support | Real-only | Real+Synthetic | Delta |
|---|---:|---:|---:|---:|
| 0 | 251 | 0.4564705882 | 0.5381355932 | +0.0816650050 |
| 1 | 420 | 0.5261744966 | 0.6167290886 | +0.0905545920 |
| 2 | 782 | 0.8318284424 | 0.8202368138 | −0.0115916287 |
| 3 | 510 | 0.7016848365 | 0.7134637514 | +0.0117789150 |
| 4 | 360 | 0.7165467626 | 0.7337461300 | +0.0171993674 |

Class 2 declined; the other four classes improved. This heterogeneity is
preserved without a causal explanation.

## Synthetic-data diagnostics

Recorded, artifact-authenticated values (not recomputed in the final review):

- Pooled balanced KID mean: `0.577493` (10 recorded draws).
- Pooled balanced FID: `382.674` (625 real vs 625 synthetic features,
  2048 dimensions).
- Macro class-conditional FID: `400.234` (complete five-class coverage).
- Macro intra-class 3-scale MS-SSIM: `0.937243` (1,000 pairs; 200/class).

The pooled FID uses singular empirical covariance matrices without
regularization; its documented finite-sample limitation remains. Expensive
feature extraction and metric replay were not repeated during the read-only
review. Synthetic quality and downstream classification utility are distinct
research dimensions.

## Verification

- 7/7 unified verification groups passed.
- No status blockers; no lineage blockers.
- Verified synthetic-pool consumption bound to the augmented selected weights.
- Paired status `LINKED_NOT_COMPLETE_EVALUATION`; `complete_evaluation=false`
  is the bounded-cell marker, not a failure of this experiment.
- Both selected classifier bundles were published, hash-matched and restored
  in a separate process (train PID `1978334`, restore PID `1981436`).
- Held-out predictions and labels persisted; ordered labels identical across
  arms; supports `251/420/782/510/360`.

## Artifact locations

Original run:
`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/runs/attempt01`

Logs:
`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/logs`

External authorization record:
`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/authorizations/cicmaldroid2020_seed42_evaluation.json`

`artifact_inventory.tsv` records exact SHA-256 and byte size for every
primary artifact, including arrays, classifier states, predictions and the
unsupported-execution authorization record. Compact text/JSON evidence is
copied into this directory; prediction arrays, classifier weights, Inception
features and the authorization record remain external.

## Interpretation boundaries

For this single cell, the augmented arm showed higher macro-F1 (about
+0.0379) and higher balanced accuracy and macro-AUPRC, with lower Brier score
and ECE, on the governed held-out real test partition. One of five classes
declined.

This does not establish statistical significance, general VAE superiority,
uniform per-class benefit, universal benefit of synthetic augmentation, or
completion of the Revision-2026 benchmark. Cross-seed reliability and
aggregate inference remain future work under the accepted protocol.

## Non-copied artifacts

Classifier weight states, prediction arrays, Inception feature arrays, pair
scores and the authorization record are not stored in Git. They remain at the
external paths above with the digests recorded in `artifact_inventory.tsv`.
No raw dataset content or synthetic images are included.

## Evidence integrity

`SHA256SUMS` covers every file in this directory except itself. Copied files
were verified against their external originals during preservation.
