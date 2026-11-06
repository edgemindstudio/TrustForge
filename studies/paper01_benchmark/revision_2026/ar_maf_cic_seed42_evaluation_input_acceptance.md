# CICMalDroid2020 seed-42 evaluation input diagnostic

**Decision.** PASS for the bounded input diagnostic.

Observed on Talon job `311073` on `talon07` with `gcs_py310`
at TrustForge revision `ef99cec7203711d12d3b8264bfb3be97df591087`.
The diagnostic printed `CIC_EVALUATION_INPUT_DIAGNOSTIC=PASS`.

| Governed split | Count | Class counts in model ID order 0–4 |
| --- | ---: | --- |
| Train | 8117 | 877, 1470, 2732, 1782, 1256 |
| Validation | 1158 | 125, 210, 390, 254, 179 |
| Test | 2323 | 251, 420, 782, 510, 360 |

The diagnostic verified the dataset configuration digest and governed
dataset provenance. It checked the source-to-model class mapping,
batch-label alignment, and canonical float32 40 × 40 × 3 images
with finite values in [0, 1] across all three splits.

Both saved AR and MAF archives matched the SHA-256 digests recorded
in [the synthesis pilot acceptance](ar_maf_cic_seed42_synthesis_acceptance.md).
Each archive contained five valid images with ordered conditioning IDs
[0, 1, 2, 3, 4].

Paired evaluator execution, semantic class fidelity, and downstream
utility remain unverified. Full Paper 1 benchmark status remains **NO-GO**.
