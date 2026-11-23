# VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-01 Production Synthesis Evidence

**Status:** Researcher-accepted bounded production evidence, preserved for
Paper 1 Revision-2026. Synthetic-pool correctness and provenance only.

## Accepted identity

- Dataset: `cicmaldroid2020`
- Generator: `vae`
- Experimental seed: 42
- Attempt: 01
- Checkpoint role: `best`
- Checkpoint run identity: `commit1aec01c_attempt02`
- BEST checkpoint generation:
  `35ca3cd756a88d6b93dcc291f02c4747`
- Historical checkpoint source commit:
  `1aec01c26fab51b87362a47e2cfa797bd428c359`
- Frozen synthesis source commit:
  `d4031a9b39565e5382d08e2b28236162498a08ca`
- Approved synthesis recipe SHA-256:
  `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9`
- Production pool manifest SHA-256:
  `08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6`

Authorization for this single execution was the researcher-approved
source-activation commit `d4031a9` plus an explicit one-job execution release.
No separate permit artifact was issued for synthesis.

## Slurm execution

Job `316980`:

- Name: `p1r1-vae-pool`
- Partition: `talon`; account: `jielun.zhang`
- Node: `talon06`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:01:15` (limit `02:00:00`)
- Submit: `2026-11-22T22:31:01`
- Start: `2026-11-22T22:31:01`
- End: `2026-11-22T22:32:16`
- Allocation: 1 node, 1 task, 8 CPUs, 32G requested, no requeue
- Batch MaxRSS: `649092K`
- Exactly one submission; no retry, no requeue, no restart

Repository log copies:

- `synthesis.stdout` — SHA-256
  `d5c2deba30b7b71b9aa76b50eeb5cbed4f1d3165dec35bb726ca1e2100321fb4`
- `synthesis.stderr` — SHA-256
  `829e953d037eb1f1af9039aafd5d6bf841d8820e100d1e48547e35c5d8c126ca`

`synthesis.stderr` contains only TensorFlow/CUDA device-initialization notices
(no CUDA device available; GPU intentionally disabled). There is no error or
failure record.

`synthesis.stdout` ends with `VAE_POOL_PUBLICATION_COMPLETE` recording
`status=COMPLETE`, `verified_batches=315`, `verified_images=10000` and the
manifest digest above.

## Production publication

External pool destination:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/pools/production/checkpoint35ca3cd756a88d6b93dcc291f02c4747_synthesis-d4031a9_attempt01/pool`

Published contents:

- 315 batch artifacts (`.npy`), 63 per class for classes 0–4.
- 2,000 samples per class; 10,000 samples total.
- `manifest.json`, 128,201 bytes, SHA-256 `08777645...43fbcc6`.
- No `FAILURE.json`; no `.manifest.pending`.
- Pool size approximately 192,184,905 bytes (196 MB).

Manifest lineage and contract:

- `schema`: `trustforge.paper01.vae-cic-pool-manifest.v1`
- `checkpoint_generation_id`: `35ca3cd756a88d6b93dcc291f02c4747`
- `checkpoint_manifest_sha256`:
  `f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`
- `checkpoint_role`: `best`
- `checkpoint_source_commit`:
  `1aec01c26fab51b87362a47e2cfa797bd428c359`
- `synthesis_source_commit`:
  `d4031a9b39565e5382d08e2b28236162498a08ca`
- `recipe_sha256`:
  `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9`
- `latent_stream_contract`:
  `trustforge.paper01.synthesis-latent-stream.v1` (PCG64, latent dimension
  100, standard normal, NumPy 2.2.6 observed/baseline)
- `model_call_batch_size`: 32
- `quota_per_class`: 2,000
- `batches_per_class`: 63
- `complete_pool`: true; `expected_total` = `observed_total` = 10,000
- `status`: `COMPLETE`; `authoritative_artifact`: true
- `canonical_output`: float32, NHWC, `[40, 40, 3]`, range `[0.0, 1.0]`

`pool_manifest.json` is a byte-identical repository copy of the authoritative
manifest. It contains the 315 per-batch filenames, sample ranges and byte
SHA-256 digests, so it serves as the complete batch integrity inventory.

## Independent verification

Independent process, run from the frozen execution worktree, reading only the
published artifacts and never calling the generator:

```text
PYTHONPATH=$WORKTREE/src:$WORKTREE <python> - <<'PY'
from trustforge.evaluation.paper01_vae_pool import VAEPoolView
view = VAEPoolView(manifest_path, manifest_sha256="08777645...",
                   family="vae", experimental_seed=42,
                   recipe_sha256="98f9ef42...",
                   checkpoint_sha256="f1fa66c2...")
view.preflight()
PY
```

Result:

- `verified_batches` = 315
- `verified_images` = 10,000
- `npy_files` = 315; `batch_record_count` = 315
- No failure or pending marker
- `VERIFIER_PASS`

`preflight()` streams every batch record and validates, per artifact:
existence, size, byte SHA-256, shape, dtype, finiteness, canonical `[0,1]`
domain, ordered sample-index coverage and total quota. A bounded
evaluator-interface spot check at positions `[0, 31, 1999, 2000, 8000, 9999]`
returned shape `(6, 40, 40, 3)`, float32, values within `[0.355, 0.534]`, and
one-hot labels summing to 1.0.

Full output: `verification.txt`, SHA-256
`13e1526e0e202174ac440c6e677d90dfab0469ebc8b44fc319c8cba7d67a4312`.

## Documented manifest-digest transcription correction

The execution-release command text contained a 62-character manifest digest
ending `...9af53bf812e`. That value is not a valid SHA-256 and would have been
rejected by the launcher's 64-hex check before synthesis.

Under explicit researcher confirmation, the submission used the accepted
64-character digest
`f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`, which
matches the on-disk BEST manifest and the accepted checkpoint record. No other
input, destination, checkpoint or source was changed. The correction is
recorded in `submission_command.txt`; the original evidence was not altered.

## Preserved limitations

- Synthetic-data fidelity and realism were not assessed.
- Requested-label fidelity was not measured beyond one-hot request
  construction; no classifier was applied.
- Downstream classification utility is not established.
- Statistical reliability across seeds or attempts is not established.
- Cross-dataset, cross-family and GPU-path behavior are not covered here.
- Scheduler accounting was collected during preservation, not by the job.
- Same-account operational custody does not protect against a malicious
  same-UID actor.

## Accepted versus not yet established

**Accepted:** production synthesis correctness, artifact completeness,
provenance lineage, and independent byte-level integrity verification for this
one CICMalDroid2020 VAE seed-42 attempt.

**Not yet established:** synthetic fidelity, requested-label fidelity,
downstream classification utility, cross-seed statistical reliability, and
complete benchmark results.

## Non-copied artifacts

Deliberately not copied into Git: the 315 generated batch files, the 196 MB
production pool, raw datasets, checkpoint model-state NPZ files, held-out test
data and any secrets. Originals remain at the external locations above,
unmodified.

## Evidence integrity

`SHA256SUMS` records the SHA-256 of every preserved file except itself. Remote
originals were re-hashed during preservation and matched the repository copies.
