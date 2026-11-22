# VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-02 Train–Checkpoint–Restore Evidence

**Status:** Accepted bounded experimental evidence, preserved for Paper 1 Revision-2026.

## Accepted identity

- Dataset: `cicmaldroid2020`
- Generator: `vae`
- Experimental seed: 42
- Attempt: 02
- Run identity: `commit1aec01c_attempt02`
- Frozen execution-source commit: `1aec01c26fab51b87362a47e2cfa797bd428c359`
- Approval-ledger commit: `83a297ad6cb486a0a21c8758b99df5609479fe9b`
- Authorization ID: `P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION`
- Approved authorization-scope SHA-256:
  `9218772997ffbb112dd8df20a0058e1e75ac8406bae05338f305075cacf40f93`

## Training evidence

Slurm job `316796`:

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:02:27`
- Node: `talon07`
- MaxRSS: `860356K`
- Completed epochs: 2
- Completed optimizer updates: 254
- Validation total loss: `4508.749372571244`

Training stdout:

- Repository copy: `training.stdout`
- SHA-256: `3f67ecbd0fc33a04ff8b35c62e54ad6d3f4d4457715b5becffc5e1c5387bf898`
- External original:
  `/home/bruno.fonkeng/TrustForge-paper01-vae-stage4-attempt02/p1r1-vae-prod.316796.out`

Training stderr:

- Repository copy: `training.stderr`
- SHA-256: `d9ef2cfe960a00645ead9d757ad414133cae9a64b79471257c4a3ade02609f8c`
- External original:
  `/home/bruno.fonkeng/TrustForge-paper01-vae-stage4-attempt02/p1r1-vae-prod.316796.err`

Training scheduler accounting:

- Repository file: `training_scheduler.sacct.txt`
- This file was collected subsequently from Slurm accounting during evidence preservation; it was not part of the original job stdout/stderr.
- It records `COMPLETED`, `0:0`, 8 CPUs, 32G requested memory, `talon07`.

## Checkpoint evidence

Publication root:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/checkpoints/production/commit1aec01c_attempt02`

BEST generation:

- Generation ID: `35ca3cd756a88d6b93dcc291f02c4747`
- Manifest SHA-256:
  `f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`
- Commit-record SHA-256:
  `f1fcd7cdc8ac7ee99edf1e794eab726a0219213b68007b3a882e5dbebf282118`

LAST generation:

- Generation ID: `0d808f873b163a2d054f87aac34f3043`
- Manifest SHA-256:
  `ba2a5231db0ef90ac9d724525e61cec9fa4bda6c0838740b52312a13cd8816ac`
- Commit-record SHA-256:
  `a98b11cfbe7119944ee13c42eadf95660ab278d713636faeb0e2741575d95f15`

Shared model-state digests:

- Encoder state:
  `c308765d65090a97ca872133ed833d2d6f975c52bb8a031aceb2005a5faa0141`
- Decoder state:
  `c7b4f3ff56a13c416f0997e4dd375812cd1db3b6bdadd60d60770a97d06557b0`
- Dataset provenance:
  `e6736e56b7bd8f2e81505d3f8b1811dfc95a0e0514c629d524025a88a03138ae`

Both roles legitimately contain identical model-state bytes because the final
epoch was also the selected best epoch. The generation identities and logical
roles remain distinct.

Repository evidence copies:

- `best_checkpoint_manifest.json`
- `last_checkpoint_manifest.json`
- `best_commit_record.json`
- `last_commit_record.json`
- `best_dataset_provenance.json`
- `last_dataset_provenance.json`
- `checkpoint_inventory.tsv`

`checkpoint_inventory.tsv` records the exact absolute path, SHA-256, size,
mode, owner and group for all ten checkpoint-publication files. Model-state
NPZ files are **not copied** into Git.

## Independent restoration evidence

Restoration Slurm job `316811`:

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:00:09`
- Node: `talon07`
- CPUs: 8
- Memory: 32G
- Wall time: 00:30:00
- Process ID: `3935002`
- Python: 3.10.19
- GPU disabled

Accepted results:

- Both committed generations verified.
- BEST governed restoration: PASS.
- Restored metadata equality: PASS.
- Restored dataset-provenance equality: PASS.
- Architecture fingerprint: PASS.
- Encoder variables: 16.
- Encoder parameters: 3,423,400.
- Decoder variables: 8.
- Decoder parameters: 734,659.
- All restored floating-point variables finite.
- All ten checkpoint-publication files unchanged.
- No optimizer state loaded.
- No training resumed.
- No synthetic samples generated.

Repository evidence copies:

- `restoration_driver.py` — SHA-256
  `091b20685ca0a2bd72abbd52d0e9d4ec602ff21b50bb7e2d3ef29aea1ffe4fd9`
- `restoration_command.sh` — SHA-256
  `7dc8366a881cfd958f6291b4d89b7736f952f54efc56e5a5f41a76f7baa9c53f`
- `restoration.stdout` — SHA-256
  `b3892fd9bb1aada05865ce126b61fb5aa191d29ea79cb95280af5b85189fe93a`
- `restoration.stderr` — SHA-256
  `9a47041f4b9e035663bea485d3bad1c7c848276c9a9019270bf5b6bce8c8ff54`
- `restoration_result.json` — SHA-256
  `167f4c69e3311bea07d95c6108d40a6ba62a8c13c7d5d1efd096f3476aa1c5c9`
- `restoration_scheduler.scontrol.txt` — SHA-256
  `40dd61bd1de8ee4dc3023f0cc094bc43c434576832dd0d8fa7032ce415a783e3`
- `restoration_scheduler.sacct.txt` — SHA-256
  `a2912c8ac8ac613471037872b6f87b0e03244feeb98a693af620ece1dae6e002`

External originals:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/diagnostics/vae-stage4-cicmaldroid2020-seed42-attempt02-job316796-restoration-20261121`

## Permit identity

The live operational permit was not copied into Git and was not reissued.

- Path:
  `/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/authorizations/vae_checkpoint_publication_permit.json`
- File SHA-256:
  `65c67f73e6b82c2de816589876980ba6f8829736f4d3a39a73566a1aef433d6a`
- Complete-permit SHA-256:
  `c2f4a0c821a831c7dea4ccaf2bfe409a62c535d9f9d66319f9dad9355cc32b2b`
- Approval commit: `83a297ad6cb486a0a21c8758b99df5609479fe9b`

## Reporting discrepancy

The original production result contained:

- Top-level `checkpoint_status=NOT_PREPARED`
- Top-level `publication_status=NOT_PUBLISHED`
- Nested `checkpoint_publication.publication_status=PUBLISHED`

The actual commit records, manifests, state files and independent restoration
corroborate the nested publication result. This is preserved as a production
reporting limitation. The CLI was not repaired and the original logs were not
altered.

## Preserved limitations

- `crash_durability_verified=False`
- GPFS crash-recovery durability remains unverified.
- Same-account operational custody does not protect against a malicious
  same-UID actor.
- Restoration does not independently authenticate checkpoint origin.
- Dataset provenance does not independently prove partition execution or
  continuous source immutability.
- Successful restoration does not establish synthetic-data fidelity or
  downstream utility.

## Non-copied artifacts

The following were deliberately not copied into Git:

- Model-state NPZ files.
- Raw datasets.
- Synthetic pools.
- The live permit.
- Large scheduler or runtime logs not listed above.

## Evidence integrity

`SHA256SUMS` contains the SHA-256 inventory for every preserved repository
evidence file except itself.
