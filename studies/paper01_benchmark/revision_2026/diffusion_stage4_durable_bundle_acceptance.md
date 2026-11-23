# Diffusion Stage 4 — Durable Checkpoint Bundle Acceptance

**Recorded:** 2026-11-23
**Status:** `DIFFUSION_STAGE4_DURABLE_BUNDLE=ACCEPTED_AND_CLOSED`
**Scope:** Bounded implementation and fixture-verification milestone only.

## A. Original scientific requirement

Research Continuity Charter §6, Stage 4:

> **Checkpoint/synthesize:** Durable, integrity-checked publication; fresh
> restoration; identity-safe, fully counted synthesis for each distinct family
> integration.

This milestone addresses only the Diffusion durable-checkpoint publication and
independent-loading implementation. It does not establish complete Stage 4
readiness and does not execute training, checkpoint publication or synthesis.

## B. Accepted implementation

Accepted source paths and hashes:

- `src/trustforge/models/paper01_revision_2026/diffusion_checkpoint_bundle.py`
  SHA-256 `710030c1984dc4abcb371156ec2f34d6b6aa8aecbb698b76a81ad5a049a1bd91`
- `tests/test_diffusion_checkpoint_bundle_revision_2026.py`
  SHA-256 `3b468e5d7eed913b8816d326721cf96e798a61f15f1eb46e1f8fbd8bb1578fef`

Accepted behavior:

- Reuses the accepted prepared best/last Diffusion checkpoint states and the
  accepted state digest and scientific-configuration definitions.
- Serializes each tensor as a separate `.npy` file (`allow_pickle=False`) with
  recorded filename, dtype, shape, byte size and SHA-256.
- A canonical manifest binds schema, dataset, experimental seed, ordered class
  mapping, class count, accepted model/DDPM scientific configuration, source
  commit, dataset provenance, and per-role selection metadata.
- Preserves the distinction between the validation-selected `best` state and
  the final `last` state, including selected epoch, completed updates and
  validation MSE.
- Verifies each prepared state against the accepted model-state digest before
  publication.
- Publishes exclusively: refuses an existing destination, creates the
  destination with restrictive permissions, writes a `.bundle.pending` marker,
  writes tensors, writes the manifest last, then removes the marker.
- Distinguishes incomplete publication: any residual pending marker or
  unexpected bundle entry is rejected by the loader.
- Loads from a published directory using a caller-supplied manifest digest,
  proving the bundle identity without dependence on the original training
  process, in-memory objects, hidden globals or a machine-specific path.
- Verifies manifest fields, identity, configuration, tensor filenames, byte
  sizes, SHA-256 values, dtypes, shapes, the accepted state digest, and the
  source commit when supplied.
- Reconstructs the existing prepared-state representation, preserving
  `publication_status="NOT_PUBLISHED"` required by the accepted
  `restore_diffusion_checkpoint_state()` contract.

No accepted training, checkpoint-selection or sampling mathematics was changed.
The accepted conditional U-Net, 1,000-step linear DDPM, epsilon-MSE objective,
Adam controls, two-epoch policy, earliest-minimum-validation-MSE selection,
best/last semantics, dataset identities, seeds and class conditioning are
unchanged.

## C. Verification evidence

OpenCode-executed fixture tests before this preservation:

```bash
pytest -q tests/test_diffusion_checkpoint_bundle_revision_2026.py
# 19 passed in 4.79s

pytest -q \
  tests/test_diffusion_checkpoint_bundle_revision_2026.py \
  tests/test_diffusion_checkpoint_state_revision_2026.py \
  tests/test_paper01_synthesis_linkage_revision_2026.py \
  tests/test_diffusion_production_entry_point_revision_2026.py
# 118 passed in 3.43s
```

Independent source review verified the complete implementation and test source,
the exact file hashes, the unchanged accepted dependency hashes, the unchanged
training gate, and the absence of production entry-point or launcher changes.
Tests were not rerun during preservation.

Coverage includes: best/last round-trip; independent restoration into a
separately constructed compatible model; role preservation; wrong
dataset/seed/class-mapping/configuration rejection; corrupted and missing
tensor rejection; manifest/state digest contradiction rejection;
partial-publication rejection; overwrite refusal; and compatibility with the
existing synthesis restoration interface.

## D. Acceptance decision

`DIFFUSION_STAGE4_DURABLE_BUNDLE=ACCEPTED_AND_CLOSED`

This closes only the bounded durable-bundle implementation and fixture
verification. It does not close Stage 4 overall, authorize runtime activation,
or assert production checkpoint-backed synthesis.

## E. Remaining original Stage 4 requirements

Not yet established:

1. Production entry-point integration with the durable checkpoint publisher.
2. Production synthesis-loader integration binding the requested checkpoint
   digest to the verified published state.
3. Authorized real-data Diffusion training and checkpoint publication.
4. Actual fresh-process production restoration.
5. Production synthesis and independently verified synthetic-pool completion.

`STAGE_4_OVERALL=IN_PROGRESS`

Implementation compatibility must not be presented as completed production
execution.

## F. Scientific preservation

`DIFFUSION_STAGE3_ENTRY_INTEGRATION=ACCEPTED_AND_CLOSED`

`STAGE5_AGGREGATION_LINKAGE=ACCEPTED_AND_CLOSED`

`VAE_CIC_SEED42_PAIRED_EVALUATION=ACCEPTED_AND_CLOSED`

`RBM_S2_01=BLOCKED`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

The accepted four datasets, seven generator families, seven seeds, fixed primary
synthetic budget, dataset partitions, model specification, classifier protocol
and statistical reporting protocol remain unchanged.

`DIFFUSION_REAL_DATASET_TRAINING_ENABLED` remains `False`.

## G. Limitations

- No actual Diffusion production checkpoint was created.
- Independent loading was fixture-tested, not demonstrated through a separately
  launched production process.
- Dataset provenance is recorded in the manifest but is not independently
  authenticated by this bundle.
- The existing synthesis restoration interface does not yet bind the requested
  checkpoint SHA-256 to the verified published state.
- No production training, synthesis or Slurm execution occurred.

These are scope limitations of the bounded milestone, not reasons to reopen the
accepted implementation.
