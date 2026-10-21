# CICMalDroid2020 Revised Paper-1 Source Contract

**Status:** VALIDATED

## Scope

This contract defines the authoritative CICMalDroid2020 source, partition
provenance, representation lineage, and revised preprocessing boundary for the
2026 Paper-1 revision.

Historical artifacts are retained as provenance evidence. They do not override
the revised partition policy or create provider-defined split authority where
none exists.

## Authoritative source

Selected source:

`~/gencys/data/CICMalDroid2020_raw/feature_vectors_syscalls_frequency_5_Cat.csv`

Observed source properties:

- file size: 14,626,377 bytes
- SHA-256:
  `3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00`
- rows: 11,598
- numeric feature count: 139
- raw label column: `Class`
- observed raw label values: 1, 2, 3, 4, 5
- task: 5-class multiclass malware-category classification

Observed class counts:

- class 1: 1,253
- class 2: 2,100
- class 3: 3,904
- class 4: 2,546
- class 5: 1,795

The raw label column is read by pandas as float64 in the audited source, but
all observed values are integer-valued class identifiers in the set 1..5.

## Feature-order contract

Historical Stage-3 artifact:

`~/gencys/data/CICMalDroid2020_stage3/feature_columns.json`

Observed properties:

- feature count: 139
- SHA-256:
  `7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f`

The ordered feature list is the representation-order contract for the revised
CIC preprocessing path.

Feature order must not be inferred from arbitrary dataframe column order.

## Historical partition lineage

Historical Stage 3 used:

- split seed: 42
- stratified split
- train: 70%
- validation: 15%
- test: 15%

Observed historical row counts:

- train: 8,118
- validation: 1,740
- test: 1,740

These partitions were created by the earlier TrustForge preparation pipeline.
They are not provider-defined partitions.

The historical Stage-3 stored labels were audited against the authoritative raw
CSV using the stored row indices.

Result:

- historical train label lineage: exact
- historical validation label lineage: exact
- historical test label lineage: exact
- all historical split label lineage: exact

Therefore the Stage-3 artifacts have verified row-level lineage to the
authoritative unsplit source, but their 70/15/15 partition is historical
TrustForge preparation state rather than provider partition authority.

## Revised partition contract

CICMalDroid2020 is governed as partition-policy **Case 3**.

The 2026 Paper-1 revision must derive train, validation, and test from the
authoritative unsplit labeled corpus using:

- train: 70%
- validation: 10%
- test: 20%
- deterministic partition seed: 42
- stratification by class label

The historical Stage-3 70/15/15 partitions must not be reused as the revised
benchmark partitions.

The historical `pipeline_paper1` train/test packaging must not be interpreted
as provider train/test provenance.

## Historical representation reconstruction

The historical CIC image-style representation was reconstructed directly from
the authoritative CSV and the Stage-3 preprocessing artifacts.

The recovered numerical execution semantics are:

1. read the 139 features in the recorded feature-column order;
2. provide feature values to sklearn `StandardScaler.transform` as float32;
3. use scaler parameters fitted on the historical training partition;
4. retain the float32 transformed output;
5. apply `numpy.tanh`;
6. append five zero-valued features, producing 144 values;
7. reshape row-major to `12 x 12 x 1`.

A reconstruction using an instantiated sklearn `StandardScaler` with the
historical fitted parameters reproduced the stored Stage-3 image tensors
bit-for-bit:

- train: exact equality; maximum absolute delta = 0
- validation: exact equality; maximum absolute delta = 0
- test: exact equality; maximum absolute delta = 0

The final five flattened positions in every audited stored tensor were exactly
zero, confirming the historical 139-to-144 zero-padding boundary.

This establishes the historical representation transformation by empirical
reconstruction even though the original preparation script was not recovered.

## Revised preprocessing boundary

The revised benchmark preserves the verified representation semantics but not
the historical split-dependent fitted state.

The revised CIC preprocessing sequence is therefore:

1. load the authoritative unsplit source;
2. validate source identity, feature order, labels, and row count;
3. derive the governed Case-3 70/10/20 partition using seed 42 and
   stratification;
4. fit a new `StandardScaler` on the revised training partition only;
5. transform revised train, validation, and test using the fitted scaler with
   the verified float32 execution semantics;
6. apply `tanh`;
7. append five zeros to obtain 144 values;
8. reshape row-major to `12 x 12 x 1`;
9. convert from the CIC source representation range `[-1, 1]` to `[0, 1]`;
10. resize from `12 x 12` to the governed canonical `40 x 40` representation
    using the revision canonicalization policy;
11. replicate the grayscale channel to produce canonical `40 x 40 x 3`;
12. expose float32 NHWC tensors in `[0, 1]`.

The historical fitted scaler must not be reused for revised experiments because
its parameters depend on the superseded historical 70/15/15 training
partition.

## Provenance boundary

The following distinctions are binding:

- raw source identity != prepared historical tensor identity
- historical partition lineage != provider-defined partition authority
- historical fitted scaler state != revised fitted scaler state
- verified representation semantics != permission to reuse historical
  split-dependent preprocessing parameters
- historical execution evidence != revised experiment evidence

Historical CIC artifacts remain preserved as provenance.

## Revision status

Authoritative source identity: VALIDATED

Feature-order lineage: VALIDATED

Historical label lineage: VALIDATED

Historical representation reconstruction: VALIDATED

Historical numerical execution semantics: VALIDATED

Revised partition classification: CASE 3

Revised partition rule: 70% / 10% / 20%, stratified, seed 42

Historical scaler reuse for revised experiments: PROHIBITED

Revised CIC adapter implementation: VALIDATED

Real-data adapter smoke test: PASS

Dedicated CIC revision regression suite: 8 / 8 PASS

Focused Paper-1 revision integration suite after package export: 35 / 35 PASS

Canonical TrustForge repository suite: 704 PASS, 3 subtests PASS

TrustForge doctor with canonical Talon logical storage roots:
0 failures, 1 expected dirty-worktree warning
