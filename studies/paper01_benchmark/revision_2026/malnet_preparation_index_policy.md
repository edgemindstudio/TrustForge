# MalNet-Image Preparation-Index Policy — Paper 1 Revision 2026

## Status

GOVERNED DESIGN — implementation not yet completed.

## Purpose

MalNet-Image contains 1,262,024 official type-level benchmark samples.

The authoritative provider split metadata occupies approximately 95.8 MiB
uncompressed and contains variable-length relative paths.

TrustForge must preserve exact provider membership and ordering without
materializing the complete path universe as resident Python string objects.

## Authority boundary

The authoritative scientific source remains:

`~/gencys/data/MalNet-Image_raw/split_info/split_info.zip`

SHA-256:

`74e3615aa848c27dc06bb9ea2ae53e415f4b842fdce8dc45ba50849db8851492`

Authoritative members:

- `split_info/type/1.0/train.txt`
- `split_info/type/1.0/val.txt`
- `split_info/type/1.0/test.txt`

The preparation index is derived infrastructure only.

It must not create, remove, reorder, relabel, deduplicate, or otherwise redefine
provider membership.

`PROVIDER SPLIT METADATA != PREPARATION INDEX`

`PREPARATION INDEX != SCIENTIFIC AUTHORITY`

## Scale evidence

Observed authoritative metadata:

- train rows: 883,416
- validation rows: 126,202
- test rows: 252,406
- total rows: 1,262,024
- uncompressed bytes: 100,429,218
- uncompressed size: approximately 95.777 MiB
- compressed size: approximately 49.567 MiB
- observed path length: 72 to 103 bytes
- mean path length: approximately 78.58 bytes

The revised adapter must avoid requiring all 1,262,024 paths as ordinary
Python string objects simultaneously.

## Governed preparation representation

For each provider split, TrustForge will derive:

1. a UTF-8 text file containing the exact non-empty provider lines in original
   provider order;
2. an `int64` byte-offset array enabling direct random access.

Conceptually:

- `train.txt`
- `train_offsets.npy`
- `validation.txt`
- `validation_offsets.npy`
- `test.txt`
- `test_offsets.npy`

These preparation artifacts must remain outside Git under governed TrustForge
data storage.

They must not replace or mutate the original provider archive.

## Text preservation rule

For every logical sample position `i`, the prepared logical line must equal the
corresponding authoritative provider line after removal only of its line
terminator.

No sorting, case conversion, path normalization, relabeling, deduplication, or
filesystem-derived reordering is permitted.

## Offset rule

Each offset array must:

- use `int64`;
- contain one starting byte offset per sample plus one terminal offset;
- have length `N + 1` for a split containing `N` samples;
- begin at byte offset `0`;
- be monotonically increasing;
- end at the exact prepared-text byte length;
- permit exact reconstruction of every provider entry.

## Runtime access

The revised MalNet adapter must:

1. memory-map the prepared split text;
2. memory-map the corresponding offset array;
3. validate requested logical positions;
4. decode only requested provider entries;
5. resolve those entries beneath the authoritative image root;
6. decode only requested PNG files;
7. canonicalize only the requested image batch.

The complete canonical MalNet dataset must not be materialized in memory.

## Path and label semantics

Every provider entry has the form:

`<type>/<family>/<sample_sha>`

The source PNG resolves as:

`malnet-images/<type>/<family>/<sample_sha>.png`

The first path component is the authoritative provider type label.

The governed TrustForge numerical encoding is defined by:

`studies/paper01_benchmark/audits/revision_2026/malnet_image_type_mapping.json`

Mapping SHA-256:

`84406057e188a35a98ec7237275bf1bce22df037d627c40d2ded5d735020b253`

The preparation layer must not introduce an independent label mapping.

## Integrity requirements

Preparation must validate and record:

- authoritative archive SHA-256;
- provider member name;
- provider row count;
- prepared row count;
- prepared text byte count;
- prepared text SHA-256;
- offset-array SHA-256;
- offset dtype;
- offset length;
- first offset;
- terminal offset;
- exact provider-line reconstruction.

Expected row counts:

- train: 883,416
- validation: 126,202
- test: 252,406

Any mismatch must fail closed.

## Mutation and regeneration

Preparation artifacts are reproducible derivatives.

They may be regenerated from the authoritative provider archive, but
regeneration must not modify:

- the provider archive;
- the raw MalNet image corpus;
- historical Paper-1 artifacts.

## Storage boundary

Large preparation artifacts remain outside Git.

Git may contain only:

- preparation code;
- tests;
- policies;
- compact provenance and integrity manifests.

## Implementation status

Preparation-index design: GOVERNED

Authoritative provider archive: VALIDATED

Provider split membership: VALIDATED

Provider ordering preservation: REQUIRED

Full Python-string materialization: NOT REQUIRED

Memory-mapped random access: REQUIRED

Preparation implementation: NOT YET COMPLETED

MalNet adapter integration: NOT YET COMPLETED
