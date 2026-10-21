# MalNet-Image Validation — Paper 1 Revision 2026

## Scope

This record documents revision-specific validation of MalNet-Image for the
2026 revision of Paper 1 under the TrustForge benchmark.

This record does not replace or rewrite historical Paper 1 evidence.

## Dataset role

MalNet-Image is being added as a large-scale malware-image benchmark dataset.

For revised Paper 1, the intended task is:

- dataset: MalNet-Image
- representation: image
- classification level: type
- official split fraction: `1.0`
- official provider split policy: train / validation / test
- TrustForge partition case: Case 1 — preserve provider partitions

## Official split metadata

The official split archive is:

`~/gencys/data/MalNet-Image_raw/split_info/split_info.zip`

The official type-level `1.0` split contains:

- train: 883,416 samples
- validation: 126,202 samples
- test: 252,406 samples
- total: 1,262,024 samples

The official split contains 47 type classes.

TrustForge will preserve these provider-defined train / validation / test
partitions exactly.

TrustForge will not recombine and resplit MalNet-Image.

## Extracted corpus validation

The extracted image corpus is located at:

`~/gencys/data/MalNet-Image_raw/malnet-images`

Observed unique PNG count:

- 1,262,025

Official type-level `1.0` unique split paths:

- 1,262,024

Every official split entry maps to an extracted PNG.

No official type-level `1.0` split entry is missing from the extracted corpus.

Exactly one extracted PNG is not referenced by the official type-level `1.0`
split:

`malnet-images/adware/mobclick/CDCC51ED70F35D9515AF6CA48B02013DA99C221A1746AC9403787F3E825A54D8.png`

The raw file must be preserved.

Structural inspection established that this file is:

- format: PNG
- dimensions: 256 × 256
- mode: RGB
- bands: R, G, B
- Pillow verification: OK
- file SHA-256:
  `f661a0b8593312b684806ece95e2a7faff1bf04a9e2812f2f7b30991e6db9cf7`

The sample SHA-256 encoded in the filename was searched across every `.txt`
member of the official split archive.

Result:

- no official split reference was found

The file is therefore a structurally valid extracted MalNet image artifact that
is outside the official split metadata universe.

Because it is not part of the official type-level `1.0` split universe, it is
not part of the revised Paper 1 MalNet-Image benchmark sample universe.

## TrustForge partition policy

MalNet-Image is governed by TrustForge Partition Policy Case 1:

- provider train -> TrustForge train
- provider validation -> TrustForge validation
- provider test -> TrustForge test

No benchmark-derived validation split is required.

The official provider split is authoritative for revised Paper 1.


## Stratified image structure audit

A deterministic stratified structure/readability audit was conducted across
the official type-level `1.0` split using partition seed 42.

Sampling policy:

- 5 images per type
- per split
- across train / validation / test
- all 47 types represented

Audit coverage:

- train: 235 verified images
- validation: 235 verified images
- test: 235 verified images
- total: 705 verified images
- errors: 0

Observed structural signature for all 705 audited images:

- format: PNG
- dimensions: 256 × 256
- mode: RGB
- bands: R, G, B
- Pillow verification: OK

This establishes broad structural consistency across all 47 official type
classes and all three official provider partitions.

A full 1,262,024-image readability sweep has not been performed and is not
currently required by the revision policy.

## Current validation status

Established:

- full image corpus acquired
- official split archive acquired
- official type-level `1.0` split selected
- official split counts established
- 47 type classes established
- official sample universe reconciled against extracted PNGs
- all official split entries present
- one unreferenced extracted PNG identified
- provider train / validation / test preservation policy approved

Not yet completed:

- final preparation adapter implementation
- integration with revised TrustForge execution paths

## Status

Dataset selected for revised Paper 1: APPROVED

Task level — type classification: APPROVED

Official split fraction — `1.0`: APPROVED

Provider train / validation / test preservation: APPROVED

Official split universe as revised Paper 1 sample universe: APPROVED

Unreferenced PNG deletion: NOT AUTHORIZED

Historical evidence mutation: NOT AUTHORIZED

Final execution integration: NOT YET COMPLETED
