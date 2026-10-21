# TrustForge Partition Policy — Paper 1 Revision 2026

## Purpose

This record defines the dataset partition contract for the 2026 revision of
Paper 1 under the TrustForge benchmark.

The goal is to preserve dataset-provider intent wherever authoritative
partitions exist while still exposing every dataset to the TrustForge benchmark
through a common logical interface:

- train
- validation
- test

This policy does not rewrite historical Paper 1 execution evidence.

## Core principle

TrustForge must respect and preserve provider-defined dataset partitions when
they exist.

Benchmark consistency is achieved through a common logical train/validation/test
contract, not by forcing every dataset through the same physical resplitting
operation.

## Partition cases

### Case 1 — Provider supplies train / validation / test

TrustForge preserves all three partitions exactly as provided.

No benchmark-created resplitting is performed.

Example:

- MalNet-Image type-level `1.0` split

Expected behavior:

- provider train -> TrustForge train
- provider validation -> TrustForge validation
- provider test -> TrustForge test

### Case 2 — Provider supplies train / test only

TrustForge preserves the provider test partition.

Validation is derived deterministically from the provider training partition.

Expected behavior:

- provider train -> TrustForge train + validation
- provider test -> TrustForge test

The validation derivation rule must be deterministic, documented, and applied
consistently to datasets in this case.

### Case 3 — Provider supplies an unsplit labeled corpus

TrustForge derives train, validation, and test partitions deterministically.

The split policy must be:

- reproducible
- label-aware where appropriate
- leakage-safe
- documented
- applied before model training

Example:

- MaleX v1 after removal of conflicting duplicate SHA-256 identities

## Dataset-specific application

### USTC-TFC2016

Current source condition:

- train / test style prepared dataset

Revision policy:

- preserve provider/prepared test
- derive validation from training data under the governed Case 2 rule

### CICMalDroid2020

Authoritative source condition:

- unsplit labeled corpus from `feature_vectors_syscalls_frequency_5_Cat.csv`
- 11,598 rows
- 139 numeric features
- 5-class labels
- historical Stage 3 train/validation/test partitions were created by the
  earlier TrustForge preparation pipeline and are not provider-defined splits
- historical `pipeline_paper1` train/test packaging is therefore not treated
  as authoritative partition provenance for the revision

Revision policy:

- derive train / validation / test directly from the authoritative unsplit
  labeled corpus under the governed Case 3 rule
- use the governed 70% / 10% / 20% train/validation/test ratios
- use deterministic partition seed 42
- stratify by class label
- do not preserve the historical Stage 3 70% / 15% / 15% split as the
  revision partition contract

### MalNet-Image

Current source condition:

- official train / validation / test partitions supplied by dataset authors

Revision policy:

- preserve official type-level `1.0` train / validation / test partitions
- do not recombine and resplit
- the official split universe is authoritative for Paper 1

A single extracted PNG was identified outside the official type-level `1.0`
split universe. The raw file is preserved but is not part of the Paper 1
benchmark sample universe.

### MaleX

Current source condition:

- labeled corpus with no authoritative train / validation / test split yet
  established for this revision

Revision policy:

- first exclude all 19 SHA-256 identities that occur in both benign and malware
  label sets
- then create a deterministic train / validation / test split under the
  governed Case 3 rule

## Common TrustForge contract

Regardless of source partitioning, every dataset must enter the benchmark core
as:

- train
- validation
- test

Downstream generator and evaluation code should consume this common logical
contract without needing to know how the source dataset originally supplied or
derived the partitions.

## Historical compatibility

The historical loader in `common/data.py` currently expects only:

- `train_data.npy`
- `train_labels.npy`
- `test_data.npy`
- `test_labels.npy`

and derives validation by splitting the supplied test set.

That historical behavior is not the target policy for the 2026 revision.

Historical scientific artifacts and accepted execution evidence must not be
silently rewritten.

Any implementation change must preserve compatibility deliberately or provide a
clear revision-specific path.

## Status

Policy principle: APPROVED

Common logical contract: APPROVED

Provider-defined split preservation: APPROVED

Validation-from-training rule for train/test-only datasets: APPROVED IN PRINCIPLE

Unsplit-corpus three-way partition rule: APPROVED IN PRINCIPLE

Exact validation fraction for Case 2: 10% of provider training partition

Exact unsplit train/validation/test ratios for Case 3: 70% / 10% / 20%

Exact deterministic partition seed: 42

Partition derivation requirements:

- Case 2 validation derivation must be deterministic and stratified by label.
- Case 3 train/validation/test derivation must be deterministic and stratified by label.
- Partition seed 42 is a dataset-partition reproducibility seed and is distinct from experimental/model seeds.

Implementation design: NOT YET COMPLETED
