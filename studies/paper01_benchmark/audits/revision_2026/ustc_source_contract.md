# Paper 1 Revision 2026 — USTC-TFC2016 Source Contract

Status: VALIDATED
Scope: Revised Paper 1 benchmark
Dataset: USTC-TFC2016

## 1. Provider-style source artifacts

Primary historical source directory:

    ~/gencys/data/USTC-TFC2016_malware

Observed arrays:

### Training

- image shape: `(81000, 1, 40, 40)`
- layout: NCHW
- image dtype: `float64`
- numeric range: `[0,1]`
- label shape: `(81000,)`
- label dtype: `int32`
- classes: `0..8`
- class count: `9000` samples per class

### Test

- image shape: `(9000, 1, 40, 40)`
- layout: NCHW
- image dtype: `float64`
- numeric range: `[0,1]`
- label shape: `(9000,)`
- label dtype: `int32`
- classes: `0..8`
- class count: `1000` samples per class

## 2. Source artifact hashes

Primary NCHW source:

### train_data.npy

    sha256=4dc2fd447fc9bc7751fb480a1372b38293b09463d6751b1af1af521d05e880b0

### test_data.npy

    sha256=dcc1b09ef5e8c82c7d93b5da712e69abb3f74f1b092a9afccba243b141c0e95f

### train_labels.npy

    sha256=1781f9681217dba24967f56a4c17b84e2b95d06e89a683976277a80de46c5416

### test_labels.npy

    sha256=9842101d83e8209ddf8d1c0b4e7bfcc6a1be3ff4b5b394fae974b6552a9d7dee

## 3. Verified NHWC convenience copy

A second directory exists:

    ~/gencys/data/USTC-TFC2016_malware_nhwc

Observed image shapes:

- train: `(81000, 40, 40, 1)`
- test: `(9000, 40, 40, 1)`

The labels are exactly identical to the primary source labels.

The NHWC image arrays were compared against:

    np.transpose(nchw, (0, 2, 3, 1))

Results:

- training transpose shape match: PASS
- training exact value equality: PASS
- training maximum absolute difference: `0.0`
- test transpose shape match: PASS
- test exact value equality: PASS
- test maximum absolute difference: `0.0`
- training labels exact equality: PASS
- test labels exact equality: PASS

Therefore the NHWC directory is treated as a verified representation-only
convenience copy of the NCHW source.

Its differing image-file hashes reflect different stored array shape/layout,
not different image values.

## 4. NHWC convenience-copy hashes

### train_data.npy

    sha256=885c9925b0757a9409698f2dbb72be37fcd3838ab938c6d1065ee7438771d8c0

### test_data.npy

    sha256=208ff0626f8309a87ba3390551bc538d3ff1a502d682bc3a089f908596e1332b

### train_labels.npy

    sha256=1781f9681217dba24967f56a4c17b84e2b95d06e89a683976277a80de46c5416

### test_labels.npy

    sha256=9842101d83e8209ddf8d1c0b4e7bfcc6a1be3ff4b5b394fae974b6552a9d7dee

## 5. Revision-2026 partition semantics

USTC-TFC2016 is governed as partition-policy Case 2:

- provider training partition is preserved as the source from which revised
  training and validation partitions are derived;
- provider test partition remains the revised benchmark test partition;
- validation is derived from 10% of provider training;
- partition seed is 42;
- splitting is stratified by class label.

Because provider training contains exactly 9000 samples per class, the
expected revised logical counts are:

- training: 8100 samples per class
- validation: 900 samples per class
- test: 1000 samples per class

Across 9 classes this yields:

- training: 72,900
- validation: 8,100
- test: 9,000

## 6. Revision-2026 canonicalization semantics

USTC source numeric convention:

    SourceNumericRange.ZERO_ONE

The revised canonical path is:

    source image
      -> explicit layout normalization
      -> float32 [0,1]
      -> preserve 40x40 spatial resolution
      -> replicate grayscale channel to RGB
      -> 40x40x3 float32 [0,1]

No spatial interpolation is required because the source is already 40x40.

No heuristic numeric-range inference is permitted.

## 7. Historical loader boundary

The historical `common.data.load_dataset_npy()` is not used for revised
Paper 1 USTC partitioning because it derives validation from the provider
test partition.

Historical loader behavior remains unchanged for provenance.

The revision-2026 path uses the governed partitioning and canonicalization
modules instead.
