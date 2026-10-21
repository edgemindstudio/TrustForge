# TrustForge Canonical Representation Policy — Paper 1 Revision 2026

## Purpose

This record defines the canonical image representation for the revised
TrustForge Paper 1 benchmark.

The revised benchmark is methodologically authoritative for new experiments.
Historical TrustForge artifacts remain preserved as historical evidence, but
questionable historical preprocessing or implementation choices do not constrain
the revised benchmark.

## Core benchmark principle

TrustForge evaluates heterogeneous malware-image datasets through a common,
controlled benchmark representation.

Dataset provenance, source files, sample identities, labels, and authoritative
provider partitions are preserved separately from the representation supplied
to the generator families.

Therefore:

SOURCE REPRESENTATION != TRUSTFORGE CANONICAL BENCHMARK REPRESENTATION

## Canonical TrustForge representation

The canonical image representation for revised Paper 1 is:

- height: 40 pixels
- width: 40 pixels
- channels: 3
- layout: NHWC
- numeric dtype presented to models: float32
- numeric range presented to models: [0, 1]

Canonical tensor shape:

`40 x 40 x 3`

Per-image dimensionality:

`40 * 40 * 3 = 4,800`

## Dataset-specific source mapping

### USTC-TFC2016

Source benchmark representation:

`40 x 40 x 1`

Canonicalization:

- preserve 40 x 40 spatial resolution
- preserve grayscale intensity information
- replicate the single grayscale channel into three identical channels

Result:

`40 x 40 x 3`

### CICMalDroid2020

Source benchmark representation:

`12 x 12 x 1`

Canonicalization:

- deterministically resize from 12 x 12 to 40 x 40
- preserve grayscale intensity information
- replicate the resulting grayscale channel into three identical channels

Result:

`40 x 40 x 3`

### MalNet-Image

Native representation:

`256 x 256 x 3`

Observed audit evidence established that the RGB channels contain distinct
information and are not redundant.

Canonicalization:

- preserve all three RGB channels
- deterministically downsample spatial dimensions from 256 x 256 to 40 x 40
- do not convert MalNet-Image to grayscale

Result:

`40 x 40 x 3`

### MaleX

Selected Paper 1 source representation:

`256 x 256 x 1` grayscale byteplot

Canonicalization:

- deterministically downsample from 256 x 256 to 40 x 40
- preserve grayscale intensity information
- replicate the resulting grayscale channel into three identical channels

Result:

`40 x 40 x 3`

## Source numeric normalization policy

TrustForge does not infer source numeric ranges heuristically.

Each dataset uses an explicit source-to-canonical numeric transformation.

### USTC-TFC2016

Observed source range:

- train: [0, 1]
- test: [0, 1]

Canonical conversion:

- cast to float32
- preserve values in [0, 1]
- defensively clip to [0, 1]

### CICMalDroid2020

Observed source range:

- train: approximately [-1, 1]
- test: approximately [-1, 1]

Canonical conversion:

`x01 = (x + 1.0) / 2.0`

Then:

- cast to float32
- defensively clip to [0, 1]

Negative values must not be truncated to zero before this transformation.

### MalNet-Image

Observed source representation:

- PNG
- uint8
- RGB

Canonical conversion:

`x01 = x / 255.0`

Then:

- cast to float32
- preserve RGB channels
- defensively clip to [0, 1]

### MaleX

Observed source representation:

- PNG
- uint8
- grayscale

Canonical conversion:

`x01 = x / 255.0`

Then:

- cast to float32
- defensively clip to [0, 1]

After spatial canonicalization, the grayscale channel is replicated into
three identical channels.

### Policy requirement

Source numeric conversion must be dataset-declared and explicit.

The revised TrustForge benchmark must not use magnitude-based heuristics such as
"if max > 1.5, divide by 255" to infer the intended source range.

## Canonicalization order

For every revised TrustForge sample, preprocessing must follow this order:

1. decode the source image or source array without modifying the raw artifact
2. convert numeric image values to float32 in [0, 1]
3. if spatial dimensions are not already 40 x 40, resize to 40 x 40 using
   bilinear interpolation with antialiasing enabled
4. preserve RGB channels for RGB-source datasets
5. for grayscale-source datasets, replicate the single canonical grayscale
   channel into three identical channels
6. verify final tensor shape is exactly 40 x 40 x 3
7. defensively clip numeric values to [0, 1]

Datasets already at 40 x 40 are not spatially resampled.

The same spatial resize operator is used whenever resizing is required,
regardless of dataset or generator family.

## Channel policy

RGB source images:

- RGB information is preserved
- channels are not collapsed to grayscale

Grayscale source images:

- grayscale information is preserved
- the single channel is replicated into R, G, and B
- no artificial color information is introduced
- R = G = B for canonicalized grayscale-source samples

This gives every generator family the same channel count while preserving the
information content of grayscale inputs.

## Spatial policy

All revised TrustForge datasets are presented to generator families at:

`40 x 40`

This size is selected as the canonical benchmark spatial resolution because:

- the complete historical seven-family TrustForge benchmark has already been
  exercised at 40 x 40 on USTC-TFC2016
- it substantially reduces the computational dimensionality of the two new
  256 x 256 datasets
- it permits a common representation across all four datasets
- it avoids requiring dataset-specific high-resolution architecture redesigns
  that would confound generator-family comparison with architecture-scaling
  research

The revised Paper 1 does not claim that 40 x 40 is an optimal malware-image
resolution.

The benchmark instead uses 40 x 40 as a controlled operating representation
for comparative evaluation.

## Source preservation

Canonicalization must never overwrite, replace, or destructively modify raw
dataset files.

Raw/native datasets remain preserved separately.

Official MalNet-Image train / validation / test membership remains unchanged.

MaleX conflict exclusions and governed partitioning occur at the sample-identity
level before benchmark execution.

## Fairness principle

Every generator family must receive the same canonical representation for a
given sample:

- same spatial size
- same channel count
- same tensor layout
- same numeric range
- same dataset partition membership

Architecture-specific hyperparameters may differ where required by the model
family, but preprocessing must not differ by generator family.

## Historical compatibility

Historical Paper 1 executions used dataset-specific representations including:

- USTC-TFC2016: 40 x 40 x 1
- CICMalDroid2020: 12 x 12 x 1

Those historical executions remain preserved for provenance.

New revised Paper 1 experiments will use the canonical 40 x 40 x 3
representation and should not be mixed with historical experimental results as
if they were produced under the same preprocessing protocol.

## Status

Canonical spatial resolution — 40 x 40: APPROVED

Canonical channel count — 3: APPROVED

Canonical tensor layout — NHWC: APPROVED

Canonical model-input dtype — float32: APPROVED

Canonical model-input range — [0, 1]: APPROVED

Grayscale-to-three-channel replication: APPROVED

MalNet RGB-channel preservation: APPROVED

Common preprocessing across all seven generator families: APPROVED

Raw dataset preservation: APPROVED

Exact deterministic spatial resize algorithm: APPROVED

- implementation family: TensorFlow image resize
- interpolation: bilinear
- antialias: enabled
- target spatial dimensions: 40 x 40
- model-facing dtype: float32
- model-facing range: [0, 1]

Canonicalization implementation: NOT YET COMPLETED

Seven-family compatibility corrections for 40 x 40 x 3: NOT YET COMPLETED
