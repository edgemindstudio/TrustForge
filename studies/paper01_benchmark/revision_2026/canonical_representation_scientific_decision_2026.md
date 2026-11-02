# Paper 1 Revision-2026 — Canonical Representation Decision

Status: CANDIDATE — PENDING SCIENTIFIC APPROVAL

## 1. Scope

This decision candidate applies to the four-dataset,
seven-generator Paper 1 Revision-2026 benchmark.

It records a shared model-facing input policy.
It does not authorize real-data training, checkpoint
publication, or production synthetic generation.

## 2. Shared canonical representation

All participating generators consume canonical images:

- Tensor shape: (B, 40, 40, 3)
- Layout: NHWC
- Numeric dtype: float32
- Numeric range: [0, 1]

The canonicalization implementation is:

common/canonicalization.py

Transformations:

1. Explicit source numeric normalization.
2. Conversion to NHWC.
3. Spatial resizing to 40 x 40 using bilinear
   interpolation with antialiasing when required.
4. Replication of single-channel grayscale data.
5. Final shape, channel, and numeric-range verification.

## 3. Dataset-specific source representations

### USTC-TFC2016

Source: grayscale malware image.

Canonicalization:
- Preserve 40 x 40 spatial resolution.
- Replicate the grayscale channel to three channels.

### CICMalDroid2020

Source: feature-derived 12 x 12 grayscale image.

Canonicalization:
- Resize to 40 x 40.
- Replicate the grayscale channel to three channels.

### MalNet-Image

Source: official RGB PNG image.

Canonicalization:
- Preserve RGB source channel semantics.
- Resize spatially to 40 x 40.
- Do not introduce grayscale conversion.

This is consistent with the existing autoregressive,
masked-autoflow, and RBM representation policies.

### MaleX

Source: 256 x 256 grayscale malware byteplot.

Canonicalization:
- Resize to 40 x 40.
- Replicate the grayscale channel to three channels.

## 4. Comparability rule

All seven generator families must receive the same
governed canonical representation for each dataset.

No generator may independently substitute a grayscale,
RGB, resized, thresholded, or alternative representation
without a separately approved experimental protocol.

The input tensor interface is common, but the original
source representation is heterogeneous across datasets.

## 5. Scientific limitations

Downsampling may remove fine-grained source information.

Replicated grayscale channels are not equivalent to
genuine three-channel RGB information.

The primary benchmark must disclose these limitations.

Any supplementary resolution-sensitivity experiment
must be explicitly identified and kept separate from
the primary comparison.

## 6. Historical preservation

Earlier grayscale-byteplot proposals remain historical
planning evidence and do not automatically supersede
the implemented canonical adapters.

Historical documents and datasets are not modified
by this decision candidate.

## 7. Evidence basis

Reviewed implementations:

- common/canonicalization.py
- Four Revision-2026 dataset adapters
- VAE implementation contract
- Autoregressive observation-model policy
- Masked-autoflow observation-domain policy
- RBM architecture policy

The repository-wide representation review scanned
76 scientific and configuration files.

## 8. Authorization

REPRESENTATION_POLICY_STATUS=CANDIDATE
REPRESENTATION_POLICY_APPROVAL=PENDING
REAL_DATASET_TRAINING_AUTHORIZED=NO
PRODUCTION_SYNTHESIS_AUTHORIZED=NO

Final scientific approval must independently address
training schedules, evaluation scope, experimental
fairness, and remaining acceptance requirements.
