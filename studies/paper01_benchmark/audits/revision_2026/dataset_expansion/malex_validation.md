# MaleX Dataset Validation Record

## Scope

This record documents the provenance, acquisition, representation, integrity,
label consistency, and Paper 1 suitability checks performed for MaleX as a
candidate Dataset D in the 2026 revision of Paper 1.

This record is revision-specific and does not modify or supersede historical
Paper 1 execution evidence.

## Source and acquisition

The working corpus was acquired from the Kaggle dataset:

- Owner: `khansaurooj`
- Dataset slug: `malex-fyp-image-dataset`

The Kaggle mirror is not the official MaleX repository and is therefore treated
as a secondary distribution source.

The mirror contains artifacts whose population counts and filenames align with
MaleX v1 metadata, including:

- 864,669 malware samples
- 179,725 benign samples
- 1,044,394 total underlying samples

The complete Kaggle archive was downloaded on UND Talon and retained at:

`~/gencys/data/MaleX_raw/malex-fyp-image-dataset.zip`

The archive passed a full ZIP integrity test with no compressed-data errors.

The complete extracted mirror is retained at:

`~/gencys/data/MaleX_raw/extracted`

Acquisition completed successfully under Slurm job:

`301250`

A completion marker is present at:

`~/gencys/data/MaleX_raw/.malex-acquisition.complete`

## Representations

The mirror contains two image representations derived from the same underlying
MaleX v1 sample identities.

### Byteplot representation

Benign:

`ben_bigram_imgs_zipped/byteplot_imgs_RxR/256`

Malware:

`mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256`

Verified counts:

- benign: 179,725 PNG files
- malware: 864,669 PNG files
- total: 1,044,394 PNG files

Representative image validation established:

- PNG format
- 256 x 256 pixels
- grayscale
- Pillow mode `L`
- one image band
- successful image integrity verification

Filename pattern:

`<SHA256>_img_256.png`

### Bigram/DCT representation

Benign:

`ben_byteplot_imgs_zipped/bigram_imgs/imgs_dct`

Malware:

`mal_bigram_imgs_zipped/bigram_imgs/imgs_dct`

Verified counts:

- benign: 179,725 PNG files
- malware: 864,669 PNG files
- total: 1,044,394 PNG files

Filename pattern:

`<SHA256>_bigram_dct.png`

The same SHA-256 sample identifiers were observed across the byteplot and
bigram/DCT representations, supporting the interpretation that these are two
representations of the same underlying binary corpus rather than two
independent datasets.

The benign top-level directory names appear inconsistent with their child
representation names. This source structure is preserved as acquired and is
not silently renamed.

## Label metadata

The extracted MaleX v1 label archive is:

`labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5`

Verified file size:

`253462319 bytes`

Verified SHA-256:

`0ef4285cc14d1edbf6f937f783bc9a3b896bff70e762907cf556d41413f9f0ff`

The extracted copy was verified byte-for-byte identical to the independently
downloaded audit copy.

The HDF5 structure contains:

- `ben_md5s`: 179,725
- `ben_mimes`: 179,725
- `ben_sha256s`: 179,725
- `mal_md5s`: 864,669
- `mal_mimes`: 864,669
- `mal_sha256s`: 864,669
- `mal_avast_labels`: 864,669
- `mal_bitdefender_labels`: 864,669

Representative benign and malware PNG SHA-256 identifiers were verified
against the corresponding HDF5 arrays.

## Cross-class conflicts

A set intersection between `ben_sha256s` and `mal_sha256s` identified:

`19`

SHA-256 identifiers present in both classes.

At least one conflicting SHA-256 was independently downloaded from both the
benign and malware byteplot directories and verified to produce byte-for-byte
identical PNG files.

Therefore these conflicts are not treated as independent observations.

### Governed preparation rule

Before any Paper 1 train/validation/test split, all SHA-256 identifiers appearing
in both the benign and malware label sets will be excluded from both classes.

Expected post-exclusion population:

- benign: 179,706
- malware: 864,650
- total: 1,044,356

This exclusion is deterministic and is intended to prevent contradictory
cross-class duplicate leakage.

## Additional included artifacts

The complete mirror also includes:

- `MaleXv1_bigram_dct_gist_m864001_b179725.hdf5`
- `MaleXv1_byteplot_gist_m863936_b179725.hdf5`
- `MaleXv1_mal_100fam_sha256s_10092017.hdf5`
- `MaleXv2-m1789632-b306244_02052018.hdf5`

These artifacts are preserved but are not automatically authorized as Paper 1
experimental inputs.

## Paper 1 representation decision

MaleX is treated as one dataset with multiple representations, not as two
independent datasets.

For the revised Paper 1 benchmark, the preferred experimental representation is:

`256 x 256 grayscale byteplot`

The bigram/DCT representation is preserved for provenance, possible future
analysis, and reproducibility, but is not currently designated as a separate
Paper 1 benchmark dataset.

## Current status

Acquisition: COMPLETE

Archive integrity: VERIFIED

Extracted population counts: VERIFIED

Representative image structure: VERIFIED

MaleX v1 label archive identity: VERIFIED

Image-to-label sample identity: VERIFIED

Cross-class conflict audit: VERIFIED

Paper 1 representation preference: BYTEPLOT

Final train/validation/test split policy: NOT YET GOVERNED

Full corpus-wide image readability audit: NOT YET COMPLETED

Official-source lineage equivalence beyond the evidence recorded above:
NOT YET PROVEN
