# MaleX Source and Partition Contract — Paper 1 Revision 2026

## Status

VALIDATED SOURCE IDENTITY AND GOVERNED PARTITION CONTRACT.

## Benchmark role

MaleX is Dataset D in the revised Paper 1 benchmark.

The selected Paper 1 representation is the MaleX v1:

`256 x 256 grayscale byteplot`

MaleX is treated as one dataset with multiple available representations.
The bigram/DCT representation is preserved for provenance but is not a separate
Paper 1 benchmark dataset.

## Distribution provenance

The working corpus was acquired from the Kaggle mirror:

- owner: `khansaurooj`
- slug: `malex-fyp-image-dataset`

The Kaggle mirror is a secondary distribution source.

Official-source lineage equivalence beyond the evidence recorded in the
revision audit has not been proven.

The complete acquired archive is retained at:

`~/gencys/data/MaleX_raw/malex-fyp-image-dataset.zip`

The extracted mirror is retained at:

`~/gencys/data/MaleX_raw/extracted`

Raw source artifacts must remain unchanged.

## Authoritative Paper 1 label metadata

The authoritative MaleX v1 identity/label artifact for this revision is:

`~/gencys/data/MaleX_raw/extracted/labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5`

File size:

`253462319 bytes`

SHA-256:

`0ef4285cc14d1edbf6f937f783bc9a3b896bff70e762907cf556d41413f9f0ff`

Relevant datasets:

- `ben_sha256s`: 179,725 entries
- `mal_sha256s`: 864,669 entries

Both arrays contain unique identities within their respective classes.

Observed uniqueness:

- benign rows: 179,725
- benign unique identities: 179,725
- malware rows: 864,669
- malware unique identities: 864,669

## Ordered source identity fingerprints

The authoritative HDF5 order is retained as provenance.

For each SHA-256 identity in source order, the identity plus a newline was
included in the sequence fingerprint.

Benign sequence SHA-256:

`30b7e6d4cead5263bb983356f6be77348b2ac48a2878b934ebfe68623b0f8424`

Malware sequence SHA-256:

`e20eff6ceaaaa3d87bcf2b185fa6ddd80615570e3677d887199dcf08754059e9`

These fingerprints bind the revision contract to the observed authoritative
identity ordering.

## Selected byteplot source roots

Benign byteplot root:

`~/gencys/data/MaleX_raw/extracted/ben_bigram_imgs_zipped/byteplot_imgs_RxR/256`

Malware byteplot root:

`~/gencys/data/MaleX_raw/extracted/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256`

Provider filename convention:

`<SHA256>_img_256.png`

Observed source representation:

- PNG
- 256 x 256
- grayscale
- Pillow mode `L`
- uint8
- one image channel

The asymmetric acquired directory naming is preserved exactly and must not be
silently renamed.

## Cross-class conflict exclusion

The authoritative benign and malware SHA-256 identity sets intersect in
exactly 19 identities.

All 19 identities are excluded from both classes before any Paper 1 split is
derived.

Conflicting identities:

1. `294dea313b606a035305e9827c6d2cb389f567a49c0a4cf9c9399504c68f5b12`
2. `2c3ddd638363b9da5491165d33b922de6cf30430b8a7187de1e5994e8524354f`
3. `366db53139dcdff142d57d58aede52274e7325bdbbd77cff8b700d8c1ad5c13f`
4. `3840424265b7ffdcf0c283ccbaed2fae2aff9850a3b69a920ffe1b7443d0d4b3`
5. `42c88fca4ce525603cbea74a28cccee8fbcaa7c7b76037425b749f078cab6d27`
6. `907e284c206c53f69aa1a0b365b93bb555c18ae8ed78af031fe85bfc0a4ef844`
7. `93cb5bfed8cac47c5fa3ecf29bfe9c4de8cf238101f4891422037f87892715f9`
8. `b49ea8a816a80757394f3a907cfa9ff72977635ddfc18c2c7a36d5bc4f83bbe0`
9. `c5d17f98bc7865c7dd4f1572d8814a8af088982c24e688c710d513a2ee734cb5`
10. `c60f6d6ea9690cc3404b7c1e6ae1f114e161245b9d541db0dd85016e1adb54dc`
11. `c8bc0247ec39802219c995d0553e98ab7125390a7130741b46c2694c6d75616f`
12. `d4c20fa8d43e7b592a95d37c386a4335f128cee06674d1393686fdeeb399543a`
13. `d9ebc53ba716dc7470dac38145085a5f3e9a04930412715941082bd6edb012af`
14. `e92aeb64d3e59a015e6c758b566708f6164797dabdaeaa8a1bb5f1f1f14a41e6`
15. `eaf1508d00e73a1d8e0214e2b4a96dc85aeb9bc6883a5894882551714bf758ca`
16. `eb61606562a3e5da03f0f2699a5a54fbb4b10c7da23a0480ee6a16e8fc92d7e4`
17. `f23c452188e2af286d1479f74756ea1792c291efd4e640553de029c8c5f139d5`
18. `f67ca74eba81e0c4f780a053ae5e3766097991f29256344210588c7454bc1216`
19. `f87077f8b738c62ce06de362d00444dc162b68245270b5cc45c7550e8195ba62`

All 19 corresponding byteplot files were observed in both benign and malware
source roots.

The exclusion is deterministic and occurs at sample-identity level.

## Post-exclusion benchmark universe

After excluding all 19 conflicting SHA-256 identities from both classes:

- benign: 179,706
- malware: 864,650
- total: 1,044,356

No additional deduplication, relabeling, or filesystem-derived filtering is
authorized by this contract.

## Governed logical class encoding

TrustForge revision-2026 model IDs are:

- benign: `0`
- malware: `1`

These IDs are computational encodings only.

The authoritative semantic labels remain:

- benign
- malware

## Corpus construction for partitioning

The governed unsplit logical corpus is constructed as:

1. authoritative benign HDF5 order after conflict exclusion;
2. authoritative malware HDF5 order after conflict exclusion.

No lexicographic sorting or filesystem traversal defines the logical corpus.

Logical corpus indices therefore range from:

`0 .. 1,044,355`

## Partition policy

MaleX is governed under TrustForge Case 3.

Partition configuration:

- partition seed: `42`
- train fraction: `0.70`
- validation fraction: `0.10`
- test fraction: `0.20`
- stratified by binary class label

The implementation uses:

`common.partitioning.split_unsplit_corpus`

which delegates to the governed `_stratified_indices` implementation.

Per class:

- indices are deterministically shuffled using `np.random.default_rng(42)`;
- train allocation uses `floor(n * 0.70)`;
- validation allocation uses `floor(n * 0.10)`;
- test receives the remaining samples;
- resulting logical indices are sorted within each final split.

## Governed partition counts

### Train

Total:

`731049`

Class counts:

- benign: 125,794
- malware: 605,255

Logical-index SHA-256:

`8c31521b7c2151d2c14bd3a9a3025ca5881bf241e27aaadc998279a9def3b32e`

Membership SHA-256:

`4c051c7e5ac388ab12e017c1f34eb1b93e650cded71de08abcad923bda7ebcf2`

### Validation

Total:

`104435`

Class counts:

- benign: 17,970
- malware: 86,465

Logical-index SHA-256:

`e0b6d6d63060f2b6bb54f244aa11ba3b5fd4e8455e2536a49e355bff1841d9ac`

Membership SHA-256:

`5fcc103c2657b811b1a85b30d40370db3a6459da9cb4f36bb9e9dec24867b91b`

### Test

Total:

`208872`

Class counts:

- benign: 35,942
- malware: 172,930

Logical-index SHA-256:

`c60e118991f4db59a30dd312c9c906a3e9fc8234b0a3fd743009b65097c4d8a7`

Membership SHA-256:

`f974c0c533ec23ef3e71b8037d22329bc5fcbc63e69d925662e8436b2429663c`

## Partition integrity

Observed partition integrity:

- train / validation overlap: 0
- train / test overlap: 0
- validation / test overlap: 0
- union count: 1,044,356
- union covers complete post-exclusion corpus: yes

The test partition remains isolated from training and validation.

## Membership fingerprint definition

For each split, the membership fingerprint is computed in final governed split
order over records encoded as:

`<numeric_label>\t<SHA256_ID>\n`

where:

- benign numeric label = `0`
- malware numeric label = `1`

The logical-index fingerprint is SHA-256 over the final split index array encoded
as little-endian signed 64-bit integers in C order.

These fingerprints make the exact split membership reproducible and auditable.

## Canonical benchmark representation

MaleX source representation:

`256 x 256 x 1`

Source numeric representation:

`uint8 [0,255]`

Canonicalization:

1. decode source PNG without modifying raw data;
2. convert explicitly using `x / 255.0`;
3. cast to float32;
4. bilinearly resize from 256 x 256 to 40 x 40 with antialiasing;
5. replicate the grayscale channel into three identical channels;
6. validate final shape and numeric range.

Final generator-facing representation:

`40 x 40 x 3`

Tensor layout:

`NHWC`

Numeric representation:

`float32 [0,1]`

## Provenance boundaries

The following distinctions are binding:

`RAW SOURCE != DERIVED PARTITION != CANONICAL BATCH`

`OBSERVED DISTRIBUTION EVIDENCE != PROVEN OFFICIAL LINEAGE`

`HISTORICAL PAPER-1 EXECUTION != REVISION-2026 EXECUTION`

Historical artifacts are preserved for provenance and are not silently
rewritten.

## Current implementation status

MaleX source identity: VALIDATED

MaleX label metadata: VALIDATED

Cross-class conflict exclusion: VALIDATED

Post-exclusion sample universe: VALIDATED

Case-3 partition policy: GOVERNED

Exact deterministic split membership: VALIDATED

Split fingerprints: VALIDATED

Byteplot representation: SELECTED

Canonical representation: GOVERNED

Revision adapter implementation: VALIDATED

Synthetic adapter regression suite: 10 / 10 PASS

Focused Paper-1 revision suite with MaleX: 60 / 60 PASS

Real-data adapter smoke validation: PASS

Real-data integration status: COMPLETE
