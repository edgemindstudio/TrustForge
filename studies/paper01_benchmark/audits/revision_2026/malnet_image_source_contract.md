# MalNet-Image Revised Paper-1 Source Contract

**Status:** VALIDATED

## Scope

This contract defines the authoritative MalNet-Image source universe,
provider-defined partition provenance, type-level classification semantics,
image representation boundary, and revision-specific preparation rules for the
2026 Paper-1 benchmark.

Historical Paper-1 evidence remains historical and is not rewritten by this
contract.

## Dataset role

Dataset:

`MalNet-Image`

Revised Paper-1 task:

- representation: image
- classification level: type
- official split fraction: `1.0`
- partition policy: Case 1
- provider train / validation / test partitions are authoritative
- number of type classes: 47

TrustForge must preserve the official provider partitions exactly.

TrustForge must not recombine and resplit MalNet-Image.

## Authoritative split metadata

Official split archive:

`~/gencys/data/MalNet-Image_raw/split_info/split_info.zip`

Observed archive properties:

- file size: 261,499,089 bytes
- SHA-256:
  `74e3615aa848c27dc06bb9ea2ae53e415f4b842fdce8dc45ba50849db8851492`
- non-directory members: 27

The authoritative revised Paper-1 split members are:

- train:
  `split_info/type/1.0/train.txt`
- validation:
  `split_info/type/1.0/val.txt`
- test:
  `split_info/type/1.0/test.txt`

Observed provider counts:

- train: 883,416
- validation: 126,202
- test: 252,406
- total: 1,262,024

## Provider split-entry format

Every audited `type/1.0` split entry has exactly three slash-separated
components:

`<type>/<family>/<sample_sha>`

For revised Paper 1:

- first component: provider type label and classification target
- second component: family metadata
- third component: sample identifier

The split entries do not include the `.png` suffix.

An official split entry resolves to the extracted image path:

`malnet-images/<type>/<family>/<sample_sha>.png`

The provider string label is authoritative scientific metadata.

A numerical model-class encoding must be explicit and deterministic, but it
must not be mistaken for provider semantic ordering.

## Authoritative extracted image corpus

Extracted image corpus:

`~/gencys/data/MalNet-Image_raw/malnet-images`

Observed extracted PNG count:

- 1,262,025

Official `type/1.0` split universe:

- 1,262,024 unique entries

Every official split entry resolves to an extracted PNG.

No official `type/1.0` split sample is missing.

Exactly one extracted PNG is outside the official split universe:

`malnet-images/adware/mobclick/CDCC51ED70F35D9515AF6CA48B02013DA99C221A1746AC9403787F3E825A54D8.png`

Observed properties of that unreferenced file:

- PNG
- 256 x 256
- RGB
- Pillow verification: OK
- SHA-256:
  `f661a0b8593312b684806ece95e2a7faff1bf04a9e2812f2f7b30991e6db9cf7`

Its sample identifier was not found in any official split text member.

The file must remain preserved in raw storage, but it is excluded from the
revised Paper-1 benchmark sample universe because it is outside the official
provider split metadata.

Deletion of the unreferenced PNG is not authorized.

## Provider split integrity

The official `type/1.0` metadata was audited directly.

### Train

- rows: 883,416
- unique rows: 883,416
- duplicate rows: 0
- type labels represented: 47
- malformed rows: 0

### Validation

- rows: 126,202
- unique rows: 126,202
- duplicate rows: 0
- type labels represented: 47
- malformed rows: 0

### Test

- rows: 252,406
- unique rows: 252,406
- duplicate rows: 0
- type labels represented: 47
- malformed rows: 0

### Cross-split integrity

- train / validation overlap: 0
- train / test overlap: 0
- validation / test overlap: 0
- unique union rows: 1,262,024
- union type labels: 47

The provider partitions are therefore mutually disjoint and collectively
cover the complete official `type/1.0` revised benchmark universe.

## Authoritative type-label universe

The 47 observed provider type labels are:

- addisplay
- addisplay++adware
- adload
- adsware
- adware
- adware++adware
- adware++grayware++virus
- adware++virus
- adwareare
- backdoor
- banker++trojan
- benign
- click
- clicker
- clicker++trojan
- clickfraud++riskware
- downloader
- dropper++trojan
- exploit
- fakeangry
- fakeapp
- fakeapp++trojan
- fakeinst++trojan
- gray
- hacktool
- malware
- malware++trj
- monitor
- ransom++trojan
- risktool++riskware++virus
- riskware
- riskware++smssend
- rog
- rootnik++trojan
- smssend
- smssend++trojan
- spr
- spy
- spy++trojan
- spyware
- trj
- troj
- trojan
- trojandownloader
- trojandropper
- virus
- worm

All 47 provider type labels occur in each of train, validation, and test.

## Structural image evidence

A deterministic stratified image audit used partition seed 42 and sampled:

- 5 images per type
- per provider split
- all 47 type classes

Coverage:

- train: 235
- validation: 235
- test: 235
- total: 705
- errors: 0

All 705 audited samples were:

- PNG
- 256 x 256
- RGB
- bands: R, G, B
- Pillow verification: OK

A separate channel-structure audit over the same 705-image design established:

- identical RGB channels: 0
- non-identical RGB channels: 705
- observed maximum inter-channel difference: 255

This supports treating MalNet-Image as genuine RGB data rather than replicated
grayscale.

This is stratified structural evidence, not a claim that every one of the
1,262,024 benchmark images has been fully decoded in a corpus-wide readability
sweep.

## Revised representation boundary

The native provider representation is treated as:

- 256 x 256
- 3 channels
- RGB
- uint8-style image intensity domain

The governed TrustForge canonical representation is:

- 40 x 40 x 3
- NHWC
- float32
- `[0, 1]`

The revised preparation path must therefore:

1. resolve samples only through official `type/1.0` split membership;
2. decode the referenced PNG as RGB;
3. convert the numeric domain explicitly to float32 `[0, 1]`;
4. preserve the three provider RGB channels;
5. downsample from 256 x 256 to 40 x 40 using the governed canonicalization
   policy;
6. expose canonical 40 x 40 x 3 float32 tensors in `[0, 1]`.

Grayscale replication must not be applied to MalNet-Image.

## Partition contract

MalNet-Image is governed as partition-policy **Case 1**.

The exact provider membership is binding:

- provider train -> TrustForge train
- provider validation -> TrustForge validation
- provider test -> TrustForge test

No benchmark-derived split seed is required to create these partitions.

Partition seed 42 may be used for deterministic audits or sampling procedures,
but it must not redefine provider membership.

## Provenance boundary

The following distinctions are binding:

- extracted corpus membership != official benchmark membership
- provider split membership != filesystem traversal order
- provider string label != TrustForge numerical model ID
- provider family metadata != revised Paper-1 type target
- raw image representation != canonical benchmark representation
- stratified readability evidence != corpus-wide readability proof
- raw preservation != permission to include unreferenced files

## Revision status

Authoritative split archive identity: VALIDATED

Official `type/1.0` task selection: VALIDATED

Provider split counts: VALIDATED

Provider split disjointness: VALIDATED

Provider split duplicate audit: VALIDATED

Official sample-universe reconciliation: VALIDATED

47-class provider type-label universe: VALIDATED

Stratified image structure audit: VALIDATED

RGB channel-structure evidence: VALIDATED

Revised partition classification: CASE 1

Provider train / validation / test preservation: REQUIRED

Unreferenced PNG benchmark inclusion: PROHIBITED

Unreferenced PNG deletion: NOT AUTHORIZED

Numerical model-class encoding: GOVERNED

Mapping artifact:

`studies/paper01_benchmark/audits/revision_2026/malnet_image_type_mapping.json`

Mapping artifact SHA-256:

`84406057e188a35a98ec7237275bf1bce22df037d627c40d2ded5d735020b253`

Numeric encoding policy:

- preserve provider string labels as authoritative scientific metadata
- lexicographically sort the complete 47-label provider type universe
- assign contiguous zero-based TrustForge model IDs `0..46`
- treat numeric IDs as computational identifiers only
- numeric IDs carry no semantic rank and are not provider-defined ordering

Mapping integrity:

- class-to-ID entries: 47
- ID-to-class entries: 47
- contiguous zero-based IDs: VALIDATED
- forward round-trip: EXACT
- reverse round-trip: EXACT
- lexicographic order: VALIDATED
- first mapping: `addisplay -> 0`
- last mapping: `worm -> 46`

Revised MalNet-Image adapter implementation: NOT YET COMPLETED
