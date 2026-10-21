# Paper 1 Revision 2026 — Gaussian Mixture Sampling Policy

`STATUS=IMPLEMENTED_AND_INTEGRATED_VALIDATED`

`GMM_SAMPLING_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=YES`

## 1. Scope

This policy governs model-consistent synthetic sampling for the revised
Paper 1 class-conditional censored diagonal Gaussian mixture model.

It governs:

- checkpoint preflight;
- sampling scientific identity;
- sampling PRNG derivation;
- mixture-component draws;
- latent diagonal-Gaussian draws;
- governed censoring into canonical image space;
- output dtype, shape, and range;
- reproducibility;
- fail-closed behavior.

This policy does not yet authorize benchmark execution.

## 2. Authoritative sampling source

Synthetic samples must be generated only from a checkpoint pair that passes
the governed checkpoint preflight.

The required checkpoint scientific identity is:

`(dataset_id, class_id, experimental_seed)`

Sampling must not:

- substitute another class;
- substitute another experimental seed;
- substitute another dataset;
- load a global fallback model;
- train a temporary replacement model;
- consume malformed or integrity-invalid checkpoint state.

`GMM_SAMPLING_CHECKPOINT_PREFLIGHT=REQUIRED`

`GMM_SAMPLING_GLOBAL_FALLBACK=NO`

## 3. Model-consistent generation semantics

For each requested synthetic vector:

1. sample one mixture component according to the learned softmax mixture
   weights;
2. sample one latent vector from that component's learned diagonal Gaussian;
3. apply the governed censored observation map coordinatewise;
4. reshape the canonical vector to `40 x 40 x 3`;
5. return canonical float32 output.

For latent coordinate `z_d`:

- if `z_d <= 0`, observed `x_d = 0`;
- if `0 < z_d < 1`, observed `x_d = z_d`;
- if `z_d >= 1`, observed `x_d = 1`.

This censoring map is part of the probabilistic observation model.

It must not be represented as external post-generation cleanup.

`GMM_SAMPLING_OBSERVATION_MAP=CENSORED_GAUSSIAN`

`GMM_POSTHOC_CLIPPING=NO`

## 4. Primary synthesis budget

The revised Paper 1 primary synthesis budget is:

`2000 synthetic samples per class per experimental seed`

`GMM_PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS=2000`

The sampler implementation may support smaller explicitly requested counts for
testing or controlled execution, but primary benchmark execution must use the
governed budget.

## 5. Sampling randomness identity

Sampling randomness must be reproducible but must not reuse an identical raw
PRNG stream merely because two distinct scientific models share the same
experimental seed.

The sampling PRNG substream is therefore derived from:

`(purpose, dataset_id, class_id, experimental_seed)`

where:

`purpose = paper01_revision2026_gmm_sampling`

`GMM_SAMPLING_PRNG_IDENTITY=PURPOSE_DATASET_CLASS_EXPERIMENTAL_SEED`

`GMM_SAMPLING_PRNG_PURPOSE=paper01_revision2026_gmm_sampling`

This separates sampling randomness across:

- datasets;
- classes;
- experimental seeds;
- unrelated future PRNG purposes.

## 6. PRNG derivation

The sampling-identity payload is represented as canonical TrustForge JSON:

- UTF-8;
- sorted keys;
- compact separators;
- no NaN or Infinity.

The payload contains exactly:

- `purpose`;
- `dataset_id`;
- `class_id`;
- `experimental_seed`.

SHA-256 is computed over those canonical JSON bytes.

The complete 256-bit digest is consumed as eight consecutive big-endian
unsigned 32-bit entropy words.

Those eight words initialize:

`numpy.random.SeedSequence`

The resulting SeedSequence initializes:

`numpy.random.default_rng`

`GMM_SAMPLING_PRNG_HASH=SHA256`

`GMM_SAMPLING_PRNG_HASH_INPUT=CANONICAL_JSON`

`GMM_SAMPLING_PRNG_DIGEST_CONSUMPTION=EIGHT_BIG_ENDIAN_UINT32_WORDS`

`GMM_SAMPLING_PRNG_SEED_SEQUENCE=NUMPY_SEEDSEQUENCE`

`GMM_SAMPLING_PRNG_GENERATOR=NUMPY_DEFAULT_RNG`

No Python built-in hash value may participate because Python hash
randomization is not a governed cross-process seed contract.

`GMM_SAMPLING_PYTHON_HASH=FORBIDDEN`

## 7. PRNG audit evidence

The selected derivation was audited for representative scientific identities:

- `(ustc_tfc2016, 0, 42)`;
- `(ustc_tfc2016, 1, 42)`;
- `(ustc_tfc2016, 0, 43)`;
- `(cicmaldroid2020, 0, 42)`;
- `(malnet_image, 0, 42)`;
- `(malex, 0, 42)`.

Observed properties:

- six identities produced six unique SHA-256 digests;
- repeated derivation of the same identity produced the same digest;
- repeated same-identity draws were bitwise identical;
- changing dataset changed the audited stream;
- changing class changed the audited stream;
- changing experimental seed changed the audited stream;
- changing the purpose string changed the digest.

This audit demonstrates the selected deterministic derivation in the current
governed implementation environment.

It does not claim a mathematical impossibility of SHA-256 collision.

`GMM_SAMPLING_PRNG_AUDIT=PASSED`

## 8. Component sampling

Mixture weights are the governed model softmax weights.

For each requested sample, the sampler draws exactly one component index from:

`Categorical(weights)`

using the governed NumPy Generator.

The sampler must validate before component sampling that weights are:

- finite;
- strictly positive;
- normalized according to the governed model-state tolerance.

No component may be silently removed because of a small learned weight.

`GMM_SAMPLING_COMPONENT_DISTRIBUTION=LEARNED_SOFTMAX_CATEGORICAL`

`GMM_SAMPLING_COMPONENT_DELETION=NO`

## 9. Latent Gaussian sampling

After selecting component `k`, the sampler draws:

`epsilon ~ N(0, I)`

and constructs:

`z = mean_k + scale_k * epsilon`

coordinatewise.

The learned means are latent Euclidean parameters and may lie outside
`[0,1]`.

The transformed scales are the governed positive learned scales.

No latent mean projection is permitted.

`GMM_SAMPLING_LATENT_DISTRIBUTION=DIAGONAL_GAUSSIAN`

`GMM_SAMPLING_LATENT_MEAN_PROJECTION=NO`

## 10. Canonical output contract

After applying the censored observation map, each generated sample must have:

- shape `40 x 40 x 3`;
- output batch shape `(B,40,40,3)`;
- dtype `float32`;
- all finite values;
- exact numerical range `[0,1]`.

Exact zeros and ones are legitimate model-generated probability mass.

The float64 latent/censoring result is cast to canonical float32 only after
the observation map.

`GMM_SAMPLING_OUTPUT_SHAPE=40x40x3`

`GMM_SAMPLING_OUTPUT_DTYPE=FLOAT32`

`GMM_SAMPLING_OUTPUT_RANGE=ZERO_ONE`

`GMM_SAMPLING_ENDPOINT_VALUES=PERMITTED`

## 11. Reproducibility contract

For fixed:

- checkpoint bytes;
- validated metadata;
- dataset identity;
- class identity;
- experimental seed;
- requested sample count;
- governed implementation/runtime dependency behavior;

sampling must be reproducible.

At minimum, permanent regression testing must demonstrate:

- same scientific identity and same checkpoint produce identical samples;
- changed experimental seed can produce different samples;
- changed class identity produces a distinct PRNG substream;
- changed dataset identity produces a distinct PRNG substream.

`GMM_SAMPLING_REPRODUCIBLE=YES`

## 12. Sampling count

Requested sample count must be an exact positive integer.

The implementation must reject:

- zero;
- negative values;
- booleans;
- floating-point coercion;
- string coercion.

`GMM_SAMPLING_COUNT_VALIDATION=EXACT_POSITIVE_INTEGER`

## 13. Checkpoint preflight

Before any random draw occurs, the implementation must invoke governed
checkpoint loading and preflight validation.

This must validate at least:

1. metadata file exists;
2. state file exists;
3. metadata schema type/version;
4. requested dataset identity;
5. requested class identity;
6. requested experimental seed;
7. state filename;
8. state SHA-256;
9. exact NPZ keys;
10. exact state shapes;
11. exact state dtypes;
12. finite parameters;
13. valid transformed scales;
14. finite normalized mixture weights;
15. selected-checkpoint metadata consistency.

No PRNG draw may occur before preflight succeeds.

`GMM_SAMPLING_PREFLIGHT_BEFORE_RANDOMNESS=YES`

## 14. Fail-closed behavior

Sampling must fail closed on:

- missing checkpoint files;
- metadata/state identity mismatch;
- SHA-256 mismatch;
- malformed metadata;
- malformed numerical state;
- non-finite model parameter;
- non-finite transformed scale;
- invalid mixture weights;
- invalid sample count;
- PRNG derivation failure;
- non-finite latent samples;
- non-finite observed samples;
- unexpected output shape;
- unexpected output dtype;
- output outside `[0,1]`.

A failure must not silently trigger:

- model substitution;
- class substitution;
- seed substitution;
- global fallback;
- posthoc clipping;
- NaN replacement;
- component deletion;
- alternate PRNG semantics.

## 15. Evaluation interpretation

GMM likelihood remains a model-training and checkpoint-selection quantity.

Primary benchmark comparison is based on governed synthetic-image and
downstream evaluation procedures.

GMM NLL must not be promoted into an ungoverned cross-dataset primary metric.

## 16. Implementation gate

Sampling implementation is complete under this governed policy.

Checkpoint persistence is implemented and integrated-validated.

Sampling is implemented and focused-validated with 16 passing tests.

The complete revised GMM stack, including checkpoint persistence and
model-consistent sampling, passes 187 integrated tests.

Therefore:

`GMM_SAMPLING_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_SAMPLING_FOCUSED_VALIDATION=PASSED`

`GMM_SAMPLING_FOCUSED_TEST_COUNT=16`

`GMM_SAMPLING_INTEGRATED_VALIDATION=PASSED`

`GMM_SAMPLING_INTEGRATED_GMM_TEST_COUNT=187`

`GMM_IMPLEMENTATION_AUTHORIZED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=YES`
