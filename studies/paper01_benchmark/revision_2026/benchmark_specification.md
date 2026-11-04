# Paper 1 Revision-2026 Benchmark Specification

Status: DRAFT — PARTIAL CONSOLIDATION
Full benchmark execution: NOT AUTHORIZED
Prepared: 2026-11-04

## 1. Purpose and authority

Consolidate the revised experiment definition without rewriting historical
policies, executions, or results. This draft records inspected decisions
and implementation behavior; it does not independently approve unresolved
policies or supersede earlier records.

Earlier general policies contain unfinished status wording. Later
dataset-specific contracts and family acceptance records must be reconciled
explicitly before this specification is frozen.

## 2. Primary benchmark scope

- Datasets: USTC-TFC2016, CICMalDroid2020, MalNet-Image, MaleX v1.
- Families: GAN, VAE, Diffusion, AR, MAF, RBM, GMM.
- Experimental repetitions: seven.
- Proposed seeds: [42, 43, 44, 45, 46, 47, 48].
- Exact seed acceptance: PENDING.
- Primary synthetic budget: 2,000 samples per class per experimental seed.
- Dataset-family-seed combinations: 196.
- Class-specific models and evaluation stages create additional work units.

## 3. Dataset identities and partitions

| Dataset ID | Task and model labels | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| ustc_tfc2016 | Nine-class prepared malware-image task; source IDs 0..8 preserved; semantic names unresolved | 72900 | 8100 | 9000 |
| cicmaldroid2020 | Five-category feature-derived classification; source IDs 1..5 map to model IDs 0..4; semantic names unresolved | 8117 | 1158 | 2323 |
| malnet_image | Official type/1.0 task, 47 provider type labels including benign | 883416 | 126202 | 252406 |
| malex | Binary classification: benign=0, malware=1 | 731049 | 104435 | 208872 |

Counts above come from accepted dataset contracts and prior validation
records; they were not recomputed during this specification review.

### USTC

Use the hash-bound prepared NCHW arrays described in ustc_source_contract.md.
Preserve their supplied test partition. Derive validation from 10% of
supplied training, stratified with partition seed 42. This review does not
independently establish the original raw-to-prepared acquisition lineage.

### CICMalDroid2020

Use the authoritative 11,598-row CSV and hash-bound ordered list of 139
features. Create revised 70/10/20 partitions with seed 42.
Do not reuse historical 70/15/15 partitions or their fitted scaler.

### MalNet-Image

Preserve official type/1.0 train/validation/test membership exactly.
Use the existing explicit lexicographic provider-label mapping to IDs 0..46.
Preserve, but exclude from the benchmark, the one unreferenced PNG.
Split-row disjointness does not establish absence of all content duplicates.
The 705-image structural audit is not a corpus-wide decode audit.

### MaleX

Use the selected grayscale byteplot representation and authoritative HDF5
identity order. Exclude all 19 conflicting identities from both classes
before splitting. Construct the logical corpus from remaining benign
identities followed by remaining malware identities.
Apply the governed 70/10/20 split with seed 42 and retain recorded
membership fingerprints. No extra filtering or relabeling is implied.

## 4. Exact derived partition algorithm

Implementation: common/partitioning.py, _stratified_indices.

- Create one NumPy default_rng(seed) per partitioning call.
- Visit classes in np.unique label order.
- Permute each class using that same advancing RNG.
- Allocate floor(n * fraction) to every partition except the last.
- Assign the remaining class samples to the final partition.
- Concatenate class allocations and sort indices within each final split.

Do not reset the RNG separately for each class.
Do not substitute another splitting library based only on matching ratios.
Experimental model seeds never redefine these fixed partitions.
MalNet provider membership does not require a benchmark resplit.

## 5. Canonical representation

All families receive float32 NHWC batches shaped (B,40,40,3), range [0,1].
Model-internal mathematical transformations require documented family policies.

Implementation: common/canonicalization.py.

1. Explicitly normalize the declared source numeric range.
2. Clip normalized values to [0,1].
3. Convert supported layouts to NHWC; USTC adapter handles NCHW conversion.
4. Resize when needed using TensorFlow bilinear resize, antialias=True.
5. Cast to float32 and replicate single-channel grayscale three times.
6. Check final spatial/channel shape and clip output to [0,1].

No numeric-range inference by image magnitude is permitted.
The shared canonicalizer does not itself explicitly reject nonfinite input.
Production adapter validation must be verified separately.

| Dataset | Source-to-canonical transformation |
| --- | --- |
| USTC | Explicit NCHW-to-NHWC conversion; float32 [0,1]; no resize; grayscale replication |
| CIC | Training-fitted scaling, tanh, five zeros, 12x12x1 reshape; (x+1)/2; resize to 40x40; grayscale replication |
| MalNet | uint8 RGB /255; resize 256x256 to 40x40; preserve distinct RGB channels |
| MaleX | uint8 grayscale /255; resize 256x256 to 40x40; grayscale replication |

The benchmark does not establish optimal resolution or 256x256 generation.
Replicated grayscale channels do not add color information.

## 6. CIC learned preprocessing

The inspected adapter:

- Loads features in the hash-bound feature order as float32.
- Derives revised partitions before fitting.
- Fits a new sklearn StandardScaler only on revised training rows.
- Reuses that scaler for all three partitions.
- Requires float32 transform output and float32 NumPy tanh output.
- Appends five zeros after tanh and reshapes row-major to 12x12x1.
- Checks source feature finiteness and final canonical output finiteness.

Zero padding occurs before conversion from [-1,1] to [0,1].
Thus padded zeros map to 0.5 before resizing, not to black pixels.

The adapter exposes scaler state in memory. Persisted scaler identity was
not demonstrated in the inspected provenance paths.
Before launch, establish reproducible fitted-state evidence tied to source,
feature order, exact training membership, implementation and environment.

## 7. Reproducibility evidence requirements

Record actual consumed settings, source/preparation identities, class mapping,
partition membership, preprocessing identity, model policy, environment,
execution source, checkpoint identity, synthesis identity and output counts.

Separate configuration declarations from consumed and derived runtime values.
No historical cache, random-weight synthesis or omitted required output may
silently substitute for the requested experiment.

## 8. Remaining specification sections

- Exact seed acceptance and evaluator/synthesis randomization policy.
- Family-specific training, selection and production sampling policies.
- Downstream classifier and comparable class-weighting policy.
- Metric definitions, feature extractors, reference splits and failure rules.
- Primary estimands, paired uncertainty analysis and multiplicity policy.
- Retry, incomplete-run and aggregation rules.
- Final authority reconciliation and executable configuration linkage.

Stage 2 remains IN_PROGRESS. This document is not a launch authorization.

## 9. Source records

Paths below are relative to studies/paper01_benchmark.

- audits/revision_2026/partition_policy.md
- audits/revision_2026/representation_policy.md
- audits/revision_2026/ustc_source_contract.md
- audits/revision_2026/cicmaldroid_source_contract.md
- audits/revision_2026/malnet_image_source_contract.md
- audits/revision_2026/malex_source_contract.md
- audits/revision_2026/malnet_image_type_mapping.json
- revision_2026/generator_compatibility_policy.md
- revision_2026/canonical_representation_scientific_decision_2026.md
- revision_2026/experimental_seed_policy.md

## 10. Family policies recovered during consolidation

This table records inspected policies and source behavior. It is not a
complete executable training configuration or operational authorization.

| Family | Training and selection evidence | Remaining specification/integration requirement |
| --- | --- | --- |
| AR | Channel-coupled boundary-censored logistic PixelCNN; caller-supplied epoch limit and patience; validation-based stopping | Recover final consumed architecture, optimizer, budget and sampling controls; reconcile aggregation and selection semantics |
| MAF | Conditional block-autoregressive flow; slope-conditioned splines; log-slope limit 0.75; maximum inverse depth 125 stages; final contract section 29 accepts training | Recover exact training budget, optimizer and selection policy; complete durable checkpoint integration |
| GMM | Class-specific five-component censored diagonal mixture; B=ceil(N_class/128); max_updates=max(100,5B); validation interval=max(20,B); minimum finite validation NLL, earliest tie | Bind current production invocation and checkpoint-backed sampling to accepted identities |
| GAN | Conditional DCGAN; batch 256; Adam lr 0.0002, beta1 0.5; D:G updates 2:1; class-sampling exponent 0.85; budget min(10000,max(500,2*ceil(N_train/256))); terminal checkpoint | Govern operational activation and production launcher |
| RBM | Shared conditional continuous-visible RBM; CD-1; batch 256; 2*ceil(N_train/256) updates; validation at 0,B,2B; final checkpoint | Production Gibbs depth remains unselected; sampling and synthesis restricted |
| VAE | Conditional convolutional VAE; latent 100; observation variance 1; KL coefficient 1; Adam lr 0.001; batch 64; two complete epochs; minimum validation total loss, earliest tie | Operational activation, publication and production synthesis remain restricted |
| Diffusion | Conditional U-Net; depth 2, base filters 64, time embedding 128; linear 1000-step DDPM, betas 0.0001..0.02; Adam lr 0.0002, beta1 0.9; batch 64; two epochs; minimum validation MSE, earliest tie | Operational activation, durable publication and production synthesis remain restricted |

### Observation-model and authorization reconciliation

- autoregressive_observation_model_policy.md explicitly rejects the older
  logit-Gaussian formulation in autoregressive_policy.md.
- autoregressive_observation_model_implementation_contract.md specifies
  censored-logistic endpoints, R->G->B coupling, centered channels x-0.5,
  tanh coupling bounds and log-scale bounds [-7,3].
- autoregressive_training.py imports coupled_rgb_censored_logistic_nll.
  This confirms active likelihood use, not complete production acceptance.
- masked_autoflow_implementation_contract.md section 29 records final
  implementation acceptance and permission for governed real-data training.
  Earlier pending statements are retained as historical development state.
- gaussian_mixture_sampling_policy.md records later integrated acceptance;
  the earlier budget-only denial is not a current blanket training denial.
- variational_autoencoder_formal_training_authorization.md explicitly
  supersedes earlier pending/denied training-methodology assessments.
- GAN, VAE and Diffusion methodological acceptance does not enable runtime
  gates or independently authorize production publication/synthesis.
- RBM training acceptance explicitly preserves production sampling restrictions.

### AR source findings requiring bounded resolution

The inspected fit_revision_pixelcnn primitive:

- Averages scalar batch losses equally rather than by batch sample count.
- Updates best weights only when val_loss < best_val - 1e-6.
- Counts other epochs toward the caller-supplied stopping patience.
- Creates checkpoint directories with exist_ok=True and writes fixed best,
  last and periodic filenames.
- Receives its optimizer, datasets and training configuration from the caller.

Before production integration, specify the intended sample aggregation,
selection tolerance, exact consumed controls and collision-safe publication.
Do not silently describe existing behavior as exact minimum-loss selection.
Do not alter historical checkpoints or execution evidence.

## 11. Pinned evaluator findings and proposed revision policy

Source inspected: gcs-core/gcs_core/val_common.py at submodule
df7b5279714d54694e598cabe3188c73297741e1.

### Confirmed source findings

- KID unconditionally applies grayscale_to_rgb to both image inputs,
  incompatible with the revised canonical three-channel contract.
- compute_all_metrics invokes KID twice. The first call is unguarded;
  the second catches exceptions and substitutes None.
- Inception initialization failures are suppressed and cached as unavailable.
- cFID omits unavailable classes from its macro average.
- Normalization infers numeric range and reshapes by element count.
- Real-only training uses balanced weights; augmented training uses none.
- ROC exceptions are replaced with recall zero.
- TensorFlow seed-setting exceptions are suppressed.
- Real and augmented image arrays are materialized and concatenated.
- Classifier macro precision/recall are referenced using generative-sounding
  legacy column terminology. Actual downstream writer mappings still need
  verification; these are not generative distribution precision/recall.
- No MS-SSIM computation appears in the inspected compute_all_metrics path.
  This does not establish absence elsewhere in the repository.

These are findings about the pinned implementation, not conclusions about
every historical execution.

### Proposed classifier protocol — acceptance pending

- EvalCNN: three same-padded 3x3 ReLU convolutions, widths 32/64/128,
  each followed by 2x2 stride-2 max pooling at canonical resolution.
- Global average pooling, dropout 0.1, K-class softmax.
- Categorical cross-entropy; Adam learning rate 0.001.
- Batch size 128; maximum 20 epochs.
- Early stopping on real-validation loss, patience 5, restore best weights.
- No class weighting in either primary comparison condition.
- Same real validation/test membership and paired evaluator seed policy.
- No test-based checkpoint selection, tuning, filtering or calibration.
- Fixed epoch limits do not equalize optimizer updates between conditions.
  Report actual samples, updates, completed epochs and stopping outcomes.
- Use bounded-memory loading and preserve deterministic sample identities.

### Proposed metric execution contract — acceptance pending

- Validate explicit canonical shapes, numeric ranges, finite values and labels.
- Fix authoritative class count and mapping independently of observed maxima.
- Compute each requested metric once and retain its configuration and status.
- Preserve RGB inputs in feature extraction.
- Pin feature-extractor identity, weights and preprocessing.
- Reuse deterministic real-reference identities across family comparisons.
- Record per-class real/synthetic counts and expected/computed coverage.
- Required metric failures prevent complete-evaluation status, while already
  completed training artifacts and metrics remain preserved.
- Optional metrics require explicit NOT_REQUESTED or UNAVAILABLE reasons.
- Never substitute failed metrics with numerical zero.
- Keep classifier precision/recall separate from generative diagnostics.
- Test-ROC recall at 1% FPR is descriptive discrimination analysis, not
  evidence of a deployable validation-selected operating threshold.

Exact KID/FID sampling, MS-SSIM definition, estimator validation,
statistical analysis and failure aggregation remain to be finalized.
Historical evaluator behavior remains preserved.

## 12. Shared execution and analysis proposal

Status: PROPOSED — NOT YET FROZEN OR IMPLEMENTATION-VERIFIED.
This section proposes new decisions, not recovered historical decisions.

### Randomization and primary comparison

- Experimental seeds: [42,43,44,45,46,47,48].
- Dataset partitions remain fixed under their existing contracts.
- Evaluator seed equals the associated experimental seed.
- Real-only and augmented classifiers use the same evaluator seed and
  initial architecture/weights; verify initial-weight identity explicitly.
- Reset supported random generators before each classifier construction.
  Record environment and determinism settings; identical seeds alone do
  not guarantee bitwise replay across environments.
- Use no class weighting in either primary condition.
- Retain the proposed EvalCNN controls in Section 11.
- An epoch traverses the entire respective training condition.
  Equal epoch limits do not equalize updates or real-sample exposure.
- Reuse a real-only baseline across families only when dataset, split,
  evaluator configuration, seed, source and environment identities match.

### Three-scale intra-class MS-SSIM proposal

- Metric identifier: ms_ssim_3scale_40x40_v1.
- Input: finite float32 canonical 40x40x3 images in [0,1].
- No resizing, grayscale conversion or range guessing inside this metric.
- TensorFlow ssim_multiscale with max_val=1.0, filter_size=7,
  filter_sigma=1.5, k1=0.01 and k2=0.03.
- Three scales, with equal power_factors=(1/3,1/3,1/3).
  This is an explicitly adapted variant, not standard five-scale MS-SSIM.
- Preserve the implementation's channel reduction semantics and record
  the exact TensorFlow version.
- Select 200 unique unordered pairs of distinct samples per class,
  uniformly without replacement from the class's possible pairs.
- Require the complete 2000/class pool for primary evaluation.
- Pair selection uses default_rng(42 + class_id), separately per class,
  on sample indices in the verified synthetic-manifest order.
- Persist pair indices and manifest identity; reuse index pairs across
  comparable pools with the same count.
- Report every class mean, pair count and an equal-weight class macro mean.
- Any missing class, failed pair or nonfinite result prevents complete
  metric status; no single-scale SSIM fallback or successful-pair omission.
- Higher similarity suggests lower within-class variation. A low value
  alone does not establish realism, label fidelity or useful diversity.
- Historical fallback values are not directly comparable to this variant.
- Acceptance requires geometry, reference agreement, reproducibility,
  class coverage and failure-path checks before production use.

### Utility estimand and uncertainty proposal

- Primary endpoint: augmented minus real-only macro-F1.
- Analyze seven paired repetition-level differences separately for each
  dataset/family, with macro-F1 expressed on the [0,1] scale.
- Publish all seven differences, their mean and sample standard deviation.
- Supporting endpoints include balanced accuracy, macro-AUPRC,
  calibration and classwise outcomes.
- Proposed primary uncertainty summary: two-sided 95% Student-t interval
  for the mean paired difference, using six degrees of freedom.
- This interval assumes independent repetition-level differences and an
  adequate normal approximation; seven repetitions cannot establish that
  assumption. Disclose its small-sample and fixed-dataset limitations.
- The interval is pointwise, not simultaneous across the benchmark.
- Do not treat synthetic samples, classes, metric subsets or reused
  baselines as additional independent repetitions.
- The proposed inference policy is estimation-only, as specified in
  Section 13; no confirmatory significance or benchmark-wide superiority claims.
- Do not use confidence-interval overlap with zero as evidence of equivalence.
- No inference policy may be selected because it produces favorable results.

### Missing-run and retry proposal

- Enumerate every expected dataset/family/seed and required stage.
- Track training, checkpoint, synthesis and evaluation statuses separately.
- Preserve every failed, interrupted and superseded attempt with its cause.
- A retry retains scientific identity but receives a fresh execution ID
  and destination; never overwrite an authoritative completed artifact.
- Select the first attempt satisfying predeclared execution and integrity
  requirements, never the attempt with the best scientific metric.
- Infrastructure retries must not alter the scientific configuration.
  Scientific changes create a new protocol identity and require an
  explicit compatibility/re-execution decision.
- Resume only when provenance and supported restoration semantics match;
  otherwise restart under a fresh attempt identity.
- Retry limits/resource escalation must be fixed before scale-out.
- Never replace missing values with zero or silently shrink the seed set.
- A cell with fewer than seven valid paired runs is INCOMPLETE.
  Available-case summaries may be shown as descriptive, explicitly
  conditional on successful runs; primary seven-run inference is withheld.
- Report completion/failure rates alongside scientific results.
- No result-driven replacement seeds.

### Remaining shared decisions

- KID/FID reference sampling, feature-weight identity and metric scope.
- Final acceptance of estimation-only reporting and its claim boundaries.
- Exact retry limits and resource policy.
- MS-SSIM implementation acceptance.
- AR/MAF complete training and sampling recipes.
- Final scientific acceptance and executable configuration linkage.

FULL_BENCHMARK_GO remains NO.

## 13. Fidelity and inference proposal

Status: PROPOSED — acceptance and implementation verification pending.

### Feature extraction

- Use one fixed Keras InceptionV3 feature extractor: include_top=False,
  pooling=avg, input_shape=(299,299,3), explicitly selected ImageNet weights.
- Record framework versions and SHA-256 of the actual weight artifact.
  Resolve and verify that artifact before evaluation; do not silently
  substitute random weights or a different feature extractor.
- Inputs are canonical RGB float32 [0,1]. Preserve all three channels.
- Resize to 299x299 with explicitly configured bilinear interpolation,
  antialias=False; multiply by 255 and use InceptionV3 preprocess_input.
- Extract features in inference mode with bounded batches.
- Inception features are generic-image diagnostics, not demonstrated
  malware-semantic representations.
- Pixel/feature preprocessing must be identical across all families.

### Class-balanced reference design

- Reference real images come from the governed validation split.
- For each dataset, set m to min(200, minimum validation class count).
  Require m >= 2 and all authoritative classes to be present.
- Select m distinct real validation samples per class using a separate
  default_rng(42 + class_id) on fixed governed split order.
- Save selected identities before family comparison and reuse them across
  all families and repetitions on that dataset.
- Select m distinct synthetic samples per class using a separate
  default_rng(42 + class_id) on verified manifest order.
- Never filter these selections using metric scores.
- The pooled reference and synthetic sets each contain K*m images,
  with equal representation of every authoritative class.
- Report m, K, per-class counts, selected identities and effective sizes.
- Missing classes prevent complete primary fidelity status.

### KID definition

- Primary supporting fidelity diagnostic: pooled class-balanced KID.
- Use unbiased degree-3 polynomial-kernel MMD-squared on the fixed features,
  gamma=1/feature_dimension and coef0=1.
- Use ten subset draws, each of size min(200,K*m), sampling without
  replacement within each draw, with default_rng(42).
- Retain every subset estimate and report their arithmetic mean.
- Draws may overlap and are not independent experimental repetitions.
  Their spread is descriptive, not a seven-seed confidence interval.
- Do not clamp negative unbiased estimates to zero.
- Explicitly call the result pooled class-balanced KID, not macro cKID.
- Record estimator, feature and subset identities; compute once.
- Validate estimator mathematics and RGB integration before execution.

### FID and cFID scope

- Secondary diagnostics use the same fixed feature sets.
- Report pooled balanced FID separately from per-class FID.
- Macro cFID is the equal-weight mean over all authoritative classes.
  Incomplete class coverage prevents complete macro-cFID status.
- Explicitly disclose limited samples relative to 2048 feature dimensions,
  singular empirical covariances and finite-sample estimator limitations.
- Validate numerical handling; do not silently alter features, add
  unreported regularization, discard failed classes or substitute metrics.
- Do not infer generator rankings solely from FID/cFID/KID.

### Inference and claim boundaries

- Proposed Paper 1 reporting is estimation-focused: all seed results,
  mean paired macro-F1 differences, sample SD and pointwise t intervals.
- No confirmatory hypothesis tests, adjusted p-values or significance stars.
- No family-wide or benchmark-wide superiority claim from pointwise intervals.
- Supporting endpoints and cross-metric correlations are descriptive.
- No pooling datasets as if they were independent seed repetitions.
- A later confirmatory testing plan requires a prospective amendment
  before the relevant outcome comparisons are examined.
- Reviewer concerns are addressed through paired effect/uncertainty
  reporting and restrained claims, not an automatic significance claim.

Remaining decisions: AR/MAF complete recipes, retry resource limits,
final acceptance of the proposed shared policies and executable linkage.
Stage 2 remains IN_PROGRESS. FULL_BENCHMARK_GO remains NO.

## 14. Shared-policy acceptance and AR/MAF recovery

Decision date: 2026-11-04.

Bruno explicitly accepted the shared methodological choices presented in
Sections 11-13: seeds 42-48, paired unweighted classifier comparison,
specified diagnostic metrics, estimation-focused paired reporting,
claim limitations, and explicit incomplete-run/retry identity rules.

SHARED_POLICY_METHODOLOGICAL_ACCEPTANCE=YES
SHARED_POLICY_IMPLEMENTATION_VERIFIED=NO
FULL_BENCHMARK_GO=NO

This acceptance supersedes pending/proposed wording only for those selected
shared choices. It does not close explicitly unresolved retry/resource
limits, feature-weight identification, numerical implementation checks,
AR/MAF recipes or production activation. Earlier proposal wording is
retained as the decision-development record.

### Recovered AR/MAF scope

The supplied historical recovery confirms accepted model mathematics and
training primitives, not complete dataset-level execution recipes.
Missing optimizer, budget, traversal, checkpoint and sampling controls
must be selected prospectively rather than inferred from tests.

The recovery is reported historical evidence. Exact original acceptance-test
commits remain unestablished; current implementation checks are separate.

AR's inspected observation contract supplies initial scale bounds [-7,3].
The recovery's broader statement that scale values were never selected
does not override this explicit contractual evidence.

Current AR constructor defaults are filters=64 and masked_layers=6,
following an initial mask-A layer. Their use in the benchmark requires
an explicit recipe decision.

MAF structural permutation seeds are (42,43,44,45,46), distinct from
experimental seeds. Its sampler resets a stateless stream per call;
production orchestration must prevent accidental latent-stream reuse
across generated batches.

No AR/MAF recipe is selected by this recovery record.
Stage 2 remains IN_PROGRESS.
