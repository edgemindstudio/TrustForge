# Paper 1 Revision 2026 — Restricted Boltzmann Class-Conditioning Policy

`STATUS=SELECTED`

`RBM_CLASS_CONDITIONING_POLICY_SELECTED=YES`

`RBM_CLASS_CONDITIONING_POLICY_REVISION=SHARED_CONDITIONAL`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs class conditioning for the revised Paper 1 Restricted
Boltzmann Machine family.

It selects how class identity participates in:

- training;
- model energy;
- hidden conditional probabilities;
- checkpoint identity;
- synthesis;
- synthetic labels.

It does not yet select hidden width, optimizer, training budget, checkpoint
format, or sampling-chain length.

## 2. Historical class-conditioning evidence

The historical Paper 1 RBM used one independently trained RBM per class.

For historical class `k`, the trainer:

1. selected training examples satisfying `label == k`;
2. created a separate RBM;
3. trained that RBM only on class-`k` data;
4. validated against class-`k` validation data;
5. stored checkpoints beneath a class-specific directory.

Historical synthesis then loaded the class-specific checkpoint and generated
samples assigned to the corresponding class.

Therefore:

`RBM_HISTORICAL_CLASS_CONDITIONING=SEPARATE_MODEL_PER_CLASS`

This historical structure remains provenance evidence.

It is not automatically authoritative for the revised four-dataset benchmark.

## 3. Revision-scale evidence against separate class models

The revised canonical visible dimensionality is:

`D=4800`

For a standard RBM with `H` hidden units, excluding any class-conditioning
parameters:

`parameter_count = D*H + D + H`

At `H=128`:

`parameter_count = 619328`

A governed training-class audit found:

### USTC

- 9 classes;
- 8100 training examples per class.

At `H=128`:

- approximately `76.46` parameters per training example.

### CICMalDroid2020

- 5 classes;
- smallest training class: `877`;
- median training class: `1470`;
- largest training class: `2732`.

At `H=128`:

- smallest-class ratio: approximately `706.19` parameters/example;
- median-class ratio: approximately `421.31`;
- largest-class ratio: approximately `226.69`.

### MalNet-Image

- 47 classes;
- smallest training class: `79`;
- median training class: `585`;
- largest training class: `619118`.

Five smallest observed training classes contained:

- 79;
- 83;
- 106;
- 106;
- 125

examples.

At `H=128`:

- smallest-class ratio: approximately `7839.59` parameters/example;
- median-class ratio: approximately `1058.68`;
- largest-class ratio: approximately `1.00`.

At larger hidden widths these ratios increase further.

### MaleX

- 2 classes;
- smaller training class: `125794`;
- larger training class: `605255`.

At `H=128`:

- smaller-class ratio: approximately `4.92` parameters/example;
- larger-class ratio: approximately `1.02`.

## 4. Separate-model decision

The historical separate-model-per-class design is rejected as the authoritative
revision design.

Reason:

The expanded benchmark contains highly imbalanced class populations, especially
MalNet-Image.

Training a full independent `D x H` RBM for classes containing only tens or
hundreds of observations would create extreme parameter-to-example ratios and
would prevent statistically efficient sharing of representation across classes.

Preserving the historical structure would therefore undermine the revised
benchmark's scalability and methodological consistency.

`RBM_REVISION_SEPARATE_MODEL_PER_CLASS=REJECTED`

## 5. Selected revision class-conditioning architecture

The revised Paper 1 RBM uses:

**one shared class-conditional RBM per dataset and experimental seed**

with:

- one shared visible-hidden weight matrix `W`;
- one shared visible bias vector `a`;
- one shared hidden bias vector `b`;
- one class-to-hidden interaction matrix `U`.

For a dataset containing `K` classes:

`W shape = (D,H)`

`a shape = (D,)`

`b shape = (H,)`

`U shape = (K,H)`

The class label is represented by a governed zero-based class identity `y`.

`RBM_REVISION_CLASS_CONDITIONING=SHARED_CLASS_TO_HIDDEN_INTERACTION`

## 6. Governed conditional energy

For canonical visible vector `v`, hidden vector `h`, and class identity `y`,
the governed conditional energy is:

`E(v,h | y) = -a^T v - b^T h - v^T W h - U[y,:]^T h`

where:

`v in [0,1]^D`

and:

`h in {0,1}^H`.

Class identity therefore changes the energy landscape through the hidden layer.

No posthoc class assignment is required to create class-aware samples.

## 7. Hidden conditional

The governed hidden conditional is:

`P(h_j=1 | v,y)`

`= sigmoid(b_j + W[:,j]^T v + U[y,j])`

Therefore class identity directly affects hidden activation probabilities.

A numerical smoke test confirmed:

- finite hidden probabilities;
- probabilities remained in `[0,1]`;
- changing class identity changed hidden probabilities;
- changing class identity changed the joint energy;
- gradients with respect to `W`, `b`, and `U` were present and finite for the
  tested hidden-probability objective.

`RBM_SHARED_CONDITIONAL_HIDDEN_SEMANTICS_SMOKE=PASS`

## 8. Visible conditional

Conditioned on hidden state `h`, the visible conditional remains the selected
bounded-exponential distribution.

For visible coordinate `i`:

`theta_i(h) = a_i + W[i,:] h`

and:

`p(v_i | h,y) = p(v_i | h)`

with:

`p(v_i | h) ∝ exp(theta_i(h) * v_i)`

for:

`0 <= v_i <= 1`.

The class variable therefore influences visible generation through the sampled
hidden state rather than by changing the visible observation family itself.

A numerical smoke test confirmed that for a fixed hidden state, the visible
natural parameter is class-independent.

`RBM_VISIBLE_CONDITIONAL_GIVEN_H_CLASS_INDEPENDENT=YES`

## 9. No class-only bias requirement

A class-only scalar term that depends on `y` but not on `v` or `h` contributes
only an additive constant when class identity is clamped for conditional
training and synthesis.

The revision therefore does not require a separate class-only bias parameter.

Any future addition of such a parameter requires separate scientific
justification.

`RBM_CLASS_ONLY_BIAS=NOT_REQUIRED`

## 10. Shared-parameter scalability evidence

The shared conditional design adds only:

`K * H`

class-conditioning parameters through `U`.

At `H=128`:

### USTC

- shared base parameters: `619328`;
- class-conditioning parameters: `1152`;
- total shared conditional parameters: `620480`.

### CICMalDroid2020

- class-conditioning parameters: `640`;
- total shared conditional parameters: `619968`.

### MalNet-Image

- class-conditioning parameters: `6016`;
- total shared conditional parameters: `625344`.

### MaleX

- class-conditioning parameters: `256`;
- total shared conditional parameters: `619584`.

Compared with separate per-class RBMs, the approximate parameter-count
reductions at `H=128` are:

- USTC: about `9x`;
- CICMalDroid2020: about `5x`;
- MalNet-Image: about `46.5x`;
- MaleX: about `2x`.

The class-conditioning overhead relative to the shared RBM base remains below
approximately one percent even for 47-class MalNet-Image.

`RBM_SHARED_CONDITIONAL_SCALABILITY_AUDIT=PASS`

## 11. Scientific identity

The minimum trained-model scientific identity is:

`(dataset_id, experimental_seed)`

The trained model itself contains all governed class-conditioning rows for that
dataset.

Class identity remains part of every synthesis request.

Therefore generation identity includes:

`(dataset_id, class_id, experimental_seed)`

The distinction must remain explicit:

`TRAINED MODEL IDENTITY != SYNTHESIS REQUEST IDENTITY`

## 12. Training semantics

Training examples retain their governed class labels.

For every observed pair `(v,y)`:

- `v` enters the shared visible representation;
- `y` selects exactly one governed row of `U`;
- positive-phase hidden probabilities use `p(h|v,y)`.

The implementation must not:

- discard labels;
- remap labels outside the governed dataset mapping;
- train separate hidden/visible matrices for each class under this policy;
- construct one unconditional model and attach labels afterward.

## 13. Negative-phase boundary

The exact negative-phase estimator remains separately governed.

However, any class-conditional Gibbs chain used for training must hold the
training class identity fixed throughout that chain unless a future policy
explicitly selects joint label sampling.

For fixed class `y`, a Gibbs transition is therefore structurally:

`v -> h ~ p(h|v,y) -> v' ~ p(v|h)`

Class identity is not resampled inside that fixed-class chain.

`RBM_FIXED_CLASS_GIBBS_CHAIN=REQUIRED_FOR_CONDITIONAL_CD_CANDIDATE`

This marker does not yet select CD depth.

## 14. Synthesis semantics

For requested class `k`:

1. load the accepted shared dataset/seed checkpoint;
2. validate that class `k` is within the governed class universe;
3. clamp class identity to `k`;
4. initialize the governed Gibbs chain;
5. sample hidden state from `p(h|v,k)`;
6. sample visible state from the governed bounded-exponential `p(v|h)`;
7. repeat according to the separately governed chain policy;
8. emit canonical synthetic samples labeled `k`.

Every generated label must match the class identity actually clamped during
generation.

## 15. No posthoc class assignment

The revised implementation must not:

- generate one unconditional sample pool;
- partition that pool into class-sized slices;
- attach class labels afterward.

Class identity must participate in the generative chain before the sample is
produced.

`RBM_POSTHOC_SYNTHETIC_LABEL_ASSIGNMENT=FORBIDDEN`

## 16. Checkpoint identity

One accepted RBM checkpoint corresponds to one:

`(dataset_id, experimental_seed)`

and contains at least:

- shared `W`;
- shared `a`;
- shared `b`;
- complete class-to-hidden matrix `U`;
- governed class count and mapping identity;
- architecture identity;
- integrity metadata.

The exact serialization format remains separately governed.

A checkpoint must fail closed if its class mapping is inconsistent with the
dataset adapter used for synthesis.

## 17. Cross-dataset isolation

Parameter sharing occurs only across classes within the same governed dataset.

No RBM parameter state may be shared across:

- USTC;
- CICMalDroid2020;
- MalNet-Image;
- MaleX

within one benchmark execution.

`RBM_CROSS_DATASET_PARAMETER_SHARING=FORBIDDEN`

## 18. Historical preservation

Historical class-specific RBM checkpoints remain historical provenance.

They must not be rewritten or relabeled as shared conditional RBM checkpoints.

Revision checkpoints must use revision-specific artifact paths.

## 19. Implementation gate

Before class-conditioning implementation may be accepted, focused evidence must
demonstrate at least:

1. one shared `W` per dataset/seed;
2. one shared `a` per dataset/seed;
3. one shared `b` per dataset/seed;
4. one `U` row per governed class;
5. governed class IDs index the correct `U` rows;
6. changing `y` can change `p(h|v,y)`;
7. changing `y` can change model energy;
8. fixed `h` leaves `p(v|h)` unchanged across classes;
9. finite class-conditioned hidden probabilities;
10. finite model energy;
11. finite gradients for trainable class-conditioning state;
12. fixed-class Gibbs chains retain the requested class;
13. synthesis class `k` clamps class `k` before generation;
14. no unconditional-pool label assignment;
15. checkpoint class mapping is identity-bound;
16. invalid class identity fails closed;
17. no cross-dataset parameter sharing;
18. canonical `(B,40,40,3)` output compatibility remains intact.

## 20. Current authorization state

The revised class-conditioning architecture is selected as:

`SHARED_CLASS_TO_HIDDEN_INTERACTION`

This supersedes the earlier provisional separate-model-per-class revision
direction before implementation.

Implementation remains unaccepted.

Real benchmark training remains unauthorized.

Therefore:

`RBM_CLASS_CONDITIONING_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
