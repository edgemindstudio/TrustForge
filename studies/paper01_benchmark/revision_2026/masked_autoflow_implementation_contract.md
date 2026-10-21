# Paper 1 Revision 2026 — Masked AutoFlow Implementation Contract

## Status

`IMPLEMENTATION_STATUS=ACCEPTED`

`MASKED_AUTOFLOW_TRAINING_AUTHORIZED=YES`

This contract defines the implementation requirements for the governed
revision-2026 Masked AutoFlow family.

It does not authorize experiment execution.

## 1. Governed identity

Benchmark family:

`maskedautoflow`

Revision implementation:

`revision_2026_block_conditional_piecewise_linear_maf`

The historical implementation:

`historical_gencys_maskedautoflow`

remains preserved for provenance and must not be modified in place.

## 2. Canonical representation

The implementation must accept model-facing tensors with:

- shape `(B, 40, 40, 3)`;
- dtype float32;
- canonical range `[0,1]`.

The canonical scalar dimensionality is:

`40 x 40 x 3 = 4800`

No canonical dimension may be dropped, duplicated, reordered ambiguously, or
silently transformed outside governed code.

## 3. Governed block representation

Canonical images must be partitioned into:

`5 x 5 = 25`

non-overlapping spatial blocks.

Each block must contain:

`8 x 8 x 3 = 192`

scalar values.

The implementation must provide explicit reversible helpers conceptually
equivalent to:

- `images_to_blocks`;
- `blocks_to_images`.

Required block tensor shape:

`(B, 25, 192)`

Required properties:

- exact shape preservation;
- deterministic ordering;
- exact round-trip recovery;
- no duplicated scalar positions;
- no omitted scalar positions.

## 4. Block ordering

The baseline governed block order is raster order over the 5 x 5 spatial grid.

Conceptually:

`B_0, B_1, ..., B_24`

Later flow stages may use deterministic whole-block permutations.

Permutations must:

- operate only on complete 192-dimensional blocks;
- be invertible;
- be reproducible from a governed architecture seed;
- expose explicit inverse permutations;
- never split or partially permute scalar values inside one block.

## 5. Autoregressive masking

For one flow stage, transform parameters for block `i` may depend on:

- class condition `y`;
- earlier blocks in that stage's governed ordering.

They must not depend on:

- the current block being transformed;
- future blocks in that stage's ordering.

The output dependency relation must therefore be strict.

The historical self-dependence defect must not be reproduced.

Regression testing must demonstrate numerically that changing only the current
block cannot alter that block's predicted transform parameters when earlier
blocks and class condition are held fixed.

Regression testing must also demonstrate that earlier blocks can influence
later-block transform parameters.

## 6. Class conditioning

The revised MAF is conditional.

Class input must participate in:

- training;
- likelihood evaluation;
- synthesis.

A class representation may use a learned embedding.

Initial governed candidate:

`CLASS_EMBED_DIM=64`

Changing only the requested class condition must be capable of changing
transform parameters.

Synthetic labels must match the class condition actually supplied during
generation.

Post-hoc assignment of labels to unconditional samples is forbidden.

## 7. Masked conditional backbone

Initial governed implementation candidate:

- block hidden width: 256;
- hidden depth: 2.

Backbone representations must be block-specific.

The network may use masked dense or equivalent masked tensor operations, but
the dependency structure must remain explicit and regression testable.

The implementation must not rely on accidental runtime ordering to enforce
causality.

## 8. Shared per-block spline decoder

The same decoder weights are shared across all 25 block positions.

Governed candidate:

`256 -> 64 -> 3072`

where:

- 256 = block hidden representation width;
- 64 = decoder bottleneck;
- 3072 = transform values for one 192-dimensional block.

The decoder must emit distinct transform parameters for each scalar position
inside the current block.

Weight sharing across block positions must not imply shared output values.

Different block representations must be capable of producing different spline
parameters.

## 9. Final spline parameterization

The selected scalar transform is:

`8-bin bounded monotonic piecewise-linear spline`

Each scalar requires:

- 8 raw bin-width parameters;
- 8 raw bin-height parameters.

Therefore:

`16 transform values per scalar`

and:

`192 x 16 = 3072`

transform values per block.

## 10. Bin normalization

Raw widths and heights must be converted into strictly positive normalized bin
sizes.

The implementation must enforce:

- all widths > 0;
- all heights > 0;
- widths sum to 1;
- heights sum to 1.

Initial governed minimum values:

`MIN_BIN_WIDTH=1e-3`

`MIN_BIN_HEIGHT=1e-3`

Any future change to these minimums is a governed configuration change.

No endpoint epsilon displacement is permitted.

## 11. Spline knot construction

For each scalar transform:

- lower x boundary = 0;
- upper x boundary = 1;
- lower y boundary = 0;
- upper y boundary = 1.

Cumulative widths define x-knots.

Cumulative heights define y-knots.

The final knots must be fixed explicitly to exact canonical boundaries to
remove floating-point accumulation drift.

Exact input zero must map to exact output zero.

Exact input one must map to exact output one.

## 12. Piecewise-linear forward transform

Within selected bin `k`:

`y = y_k + slope_k * (x - x_k)`

where:

`slope_k = height_k / width_k`

The per-scalar forward log absolute Jacobian determinant is:

`log_abs_det = log(slope_k)`

Required properties:

- monotonicity;
- bounded output;
- finite output;
- finite log determinant;
- exact canonical boundaries;
- no clipping.

## 13. Piecewise-linear inverse transform

Within selected output bin `k`:

`x = x_k + (y - y_k) / slope_k`

The inverse log absolute Jacobian determinant is:

`inverse_log_abs_det = -log(slope_k)`

Required properties:

- bounded output;
- finite output;
- exact canonical boundaries;
- numerically stable forward/inverse round trip;
- no clipping.

## 14. Knot convention

Internal spline knots are measure-zero locations where the piecewise derivative
changes.

The implementation must use one deterministic bin-selection convention at
internal knots.

The same convention must be used consistently in forward and inverse logic.

Regression tests must include internal-knot values.

## 15. One block-autoregressive flow stage

For one stage:

1. obtain governed ordered block representation;
2. compute masked conditional hidden representations;
3. decode spline parameters for each block;
4. apply scalar spline transforms to all 192 values inside each block;
5. sum scalar log-Jacobian terms;
6. apply or record the governed block permutation semantics.

Forward density evaluation may compute all block parameters in parallel because
the strict mask ensures valid autoregressive dependencies.

## 16. Inverse of one flow stage

Inverse synthesis must recover blocks sequentially in governed autoregressive
order.

For each block:

1. compute parameters from already recovered earlier blocks plus class;
2. invert all 192 scalar spline transforms in parallel;
3. write the recovered block into the current state.

Therefore one flow requires at most:

`25 sequential autoregressive block stages`

## 17. Full flow stack

Initial governed candidate:

`NUM_FLOWS=5`

Maximum inverse autoregressive depth:

`25 x 5 = 125 block stages`

This bound must be explicit in implementation and regression tests.

The implementation must not fall back to scalar-sequential inversion over all
4,800 canonical dimensions.

## 18. Base distribution

The selected bounded transform operates directly on `[0,1]`.

The initial implementation candidate uses a factorized Uniform(0,1) base on
the final transformed coordinates.

For one scalar transformed coordinate `z`:

`log p_base(z) = 0`

for `z` in `[0,1]`.

Therefore the flow log likelihood is determined by the accumulated log
Jacobian under the bounded transform stack.

No Gaussian-base logit transform is part of the governed design.

No endpoint epsilon is part of the governed design.

No universal dequantization is part of the governed design.

## 19. Conditional log probability

The model must expose a conditional log-probability operation conceptually
equivalent to:

`log_prob(x, y)`

where:

- `x` is canonical image data;
- `y` is the class condition.

The result must be one finite scalar log probability per sample.

The likelihood must not ignore `y`.

## 20. Sampling

Sampling must conceptually:

1. draw governed Uniform(0,1) latent values;
2. apply inverse flow stages in reverse order;
3. use the requested class condition at every conditional stage;
4. reconstruct `(B,40,40,3)` canonical tensors.

Required output properties:

- float32;
- finite;
- shape `(B,40,40,3)`;
- values in `[0,1]` by construction;
- no post-generation clipping.

## 21. Randomness

All stochastic sampling must be seed-governed.

Fixed seed plus identical model state and identical class requests must produce
identical samples.

Changed seed must be capable of producing different stochastic samples.

Randomness used for:

- latent sampling;
- architecture permutations, if seed-derived;
- training shuffling;

must have explicit and reproducible seed semantics.

## 22. Numerical requirements

The implementation must guard against:

- zero-width bins;
- zero-height bins;
- non-finite slopes;
- non-finite log determinants;
- non-finite loss;
- non-finite gradients.

Silent replacement of invalid values is forbidden.

Invalid numerical state must fail closed.

## 23. Training

The training objective is conditional negative log likelihood:

`NLL = -mean(log_prob(x, y))`

Training must:

- use real class labels;
- preserve canonical `[0,1]` inputs;
- use no universal dequantization;
- use no endpoint epsilon;
- use no post-hoc canonical clipping.

Initial optimizer and batch-size choices are not yet locked.

Training batch size must be selected after memory/throughput smoke testing.

## 24. Historical isolation

The following historical files remain historical:

- `maskedautoflow/models.py`;
- `maskedautoflow/train.py`;
- `maskedautoflow/sample.py`;
- `maskedautoflow/pipeline.py`;
- `adapters/maskedautoflow_adapter.py`.

The revision implementation must live under the revision-specific model
namespace and must not rewrite historical artifacts in place.

## 25. Required implementation modules

The revision implementation should be separated conceptually into:

- block packing/unpacking;
- bounded spline primitive;
- masked conditional block network;
- block-MAF flow stage;
- multi-flow model;
- training;
- sampling.

Exact file boundaries may differ if separation remains clear and testable.

## 26. Required regression gates

Before training authorization, the permanent regression suite must demonstrate:

1. `(B,40,40,3)` -> `(B,25,192)` block shape;
2. exact image/block round trip;
3. no dropped or duplicated scalar locations;
4. reproducible raster block ordering;
5. reproducible seeded whole-block permutations;
6. valid inverse permutations;
7. positive finite bin widths;
8. positive finite bin heights;
9. widths sum to one;
10. heights sum to one;
11. exact zero maps to exact zero;
12. exact one maps to exact one;
13. interior forward values remain in `[0,1]`;
14. inverse values remain in `[0,1]`;
15. finite forward log Jacobians;
16. finite inverse log Jacobians;
17. forward/inverse round-trip error within governed tolerance;
18. forward/inverse log-Jacobian cancellation within governed tolerance;
19. internal-knot behavior is deterministic;
20. current block cannot influence its own parameters;
21. future blocks cannot influence earlier-block parameters;
22. earlier blocks can influence later-block parameters;
23. class condition changes conditional parameters;
24. shared decoder can produce different parameters for different blocks;
25. conditional `log_prob(x,y)` is finite;
26. changing `y` can change `log_prob(x,y)`;
27. one real training step yields finite loss;
28. one real training step yields finite gradients;
29. trainable parameters update after an optimizer step;
30. fixed-seed sampling is reproducible;
31. changed seed changes stochastic samples;
32. generated samples are finite;
33. generated samples stay in `[0,1]` without clipping;
34. generated samples preserve `(B,40,40,3)` shape;
35. requested class condition participates during sampling;
36. one flow uses at most 25 sequential inverse block stages;
37. five flows use at most 125 sequential inverse block stages;
38. no historical scalar-sequential 4,800-step inverse path is used;
39. no endpoint epsilon API exists;
40. no universal dequantization path exists in the revision implementation.

## 27. Authorization gate

Implementation work may begin under this contract.

Real dataset training remains unauthorized until all required structural,
numerical, conditioning, sampling, and training regression gates pass.

Current authoritative state:

`MAF_IMPLEMENTATION_AUTHORIZED=YES`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`

## 28. Numerical-conditioning amendment

The initial implementation contract described 16 scalar transform outputs as:

- 8 raw widths;
- 8 raw heights.

Five-flow composition testing demonstrated that independently normalized widths
and heights permit unnecessarily extreme segment slopes and poor float32
conditioning across the composed flow stack.

The authoritative implementation therefore keeps the same 16-output budget but
changes the second eight outputs to bounded relative-slope controls.

Per scalar:

- outputs 0-7: raw width parameters;
- outputs 8-15: raw relative-slope controls.

Widths must remain strictly positive and normalized to one.

Relative-slope controls must use:

`bounded_log_relative_slope = MAF_LOG_SLOPE_LIMIT * tanh(raw_control)`

with:

`MAF_LOG_SLOPE_LIMIT=0.75`

and:

`relative_slope = exp(bounded_log_relative_slope)`

Heights must then be derived as:

`unnormalized_height_i = width_i * relative_slope_i`

`height_i = unnormalized_height_i / sum_j(unnormalized_height_j)`

The implementation must not independently softmax height logits.

The resulting segment slope is:

`slope_i = height_i / width_i`

The theoretical governed slope envelope is:

`exp(-1.5) <= slope_i <= exp(1.5)`

approximately:

`0.22313 <= slope_i <= 4.48169`

This amendment supersedes Sections 9-10 only where those sections describe
the second eight outputs as independent height parameters.

All other bounded-domain, invertibility, endpoint, decoder-size, and flow-family
requirements remain unchanged.

Additional regression requirements:

41. relative-slope controls map to finite positive relative slopes;
42. derived heights are finite and strictly positive;
43. derived heights sum to one;
44. observed segment slopes remain inside the governed theoretical envelope;
45. five-flow round-trip maximum absolute error remains below `2e-5` under the
    governed deterministic regression case;
46. five-flow forward/inverse log-Jacobian cancellation remains below `1e-3`
    under the governed deterministic regression case.

Current authoritative state:

`MAF_SPLINE_PARAMETERIZATION=SLOPE_CONDITIONED`

`MAF_LOG_SLOPE_LIMIT=0.75`

`MAF_IMPLEMENTATION_AUTHORIZED=YES`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`


## 29. Final implementation acceptance and training authorization

The revision-2026 Masked AutoFlow implementation completed its governed
implementation acceptance gate.

Accepted evidence includes:

- syntax validation passed;
- diff/whitespace validation passed;
- corrected prohibited-active-pattern audit passed;
- legacy identifiers `DEQUANT_NOISE`, `canonical_to_logit`, `raw_heights`,
  and `MIN_BIN_HEIGHT` are absent from the active revision implementation;
- governed implementation constants passed;
- consolidated focused MAF regression suite passed:
  `34 passed`;
- five-flow bounded composition passed the governed numerical-conditioning
  requirements;
- conditional log probability is finite and class-sensitive;
- fixed-seed synthesis is reproducible;
- generated values remain in `[0,1]` without clipping;
- maximum inverse autoregressive depth remains 125 block stages;
- conditional NLL is finite;
- gradients are finite;
- optimizer steps update trainable parameters;
- canonical training inputs are not mutated.

The numerical-conditioning correction is authoritative:

`MAF_SPLINE_PARAMETERIZATION=SLOPE_CONDITIONED`

`MAF_LOG_SLOPE_LIMIT=0.75`

The implementation is therefore accepted for revised Paper 1 experiment
execution.

Training authorization applies only to the revision-2026 implementation and
does not authorize reuse of the historical MAF implementation.

Historical artifacts remain provenance-only.

This authorization means that real dataset training may begin under the
governed revision protocol. It does not mean that any four-dataset benchmark
run has already occurred or been accepted.

`MAF_IMPLEMENTATION_ACCEPTANCE=PASS`

`MAF_REAL_DATASET_TRAINING_MAY_BEGIN=YES`
