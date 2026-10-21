# Paper 1 Revision 2026 — Candidate Block-Conditional MAF Architecture

## Status

`ARCHITECTURE_STATUS=SELECTED`

`MASKED_AUTOFLOW_TRAINING_AUTHORIZED=YES`

This document specifies the candidate structural replacement for the historical
Paper 1 Masked AutoFlow implementation.

It does not yet authorize training.

The remaining observation-domain and exact-boundary treatment must be governed
separately before implementation authorization.

## 1. Family identity

The replacement remains within the Masked Autoregressive Flow family.

Each flow stage is an autoregressive affine transformation with a triangular
Jacobian.

The revision changes the autoregressive unit from one scalar canonical
dimension to one governed spatial block of canonical dimensions.

This is therefore a block-autoregressive MAF rather than an inverse
autoregressive flow, coupling flow, RealNVP model, or other normalizing-flow
family.

## 2. Canonical input

Model-facing benchmark representation:

- height: 40;
- width: 40;
- channels: 3;
- flattened dimensionality: 4800;
- dtype: float32;
- canonical range: [0,1].

The flow may use a separately governed internal continuous representation.

## 3. Spatial block partition

Each canonical image is divided into non-overlapping spatial blocks:

`8 x 8 x 3`

Each block therefore contains:

`8 x 8 x 3 = 192`

scalar dimensions.

Because:

`40 / 8 = 5`

there are:

`5 x 5 = 25`

spatial blocks per image.

The 25 blocks form the autoregressive units.

No canonical dimension is dropped or duplicated.

## 4. Block-autoregressive factorization

Let the ordered blocks be:

`B_1, B_2, ..., B_25`

For one flow stage, the affine parameters for block `B_i` may depend on:

- class condition `y`;
- blocks `B_1 ... B_(i-1)`.

They must not depend on:

- the current block `B_i`;
- any later block `B_(i+1) ... B_25`.

For each block, the flow predicts 192 location parameters and 192 log-scale
parameters.

All 192 scalar dimensions inside the current block are transformed in
parallel once their parameters have been obtained.

This yields a block-lower-triangular Jacobian.

The log determinant is therefore exactly the sum of the governed per-dimension
log scales.

## 5. Mask semantics

The mask construction must distinguish hidden-layer and output-layer
connectivity.

Hidden-layer connectivity may use inclusive autoregressive degree relations
where mathematically valid.

Output parameters for autoregressive block `i` must use a strict relation that
excludes the current block.

Regression tests must prove that changing block `i` while holding all earlier
blocks fixed cannot change the predicted affine parameters for block `i`.

The historical inclusive-output-mask defect must not be reproduced.

## 6. Class conditioning

The model is explicitly conditional.

Class identity is an external conditioning variable and is not part of the
autoregressive ordering.

The class condition may be represented by a learned embedding and injected
into masked hidden representations and/or affine-parameter heads.

Required semantics:

- class labels participate during training;
- class labels participate during likelihood evaluation;
- requested class labels participate during synthesis;
- changing only the class condition must be capable of changing generated
  distribution parameters;
- samples are emitted with the same class condition used to generate them.

Post-hoc assignment of labels to unconditional samples is forbidden.

## 7. Flow stack

Candidate initial configuration:

- number of flow stages: 5;
- autoregressive blocks per stage: 25;
- hidden width: 256;
- hidden depth: 2;
- class embedding dimension: 64.

These values are implementation candidates, not yet benchmark hyperparameter
locks.

Between flow stages, deterministic seeded block permutations may be used to
increase dependency coverage.

Permutations must operate on whole 192-dimensional blocks so that the governed
block structure remains explicit.

Permutation construction must be reproducible from the experimental seed or a
separately governed architecture seed.

## 8. Inverse complexity

Historical scalar-order MAF:

`4800 dimensions x 5 flows = 24000`

sequential MADE evaluations per inverse batch.

Candidate block-MAF:

`25 blocks x 5 flows = 125`

sequential block-MADE evaluations per inverse batch.

Structural reduction:

`24000 / 125 = 192`

The candidate therefore reduces the sequential inverse depth by a factor of:

`192x`

without replacing MAF with a different normalizing-flow family.

## 9. Forward and inverse semantics

Forward density evaluation may compute all affine parameters for one flow
stage in parallel under the governed block mask.

Inverse synthesis proceeds sequentially over 25 blocks for each flow stage.

Within one block, all 192 scalar values are recovered in parallel.

Across five flow stages, the maximum governed sequential inverse depth is:

`125 block stages`

excluding fixed permutations and bounded-domain transforms.

## 10. Conditional base distribution

The initial candidate uses a standard diagonal normal latent base.

Class dependence is represented through the conditional autoregressive
transforms rather than through post-hoc labels.

A class-dependent base distribution is not required unless later evidence
shows that it is needed.

## 11. Observation-domain issue remains open

The canonical observations include exact zero, exact one, quantized values,
and legitimate continuous interior values.

A continuous normalizing flow with an ordinary bounded bijection cannot assign
ordinary continuous density directly to endpoint atoms without an explicit
policy.

Therefore this architecture document does not choose:

- universal dequantization;
- endpoint epsilon clipping;
- boundary jitter;
- logit clipping;
- post-generation clipping.

The observation-domain transform must be governed separately before model
implementation is authorized.

Any selected transform must define:

1. training-time treatment of exact zero;
2. training-time treatment of exact one;
3. treatment of continuous interior values;
4. inverse-generation semantics;
5. whether generated values can reach exact endpoints;
6. whether any stochastic preprocessing is used;
7. whether its randomness is governed by the experimental seed;
8. what likelihood is actually optimized;
9. why generated samples remain valid canonical benchmark inputs.

## 12. Required structural regression tests

Before training authorization, tests must demonstrate at minimum:

1. canonical input shape 40 x 40 x 3 maps to exactly 25 governed blocks;
2. each block contains exactly 192 scalar dimensions;
3. flatten/block/unblock operations are exactly invertible;
4. no canonical dimension is lost or duplicated;
5. affine output shape is correct;
6. positive finite scales;
7. output block `i` has no dependency on current block `i`;
8. output block `i` has no dependency on future blocks;
9. earlier blocks can influence later-block parameters;
10. class condition changes conditional parameters;
11. forward and inverse agree numerically on finite interior test data;
12. log determinants are finite;
13. five-flow inverse uses at most 125 sequential autoregressive block stages;
14. fixed-seed structural masks and block permutations are reproducible;
15. changed architecture seed changes governed permutations where expected;
16. finite loss and finite gradients occur in a real training step;
17. generated tensors have canonical 40 x 40 x 3 shape;
18. final governed observation transform keeps generated values within [0,1].

## 13. Authorization checkpoint

Architecture selection alone did not authorize experimentation at this
checkpoint.

Decision state at the architecture-selection checkpoint:

`CHECKPOINT_BLOCK_CONDITIONAL_MAF_ARCHITECTURE_SELECTED_FOR_EVALUATION=YES`

`CHECKPOINT_MAF_OBSERVATION_DOMAIN_POLICY_COMPLETE=NO`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`

These checkpoint markers preserve the state before later observation-domain
and transform decisions.

## 14. Spline-parameter decoder scale analysis

A naive per-scalar eight-bin spline head would require:

- 16 predicted values per canonical scalar;
- 76,800 outputs per flow;
- 19,737,600 output-head parameters per flow at hidden width 256;
- 98,688,000 output-head parameters across five flows;
- approximately 376.46 MiB of float32 head weights alone.

This is rejected as unnecessarily large.

Sharing one spline across an entire 192-dimensional block would reduce
parameter count substantially, but would also force all scalar positions
inside that block to use the same monotonic transform.

Sharing only by RGB channel would still impose substantial within-block
parameter tying.

Neither form of coarse spline sharing is selected as the primary candidate.

## 15. Shared per-block scalar-spline decoder

The selected candidate parameterization uses:

- one hidden representation per autoregressive block;
- one decoder network shared across all 25 block positions;
- distinct spline parameters for every one of the 192 scalar dimensions
  inside the current block.

For eight-bin piecewise-linear sizing, the decoder predicts:

`192 x 16 = 3072`

values for one block representation.

The same decoder weights are reused for every block.

This preserves distinct per-scalar transforms while preventing decoder
parameter count from scaling by 25 separate block heads.

## 16. Decoder bottleneck candidate

The current decoder candidate is:

`256 -> 64 -> 3072`

where:

- 256 is the block hidden representation width;
- 64 is a shared decoder bottleneck;
- 3072 is the eight-bin piecewise-linear sizing target for one
  192-dimensional block.

Observed sizing:

- parameters per flow: 216,128;
- parameters across five flows: 1,080,640;
- float32 decoder-weight memory across five flows: approximately 4.12 MiB.

This is substantially smaller than both:

- the historical-style global affine head; and
- the naive global per-scalar spline head.

The 64-unit bottleneck is therefore selected as the current implementation
candidate.

It remains a benchmark hyperparameter candidate rather than a final empirical
tuning result.

`MAF_SHARED_BLOCK_DECODER_BOTTLENECK=64`

## 17. Activation-memory consequence

Decoder parameter sharing reduces model weights but does not remove the need
to materialize per-scalar spline parameters during execution.

For the eight-bin sizing prototype, full spline-parameter activation storage
is approximately:

- batch 64: 18.75 MiB;
- batch 128: 37.50 MiB;
- batch 256: 75.00 MiB.

These values describe raw spline-parameter activations only and do not include:

- masked-network hidden activations;
- gradients;
- optimizer state;
- additional flow-stage tensors;
- TensorFlow runtime overhead.

Therefore the historical batch size of 256 is not inherited automatically.

Revision training batch size must be selected after an implementation-level
memory and throughput smoke test.

`MAF_TRAINING_BATCH_SIZE_LOCKED=NO`

## 18. Preserved semantics

The shared decoder does not alter the governed autoregressive structure.

For block `i`:

1. the masked conditional backbone computes a representation using only
   earlier blocks and class condition;
2. the shared decoder maps that representation to scalar-specific transform
   parameters for block `i`;
3. all 192 scalar transforms inside block `i` are applied in parallel.

The decoder is shared across block positions, but its input representation is
block-specific.

Therefore different blocks may receive different spline parameters even
though the decoder weights are shared.

Required regression tests must verify this property explicitly.

## 19. Updated architecture state

Current structural candidate:

- canonical shape: 40 x 40 x 3;
- 25 spatial blocks;
- 192 scalar dimensions per block;
- five block-autoregressive flow stages;
- hidden block representation width: 256;
- shared decoder bottleneck: 64;
- scalar-specific bounded monotonic transforms;
- class conditioning;
- maximum sequential inverse depth: 125 block stages.

Still unresolved:

- final spline parameterization;
- final spline bin count;
- final hidden depth;
- final training batch size;
- final optimization hyperparameters.

Decision state at the shared-decoder checkpoint:

`CHECKPOINT_BLOCK_CONDITIONAL_MAF_ARCHITECTURE_SELECTED_FOR_EVALUATION=YES`

`CHECKPOINT_MAF_SHARED_BLOCK_DECODER_SELECTED_FOR_EVALUATION=YES`

`CHECKPOINT_FINAL_SPLINE_PARAMETERIZATION_SELECTED=NO`

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`

These checkpoint markers preserve the state immediately before final
bounded-transform selection.

## 20. Final bounded-transform selection

The final implementation candidate uses:

`8-bin bounded monotonic piecewise-linear splines`

for each scalar canonical dimension.

Each scalar requires:

- eight raw bin-width parameters;
- eight raw bin-height parameters.

Therefore each scalar uses:

`16 transform parameters`

and each 192-dimensional block requires:

`192 x 16 = 3072`

transform outputs.

The governed shared decoder remains:

`256 -> 64 -> 3072`

The transform acts directly on the closed canonical domain `[0,1]`.

No endpoint epsilon, universal dequantization, or post-generation clipping is
part of the revision implementation.

Current structural decision state:

`FINAL_SPLINE_PARAMETERIZATION_SELECTED=YES`

`MAF_SPLINE_TYPE=PIECEWISE_LINEAR`

`MAF_SPLINE_BINS=8`

`MAF_SHARED_BLOCK_DECODER_BOTTLENECK=64`

`MAF_OBSERVATION_DOMAIN_POLICY_COMPLETE=YES`

Implementation and training authorization remain separate gates.

`CHECKPOINT_MASKED_AUTOFLOW_TRAINING_AUTHORIZED=NO`
