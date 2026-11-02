# Paper 1 Revision 2026 — Restricted Boltzmann Architecture Policy

`STATUS=SELECTED`

`RBM_ARCHITECTURE_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs the structural architecture of the revised Paper 1
Restricted Boltzmann Machine.

It selects:

- visible dimensionality;
- hidden-unit count;
- shared versus dataset-specific architecture;
- class-conditioning parameter structure.

It does not yet select:

- numerical precision;
- initialization;
- optimizer;
- contrastive-divergence depth;
- training traversal;
- training budget;
- checkpoint policy;
- production Gibbs-chain length.

## 2. Governed visible dimensionality

The revised canonical representation is:

`40 x 40 x 3`

Therefore:

`D = 4800`

Every revised RBM consumes all 4800 canonical scalar values.

No dimensionality reduction, grayscale collapse, thresholding, or dataset-specific
visible width is authorized.

`RBM_VISIBLE_DIMENSION=4800`

## 3. Historical hidden-width evidence

Historical Paper 1 configuration intended:

`H=512`

for the RBM.

However, recovered historical execution evidence showed that at least the
recovered USTC seed-42 artifact-producing execution instantiated:

`H=256`

because the trainer failed to consume the intended nested RBM configuration.

Therefore neither historical 512 nor historical-runtime 256 is automatically
authoritative for the revision.

The revision architecture is selected from current governed benchmark evidence.

## 4. Candidate widths audited

The revision evaluated the following fixed hidden-width candidates:

- `H=128`;
- `H=256`;
- `H=512`.

The shared conditional architecture contains:

- visible-hidden matrix `W` with shape `(4800,H)`;
- visible bias `a` with shape `(4800,)`;
- hidden bias `b` with shape `(H,)`;
- class-to-hidden interaction matrix `U` with shape `(K,H)`.

For a dataset containing `K` classes:

`parameter_count = 4800*H + 4800 + H + K*H`

## 5. Dataset-scale parameter evidence

### H = 128

Observed total parameters:

- USTC: `620480`;
- CICMalDroid2020: `619968`;
- MalNet-Image: `625344`;
- MaleX: `619584`.

Observed parameters per training example:

- USTC: approximately `8.5114`;
- CICMalDroid2020: approximately `76.3790`;
- MalNet-Image: approximately `0.7079`;
- MaleX: approximately `0.8475`.

Model-state size is approximately:

`2.36–2.39 MiB`

in float32, excluding optimizer state and runtime activations.

### H = 256

Observed parameters per training example:

- USTC: approximately `16.9569`;
- CICMalDroid2020: approximately `152.1666`;
- MalNet-Image: approximately `1.4103`;
- MaleX: approximately `1.6885`.

Model-state size is approximately:

`4.71–4.75 MiB`

in float32.

### H = 512

Observed parameters per training example:

- USTC: approximately `33.8480`;
- CICMalDroid2020: approximately `303.7418`;
- MalNet-Image: approximately `2.8152`;
- MaleX: approximately `3.3704`.

Model-state size is approximately:

`9.40–9.49 MiB`

in float32.

## 6. Dominant compute scaling

The dominant visible-hidden dense interaction contains:

`D * H`

weight terms per matrix direction.

Therefore:

- `H=128`: `614400`;
- `H=256`: `1228800`;
- `H=512`: `2457600`.

Increasing from 128 to 256 approximately doubles this dominant interaction
cost.

Increasing from 128 to 512 approximately quadruples it.

This cost multiplies across:

- positive-phase hidden evaluation;
- negative-phase Gibbs transitions;
- minibatches;
- training updates;
- validation;
- seven experimental seeds;
- four datasets.

## 7. Selected hidden width

The authoritative revised hidden width is:

`H=128`

Therefore:

`RBM_HIDDEN_UNITS=128`

The selection is benchmark-wide.

No dataset-specific hidden width is authorized.

## 8. Selection rationale

`H=128` is selected because it:

1. preserves a nontrivial latent binary representation;
2. maintains the selected RBM family identity;
3. substantially reduces parameter count relative to 256 or 512 hidden units;
4. substantially reduces dominant Gibbs/training matrix cost;
5. avoids worsening the already highest parameter-to-example pressure observed
   on CICMalDroid2020;
6. remains easily scalable for MalNet-Image and MaleX;
7. avoids dataset-specific architecture tuning;
8. avoids selecting architecture using downstream benchmark test performance;
9. provides one uniform governed architecture across all four datasets.

No current revision evidence demonstrates that 256 or 512 hidden units are
required to make the RBM scientifically usable for Paper 1.

Therefore additional width is not justified merely by historical configuration
precedent.

## 9. Uniformity rule

All four revised datasets must use:

`H=128`

for the primary Paper 1 RBM benchmark.

The benchmark must not silently use:

- 128 for one dataset;
- 256 for another;
- 512 for another

based on dataset size or observed downstream performance.

Any future architecture-scaling study is outside the primary Paper 1 benchmark
unless separately governed.

`RBM_DATASET_SPECIFIC_HIDDEN_WIDTH=FORBIDDEN`

## 10. Class-conditioning architecture

For dataset class count `K`, the shared conditional state is:

`W: (4800,128)`

`a: (4800,)`

`b: (128,)`

`U: (K,128)`

Class identity enters through:

`U[y,:]`

in the hidden conditional.

The visible conditional remains the selected bounded-exponential conditional.

## 11. Parameter counts under selected architecture

At `H=128`, the shared unconditional core contains:

`619328`

parameters.

Class-conditioning overhead is:

`K * 128`

parameters.

Therefore authoritative totals are:

- USTC (`K=9`): `620480`;
- CICMalDroid2020 (`K=5`): `619968`;
- MalNet-Image (`K=47`): `625344`;
- MaleX (`K=2`): `619584`.

These counts exclude checkpoint metadata and optimizer state.

## 12. Architectural invariants

The implementation must preserve:

- one visible layer of width 4800;
- one hidden layer of width 128;
- one shared visible-hidden matrix;
- one shared visible bias;
- one shared hidden bias;
- one class-to-hidden interaction row per governed class;
- no hidden multilayer stack;
- no convolutional replacement;
- no dataset-specific width changes.

Adding additional hidden layers or replacing the RBM with a deep belief network
or other multilayer energy model would constitute a different generator
architecture and requires separate governance.

## 13. No post-hoc capacity tuning

The primary benchmark must not increase or decrease hidden width after observing
test-set benchmark metrics.

Architecture selection occurs before revised production training.

Any future sensitivity study over hidden width must:

- be explicitly labeled as an architecture sensitivity analysis;
- remain separate from the locked primary benchmark result;
- avoid rewriting the primary architecture identity.

`RBM_PRIMARY_WIDTH_POSTHOC_TUNING=FORBIDDEN`

## 14. Implementation gate

Before architecture implementation may be accepted, focused evidence must
demonstrate at least:

1. visible dimension equals 4800;
2. hidden dimension equals 128;
3. `W` shape equals `(4800,128)`;
4. `a` shape equals `(4800,)`;
5. `b` shape equals `(128,)`;
6. `U` shape equals `(K,128)` for the governed dataset;
7. all four datasets instantiate the same hidden width;
8. no dataset-specific hidden-width override exists;
9. canonical `(B,40,40,3)` inputs flatten reversibly to `(B,4800)`;
10. generated `(B,4800)` states reshape exactly to `(B,40,40,3)`;
11. architecture state remains finite after initialization;
12. class conditioning remains active through `U`.

## 15. Current authorization state

The revised RBM architecture is selected as:

`D=4800`

`H=128`

with shared class-to-hidden conditioning.

Architecture implementation remains unaccepted.

Real benchmark training remains unauthorized.

Therefore:

`RBM_ARCHITECTURE_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
