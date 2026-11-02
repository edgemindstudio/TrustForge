# Paper 1 Revision 2026 — Restricted Boltzmann Optimizer Policy

`STATUS=SELECTED`

`RBM_OPTIMIZER_POLICY_SELECTED=YES`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs parameter updates for the revised Paper 1 Restricted
Boltzmann Machine.

It selects:

- optimizer family;
- primary learning rate;
- weight-decay behavior;
- update direction semantics.

It does not yet select:

- class-balanced training traversal;
- epoch or update budget;
- validation/checkpoint rule;
- synthesis-chain length.

## 2. Historical optimizer evidence

Historical CD training used direct manual SGD-style updates.

The historical implementation applied:

`parameter <- parameter + learning_rate * CD_statistic`

for:

- visible-hidden weights;
- visible biases;
- hidden biases.

When enabled, historical weight decay modified only the weight statistic.

Adam was used only by the separate historical MSE training mode.

Therefore:

`RBM_HISTORICAL_CD_OPTIMIZER=MANUAL_SGD_STYLE`

`RBM_HISTORICAL_ADAM_CD_PATH=NO`

Historical source defaults included:

`learning_rate=1e-3`

and:

`weight_decay=0`

while historical runtime configuration could override the learning rate.

Historical intended nested RBM configuration specified:

`LR=5e-4`

but that historical value is not automatically authoritative.

## 3. Revised optimizer family

The revised primary optimizer is direct SGD-style application of the governed
contrastive-divergence statistics.

No momentum accumulator is used.

No Adam first- or second-moment state is used.

No RMSProp accumulator is used.

Therefore:

`RBM_OPTIMIZER=MANUAL_SGD`

This preserves a direct relationship between the governed CD estimator and the
parameter update.

## 4. Learning-rate candidates audited

The revision evaluated one-update scales:

- `1e-4`;
- `2e-4`;
- `5e-4`;
- `1e-3`;
- `2e-3`.

The audit used:

- revised bounded-exponential visible conditionals;
- CD-1;
- batch size 256;
- float64 internal arithmetic;
- selected initialization;
- zero weight decay.

All candidate updates remained finite in the one-step audit.

## 5. One-update scale evidence

At `5e-4`, representative weight-step norm relative to current weight norm was
approximately:

- USTC: `0.00160`;
- CICMalDroid2020: `0.00080`;
- MalNet-Image: `0.00250`;
- MaleX: `0.00076`.

At `1e-3`, these values approximately doubled.

At `2e-3`, MalNet-Image reached approximately:

`W_step / W_norm = 0.0100`

in a single initial update.

The revision therefore rejected `2e-3` as unnecessarily aggressive for the
primary benchmark.

## 6. Short-horizon stability audit

A 50-update fixed-batch numerical stress audit compared:

- `5e-4`;
- `1e-3`.

Both learning rates:

- kept all model state finite;
- avoided hidden-probability saturation below `0.01`;
- avoided hidden-probability saturation above `0.99`.

However, `1e-3` produced materially stronger parameter and hidden-logit drift.

Representative MalNet-Image results after 50 updates were:

### learning rate 5e-4

- weight-norm ratio: approximately `1.0106`;
- maximum hidden-logit magnitude: approximately `1.93`.

### learning rate 1e-3

- weight-norm ratio: approximately `1.0562`;
- maximum hidden-logit magnitude: approximately `3.86`.

USTC showed the same direction of effect, with greater weight growth and
hidden-logit expansion under `1e-3`.

## 7. Selected learning rate

The revised primary learning rate is:

`5e-4`

Therefore:

`RBM_LEARNING_RATE=5e-4`

This value is selected because it:

1. produces finite governed updates across all four datasets;
2. preserves conservative initial parameter movement;
3. remains stable across the short-horizon stress audit;
4. produces less parameter drift than `1e-3`;
5. avoids the unnecessarily aggressive scale observed at `2e-3`;
6. requires no adaptive optimizer state.

The fact that `5e-4` also matches the historical intended RBM learning rate is
incidental corroboration rather than the governing justification.

`RBM_LEARNING_RATE_SELECTION_BASIS=REVISION_STABILITY_EVIDENCE`

## 8. Weight decay

The revised primary RBM uses:

`weight_decay=0`

Therefore:

`RBM_WEIGHT_DECAY=0`

No revision-specific evidence currently establishes that explicit L2-style
weight shrinkage is required.

Adding weight decay would introduce another model-specific regularization
hyperparameter without demonstrated benchmark necessity.

`RBM_WEIGHT_DECAY_PRIMARY=DISABLED`

## 9. Update equations

Let the governed positive-minus-negative CD statistics be:

- `dW`;
- `da`;
- `db`;
- `dU`.

The primary update is:

`W <- W + lr * dW`

`a <- a + lr * da`

`b <- b + lr * db`

`U <- U + lr * dU`

with:

`lr = 5e-4`.

No momentum term is added.

No adaptive per-parameter rescaling is added.

## 10. Gradient clipping

Gradient or update clipping is not selected for the primary implementation.

The current audits did not produce non-finite or explosive update statistics at
the selected learning rate.

Therefore:

`RBM_UPDATE_CLIPPING=DISABLED`

If future implementation evidence demonstrates a numerical instability,
training must fail closed and the policy must be reconsidered rather than
silently introducing clipping.

## 11. Learning-rate schedule

The primary revision uses a constant learning rate.

No warmup, cosine decay, step decay, or adaptive schedule is selected.

Therefore:

`RBM_LEARNING_RATE_SCHEDULE=CONSTANT`

This keeps optimizer behavior uniform and auditable across datasets and seeds.

## 12. Optimizer-state checkpointing

Because manual SGD has no momentum or adaptive accumulator state, authoritative
optimizer state consists only of:

- governed learning-rate configuration;
- update count / training position required by execution governance.

There is no hidden Adam- or momentum-style optimizer tensor state.

`RBM_ADAPTIVE_OPTIMIZER_STATE=ABSENT`

## 13. Numerical requirements

Every update must:

- operate in float64;
- receive finite CD statistics;
- produce finite updated `W`;
- produce finite updated `a`;
- produce finite updated `b`;
- produce finite updated `U`.

Any non-finite update must fail closed.

No automatic learning-rate reduction may silently recover from a failed update.

## 14. Reproducibility

The optimizer itself introduces no additional randomness.

All stochasticity comes from governed:

- training traversal;
- hidden-state sampling;
- visible-state sampling.

For identical governed execution identity and stochastic streams, the optimizer
must produce identical parameter updates.

## 15. Implementation gate

Before optimizer implementation may be accepted, focused evidence must
demonstrate at least:

1. optimizer family is manual SGD;
2. learning rate is exactly `5e-4`;
3. learning rate is constant;
4. weight decay is exactly zero;
5. no Adam optimizer is instantiated;
6. no momentum state exists;
7. no RMSProp state exists;
8. no update clipping path exists;
9. `W`, `a`, `b`, and `U` all receive governed CD updates;
10. all update arithmetic is float64;
11. non-finite update statistics fail closed;
12. historical generic `LR` fallback cannot override the revision policy.

## 16. Current authorization state

The revised RBM optimizer is selected as:

- manual SGD;
- constant learning rate `5e-4`;
- zero weight decay;
- no momentum;
- no adaptive optimizer;
- no clipping.

Therefore:

`RBM_OPTIMIZER_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
