# Paper 1 Revision 2026 — Gaussian Mixture Optimizer Policy

`STATUS=IMPLEMENTED_AND_FOCUSED_VALIDATED`

`GMM_OPTIMIZER_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

The governed optimizer implementation is now present in:

`src/trustforge/models/paper01_revision_2026/gaussian_mixture_optimizer.py`

and its focused validation suite passed 16 tests.

The implementation validates:

- exact SGD with learning rate `0.001`;
- zero momentum;
- no gradient clipping;
- governed float64 parameter and gradient arithmetic;
- acceptance of balanced production batches up to size 128;
- finite loss and gradient requirements;
- candidate-state validation before mutation;
- fail-closed handling of invalid or non-finite inputs;
- no latent-mean projection into `[0,1]`;
- finite normalized mixture weights and transformed scales after update.

This status advancement applies only to the optimizer primitive.

Overall Gaussian-mixture training remains unauthorized until the complete
training loop, validation/checkpoint selection integration, checkpoint
persistence, and sampling implementation are completed and validated.

## 1. Scope

This policy governs the primary numerical optimizer for the revised Paper 1
class-conditional censored diagonal Gaussian mixture model.

It does not authorize real benchmark training.

Epoch construction, training-data traversal, validation evaluation,
convergence criteria, checkpoint selection, and final implementation
acceptance remain separately governed.

## 2. Optimization objective

The revised GMM optimizes the class-conditional censored mixture negative
log likelihood.

For each training class, optimization acts on:

- five mixture logits;
- five 4800-dimensional latent Gaussian mean vectors;
- five 4800-dimensional raw-scale vectors.

The observation likelihood incorporates:

- lower-endpoint probability mass at 0;
- Gaussian interior density on `(0,1)`;
- upper-endpoint probability mass at 1.

The optimizer must therefore act on the censored likelihood itself.

It must not optimize an ordinary unconstrained Gaussian-mixture likelihood
and censor only during generation.

## 3. Candidate optimizer evidence

The revision evaluated:

- Adam with learning rate `1e-3`;
- Adam with learning rate `3e-4`;
- SGD with learning rate `1e-3`;
- SGD with learning rate `3e-4`;
- SGD with learning rate `1e-4`.

All tested candidates produced finite losses, finite gradients, finite
parameters, and valid mixture weights under the audited synthetic
revision-scale workloads.

### 3.1 Twenty-step synthetic comparison

At the governed revision dimensionality:

`K=5`

`D=4800`

`batch_size=128`

the initial full synthetic NLL was approximately:

`7338.06`

After 20 updates:

- Adam `1e-3`: approximately `7064.18`;
- Adam `3e-4`: approximately `7253.48`;
- SGD `1e-3`: approximately `6461.91`.

SGD `1e-3` produced the strongest early reduction while remaining finite.

## 4. Longer-horizon synthetic evidence

A 100-step audit compared:

- Adam `1e-3`;
- SGD `1e-3`;
- SGD `3e-4`;
- SGD `1e-4`.

Every candidate decreased the full synthetic evaluation NLL at every
10-step evaluation point.

Final NLL values were approximately:

- Adam `1e-3`: `6140.10`;
- SGD `1e-3`: `5011.39`;
- SGD `3e-4`: `6122.86`;
- SGD `1e-4`: `6941.90`.

SGD `1e-3` again produced the strongest reduction.

No evaluation-interval increase was observed for SGD `1e-3`.

## 5. Real-data initial-gradient evidence

Initial censored-likelihood gradients were audited using representative
revised canonical classes:

- USTC class 0;
- CICMalDroid class 0;
- MalNet-Image class 12;
- MalNet-Image class 4;
- MaleX class 0;
- MaleX class 1.

All gradients were finite.

Observed total initial gradient norms ranged from approximately:

`126`

to:

`911`

depending on class and dataset.

The variation was dominated by the 24000 latent-mean parameters.

The maximum observed individual mean-gradient magnitude was approximately:

`22.29`.

At learning rate `1e-3`, this corresponds to a maximum audited
single-coordinate first-order update magnitude of approximately:

`0.0223`.

No evidence from the initial-gradient audit required gradient clipping.

## 6. Real-data short-horizon evidence

SGD with learning rate `1e-3`, batch size 128, and no gradient clipping was
audited for 20 updates on representative real revised canonical classes.

### USTC class 0

Evaluation NLL decreased monotonically from approximately:

`2764.08`

to:

`795.23`.

### CICMalDroid class 0

Evaluation NLL decreased monotonically from approximately:

`-7315.52`

to:

`-7549.58`.

### MalNet-Image class 12

This is the smallest revised MalNet training class.

Evaluation NLL decreased monotonically from approximately:

`-1798.87`

to:

`-3683.36`.

### MalNet-Image class 4

This is the largest revised MalNet training class.

Evaluation NLL decreased monotonically from approximately:

`334.04`

to:

`-1668.74`.

### MaleX class 0

Evaluation NLL decreased monotonically from approximately:

`-2065.22`

to:

`-2312.81`.

### MaleX class 1

Evaluation NLL decreased monotonically from approximately:

`-2278.27`

to:

`-2516.43`.

Across all six real-data audits:

- no evaluation-interval increase occurred;
- mixture weights remained finite;
- component scales remained finite;
- the governed minimum scale remained satisfied;
- latent means remained finite;
- no gradient clipping was required for numerical stability.

## 7. Selected optimizer

The selected primary optimizer is:

`SGD`

with:

`GMM_LEARNING_RATE=0.001`

No momentum is selected at this stage.

No gradient clipping is selected.

`GMM_OPTIMIZER=SGD`

`GMM_SGD_MOMENTUM=0`

`GMM_GRADIENT_CLIPPING=NO`

The optimizer acts directly on the float64 trainable parameters governed by
the numerical-precision policy.

## 8. Latent-mean semantics

The Gaussian component means are latent Euclidean parameters.

They are not canonical image pixels.

Therefore component means are not constrained to `[0,1]`.

A latent mean below 0 or above 1 is valid under the selected censored
observation model.

The bounded observation semantics are provided by censoring:

- latent values `<=0` map to observed 0;
- latent values in `(0,1)` remain interior values;
- latent values `>=1` map to observed 1.

The implementation must therefore not clip or project latent means into
`[0,1]`.

`GMM_LATENT_MEAN_BOUNDED_TO_UNIT_INTERVAL=NO`

## 9. Batch-size evidence and production interpretation

All primary optimizer prototypes used:

`GMM_OPTIMIZER_PROTOTYPE_BATCH_SIZE=128`

at:

- `K=5`;
- `D=4800`;
- float64 internal likelihood arithmetic.

This prototype batch size remained numerically stable in the audited
workloads.

Production traversal is governed separately by the balanced-batch traversal
policy.

The production contract therefore uses:

`GMM_TRAIN_BATCH_SIZE_MAX=128`

rather than requiring every optimizer update to contain exactly 128 examples.

For a class with training count `N`, the traversal policy defines:

`B = ceil(N / 128)`

and partitions each seeded class permutation into `B` balanced batches whose
sizes differ by at most one.

Therefore the optimizer must accept governed batch sizes:

`1 <= batch_size <= 128`

without changing its learning rate, momentum, precision, or gradient policy.

The earlier optimizer evidence at exact batch size 128 remains prototype
stability evidence; it is not a requirement that every production batch have
exactly 128 examples.

`GMM_OPTIMIZER_BATCH_GEOMETRY_SOURCE=TRAVERSAL_POLICY`

## 10. Failure policy

Training must fail closed on:

- non-finite loss;
- missing gradient;
- non-finite gradient;
- non-finite parameter;
- non-finite transformed scale;
- mixture-weight normalization failure.

No numerical failure may trigger silent:

- clipping;
- NaN replacement;
- optimizer substitution;
- learning-rate substitution;
- component deletion;
- global-model fallback.

## 11. Governed optimizer constants

`GMM_OPTIMIZER=SGD`

`GMM_LEARNING_RATE=0.001`

`GMM_SGD_MOMENTUM=0`

`GMM_OPTIMIZER_PROTOTYPE_BATCH_SIZE=128`

`GMM_TRAIN_BATCH_SIZE_MAX=128`

`GMM_OPTIMIZER_BATCH_GEOMETRY_SOURCE=TRAVERSAL_POLICY`

`GMM_GRADIENT_CLIPPING=NO`

`GMM_LATENT_MEAN_BOUNDED_TO_UNIT_INTERVAL=NO`

`GMM_OPTIMIZER_POLICY_SELECTED=YES`

## 12. Authorization state

The optimizer policy is selected.

Training-data traversal, epoch semantics, validation-based stopping,
checkpoint selection, and production implementation remain unaccepted.

Therefore:

`GMM_OPTIMIZER_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_OPTIMIZER_FOCUSED_VALIDATION=PASSED`

`GMM_OPTIMIZER_FOCUSED_TEST_COUNT=16`

`GMM_IMPLEMENTATION_AUTHORIZED=PARTIAL_INITIALIZATION_TRAVERSAL_BUDGET_AND_OPTIMIZER_ONLY`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
