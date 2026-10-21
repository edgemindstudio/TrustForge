# Paper 1 Revision 2026 — Gaussian Mixture Numerical Precision Policy

`STATUS=SELECTED`

`GMM_PRECISION_POLICY_SELECTED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`

## 1. Scope

This policy governs numerical precision for the revised Paper 1
class-conditional censored diagonal Gaussian mixture model.

It does not authorize real dataset training.

Optimizer choice, minibatch policy, convergence criteria, checkpointing,
and final implementation acceptance remain separately governed.

## 2. Canonical input representation

The governed Paper 1 canonical representation remains:

`40 x 40 x 3`

`float32`

`[0,1]`

The GMM does not alter the benchmark-wide canonical representation.

Canonical dataset adapters therefore continue to emit float32 images.

## 3. Internal GMM training precision

The revised GMM must cast canonical training batches from float32 to
float64 before censored-likelihood evaluation.

The following quantities use float64 internally:

- mixture logits;
- component means;
- raw scale parameters;
- transformed component scales;
- Gaussian standardized values;
- Gaussian log densities;
- endpoint log-CDF probabilities;
- endpoint log-survival probabilities;
- component log likelihoods;
- mixture `logsumexp`;
- negative log likelihood;
- gradients;
- optimizer state where supported.

`GMM_INTERNAL_DTYPE=FLOAT64`

`GMM_PARAMETER_DTYPE=FLOAT64`

`GMM_LIKELIHOOD_DTYPE=FLOAT64`

## 4. Tail-precision evidence

The branch-safe differentiable Gaussian log-CDF implementation was audited
against `scipy.special.log_ndtr`.

Across standardized values from -100 through +100:

### float32

- all values finite;
- all gradients finite;
- maximum absolute error versus SciPy:
  approximately `2.82913e-4`;
- mean absolute error:
  approximately `2.68301e-5`.

### float64

- all values finite;
- all gradients finite;
- maximum absolute error versus SciPy:
  approximately `1.42891e-8`;
- mean absolute error:
  approximately `7.97574e-10`.

Both precisions were numerically functional.

Float64 provided substantially stronger agreement in the extreme Gaussian
tails used directly by the censored observation model.

## 5. Revision-scale likelihood evidence

A revision-scale audit used:

- `D=4800`;
- `K=5`;
- endpoint-heavy bounded observations;
- the governed minimum scale;
- censored diagonal mixture likelihood.

Float32 and float64 both produced finite sample log likelihoods.

Observed differences were:

- maximum absolute log-likelihood difference:
  approximately `0.00568`;
- mean absolute difference:
  approximately `0.00358`;
- maximum relative difference:
  approximately `8.08e-8`.

This establishes that float32 is broadly numerically coherent, but does not
remove the stronger tail-accuracy advantage of float64.

## 6. Selection rationale

The censored GMM directly models endpoint probability through Gaussian tail
probabilities.

Those probabilities are not auxiliary diagnostics; they are part of the
training objective.

Because:

1. endpoint mass is substantial in USTC and MalNet-Image;
2. the likelihood sums across 4800 dimensions;
3. stable tail evaluation is scientifically central to the selected
   observation model;
4. the GMM contains relatively few trainable parameters;
5. minibatch training bounds resident observation memory;

the revision selects float64 for internal GMM optimization.

The selection favors numerical robustness over the modest memory and
throughput advantage of float32 for this model family.

## 7. Memory semantics

The complete canonical dataset must not be materialized in float64.

Only the current bounded training or evaluation batch may be cast to
float64.

This preserves the scalable streaming design required for MalNet-Image and
MaleX.

`GMM_FULL_DATASET_FLOAT64_MATERIALIZATION=NO`

`GMM_BATCH_CAST_FLOAT32_TO_FLOAT64=YES`

## 8. Generation precision

Latent Gaussian sampling and censoring are performed using float64 model
parameters.

The model-consistent censoring map must produce values inside `[0,1]`
before export.

Generated arrays may then be converted to canonical float32 for downstream
benchmark storage and evaluation.

The float32 conversion is a representation conversion only.

It must not perform clipping or other distribution-changing repair.

`GMM_GENERATION_INTERNAL_DTYPE=FLOAT64`

`GMM_CANONICAL_OUTPUT_DTYPE=FLOAT32`

`GMM_OUTPUT_POSTHOC_CLIPPING=NO`

## 9. Numerical failure policy

Training and generation must fail closed on:

- non-finite parameters;
- non-finite scales;
- non-finite log probabilities;
- non-finite loss;
- non-finite gradients;
- generated values outside `[0,1]` before canonical output conversion.

No NaN, infinity, or domain violation may be silently replaced, clipped, or
sanitized.

## 10. Governed precision constants

`GMM_CANONICAL_INPUT_DTYPE=FLOAT32`

`GMM_INTERNAL_DTYPE=FLOAT64`

`GMM_PARAMETER_DTYPE=FLOAT64`

`GMM_LIKELIHOOD_DTYPE=FLOAT64`

`GMM_GENERATION_INTERNAL_DTYPE=FLOAT64`

`GMM_CANONICAL_OUTPUT_DTYPE=FLOAT32`

`GMM_BATCH_CAST_FLOAT32_TO_FLOAT64=YES`

`GMM_FULL_DATASET_FLOAT64_MATERIALIZATION=NO`

`GMM_OUTPUT_POSTHOC_CLIPPING=NO`

`GMM_PRECISION_POLICY_SELECTED=YES`

## 11. Authorization state

The numerical precision policy is selected.

The complete training procedure and production implementation remain
unaccepted.

Therefore:

`GMM_IMPLEMENTATION_AUTHORIZED=NO`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=NO`
