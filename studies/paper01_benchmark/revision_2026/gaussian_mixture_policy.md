# Paper 1 Revision 2026 — Gaussian Mixture Model Policy

`STATUS=FULL_IMPLEMENTATION_INTEGRATED_VALIDATED`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=YES`

## 1. Scope

This policy governs the Gaussian Mixture Model family for the revised
Paper 1 TrustForge benchmark.

Historical Gaussian Mixture artifacts and implementations remain preserved
for provenance.

The revision implementation may differ from the historical implementation
where required for scientific consistency, bounded-domain correctness,
reproducibility, or scalability.

## 2. Historical Paper 1 execution identity

The historical Paper 1 GMM execution path was:

1. `app.main train --model gaussianmixture`
2. `gaussianmixture.train.main(...)`
3. `gaussianmixture.train.run_train(...)`
4. one scikit-learn `GaussianMixture` fitted per class
5. `app.main synth --model gaussianmixture`
6. `GMMAdapter`
7. `gaussianmixture.sample.synth(...)`
8. class-specific checkpoint sampling
9. clipping generated values into `[0,1]`
10. artifact persistence and evaluation

The richer `gaussianmixture.pipeline.GaussianMixturePipeline` was not the
historical Paper 1 execution path established by the revision audit.

## 3. Historical model structure

The scoped historical Paper 1 checkpoints establish that the primary GMM
was class conditional.

For each class, the historical implementation fitted a separate mixture.

The audited Paper 1 checkpoints used:

- covariance type: diagonal;
- mixture components: 5;
- USTC historical dimensionality: 1600;
- CICMalDroid historical dimensionality: 144;
- all audited checkpoints converged.

The historical Paper 1 GMM therefore must not be described as a
full-covariance GMM.

## 4. Historical bounded-domain defect

The historical GMM used ordinary Gaussian components in unconstrained
Euclidean space.

Generated values were subsequently clipped to `[0,1]`.

Direct checkpoint audits demonstrated that this clipping was not an
incidental numerical safeguard.

For USTC historical checkpoints:

- approximately 43.44% of sampled scalar values were outside `[0,1]`;
- 100% of sampled images contained at least one out-of-range value.

For CICMalDroid historical checkpoints:

- approximately 41.49% of sampled scalar values were outside `[0,1]`;
- 100% of sampled images contained at least one out-of-range value.

Most violations were below zero.

Therefore the historical generator distribution and the persisted synthetic
distribution were materially different.

Post-hoc clipping is not accepted as the revised observation model.

## 5. Revised canonical-domain evidence

All revised generators consume the governed canonical representation:

`40 x 40 x 3`

`float32`

`[0,1]`

`D=4800`

The endpoint-mass audit established materially different support structures
across datasets.

### USTC

Full revised training split audited.

Approximately:

- 84.83% exact zero;
- 0.16% exact one;
- 84.99% total endpoint mass.

Every audited image contained an endpoint value.

### CICMalDroid2020

Full revised training split audited.

Approximately:

- 0% exact zero;
- 0.0013% exact one;
- 99.9987% interior values.

### MalNet-Image

A deterministic evenly spaced sample of 20,000 training images was audited.

Approximately:

- 64.76% exact zero;
- 0% exact one;
- 64.76% endpoint mass.

Every audited image contained an endpoint value.

### MaleX

A deterministic evenly spaced sample of 20,000 training images was audited.

Approximately:

- 3.12% exact zero;
- 0.02% exact one;
- 3.14% endpoint mass.

Approximately 23.44% of audited images contained at least one endpoint value.

These results establish that a universal continuous-only observation model
would erase substantial observed endpoint mass for some benchmark datasets.

## 6. Rejected revision alternatives

### 6.1 Ordinary diagonal Gaussian mixture plus post-hoc clipping

Rejected.

Reason:

The historical audit demonstrated that clipping materially changes the
generated distribution.

### 6.2 Full-covariance Gaussian mixture at D=4800

Rejected as the primary revision design.

Reason:

The historical Paper 1 benchmark did not require full covariance, and the
revision does not introduce full covariance merely for architectural
complexity.

At D=4800, full covariance would greatly increase parameter, memory,
estimation, and numerical-conditioning cost without evidence that it is
required for the benchmark objective.

### 6.3 Truncated continuous Gaussian mixture

Rejected as the primary revision design.

Reason:

A continuously truncated distribution on the open or closed bounded interval
does not provide positive atomic probability for observed exact endpoint
values.

USTC and MalNet-Image exhibit substantial exact endpoint mass.

### 6.4 Universal epsilon/logit preprocessing

Rejected.

Reason:

It would alter exact endpoints and impose a transformation whose consequences
differ substantially across datasets with very different endpoint structure.

## 7. Selected revision observation model

The selected revision family is:

**class-conditional censored diagonal Gaussian mixture model**

Each class receives its own mixture model.

Primary component count:

`GMM_COMPONENTS=5`

Primary dimensionality:

`GMM_DIMENSION=4800`

Covariance structure:

`GMM_COVARIANCE=DIAGONAL`

Observation support:

`[0,1]`

For latent scalar variable:

`z ~ Normal(mu, sigma)`

the observed scalar is:

`x = 0` when `z <= 0`

`x = z` when `0 < z < 1`

`x = 1` when `z >= 1`

Therefore the observation model contains:

- positive probability mass at 0;
- ordinary Gaussian density on `(0,1)`;
- positive probability mass at 1.

The censoring map is part of the probability model itself.

It is not post-hoc artifact repair.

## 8. Revised likelihood

For one component and one scalar observation:

### Lower endpoint

For `x = 0`:

`log p(x=0) = log Phi((0-mu)/sigma)`

### Interior

For `0 < x < 1`:

`log p(x) = log Normal(x | mu, sigma)`

### Upper endpoint

For `x = 1`:

`log p(x=1) = log(1-Phi((1-mu)/sigma))`

For diagonal components, scalar log probabilities sum across the 4800
dimensions.

For a K-component mixture:

`log p(x) = logsumexp_k(log pi_k + log p_k(x))`

Mixture weights are normalized using softmax.

## 9. Training objective

The revised GMM must optimize the censored mixture likelihood.

It must not:

- fit an ordinary unconstrained Gaussian likelihood and censor only at
  generation time;
- treat censoring as evaluation-only cleanup;
- use the historical clipping-based sampling semantics as the revision
  training objective.

The primary training objective is mean conditional negative log likelihood
within each class.

The final optimization procedure, initialization strategy, scale floor,
learning rate, stopping criterion, and batch policy remain implementation
decisions that must be governed and regression-tested before real training.

## 10. Generation semantics

Generation is model consistent:

1. sample mixture component according to learned class-specific weights;
2. sample latent diagonal Gaussian vector;
3. apply the governed censoring observation map;
4. return canonical `40 x 40 x 3` values in `[0,1]`.

Exact zeros and ones may therefore arise from model probability mass.

No external post-hoc clipping stage is part of the revised generator
pipeline.

## 11. Prototype feasibility evidence

A scalar censored-Gaussian prototype demonstrated:

- finite lower-endpoint log probability;
- finite upper-endpoint log probability;
- finite interior log density;
- finite gradients;
- bounded model-consistent samples.

A full revision-scale prototype used:

- 5 mixture components;
- D=4800;
- endpoint-heavy synthetic input;
- 48,005 parameter slots for one class-specific mixture.

It demonstrated:

- finite censored diagonal mixture log likelihood;
- finite mixture-logit gradients;
- finite mean gradients;
- finite log-scale gradients;
- normalized mixture weights;
- stable `logsumexp` mixture evaluation;
- bounded model-consistent sampling;
- exact generated endpoint values.

## 12. Family identity

The revision remains within the Gaussian Mixture Model family.

The benchmark manuscript must describe it accurately as a:

**class-conditional censored diagonal Gaussian mixture model**

It must not be described as the historical sklearn GMM implementation.

The historical and revised implementations share the Gaussian-mixture family
identity but differ in their observation model and optimization procedure.

## 13. Current authorization state

The scientific design is selected.

The revised in-memory training stack is now implemented and integrated
validation has passed.

The implemented training stack includes:

- censored diagonal Gaussian-mixture observation semantics;
- governed float64 model state and likelihood arithmetic;
- deterministic class-model initialization;
- exact-integer initialization identity validation;
- bounded initialization sample selection;
- greedy k-means++ mean initialization;
- governed variance floor and scale parameterization;
- balanced deterministic class-specific traversal;
- governed training-budget and validation schedule;
- momentum-free SGD with learning rate `0.001`;
- fail-closed optimizer candidate-state validation;
- full class-specific example-weighted validation NLL;
- step-0 model-selection candidacy;
- lowest-validation-NLL checkpoint selection;
- earliest-step tie breaking;
- reconstruction of the selected in-memory model rather than automatic use
  of the final optimizer state.

The current integrated revised-GMM training-stack regression suite passes:

`159 / 159`

tests across observation, model state, initialization, traversal, budget,
optimizer, validation, and in-memory class training.

Checkpoint persistence/publication and model-consistent sampling remain
separately governed and are not yet implemented and accepted.

Therefore the revised GMM is not yet authorized for benchmark training
execution.

`GMM_OBSERVATION_MODEL=CENSORED_DIAGONAL_GAUSSIAN_MIXTURE`

`GMM_COMPONENTS=5`

`GMM_DIMENSION=4800`

`GMM_CLASS_CONDITIONAL=YES`

`GMM_POSTHOC_CLIPPING=NO`

`GMM_DESIGN_SELECTED=YES`

`GMM_TRAINING_STACK_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_TRAINING_STACK_INTEGRATED_VALIDATION=PASSED`

`GMM_TRAINING_STACK_INTEGRATED_TEST_COUNT=159`

`GMM_CHECKPOINT_PERSISTENCE_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_SAMPLING_IMPLEMENTATION_STATUS=IMPLEMENTED`

`GMM_FULL_STACK_INTEGRATED_VALIDATION=PASSED`

`GMM_FULL_STACK_INTEGRATED_TEST_COUNT=187`

`GMM_SAMPLING_FOCUSED_VALIDATION=PASSED`

`GMM_SAMPLING_FOCUSED_TEST_COUNT=16`

`GMM_IMPLEMENTATION_AUTHORIZED=YES`

`GAUSSIAN_MIXTURE_TRAINING_AUTHORIZED=YES`
