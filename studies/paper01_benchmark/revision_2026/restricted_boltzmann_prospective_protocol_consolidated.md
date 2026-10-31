# Paper 1 Revision 2026 — Consolidated Prospective RBM Sampling Assessment

STATUS=PROTOCOL_DRAFT
PROTOCOL_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO

## 1. Scientific purpose

Establish whether separately authorized prospective evidence can
support a defensible fixed production Gibbs-chain depth for the
revised continuous-visible, class-conditional RBM.

The completed confirmatory investigation remains closed.

CONFIRMATORY_ASSESSMENT=INDETERMINATE
CONFIRMATORY_INVESTIGATION_CLOSED=YES

## 2. Governing decisions preserved

The production initialization remains:

h0 = exact zero vector
v0 ~ bounded continuous p(v | h0)

The requested class remains clamped throughout sampling.

Training CD depth does not determine production synthesis depth.

Historical Bernoulli-visible checkpoints are excluded.

The preferred production decision is one fixed common depth
across the benchmark datasets.

Adaptive per-sample stopping and post-hoc metric-based
depth optimization are forbidden.

Invalid numerical states, clipping, class switching,
and identity-unstable randomness require failure closure.

## 3. Prospective scientific questions

A. Does the class-conditional visible-state distribution
exhibit material differences across increasing Gibbs depths?

B. Does initialization dependence persist?

C. Is the diagnostic evidence sufficiently sensitive to
support a scientifically meaningful stability conclusion?

D. Can one fixed common depth be defended across the required
dataset and class evidence within a practical compute budget?

E. What limitations remain concerning approximation of the
stationary model distribution?

## 4. Proposed initialization groups

Group A:
h0 = zero vector
v0 ~ p(v | h0)

Group B:
h0 = one vector
v0 ~ p(v | h0)

Group C:
h0 components ~ Bernoulli(0.5)
v0 ~ p(v | h0)

Only Group A is the governed production initialization.

Groups B and C are proposed diagnostic instruments.
They are not approved for execution.

All groups would use the same trained model,
class conditioning, and Gibbs transition kernel.

## 5. Candidate distributional diagnostics

The candidate primary statistic is paired unbiased
RBF-kernel MMD-squared for independently sampled
trajectory pairs with repeated depth observations.

The statistic has passed mathematical regression tests.

Its bandwidth, feature representation, inferential
calibration, and equivalence margin remain unselected.

Alternative initialization groups require appropriately
designed between-group distributional comparisons.

Adjacent-depth stability alone does not prove mixing.

## 6. Statistical findings retained

The biased empirical MMD-squared percentile bootstrap
showed inadequate coverage at population equality.

The distribution-free Hoeffding bound was too conservative
for practical small-margin decisions.

Paired unbiased MMD-squared performed consistently
with theoretical expectations in controlled correlated
Gaussian simulations.

Its paired percentile-bootstrap coverage was promising
in limited synthetic experiments, but general inferential
validity remains unestablished.

No existing statistical method is approved for a
production convergence or equivalence decision.

## 7. Preliminary compute and storage envelope

The four benchmark datasets contain 63 classes in total.

Previously evaluated planning scenarios include
16, 32, 64, and 128 independent trajectories per class.

The seven historical observation depths were:
1, 2, 5, 10, 20, 50, 100.

These values are planning references, not independently
selected prospective experimental settings.

Historical confirmatory runtimes were:
CICMalDroid2020: 160 trajectories, 540 seconds.
MaleX: 64 trajectories, 202 seconds.

Additional initialization groups, longer transition
horizons, full-state retention, and distributional
analysis require separate budget accounting.

## 8. Evidence requirements

A future approved experiment must retain:

- dataset and class identities;
- experimental seeds and checkpoint-state hashes;
- exact repository revision;
- initialization-group identities;
- independent trajectory and random-stream identities;
- transition counts and observation depths;
- full visible-state vectors;
- numerical-validity checks;
- file integrity hashes;
- reproducible analysis configuration;
- explicit uncertainty and sensitivity results.

Historical confirmatory evidence must remain unchanged.

## 9. Required prospective decisions before execution

1. Specify checkpoint and dataset coverage.
2. Specify class and seed coverage.
3. Fix diagnostic initialization rules.
4. Fix depths and maximum transition budget.
5. Justify independent trajectory counts.
6. Fix evidence format and storage ceiling.
7. Select a defensible inferential method.
8. Establish independent sensitivity requirements.
9. Justify any stability or equivalence margin.
10. Define multiplicity handling across classes,
    datasets, depths, and initialization groups.
11. Specify explicit decision outcomes.
12. Obtain separate execution authorization.

## 10. Decision outcomes

SUPPORTED:
All prespecified requirements are satisfied,
including sufficient distributional, initialization,
sensitivity, and computational evidence.

CONTRADICTED:
Prespecified diagnostics establish material
failure of required stability or mixing conditions.

INDETERMINATE:
Evidence is insufficient, diagnostics conflict,
or uncertainty prevents a defensible decision.

SUPPORTED must not mean mathematically proven
convergence to the stationary distribution.

No outcome is selected by this draft.

## 11. Remaining implementation acceptance gates

A. Separately governed production chain-depth decision.
B. Final production sampling integration.
C. Focused RBM stack validation.
D. Full repository regression.
E. Explicit RBM training-authorization review.

No gate is waived by prospective protocol development.

## 12. Current authorization

PROSPECTIVE_PROTOCOL_APPROVED=NO
DIAGNOSTIC_INITIALIZATIONS_APPROVED=NO
STATISTICAL_INFERENCE_METHOD_APPROVED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
PRODUCTION_BURN_IN_SELECTED=NO
RBM_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO
RBM_IMPLEMENTATION_AUTHORIZED=NO
RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO


## 13. Proposed staged experimental design

STATUS=DESIGN_PROPOSAL_ONLY
EXPERIMENTAL_EXECUTION_AUTHORIZED=NO

### 13.1 Stage F: diagnostic feasibility

Purpose:
Validate evidence retention, numerical integrity,
initialization-group separation, computational cost,
and the sensitivity of candidate statistical diagnostics.

Proposed scope:
- CICMalDroid2020 and MaleX pilot checkpoints;
- all seven classes across those two datasets;
- experimental seed 42;
- initialization groups A, B, and C;
- 16 independent trajectories per class per group;
- observation depths 0, 1, 2, 5, 10, 20, 50, 100.

These are proposed settings, not approved settings.

Depth zero denotes the initial visible state and is
diagnostic evidence, not a production synthesis candidate.

Stage F must retain complete visible-state vectors,
checkpoint provenance, trajectory identities, and hashes.

Stage F may test analysis software and operational
feasibility but must not select a production Gibbs depth,
establish post-hoc equivalence margins, or silently
become confirmatory evidence.

No new sampling may begin without separate approval.

### 13.2 Stage F advancement criteria

Advancement requires:
- validated independent trajectory identities;
- correct class conditioning and bounded float64 states;
- reproducible initialization groups and transitions;
- complete evidence retention and integrity validation;
- measured runtime and storage consumption;
- a credible uncertainty-calibration strategy;
- demonstrated diagnostic sensitivity under
  independently defined synthetic alternatives;
- an explicit independent approval decision.

Failure or ambiguity prevents advancement.

### 13.3 Stage C: prospective assessment

Stage C is a separate confirmatory assessment proposal.

The preferred coverage is all four revised benchmark
datasets and all classes, with seed coverage independently
justified before authorization.

Candidate planning settings include 64 independent
trajectories per class per initialization group.

No final trajectory count, depth horizon, or seed count
is approved by this draft.

The confirmatory statistical method, bandwidth policy,
equivalence margin, multiplicity treatment, and decision
criteria must be fixed before Stage C evidence is generated.

Feasibility findings must not be used to retroactively
change the criteria after observing Stage C results.

### 13.4 Independence and comparisons

Distinct sample identities must define independent
trajectories within each initialization group.

Repeated depths from one trajectory are dependent
and must remain paired during within-group analysis.

Between-group analysis must account for its actual
sampling design and must not assume paired observations
unless a valid coupling was explicitly constructed.

The production-reference Group A must remain separate
from diagnostic-only Groups B and C.

### 13.5 Resource planning

For Stage F, the proposed trajectory count is:

7 classes * 3 groups * 16 trajectories = 336 trajectories.

Eight retained observation depths, 4,800 visible
coordinates, and float64 storage give approximately
103.2 MB of raw visible-state arrays before metadata,
compression, intermediate files, or analysis overhead.

For Stage C, an illustrative one-seed scenario is:

63 classes * 3 groups * 64 trajectories = 12,096 trajectories.

Eight retained depths would occupy approximately
3.72 GB of raw float64 visible-state arrays before
additional evidence and processing overhead.

These are arithmetic planning scenarios, not approved
compute budgets or guaranteed storage requirements.

Slurm allocation, walltime limits, I/O costs, and
checkpoint availability require independent verification.

### 13.6 Statistical approval gate

Paired unbiased MMD-squared remains a candidate.

The encouraging Gaussian bootstrap experiments do
not establish universally valid confidence bounds.

Before confirmatory execution, independently justified
uncertainty calibration and equivalence criteria
are mandatory.

A narrow interval or failure to detect a difference
must not be interpreted as evidence of equivalence.

Initialization-group agreement and adjacent-depth
stability are necessary diagnostic considerations,
not proofs of stationary-distribution convergence.

### 13.7 Decision and stop rules

Stage F outcomes:
FEASIBLE, NOT_FEASIBLE, or INDETERMINATE.

Stage C outcomes:
SUPPORTED, CONTRADICTED, or INDETERMINATE.

A SUPPORTED outcome may justify subsequent
production-depth governance review; it does not
automatically authorize RBM production sampling,
training, or benchmark execution.

Unresolved scientific criteria, conflicting diagnostics,
incomplete evidence, numerical failures, or exceeded
resource caps require stopping or an INDETERMINATE
decision as appropriate.

### 13.8 Outstanding decisions

STAGE_F_SCOPE_APPROVED=NO
STAGE_F_EXECUTION_AUTHORIZED=NO
STAGE_C_SCOPE_APPROVED=NO
STAGE_C_EXECUTION_AUTHORIZED=NO
STATISTICAL_INFERENCE_METHOD_APPROVED=NO
MMD_BANDWIDTH_POLICY_APPROVED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
MULTIPLICITY_POLICY_SELECTED=NO
CONFIRMATORY_SEED_COVERAGE_SELECTED=NO
COMPUTE_BUDGET_APPROVED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
RBM_TRAINING_AUTHORIZED=NO
