# RBM Prospective Distributional Assessment — Statistical Design Annex

STATUS=DESIGN_PROPOSAL
PROTOCOL_APPROVED=NO
NEW_SAMPLING_AUTHORIZED=NO
RBM_TRAINING_AUTHORIZED=NO

## 1. Relationship to previous investigations

This annex supplements the prospective RBM distributional-assessment
proposal. It does not amend, replace, or reopen the completed exploratory
or confirmatory investigations.

CONFIRMATORY_ASSESSMENT=INDETERMINATE
CONFIRMATORY_INVESTIGATION_CLOSED=YES
PRODUCTION_GIBBS_DEPTH_SELECTED=NO

## 2. Resource-planning evidence

The historical confirmatory audits used four allocated CPUs and 16 GB
of requested memory.

CICMalDroid2020: 160 trajectories in 540 seconds.
MaleX: 64 trajectories in 202 seconds.

These are historical observations, not validated new-study runtimes.

The revised four-dataset benchmark contains 63 classes:
USTC=9, CIC=5, MalNet-Image=47, MaleX=2.

For seven recorded depths and 4,800 float64 visible coordinates:

replicates | trajectories | raw GiB | projected wall hours
16 | 1008 | 0.2523 | 0.884–0.945
32 | 2016 | 0.5047 | 1.768–1.890
64 | 4032 | 1.0094 | 3.535–3.780
128 | 8064 | 2.0187 | 7.070–7.560

All projections assume historical linear scaling, one initialization
scheme, and no separate distributional-analysis cost.

No replicate count is selected.

## 3. Statistical questions

Question A: Can distribution-sensitive comparisons identify meaningful
changes across increasing Gibbs depths within each class?

Question B: Does initialization dependence remain detectable?

Question C: Is there evidence supporting one computationally practical
sampling depth across the required datasets and classes?

Question D: Are the diagnostics sufficiently sensitive to justify
their own stability conclusions?

## 4. Candidate analysis framework

Consider a predefined kernel two-sample statistic such as MMD-squared,
evaluated on full canonical visible vectors or a justified fixed
representation.

The kernel, scaling, representation, uncertainty procedure, and
multiple-comparison policy must be selected before execution.

High-dimensional sensitivity must be investigated without selecting
a diagnostic merely because it favors convergence.

Prefix-coupled samples at different depths are dependent.
Statistical inference must preserve trajectory pairing.

Independent trajectories, not scalar image coordinates, are the
replication units.

A nonsignificant difference does not establish equivalence.
A numerical equivalence margin requires an independent scientific
justification and sensitivity assessment.

## 5. Initialization design

The governed production initialization remains model-based.

Any additional diagnostic initialization distribution requires
separate scientific justification, explicit identification, and
authorization before execution.

Diagnostic initialization must not silently replace production
initialization or use held-out real examples.

## 6. Prospective planning constraints

Before approving a sampling experiment, specify:

- exact checkpoint and dataset identities;
- class coverage and study seeds;
- independent trajectory count and justification;
- initialization groups;
- transition depths;
- raw-state data format and integrity verification;
- primary statistic and sensitivity checks;
- uncertainty and multiplicity handling;
- scientific equivalence or stability criteria;
- compute and storage ceilings;
- explicit SUPPORTED, CONTRADICTED, and INDETERMINATE outcomes.

No convergence proof is claimed.

## 7. Authorization state

STATISTICAL_DESIGN_FINALIZED=NO
REPLICATE_COUNT_SELECTED=NO
DISTRIBUTIONAL_TEST_SELECTED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
NEW_SAMPLING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO


## 8. Empirical bootstrap coverage limitation

STATUS=NEGATIVE_METHODOLOGICAL_FINDING
EXISTING_PERCENTILE_BOOTSTRAP_APPROVED_FOR_EQUIVALENCE=NO

An exploratory coverage experiment evaluated the existing paired-index
percentile bootstrap around biased empirical RBF MMD-squared.

The experiment used artificial one-dimensional Gaussian populations,
32 observations per group, 120 independent Monte Carlo experiments
per condition, 80 bootstrap resamples per experiment, and a nominal
95 percent interval.

Results:

mean_difference | true_MMD2 | mean_estimated_MMD2 | coverage
0.0 | 0.0000000000 | 0.0264777379 | 0/120
0.5 | 0.0471239558 | 0.0778145275 | 105/120
1.0 | 0.1772676349 | 0.2044112079 | 112/120

The identical-population case exhibited zero interval coverage.
The existing procedure therefore cannot be accepted as a reliable
population-MMD2 interval near equality.

This is an exploratory finite simulation, not a comprehensive
coverage characterization.

The mathematical bootstrap implementation and regression tests
remain historically valid. Passing implementation tests does not
establish inferential validity.

Required next steps:

1. Preserve this negative finding.
2. Investigate unbiased or bias-corrected MMD estimation.
3. Validate uncertainty procedures under the equality boundary.
4. Preserve paired trajectory dependence during inference.
5. Demonstrate sensitivity and coverage before defining
   any production equivalence decision.
6. Do not select Gibbs depth from the existing intervals.

BOOTSTRAP_INFERENCE_VALIDATED=NO
PRODUCTION_EQUIVALENCE_TEST_SELECTED=NO
PROSPECTIVE_PROTOCOL_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
RBM_TRAINING_AUTHORIZED=NO


## 9. Bounded-kernel concentration feasibility

STATUS=NEGATIVE_RESOURCE_FEASIBILITY_FINDING

The paired unbiased RBF MMD-squared estimator can be expressed as
a second-order U-statistic over independent trajectory pairs.

For an RBF kernel in [0,1], its symmetric pairwise kernel lies
in [-2,2].

A conservative two-sided Hoeffding confidence radius is:

epsilon(n, alpha) =
sqrt(8 * ln(2/alpha) / floor(n/2))

For alpha=0.05, the calculated radii are:

trajectories | confidence_radius
16 | 1.920646
32 | 1.358102
64 | 0.960323
128 | 0.679051
256 | 0.480161
512 | 0.339525
1024 | 0.240081
2048 | 0.169763

These results indicate that the distribution-free bound is too
conservative for practical small-margin equivalence decisions
at the previously considered trajectory counts.

This does not invalidate paired unbiased MMD-squared.

Further work must evaluate alternative calibrated uncertainty
methods, scientifically justified margins, class-level
sensitivity, and evidence concerning stationary-distribution
approximation.

HOEFFDING_BOUND_PRIMARY_METHOD_SELECTED=NO
PAIRED_MMD_CANDIDATE_RETAINED=YES
UNCERTAINTY_METHOD_APPROVED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
PROSPECTIVE_PROTOCOL_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
RBM_TRAINING_AUTHORIZED=NO


## 10. Paired-bootstrap correlation robustness

STATUS=EXPLORATORY_POSITIVE_FINDING
PAIRED_BOOTSTRAP_PRODUCTION_APPROVED=NO

An exploratory synthetic experiment evaluated percentile-bootstrap
interval coverage for paired unbiased RBF MMD-squared at the
equal-marginal-distribution boundary.

The experiment used 64 independent Gaussian trajectory pairs,
120 Monte Carlo repetitions per correlation condition,
100 paired bootstrap resamples per repetition,
and nominal 95 percent confidence intervals.

Results:

correlation | coverage | mean_upper_bound
-0.80 | 119/120 | 0.0922809194
0.00 | 118/120 | 0.0625385062
0.80 | 114/120 | 0.0216562706
0.99 | 117/120 | 0.0012634970

Observed coverage was near or above the nominal level in
these limited simulations.

The upper bound narrowed substantially as positive
within-trajectory correlation increased.

This behavior is not sufficient evidence of Gibbs-chain mixing.
Slow-moving chains can produce highly correlated states at
adjacent depths even while remaining far from stationarity.

The earlier biased-MMD bootstrap coverage failure and
the Hoeffding resource-feasibility limitation remain valid.

Prospective assessment requires independent investigation
of initialization-memory loss and mixing behavior.

PAIRED_UNBIASED_MMD_CANDIDATE_RETAINED=YES
PAIRED_BOOTSTRAP_INFERENCE_FULLY_VALIDATED=NO
INITIALIZATION_MEMORY_DIAGNOSTIC_REQUIRED=YES
STATIONARITY_ESTABLISHED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
PROSPECTIVE_PROTOCOL_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
RBM_TRAINING_AUTHORIZED=NO


## 11. Prospective initialization-memory design

STATUS=PROPOSED_ONLY
ALTERNATIVE_DIAGNOSTIC_INITIALIZATIONS_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO

### 11.1 Purpose

Investigate whether the revised class-conditional RBM Gibbs chain
exhibits persistent dependence on its starting distribution.

This investigation is separate from the completed confirmatory
chain-length assessment.

### 11.2 Production reference initialization

Group A:

h0 = exact zero vector
v0 ~ p(v | h0)

This is the existing governed production initialization.

PRODUCTION_INITIALIZATION_CHANGED=NO

### 11.3 Candidate diagnostic initializations

Group B:

h0 = exact one vector
v0 ~ p(v | h0)

Group C:

h0 components ~ independent Bernoulli(0.5)
v0 ~ p(v | h0)

Both groups are diagnostic-only proposals.

They must not replace production initialization or introduce
empirical-data initialization.

### 11.4 Common transition kernel

All groups must use the same trained model, checkpoint,
class identity, and conditional Gibbs transition mathematics.

At each transition:

h_next ~ p(h | v_current, class)
v_next ~ p(v | h_next)

Diagnostic starting conditions must not modify the model
parameters or class-conditioning semantics.

### 11.5 Evidence requirements

Before execution, a prospective protocol must define:

- dataset, seed, class, and checkpoint coverage;
- independently governed initialization random streams;
- trajectory replication and independence assumptions;
- common transition depths and maximum computational budget;
- full visible-state retention and integrity verification;
- class-stratified comparisons between initialization groups;
- comparison against later-depth reference distributions;
- distribution-sensitive diagnostics and uncertainty procedures;
- scientifically justified acceptance criteria;
- explicit inconclusive and contradictory outcomes.

### 11.6 Interpretation safeguards

Agreement between initialization groups does not prove
stationarity.

Adjacent-depth stability does not prove mixing.

Strong within-trajectory correlation does not prove mixing.

Three starting distributions do not guarantee coverage of
all relevant modes of the RBM stationary distribution.

No single diagnostic is sufficient to select the
production Gibbs-chain depth.

### 11.7 Authorization boundary

INITIALIZATION_MEMORY_PROTOCOL_FINALIZED=NO
ALTERNATIVE_DIAGNOSTIC_INITIALIZATIONS_APPROVED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
PRODUCTION_INITIALIZATION_CHANGED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
PRODUCTION_BURN_IN_SELECTED=NO
RBM_TRAINING_AUTHORIZED=NO
