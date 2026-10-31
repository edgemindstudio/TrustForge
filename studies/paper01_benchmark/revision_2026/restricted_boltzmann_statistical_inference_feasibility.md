# Paper 1 Revision 2026 — RBM Statistical Inference Feasibility

STATUS=EXPLORATORY_METHODOLOGICAL_EVIDENCE
STATISTICAL_INFERENCE_METHOD_APPROVED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
RBM_TRAINING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO

## 1. Purpose and historical boundary

This report consolidates controlled mathematical and simulation
investigations concerning uncertainty quantification for the
paired-unbiased RBF MMD-squared estimator.

It supplements, but does not supersede:

- restricted_boltzmann_distributional_design_annex.md
- restricted_boltzmann_prospective_distributional_assessment.md
- restricted_boltzmann_prospective_protocol_consolidated.md

The historical RBM confirmatory assessment remains INDETERMINATE.
Its investigation remains closed.

No production convergence, equivalence, Gibbs-depth, burn-in,
sampling, or training decision follows from this report.

## 2. Statistical estimand and dependence structure

Let Z_i=(X_i,Y_i) be independent and identically distributed
trajectory-pair units. Dependence between X_i and Y_i is permitted.

Define the symmetric kernel

h(Z_i,Z_j) =
    k(X_i,X_j) + k(Y_i,Y_j)
    - k(X_i,Y_j) - k(Y_i,X_j).

The paired-unbiased estimator is

U_n = sum_{i != j} h(Z_i,Z_j) / [n(n-1)].

For Gaussian RBF k in [0,1], the kernel h lies in [-2,2].

Population MMD-squared is nonnegative, but the unbiased
finite-sample estimator may be negative.

Independent trajectory-pair units are required for the
stated variance and multinomial analyses. Independence
across depths within a trajectory is not assumed.

## 3. Exact variance decomposition

For theta=E[h(Z_1,Z_2)], define

zeta1 = Var(E[h(Z_1,Z_2) | Z_1])
zeta2 = Var(h(Z_1,Z_2)).

Then

Var(U_n) =
    4(n-2)zeta1/[n(n-1)]
    + 2zeta2/[n(n-1)].

At exact equality of marginal distributions, zeta1=0
for the population MMD-squared kernel.

The degenerate and nondegenerate regimes therefore
require careful treatment in uncertainty inference.

## 4. Controlled three-state reference model

A stationary, nonreversible transition matrix was used:

    [0.1, 0.9, 0.0]
    [0.0, 0.1, 0.9]
    [0.9, 0.0, 0.1]

The stationary distribution is uniform on three states.
State values are 0, 0.5, and 1.

For epsilon in {0, 0.10, 0.25, 0.50}, transformed
Y values use (1-epsilon)*value + epsilon.

The Gaussian RBF bandwidth is 0.5.

The joint distribution has nine possible ordered
states, six of which have positive probability.

Paired observations are generally not exchangeable
within a pair. The ordered pair units are independent
and identically distributed across sample indices.

This is a controlled reference model, not an RBM model
or empirical evidence of Gibbs-chain convergence.

## 5. Exact population moments at n=64

epsilon | theta | zeta1 | zeta2 | Var(U_n)
0.00 | 0.000000000000 | 0.000000000000 | 0.650860940403 | 0.000322847689
0.10 | 0.005793967635 | 0.002270904193 | 0.647541353789 | 0.000460879699
0.25 | 0.037624413397 | 0.014929078475 | 0.606153022264 | 0.001218927953
0.50 | 0.150077229476 | 0.057033674227 | 0.473071826775 | 0.003742682257

Exact population moments were calculated by enumerating
the ordered-pair state distribution.

A separate 2,000-repetition Monte Carlo investigation
produced empirical variance deviations of approximately
+7.90%, +4.52%, +4.39%, and +2.22%, respectively.

These simulation results corroborate, but do not prove
general implementation correctness.

## 6. Exact multinomial representation

Let c_a count observed paired units in state a, and H
be the state-level symmetric kernel matrix.

Then

U_n = [c^T H c - sum_a c_a H_aa] / [n(n-1)].

Numerical comparisons with direct sample evaluation
agreed to approximately 5.6e-17 or better in the
tested configurations.

With six supported states, the counts of possible
multinomial configurations are:

n=16: 20,349
n=32: 435,897
n=64: 11,238,513
n=128: 321,402,081

Exact enumeration was performed at n=16 only.

## 7. Exact finite-sample estimator distribution, n=16

All 20,349 supported count configurations were included.
Their multinomial probabilities normalized to one.
Exact enumerated estimator means and variances matched
the independently derived population formulas.

epsilon | theta | median(U_n) | Q0.95(U_n) | Q0.975(U_n)
0.00 | 0.000000 | -0.021617 | 0.143705 | 0.206289
0.10 | 0.005794 | -0.020545 | 0.162312 | 0.218664
0.25 | 0.037624 | 0.010695 | 0.225177 | 0.286697
0.50 | 0.150077 | 0.120416 | 0.408959 | 0.474973

At equality the estimator distribution is asymmetric,
and its median is negative despite an exact mean of zero.

Sampling-distribution quantiles must not be misrepresented
as confidence bounds for an unknown population parameter.

## 8. Normal-jackknife upper confidence bound

A candidate exploratory bound was

U_jack = max(0, U_n + z_0.95 * sqrt(V_jack)).

At n=16, exact multinomial coverage was:

epsilon | population MMD2 | exact coverage
0.00 | 0.000000000 | 1.0000000000
0.05 | 0.001420695 | 0.8778578305
0.10 | 0.005793968 | 0.8484468503
0.25 | 0.037624413 | 0.8599098326
0.50 | 0.150077229 | 0.8512391293

The zero-clipping rule guarantees coverage at theta=0.
Thus the 100% equality coverage is not evidence of
correct calibration.

Undercoverage at every tested positive alternative
rules out this particular normal-jackknife procedure
for the proposed production inference role.

The jackknife variance estimate also substantially
overestimated exact variance near equality in
2,000-repetition exploratory simulations.

NORMAL_JACKKNIFE_UPPER_BOUND_APPROVED=NO

## 9. Efficiency of alternative estimators

Disjoint-block estimators discarded comparisons between
blocks and suffered substantial variance inflation.

For n=64 and block sizes 2, 4, 8, and 16, observed
variance ratios relative to the full U-statistic
were approximately:

epsilon=0.00: 69.93, 23.68, 9.71, 4.41.
epsilon=0.25: 15.56, 5.73, 3.00, 1.77.

Independent matching uses n/2 disjoint kernel evaluations
and has variance zeta2/(n/2).

At population equality, the exact matching-to-full-U
variance ratio is n-1.

Thus the ratio is 15, 31, 63, or 127 for
n=16, 32, 64, or 128, respectively.

Neither disjoint blocks nor independent matching is
recommended as the primary estimator at the currently
proposed small trajectory counts.

## 10. Concentration and bootstrap findings

The prior two-sided bounded-U-statistic Hoeffding
radius was too conservative for practical small-margin
equivalence decisions at proposed trajectory counts.

An illustrative independent-matching empirical-Bernstein
calculation likewise produced a large additive
correction: approximately 4.9185 at n=16 and 1.1106
at n=64.

This calculation was not an approved confidence theorem
implementation; empirical variances were illustrative.

Earlier biased-MMD paired-percentile-bootstrap experiments
had severe coverage failures at equality.

Limited paired-unbiased bootstrap studies gave mixed
coverage results under different controlled conditions.
Their inferential validity remains unestablished.

A paired wild-bootstrap equality test showed promising
exploratory calibration in selected synthetic models.
Failure to reject equality is not an equivalence finding.

## 11. Required properties of a future inference procedure

A future candidate must:

1. Define the target population MMD-squared estimand.
2. Preserve independence at the trajectory-pair-unit level.
3. Specify the RBF bandwidth and representation policy.
4. Establish a justified one-sided upper confidence bound.
5. Address equality-boundary degeneracy and nearby alternatives.
6. Demonstrate coverage across independently defined models.
7. Establish sensitivity at scientifically justified margins.
8. Handle multiplicity across the planned comparisons.
9. Define explicit indeterminate and contradictory outcomes.
10. Receive separate approval before confirmatory execution.

Equivalence would require a justified positive margin Delta
and an upper confidence bound U satisfying the relevant
coverage requirement. A declaration may be considered only
when U < Delta, subject to broader scientific safeguards.

Agreement between sampled distributions does not prove
that the RBM Gibbs chain has reached stationarity.

## 12. Outstanding decisions and authorization

FULL_PAIRED_UNBIASED_MMD_CANDIDATE_RETAINED=YES
NORMAL_JACKKNIFE_UPPER_BOUND_APPROVED=NO
INDEPENDENT_MATCHING_PRIMARY_METHOD_SELECTED=NO
DISJOINT_BLOCK_PRIMARY_METHOD_SELECTED=NO
HOEFFDING_BOUND_PRIMARY_METHOD_SELECTED=NO
STATISTICAL_INFERENCE_METHOD_APPROVED=NO
MMD_BANDWIDTH_POLICY_APPROVED=NO
EQUIVALENCE_MARGIN_SELECTED=NO
MULTIPLICITY_POLICY_SELECTED=NO
PROSPECTIVE_PROTOCOL_APPROVED=NO
STAGE_F_EXECUTION_AUTHORIZED=NO
STAGE_C_EXECUTION_AUTHORIZED=NO
NEW_RBM_SAMPLING_AUTHORIZED=NO
RBM_TRAINING_AUTHORIZED=NO
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
PRODUCTION_BURN_IN_SELECTED=NO

This report records methodological feasibility evidence only.
It grants no experimental or production authority.
