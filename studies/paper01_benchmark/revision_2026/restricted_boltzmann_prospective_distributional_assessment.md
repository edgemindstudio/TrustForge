# Paper 1 Revision 2026 — Prospective RBM Distributional Assessment

STATUS=PROPOSAL_ONLY
PROTOCOL_APPROVED=NO
NEW_SAMPLING_AUTHORIZED=NO
RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO

## 1. Purpose

Define a prospective investigation of whether the revised continuous-visible,
class-conditional RBM admits a defensible, fixed production Gibbs-chain depth
under computationally bounded sampling.

This proposal does not select a production depth or authorize execution.

## 2. Historical scientific boundary

The completed pilot and independent confirmatory investigations remain
historically valid and unchanged.

CONFIRMATORY_ASSESSMENT=INDETERMINATE
CONFIRMATORY_INVESTIGATION_CLOSED=YES
PRODUCTION_GIBBS_DEPTH_SELECTED=NO

The earlier evidence does not establish distributional convergence.
No previously examined criterion may be described retroactively as
preregistered or independently validated.

Historical artifacts and checkpoints must remain unchanged.

## 3. Scientific questions

A. Are the class-conditional visible-state distributions sufficiently stable
across candidate Gibbs depths under a predefined diagnostic criterion?

B. Is any apparent stability robust to independently specified
model-compatible initialization conditions?

C. Does the evidence support one common production depth across the
required benchmark datasets and classes?

D. Is the candidate depth computationally practical at the revised
benchmark scale?

## 4. Candidate diagnostics

Candidate diagnostics for prospective review include:

- distribution-sensitive depth comparisons;
- uncertainty estimates using independent trajectories as replicates;
- class-specific assessments;
- initialization-dependence diagnostics;
- between-chain and within-chain behavior;
- numerical validity and boundedness;
- sensitivity to the diagnostic representation and kernel choices.

No diagnostic has yet been selected as an authoritative acceptance test.

## 5. Evidence retention

A future separately authorized implementation should retain:

- exact dataset, seed, checkpoint, and repository identities;
- class and trajectory identities;
- governed initialization identity;
- Gibbs depth and transition count;
- full visible-state vectors in a separately defined compact format;
- file integrity hashes;
- numerical-validity summaries;
- immutable linkage between raw evidence and derived analyses.

Storage format, dtype, artifact size, and retention limits must be
specified before execution.

## 6. Prospective decision requirements

Before new sampling, a separate approved protocol must define:

1. initialization groups and independence assumptions;
2. datasets, classes, seeds, and checkpoints;
3. Gibbs depths and computational budget;
4. primary distributional statistic;
5. uncertainty procedure and multiple-comparison treatment;
6. scientifically justified stability or equivalence margins;
7. decision rule for a common depth;
8. explicit INDETERMINATE and CONTRADICTED outcomes;
9. evidence-integrity and leakage safeguards;
10. authorization and acceptance boundaries.

Insufficient evidence must not be interpreted as convergence.

## 7. Explicit prohibitions

This proposal does not authorize:

- additional RBM training or sampling;
- alterations to historical checkpoints or audit artifacts;
- retroactive threshold selection;
- post-hoc dataset-specific production depths;
- adaptive per-sample stopping;
- selection using final benchmark metrics;
- production synthesis or benchmark execution.

## 8. Current decision

PROSPECTIVE_METHOD_STATUS=UNAPPROVED_PROPOSAL
PRODUCTION_GIBBS_DEPTH_SELECTED=NO
PRODUCTION_BURN_IN_SELECTED=NO
RBM_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO
RBM_IMPLEMENTATION_AUTHORIZED=NO
RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO
