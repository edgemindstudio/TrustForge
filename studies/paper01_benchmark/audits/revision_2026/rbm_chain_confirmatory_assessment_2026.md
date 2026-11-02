# Paper 1 Revision 2026 — RBM Confirmatory Chain Assessment Decision

`ASSESSMENT_STATUS=FINAL_BOUNDED_REVIEW`

`CONFIRMATORY_ASSESSMENT=INDETERMINATE`

`PRODUCTION_GIBBS_DEPTH_SELECTED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Evidence reviewed

The assessment covers the two required revision-specific seed-42 pilot
checkpoints and their independent confirmatory chain-audit artifacts:

- CICMalDroid2020: Slurm job 308233, five classes, 32 trajectories
  per class, 160 trajectories total.
  Evidence SHA-256:
  `9aad7749d426c49a743ca10d68cd1255b6a2cbeef6fd83da21ba6cd5720aba65`
- MaleX: Slurm job 308234, two classes, 32 trajectories per class,
  64 trajectories total.
  Evidence SHA-256:
  `458d50b6c75db9f0f5b09fbcba5a3695b9a1dab821d803a4e8b7ee58b4e17b40`

Both audits evaluated depths 1, 2, 5, 10, 20, 50, and 100 using
independent sample identities with sample-index offset 100000.

Both Slurm jobs completed with exit code 0:0. Confirmatory JSON evidence
passed provenance, coverage, numerical-validity, and selection-safeguard
checks.

## 2. Observed diagnostic behavior

CICMalDroid2020 exhibited relatively stable dataset-level visible means,
visible-coordinate variance, and hidden activation across candidate depths.
Some individual class transitions varied; no common sustained class-level
direction was established.

MaleX exhibited material early hidden-activation changes in both classes.
At depth 1, class means were approximately 0.0979 and 0.1025; at depth
20 they were approximately 0.1536 and 0.1506.

Paired hidden-activation changes from depth 1 to depth 2 were:

- MaleX class 0: +0.034912, descriptive normal-approximation 95%
  interval half-width 0.010849.
- MaleX class 1: +0.024170, descriptive normal-approximation 95%
  interval half-width 0.011200.

Later-depth hidden activation fluctuated without a demonstrated
equivalence region. Aggregate visible-state summaries varied relatively
little, but these summaries cannot establish full-distribution
convergence or adequate mixing.

Substantial individual-state displacement does not, by itself,
establish continuing distributional drift.

## 3. Decision

`CONFIRMATORY_ASSESSMENT=INDETERMINATE`

The governed evidence does not establish the shortest stable common
Gibbs depth across both required datasets and all classes.

The methodology was fixed after exploratory pilot observations and
before the independent confirmatory runs. No numerical equivalence
margin existed before the pilots; none is retroactively presented as
pre-pilot or independently validated.

A confidence interval containing zero is not evidence of equivalence.
The successful execution and numerical validity of the audits do not
independently establish sampling convergence.

Therefore:

`PRODUCTION_GIBBS_DEPTH_SELECTED=NO`

`PRODUCTION_BURN_IN_SELECTED=NO`

`RBM_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 4. Scope and closure

This decision does not establish that the RBM architecture or
checkpoint-training implementation is invalid. It establishes that
the current evidence is insufficient for production chain-depth
selection under the governing protocol.

No additional sampling, retraining, changed convergence threshold,
dataset-specific production depth, adaptive stopping, or benchmark-metric
optimization is authorized by this assessment.

Any future proposal to address the unresolved depth selection must
be separately justified, explicitly governed, and distinguished from
the completed confirmatory assessment.

The present bounded confirmatory investigation is closed.

## 5. Remaining acceptance gates

1. A separately governed, defensible production chain-depth decision.
2. Final production sampling integration.
3. Focused RBM stack validation.
4. Full repository regression.
5. Explicit RBM training authorization review.

`CONFIRMATORY_INVESTIGATION_CLOSED=YES`

`PRODUCTION_GIBBS_DEPTH_SELECTED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
