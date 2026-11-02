# Paper 1 Revision 2026 — RBM Confirmatory Chain Assessment

`STATUS=POST_PILOT_PRE_CONFIRMATION_METHOD`

`RBM_PRODUCTION_CHAIN_LENGTH_SELECTED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Purpose

Evaluate whether independent confirmation supports the shortest
evidence-supported common Gibbs transition depth under the existing
RBM sampling-chain audit protocol.

This methodology was developed after inspecting the CICMalDroid2020
and MaleX seed-42 pilot audits. Those pilot observations are exploratory
and must not be presented as independent confirmation.

## 2. Fixed confirmatory design

- Datasets: CICMalDroid2020 and MaleX.
- Checkpoints: existing authoritative seed-42 checkpoints.
- CIC classes: all five governed class identities.
- MaleX classes: both governed class identities.
- New independent trajectories: 32 per class.
- Sample-index offset: 100000.
- Audited depths: 1, 2, 5, 10, 20, 50, 100.
- Initialization: existing governed model-based initialization.
- Prefix coupling: required within each trajectory.
- Execution: Slurm compute node only.
- Excluded nodes: talon06, talon12, talon14.
- Final benchmark outcome metrics: prohibited for depth selection.
- Existing pilot artifacts: preserved unchanged.
- New evidence: separate, non-overwriting JSON artifacts.

The replicate count and sample-index offset are confirmatory design
choices, not requirements inherited from the original protocol.

## 3. Confirmatory measurements

For each dataset and class, evaluate all audited depths using:

1. finite-state and visible-range validity;
2. visible-mean trajectory summaries;
3. visible-coordinate variance across independent trajectories;
4. sampled hidden-activation means;
5. paired signed changes between adjacent audited depths;
6. paired initialization and previous-depth displacement;
7. changes persisting across later candidate depths.

Treat independent trajectories as the replication units. Do not treat
individual image coordinates or hidden units as independent replicates.

Report descriptive effect sizes and uncertainty intervals where
appropriate. Preserve the distinction between:

- per-trajectory visible variation;
- coordinate-wise variation across trajectories;
- individual-state displacement;
- distributional changes.

## 4. Scientific interpretation rules

A common production depth must be justified across both audited
datasets and their classes.

The selection principle remains:

`SHORTEST_EVIDENCE_SUPPORTED_STABLE_DEPTH`

No numerical equivalence margin existed before the exploratory pilot.
Consequently, a nonsignificant difference or an uncertainty interval
containing zero is not, by itself, evidence of equivalence.

No convergence threshold will be retroactively described as
pre-pilot or independently validated.

Hidden-state stability alone cannot establish visible-distribution
convergence. Visible-mean stability alone cannot establish mixing.

The confirmatory assessment must review the full diagnostic pattern,
including the possibility of later-depth drift.

## 5. Decision outcomes

The confirmatory assessment must report one of:

- SUPPORTED: the combined governed evidence defensibly supports a
  shortest stable common depth, with explicit reasoning and limits;
- CONTRADICTED: the evidence shows material systematic change
  inconsistent with a proposed depth;
- INDETERMINATE: the evidence cannot distinguish stability from
  meaningful unresolved change.

These labels describe the evidence assessment, not automatic
training authorization.

Without a defensible stability justification, the outcome remains
INDETERMINATE and production depth remains unselected.

No repeated confirmatory jobs or changed decision criteria are
automatically authorized after an inconclusive result.

## 6. Scope and safeguards

This protocol does not authorize:

- retraining either checkpoint;
- changing training budgets;
- modifying the RBM architecture;
- altering existing audit artifacts;
- using downstream classifier accuracy, FID, KID, or MS-SSIM
  to choose Gibbs depth;
- selecting different depths post-hoc for individual datasets;
- adaptive per-sample production stopping;
- final RBM implementation or training authorization.

Any future additional evidence requirement must be separately
justified and reviewed.

## 7. Remaining acceptance sequence

After evidence assessment:

1. separately review and govern any supported production chain depth;
2. complete final sampling integration;
3. run focused RBM stack validation;
4. run full repository regression;
5. perform explicit training authorization review.

`CONFIRMATORY_METHOD_STATUS=DEFINED_NOT_EXECUTED`
`PRODUCTION_GIBBS_DEPTH_SELECTED=NO`
