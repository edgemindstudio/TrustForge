# Paper 1 Revision 2026 — Restricted Boltzmann Sampling-Chain Length Audit Protocol

`STATUS=AUDIT_PROTOCOL_SELECTED`

`RBM_SAMPLING_CHAIN_AUDIT_PROTOCOL_SELECTED=YES`

`RBM_PRODUCTION_BURN_IN_SELECTED=NO`

`RBM_PRODUCTION_CHAIN_LENGTH_SELECTED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`

## 1. Purpose

This protocol defines how the production Gibbs-chain length for the revised
Paper 1 Restricted Boltzmann Machine will be selected.

It deliberately does not preselect a production burn-in length.

The final choice must be supported by revision-specific empirical evidence.

## 2. Historical values are non-authoritative

Historical synthesis commonly used a Gibbs depth derived from historical
`CD_K`, often effectively one transition.

Training CD depth and synthesis-chain length solve different problems.

Therefore:

`RBM_HISTORICAL_CD_K_CHAIN_LENGTH_AUTHORITY=NO`

`RBM_TRAINING_CD1_IMPLIES_SYNTHESIS_ONE_STEP=NO`

## 3. Audit execution location

The chain-length audit may require repeated canonical data access, pilot
training, and many Gibbs transitions.

It must therefore execute on a Slurm compute node.

It must not execute as a full-data workload on a Talon login node.

Therefore:

`RBM_CHAIN_AUDIT_EXECUTION=SLURM_COMPUTE_NODE`

`RBM_CHAIN_AUDIT_LOGIN_NODE_HEAVY_WORK=FORBIDDEN`

## 4. Governed chain initialization

Every audited chain must use the selected revision initialization:

- `h_0 = 0`;
- `v_0 ~ p(v|h=0)`;
- initial visible natural parameter `theta=a`;
- bounded-exponential float64 visible sampling;
- requested class clamped before the first hidden draw.

Therefore:

`RBM_CHAIN_AUDIT_INITIALIZATION=GOVERNED_MODEL_BASED`

## 5. Candidate transition depths

The primary candidate Gibbs transition depths are:

`1,2,5,10,20,50,100`

Therefore:

`RBM_CHAIN_AUDIT_DEPTHS=1,2,5,10,20,50,100`

These candidates provide dense coverage near the historical one-step regime
while also testing substantially longer chains.

## 6. Prefix-coupled audit chains

Where practical, candidate depths must be evaluated from the same underlying
chain trajectory.

For one governed initialization, the state at depth:

- 1;
- 2;
- 5;
- 10;
- 20;
- 50;
- 100

must be captured from one prefix-coupled trajectory rather than regenerated
from unrelated random starts.

This reduces unnecessary Monte Carlo variation in depth comparisons.

Therefore:

`RBM_CHAIN_AUDIT_PREFIX_COUPLING=REQUIRED`

## 7. Independent replicate requirement

Prefix coupling does not replace independent replication.

Each audited model/class condition must use multiple independent governed
sampling identities.

Therefore:

`RBM_CHAIN_AUDIT_INDEPENDENT_REPLICATES=REQUIRED`

A single chain trajectory is insufficient evidence for production selection.

## 8. Required diagnostics

At each candidate depth, the audit must record at least:

1. finite-state status;
2. visible range validity;
3. mean visible value;
4. visible-coordinate variance summary;
5. hidden activation mean;
6. change in visible mean relative to the previous candidate depth;
7. change in visible variance relative to the previous candidate depth;
8. sample-to-initialization displacement;
9. sample-to-previous-depth displacement.

No single diagnostic alone establishes convergence.

## 9. Distributional depth comparison

For each candidate depth, aggregate samples across independent chains.

Compare adjacent audited depths using at least:

- absolute mean difference;
- relative variance difference;
- per-sample L1 or L2 displacement summary.

Where computationally practical, an additional distributional discrepancy
metric may be recorded.

Therefore:

`RBM_CHAIN_AUDIT_ADJACENT_DEPTH_COMPARISON=REQUIRED`

## 10. Initialization-memory diagnostic

The audit must explicitly measure whether emitted states remain strongly tied
to the governed initialization.

At minimum, record:

`mean(||v_t - v_0||)`

under a fixed documented norm and normalization.

The purpose is not to require maximal displacement.

It is to identify obviously insufficient chains that remain nearly unchanged
from initialization.

Therefore:

`RBM_CHAIN_AUDIT_INITIALIZATION_MEMORY=REQUIRED`

## 11. Class-conditioning diagnostic

The audit must include multiple governed class identities.

For fixed checkpoint and matched sampling identities, changing the clamped
class must affect the hidden transition through the appropriate row of `U`.

The audit must verify that class identity remains clamped throughout every
trajectory.

Therefore:

`RBM_CHAIN_AUDIT_CLASS_CONDITIONING=REQUIRED`

`RBM_CHAIN_AUDIT_CLASS_SWITCHING=FORBIDDEN`

## 12. Dataset coverage

The final chain-length decision must not be based on only one dataset.

At minimum the audit must cover:

- one balanced/moderate dataset;
- one highly imbalanced expanded dataset.

The preferred final audit covers all four revised benchmark datasets once
revision training implementation is available.

Therefore:

`RBM_CHAIN_AUDIT_SINGLE_DATASET_SELECTION=FORBIDDEN`

## 13. Checkpoint evidence

The production chain-length decision should be based on trained revised RBM
checkpoints or explicitly identified revision pilot checkpoints.

Historical Bernoulli-visible checkpoints are not valid evidence for bounded
continuous-visible mixing behavior.

Therefore:

`RBM_CHAIN_AUDIT_HISTORICAL_CHECKPOINTS=FORBIDDEN`

## 14. Selection principle

Select the smallest audited transition depth after which additional audited
depth increases show no material systematic change in the governed diagnostics
across the required dataset/class evidence.

The decision must favor the shortest depth supported by the evidence because
unnecessary Gibbs transitions increase benchmark compute without increasing
scientific value.

Therefore:

`RBM_CHAIN_LENGTH_SELECTION_PRINCIPLE=SHORTEST_EVIDENCE_SUPPORTED_STABLE_DEPTH`

## 15. No adaptive per-sample stopping

Production synthesis must not run a different chain length for each generated
sample based on an online convergence heuristic.

Once selected, the primary benchmark uses one fixed governed transition depth.

Therefore:

`RBM_PRODUCTION_PER_SAMPLE_ADAPTIVE_CHAIN_LENGTH=FORBIDDEN`

## 16. No dataset-specific post-hoc tuning

The primary benchmark should use one common production chain length across
datasets unless the audit produces compelling evidence that a common length is
scientifically invalid.

Dataset-specific depths must not be selected merely to improve downstream
metrics.

Therefore:

`RBM_DATASET_SPECIFIC_CHAIN_LENGTH_POSTHOC_TUNING=FORBIDDEN`

`RBM_COMMON_CHAIN_LENGTH_PREFERRED=YES`

## 17. Thinning

No thinning interval is selected by this protocol.

If production synthesis uses independent chains initialized separately for each
sample, thinning is unnecessary by construction.

If a future continuation-chain design is considered, thinning must be governed
separately.

Therefore:

`RBM_PRODUCTION_THINNING_SELECTED=NO`

## 18. Benchmark metric isolation

Chain length must not be chosen by inspecting:

- downstream classifier accuracy;
- FID;
- KID;
- MS-SSIM;
- final manuscript ranking among generator families.

Those are benchmark outcomes, not hyperparameter-selection targets for this
audit.

Therefore:

`RBM_CHAIN_LENGTH_SELECTED_ON_FINAL_BENCHMARK_METRICS=FORBIDDEN`

## 19. Audit failure conditions

The audit fails closed if any candidate produces:

- non-finite state;
- out-of-range visible values;
- class switching;
- silent clipping;
- thresholding;
- historical Bernoulli-visible fallback;
- identity-unstable randomness.

Such failures must be corrected before chain-length selection.

`RBM_CHAIN_AUDIT_FAILURE_ACTION=FAIL_CLOSED`

## 20. Current decision state

The audit protocol is selected.

The production chain length is intentionally unresolved until revision-specific
compute-node evidence is available.

Therefore:

`RBM_PRODUCTION_BURN_IN_SELECTED=NO`

`RBM_PRODUCTION_CHAIN_LENGTH_SELECTED=NO`

`RBM_SAMPLING_IMPLEMENTATION_AUTHORIZED=NO`

`RBM_IMPLEMENTATION_AUTHORIZED=NO`

`RESTRICTED_BOLTZMANN_TRAINING_AUTHORIZED=NO`
