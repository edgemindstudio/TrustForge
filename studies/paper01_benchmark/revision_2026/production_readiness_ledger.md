# TrustForge Paper 1 Production Readiness Ledger

Updated: 2026-11-04

## Objective

Establish an evidence-backed decision to execute the Revision-2026
train -> synthesize -> evaluate benchmark, preserve historical research,
and provide a documented reproduction workflow for other researchers.

## Working agreement

- The assistant inspects, explains, and supplies commands.
- Bruno executes changes, commits, pushes, merges, and jobs.
- No independent assistant mutations or job submissions.
- Each action states its purpose, effect, and acceptance condition.
- Reported evidence is distinguished from inspected and executed evidence.
- Accepted methodology is reopened only for a concrete contradiction.
- Preserve historical artifacts and source-freeze records unchanged.
- No global path replacement, indiscriminate staging, or cleanup.
- Stop adding speculative requirements once mandatory gates pass.

## Source baseline

- Repository: edgemindstudio/TrustForge
- Branch: engineering/paper01-production-readiness
- Research baseline: 42ac1fa794b6a2aeb7894881cc0cdb61ff77fa38
- Preservation commit: 8a67b7ef4304c1b5edc9807f903f4ed0a533f094
- gcs-core gitlink: df7b5279714d54694e598cabe3188c73297741e1
- Academic author: Bruno Fonkeng <bruno.fonkeng@und.edu>

The preservation commit contains 11 operational/provenance additions.
It does not establish production acceptance.

## Stage plan

| Stage | Purpose and exit evidence | Status |
| --- | --- | --- |
| 1. Establish and preserve state | Reconcile Git/Talon, preserve relevant source, inventory external scripts and artifact identities | COMPLETE |
| 2. Freeze experimental specification | Version datasets, splits, preprocessing, seeds, budgets, evaluation and analysis rules | PENDING |
| 3. Trace production execution | Verify requested-to-executed identities, imports, configuration precedence, fallback/resume and failure behavior | PENDING |
| 4. Complete checkpoint and synthesis integration | Verify durable restoration and correct labeled sampling for every distinct family path | PENDING |
| 5. Validate evaluation and reporting | Resolve weighting, classifier specification, memory, metrics, pairing, aggregation and statistics | PENDING |
| 6. Establish professional reproduction workflow | Test installation, submodules, data preparation, quick start, documentation and paper traceability | PENDING |
| 7. End-to-end acceptance | Bounded production-path runs plus identity, missing-output and incomplete-checkpoint rejection checks | PENDING |
| 8. Freeze and launch | Accepted launch revision, configuration, environment, matrix, resources, retries and GO decision | PENDING |
| 9. Verify results and release | Reconcile runs, regenerate tables/figures, audit claims and preserve reproduction release | PENDING |

## Evidence vocabulary

REPORTED: stated in a handoff, not independently verified here.
INSPECTED: current source or records reviewed.
EXECUTION_VERIFIED: a bounded check ran and its output was reviewed.
BLOCKED: a required dependency or decision remains unresolved.
COMPLETE: the stated acceptance condition is satisfied.
PENDING: not yet established.

Evidence always has a scope. File existence, scheduler success and unit
tests alone do not establish complete scientific execution.

## Current evidence and gap register

| ID | Requirement | Status and evidence | Remaining action |
| --- | --- | --- | --- |
| S1 | Repository identity | EXECUTION_VERIFIED: Talon academic remote and local author checked | Retain identity on future mutations |
| S2 | Preservation baseline | COMPLETE: 11 additions committed in 8a67b7e; local/remote full hashes match; GitHub branch independently checked | Preserve later work in focused commits |
| S3 | Tracked working tree | EXECUTION_VERIFIED: empty staged/unstaged summaries after preservation | Recheck before subsequent mutations |
| S4 | Submodule | INSPECTED: Talon reports gcs-core at pinned df7b527 | Verify internal working-tree state and evaluator source |
| S5 | Existing v3 source freeze | EXECUTION_VERIFIED at 42ac1fa: baseline, all three file hashes and sizes matched | Retain as historical attestation |
| S6 | External artifact roots | EXECUTION_VERIFIED: both historical roots and revision root exist; revision root lists gmm and rbm | Inventory relevant manifests/checkpoints; verify validity separately |
| S7 | External validation scripts | INSPECTED: 19 immediate entries listed outside Git | Identify scripts required to reproduce accepted evidence and preserve appropriately |
| S8 | Failed GMM source lineage | INSPECTED: historical runner/launcher hashes match v2 records; differences explain adapter label access correction and v3 selection | Bind execution outcomes separately |
| P1 | Family training methodology | REPORTED 7/7; selected formal authorization records directly inspected | Verify remaining records as needed; do not restart hardening |
| P2 | Exact seven seeds | INSPECTED: [42,43,44,45,46,47,48] remains proposed | Accept and freeze before full execution |
| P3 | Production source binding | BLOCKED: GMM runner/launcher require HEAD 42ac1fa; current HEAD changed | Implement deliberate revision binding without rewriting historical freezes |
| P4 | Runtime activation | INSPECTED: GAN/Diffusion disabled; VAE disabled per handoffs/contract | Verify and integrate each governed operational path |
| P5 | RBM production sampling | BLOCKED: inspected training authorization withholds synthesis and leaves Gibbs depth unselected | Resolve bounded production sampling policy |
| P6 | Checkpoint-backed synthesis | PENDING across all seven families | Verify distinct persistence/restoration/sampling paths |
| E1 | Downstream class weighting | REPORTED confound: real-only weighted, augmented unweighted | Inspect pinned evaluator and freeze comparable policy |
| E2 | Large-dataset evaluation | REPORTED materialization risk | Verify bounded-memory execution |
| E3 | Required metrics and silent skipping | PENDING comprehensive production-path audit | Verify explicit completeness and failure records |
| E4 | Statistical and missing-run rules | PENDING final policy | Freeze before examining new benchmark outcomes |
| R1 | Reproduction from clean environment | PENDING | Test documented setup and bounded workflow |

## External locations

These locations were observed, not validated as complete scientific outputs.

- Historical USTC: ~/gencys/artifacts_paper1_final_rerun
- Historical CIC: ~/gencys/artifacts_paper1_cicmaldroid
- Revision artifacts: ~/trustforge_revision_2026_artifacts/paper01
- External scripts: ~/trustforge_pretraining_validation
- Preservation material: ~/trustforge_research_preservation

Do not move large artifacts into Git. Document configurable locations and
verify identity links between source, configuration, data and outputs.

## Scientific boundaries

Current target: four datasets, seven families, seven repetitions,
2,000 synthetic samples per class, canonical 40x40x3 float32 [0,1].

Exact seeds remain proposed. Larger native source images do not establish
high-resolution generation. Historical two-dataset/three-seed results
must not be combined with revised results as equivalent experiments.

Methodological authorization does not establish runtime activation,
training completion, durable publication, synthesis, or evaluation.

## Launch decision

FULL_BENCHMARK_GO=NO

Mandatory execution, synthesis and evaluation gates remain unresolved.
A bounded pilot is assessed separately from full scale-out.

## Stage 1 closure evidence — 2026-11-04

Scope: source preservation and initial external-artifact inventory.
Completion does not establish production readiness.

- Ledger commit dadf16c8945f7f628189cce8736c86cbef6e0dcf was
  explicitly verified against the remote branch.
- gcs-core internal working tree was observed clean at the pinned revision.
- Nineteen external validation scripts were copied byte-for-byte into
  audits/revision_2026/external_validation_snapshot_20261104.
  The snapshot manifest records original paths, sizes and SHA-256 hashes.
- External originals and model artifacts remain in place.
- All four observed checkpoint state hashes matched their metadata.

| Checkpoint | State SHA-256 | Observed bytes | Recorded selected step |
| --- | --- | --- | --- |
| GMM MaleX class0 seed42 pilot attempt01 | ebf212de13358a6bc68990ebb4de43b42755177b5134b0b107d3465aad306109 | 384800 | 4915 |
| GMM MaleX class1 seed42 production attempt02 | e1abc951d55d427017dd4ea9be9c9aa415a68b963f86c10c23a927a734c432b2 | 384800 | 23645 |
| RBM CICMalDroid2020 seed42 | 10f64703515290ea1493b731bb8e941f7b200c6ad18a61acfc8a1be8bd17cd72 | 4960702 | 64 |
| RBM MaleX seed42 | 18f8ddf4f1d13bd8a9c660c968019a22421f097a40068c7a7be31f29052231b8 | 4957630 | 5712 |

The observed sizes are inventory measurements, not metadata size claims.
Recorded GMM selection agrees with minimum recorded validation NLL.
Recorded RBM selection agrees with the final-update policy.

### Carried-forward requirements

- Stage 3: establish execution-source and dataset-preparation provenance
  for reusable checkpoints; metadata alone does not supply these links.
- Stage 4: verify fresh-process restoration and synthesis compatibility.
- Stage 6: inspect snapshot dependencies and establish a reproducible
  environment; archived scripts are not accepted production entry points.
- Stage 3: replace future GMM execution binding deliberately; existing
  launchers still require the historical 42ac1fa HEAD.
- Stages 2–8: unresolved scientific and production gates remain unchanged.

Earlier register rows record the evidence available when first written.
This dated closure supplements S4, S6 and S7 without retroactively
claiming that checkpoint restoration or historical execution was verified.

Next stage: freeze the Paper 1 experimental specification.
FULL_BENCHMARK_GO remains NO.
