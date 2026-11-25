# TrustForge Paper 1 Production Readiness Ledger

Updated: 2026-11-13 (America/New_York); status reconciliation at 0f0a594

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
| 2. Freeze experimental specification | Version datasets, splits, preprocessing, seeds, budgets, evaluation and analysis rules | IN_PROGRESS |
| 3. Trace production execution | Verify requested-to-executed identities, imports, configuration precedence, fallback/resume and failure behavior | IN_PROGRESS |
| 4. Complete checkpoint and synthesis integration | Verify durable restoration and correct labeled sampling for every distinct family path | IN_PROGRESS — CIC MAF seed-42 full pool publication and streaming verification accepted; benchmark-wide closure remains open |
| 5. Validate evaluation and reporting | Resolve weighting, classifier specification, memory, metrics, pairing, aggregation and statistics | IN_PROGRESS — fixture evaluation and fresh-process consumption paths verified; real-data evaluation and completion evidence remain open |
| 6. Establish professional reproduction workflow | Test installation, submodules, data preparation, quick start, documentation and paper traceability | PENDING — clean-environment bounded workflow and reviewer traceability map not yet executed |
| 7. End-to-end acceptance | Bounded production-path runs plus identity, missing-output and incomplete-checkpoint rejection checks | PENDING — no bounded integrated real-data production path and rejection suite yet accepted |
| 8. Freeze and launch | Accepted launch revision, configuration, environment, matrix, resources, retries and GO decision | PENDING/BLOCKED after prerequisite gates — no accepted revision/configuration/resource freeze or GO decision |
| 9. Verify results and release | Reconcile runs, regenerate tables/figures, audit claims and preserve reproduction release | PENDING AFTER AUTHORIZED EXECUTION — no authorized full matrix exists to verify or release |

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
| P2 | Exact seven seeds | COMPLETE for methodological selection: [42,43,44,45,46,47,48] explicitly accepted; benchmark specification Section 14 | Verify consumed seed identities during execution |
| P3 | Production source binding | BLOCKED: GMM runner/launcher require HEAD 42ac1fa; current HEAD changed | Implement deliberate revision binding without rewriting historical freezes |
| P4 | Runtime activation | INSPECTED: GAN/Diffusion disabled; VAE disabled per handoffs/contract | Verify and integrate each governed operational path |
| P5 | RBM production sampling | BLOCKED: inspected training authorization withholds synthesis and leaves Gibbs depth unselected | Resolve bounded production sampling policy |
| P6 | Checkpoint-backed synthesis | PARTIAL: CIC MAF seed-42 full pool publication and streaming verification accepted at 0f0a594; other families/seeds remain open | Verify distinct persistence/restoration/sampling paths and complete benchmark pool coverage |
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

Exact seeds 42-48 are accepted. Larger native source images do not establish
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

## AR/MAF bounded feasibility and integration selection

Date: 2026-11-04.

- Original AR job 310948 failed in the diagnostic warm-up due to mixed
  NumPy/TensorFlow inputs; correction was confined to the harness.
- MAF job 310949 and AR retry 310951 passed three-update/one-image checks.
- Batch-eight jobs 310954 (AR) and 310955 (MAF) completed successfully.
- AR eight-image sampling: 118.163868 seconds; process peak RSS 2248612 KiB.
- MAF eight-image sampling: 5.325420 seconds; process peak RSS 2012612 KiB.
- Scheduler MaxRSS was inconsistent with process RSS; do not use it for sizing.
- Node differences prevent a controlled cross-run batching-speedup claim.

Select the prospective recipe for implementation: training batch 64,
two complete epochs, natural class distribution, Adam controls from the
candidate, sample-weighted full validation after each epoch, earliest
minimum-NLL checkpoint, no early stopping, synthesis batch 8.
This selection preserves accepted model mathematics.

The original candidate remains unchanged. ar_maf_integration_recipe.json
records the selected settings, evidence identity and outstanding requirements.

These short synthetic checks do not establish sustained stability,
real-data integration, generalization, convergence or full-scale runtime.
Synthesis stream identity, checkpoint restoration, resource/retry rules
and production acceptance remain open. Stage 2 remains IN_PROGRESS.
FULL_BENCHMARK_GO remains NO.


## AR/MAF engineering integration status at b44e060

This section updates current status without rewriting earlier execution
records or treating component tests as complete production acceptance.

### Accepted methodology

Benchmark specification Section 14 records Bruno's acceptance of the
shared policies in Sections 11-13. This includes seeds 42-48, paired
unweighted classifier comparison, specified diagnostics, estimation-focused
reporting, and explicit incomplete-run/retry identity rules.

E1 and E4 therefore retain implementation and execution requirements;
their shared methodological choices are no longer awaiting selection.
Stage 2 remains IN_PROGRESS because the complete executable specification
and remaining family-specific decisions are not all frozen.

### Execution evidence

- Jobs 310961 (AR) and 310962 (MAF) passed actual-model synthetic
  two-epoch integration: four optimizer updates, validation-state guards,
  and in-memory best/last restoration.
- Evidence: audits/revision_2026/ar_maf_training_integration_483e3ec.json.
- Jobs 310978 (AR) and 310979 (MAF) passed fresh-process restoration
  using two-class models and diagnostic checkpoint artifacts.
- Evidence: audits/revision_2026/ar_maf_checkpoint_roundtrip_182cd92.json.
- Observed inventories: AR 32 tensors / 955962 archive bytes;
  MAF 70 tensors / 8422022 archive bytes.
- These jobs did not exercise the subsequently added complete manifest
  and publication path with governed real-data training.

### Implemented and component-tested

- Canonical traversal and fixed-batch synthesis seed assignment.
- Sample-weighted two-epoch training with strict minimum-loss selection.
- Actual-family batch bindings with optimizer and validation-state guards.
- Bounded state codec, ordered snapshots, identity and manifest contracts.
- Descriptor-anchored bundle verification and exclusive publication.
- Same-adapter preflight/input binding and canonical/integer label agreement.
- Selected model/optimizer setup and restricted training-to-checkpoint runner.

Reported focused checks, each with its own scope:
- Checkpoint component suites: 116 passed.
- Full checkpoint fixture integration with validators active: 12 passed.
- Input/preflight suites: 63 passed.
- Setup orchestration using model/TensorFlow stand-ins: 5 passed.
- Runner orchestration using stand-ins: 10 passed.

These counts are separate suite observations, not a combined fresh test run.
Constructed provenance records do not establish actual source verification.

### Mandatory remaining AR/MAF checks

- Actual governed source preflight and consumed-label integration.
- Import locations bound to the captured checkout.
- Actual-model execution through complete manifest/publication/loading.
- Architecture/source and environment identities sufficient for restoration.
- Checkpoint-backed synthesis and nonrepeating stream integration.
- Resource/retry acceptance and bounded production-path validation.

ar_maf_training_runner.TRAINING_EXECUTION_ENABLED remains False.
No existing family authorization gates were enabled.
Stages 3 and 4 are IN_PROGRESS; neither is complete.

FULL_BENCHMARK_GO remains NO.


## Production-readiness reconciliation at f276fa9

Evidence cutoff: f276fa9dfe9640b27200f02672f412a80a2a64e1.

This section supplies the current consolidated status and supersedes older
status wording only within the evidence scopes stated below. Earlier entries,
failed attempts, source freezes and historical results remain preserved.

INSPECTED below means committed records were reviewed. It does not mean the
recorded Talon execution was rerun during this reconciliation. REPORTED means
supporting execution evidence has not been independently inspected in this
reconciliation. PENDING and BLOCKED identify requirements still open.

### Goal and success conditions

Establish an evidence-backed decision to execute the complete four-dataset,
seven-family, seven-seed train -> synthesize -> evaluate benchmark and provide
a documented reproduction workflow without rewriting historical research.

The locked target is seeds 42-48, 2000 synthetic samples per class per seed,
and canonical float32 NHWC 40x40x3 inputs in [0,1]. There are 196
dataset/family/seed combinations, plus class-specific work units.

Methodological acceptance, operational activation, training completion,
checkpoint publication, independent restoration, synthesis completion and
evaluation completion remain separate evidence states.

### Authority reconciliation

- Benchmark specification Section 14 accepts the selected shared policies
  in Sections 11-13. Earlier proposed/pending wording does not reopen those
  methodological choices.
- Seeds 42-48, paired unweighted classifier comparison, selected diagnostics,
  estimation-focused reporting and incomplete-run identity rules are accepted.
- ar_maf_integration_recipe.json selects implementation settings, including
  two training epochs, Adam controls, validation selection, batch-eight
  synthesis, stream derivation and bounded retry rules.
- Its production_execution_authorized value remains false. Recipe selection
  does not establish full operational acceptance.
- P6 is no longer wholly pending: bounded CIC seed-42 checkpoint-backed
  restoration and synthesis passed for AR and MAF.
- E1 and E4 retain implementation requirements; policy selection is complete.
- The earlier AR/MAF checklist is partially satisfied by later CIC evidence.
  Remaining dataset/seed and production-path coverage must be assessed
  separately rather than repeating completed checks by default.

### Current family readiness matrix

Paths in this section are relative to studies/paper01_benchmark unless
explicitly rooted elsewhere.

| Family | Inspected acceptance/evidence | Remaining mandatory gate |
| --- | --- | --- |
| GAN | revision_2026/gan_formal_training_authorization.md accepts methodology and implementation; operational flags disabled | Verify governed launcher, dataset preflight, controlled activation, actual publication/restoration and synthesis |
| VAE | revision_2026/variational_autoencoder_formal_training_authorization.md accepts methodology; real-data activation and production publication/synthesis withheld | Verify actual real-data production path through persistence, independent restoration and synthesis |
| Diffusion | revision_2026/diffusion_formal_training_authorization.md accepts methodology; approved-width in-memory restoration evidence is bounded | Verify controlled activation, filesystem persistence/restoration and production synthesis |
| AR | Selected recipe; CIC seed-42 restoration/synthesis acceptance; batch-eight timing, recipe-report and class-four range boundary checks | Complete checkpoint-backed pool publication, evaluator integration, remaining dataset/seed operational coverage and resource acceptance |
| MAF | Selected recipe; CIC seed-42 restoration/synthesis acceptance; full pool publication and streaming verification accepted (array 313821, finalizer 313832) | Complete evaluator integration, remaining dataset/seed operational coverage and resource acceptance; authenticate generator lineage separately |
| RBM | revision_2026/restricted_boltzmann_training_authorization.md accepts training; Stage 1 records CIC/MaleX checkpoint integrity | BLOCKED: production Gibbs depth unselected; production sampling/synthesis withheld |
| GMM | revision_2026/gaussian_mixture_sampling_policy.md accepts integrated checkpoint/sampling implementation; Stage 1 records MaleX checkpoint integrity | Resolve current execution-source binding, reusable-checkpoint provenance and actual production-path acceptance |

Seven-family methodological acceptance does not establish seven-family
production readiness. No new convergence or sample-quality threshold is
introduced by this reconciliation.

### Later CIC evidence and its limits

| Evidence | Accepted scope | Remaining limitation |
| --- | --- | --- |
| revision_2026/ar_maf_cic_seed42_synthesis_acceptance.md; acceptance ef99cec; jobs 311064/311065 | Verified best-checkpoint restoration, full CIC validation NLL replay, five valid images/family, same-seed replay and alternate-seed variation | Not a complete 2000/class pool or full benchmark |
| revision_2026/ar_maf_cic_seed42_evaluation_input_acceptance.md; acceptance 12f8a708; job 311073 | All CIC splits and both five-image archives passed input checks | Not complete evaluator acceptance |
| audits/revision_2026/cic_paired_evaluator_smoke_311308; acceptance 2dcb080 | Paired starting weights, shared validation/test, two epochs/arm and five synthetic images/augmented arm | Engineering smoke only; not full protocol or large-dataset memory acceptance |
| audits/revision_2026/cic_batch8_311471_311472; acceptance 56c8ed9 | One checkpoint-backed class-0 batch/family; AR about 125.20 s, MAF about 3.17 s | Single-batch timing does not guarantee full-run resources |
| audits/revision_2026/cic_ar_partition_311540; acceptance 09911ef | Bounded two-worker partition diagnostic | Not complete pool coverage |
| audits/revision_2026/cic_ar_recipe_contract_311628; acceptance dd047b9 | Recipe identity and four archives verified; finalizer correctly rejected incomplete coverage without publication | Full pool pending; preceding 311611 startup failure remains unresolved |
| audits/revision_2026/cic_ar_range_boundary_311664; acceptance f276fa9 | Class 4, batches [248,250), seeds 24749/24750 passed | Full checkpoint-backed AR pool and evaluator integration pending |
| jobs 313819/313821/313832; commit 0f0a594 | CIC MAF seed-42 two-batch range, full 1,250-batch pool publication and streaming preflight | Other families/seeds, generator-lineage authentication, real-split classifier and evaluator remain open |

The handoff reports bounded component fixture passes for the pool validator
(35ee7c8, job 311469), publisher (1e96d05, job 311470), and partition finalizer
(bf614c8, job 311610). These remain REPORTED in this reconciliation; fixtures
do not establish full checkpoint-backed pool publication or crash durability.

At the evidence cutoff, slurm/paper01_revision_2026/ar_cic_seed42_pool.slurm
is absent from the committed tree. The proposed 50-task, 25-batch/task,
90-minute/task, concurrency-two launcher was not accepted or submitted in
the observed evidence.

### Shared readiness matrix

| Gate | Current status | Remaining acceptance condition |
| --- | --- | --- |
| Preservation / Stage 1 | COMPLETE within recorded closure scope | Recheck current Talon state before mutations; preserve historical evidence |
| Dataset contracts / Stage 2 | INSPECTED specification and governed CIC input acceptance; four-dataset job 310126 remains REPORTED here | Bind consumed data, partitions, preprocessing and class identities; resolve outstanding fitted-state evidence |
| Executable specification / Stage 2 | IN_PROGRESS; shared methodology and AR/MAF recipe selected | Complete configuration linkage and remaining family-specific sampling decisions |
| Execution / Stage 3 | IN_PROGRESS; later CIC AR/MAF evidence advances bounded paths | Verify remaining family launchers, source/import identities, activation and failure behavior |
| Checkpoint/synthesis / Stage 4 | IN_PROGRESS; CIC AR/MAF bounded restoration/synthesis accepted; CIC MAF seed-42 full pool publication and streaming verification accepted (array 313821, finalizer 313832, commit 0f0a594) | Close accepted full-pool publication and streaming verification for remaining AR and other family/seed paths; retain benchmark-wide Stage 4 closure until every distinct persistence/restoration/sampling path is accepted |
| Evaluation/reporting / Stage 5 | IN_PROGRESS; policy accepted, paired fixtures and unified validation verified, including fresh-process consumption acceptance in job 313741 | Complete real-data evaluation with authenticated splits/pool; complete-result publication; bounded-memory execution at required scale; evidence-backed aggregation and completion denominators |
| Reproduction / Stage 6 | PENDING | Execute the documented clean-environment installation, submodule/data preparation and one bounded train->synthesize->evaluate workflow from a fresh checkout; retain a reviewer requirement-to-evidence map |
| End-to-end / Stage 7 | PENDING | Execute one bounded integrated production path using governed real data and accepted checkpoint/pool identities, with missing-output, incomplete-checkpoint, identity-mismatch and test-access rejection checks |
| Freeze/launch / Stage 8 | BLOCKED | After prerequisite gates, freeze accepted revision, configuration, environment, matrix, resources and retries, then record an evidence-backed GO or NO-GO; fixture evidence cannot justify GO |
| Results/release / Stage 9 | PENDING AFTER AUTHORIZED EXECUTION | After authorized execution, reconcile every expected/observed run, regenerate tables/figures, audit claims and preserve the reproduction release |

Status reconciliation at 0f0a594: Stage 4 now includes accepted CIC MAF
seed-42 full-pool publication and streaming verification only; it does not
close benchmark-wide Stage 4. Stage 5 remains IN_PROGRESS with fixture-level
paired evaluation, publication/restoration, unified validation and
fresh-process consumption evidence, but real-data evaluation, complete-result
publication, required-scale bounded-memory execution and evidence-backed
aggregation/completion denominators remain open. Stages 6-9 retain their
original exit criteria; their remaining acceptance conditions are stated in
the matrix above. `complete_evaluation=false`; `FULL_BENCHMARK_GO=NO`.

Four-dataset validation job 310126 is reported as COMPLETED with counts:
USTC 72900/8100/9000; CIC 8117/1158/2323; MalNet 883416/126202/252406;
MaleX 731049/104435/208872 (train/validation/test).
These counts agree with the inspected specification; this reconciliation
does not independently authenticate that job's execution output.

### Finite remaining acceptance sequence

1. Reconcile this central ledger and review/commit the bounded status update.
2. Close Stage 2 executable-specification gaps, including RBM sampling policy,
   preprocessing identities and accepted-setting-to-runtime linkage.
3. Close Stages 3-4 for each distinct family execution/checkpoint/sampling path.
4. Close Stage 5 production evaluation, metric and reporting implementation.
5. Close Stage 6 clean reproduction and reviewer traceability.
6. Close Stage 7 bounded integrated production-path acceptance.
7. Close Stage 8 source/configuration/environment/matrix/resource/retry freeze
   and record an evidence-backed GO or NO-GO.
8. After authorized execution, close Stage 9 results and release acceptance.

This sequence organizes existing requirements; it does not add retrospective
scientific gates. Completed bounded evidence is reused within its valid scope.
Each subsequent command must state its parent stage, purpose and acceptance
condition. Resource decisions precede larger synthesis submissions.

FULL_BENCHMARK_GO=NO

Next action after this ledger update is reviewed and committed:
perform a read-only Stage 2 gap audit linking accepted settings to executable
configuration and identifying the exact unresolved decisions. Do not submit
the proposed full CIC pool as part of ledger reconciliation.


## Stage 2 bounded source audit at 2e52664

Source revision: 2e52664d5bf867799e3f37ae8b9f6d82d17fbdf4.

Scope: source inspection and tracked-file content searches. No model execution,
dataset computation, runtime activation or synthesis occurred. Successful
inventory/search commands establish inspection completion, not implementation
or production acceptance. Stage 2 remains IN_PROGRESS.

### Established links

- AR/MAF core implements two complete epochs, sample-weighted losses,
  complete split coverage and strict minimum validation NLL selection.
  Strict comparison retains the earliest epoch on ties.
- AR/MAF traversal uses batch 64 and SeedSequence([experimental_seed, epoch]),
  retains the final partial batch and traverses validation in governed order.
- AR batch primitives consume likelihood bounds [-7,3]. Their configuration's
  patience field does not activate early stopping: legacy fit() is bypassed.
- AR/MAF synthesis seed assignment implements the selected fixed-batch formula.
  This does not establish complete checkpoint-backed pool execution.
- VAE and Diffusion inspected boundaries/runners link selected training controls
  to traversal, seeded stochastic operations and validation selection.
- GAN governed runner consumes its run plan, checks discriminator/generator
  update accounting and publishes only after terminal accounting.
- RBM budget implements batch 256, 2B updates, validation at 0/B/2B and final
  authoritative update.
- GMM budget implements max(100,5B) updates, max(20,B) validation interval and
  six validation observations including step zero.

### Sampling and preprocessing findings

- gan_class_sampling_policy.md governs real-training example selection,
  not synthetic-generation quotas or a production synthesis recipe.
- diffusion_governed_sampling.py provides reverse-sampling mechanics but
  explicitly excludes checkpoint loading, provenance, publication and
  production authorization. Numerical stream identity is dataset-independent;
  the caller's production identity contract remains to be established.
- VAE records describe standard-normal prior draws, decoder-mean generation,
  canonical conversion and selected-best checkpoint use. Do not describe
  decoder-mean generation as a full Gaussian observation draw.
- variational_autoencoder_checkpoint_committed_generation.py concerns
  checkpoint publication generations, not image synthesis.
- CIC adapter fits StandardScaler on revised training rows only.
  Inspected dataset provenance binds source hashes and partition policy,
  but fitted scaler-state identity was not established.
- Content-search results do not prove repository-wide absence. Searches were
  restricted to the stated paths and patterns; historical code is not
  automatically a Revision-2026 production consumer.

### Remaining Stage 2 register

| ID | Item | Classification | Closure condition |
| --- | --- | --- | --- |
| S2-01 | RBM production Gibbs depth | BLOCKED scientific decision | Separately governed prospective sampling decision preserving the closed INDETERMINATE assessment |
| S2-02 | GAN/VAE/Diffusion synthesis recipes | PENDING consolidation/integration | Bind checkpoint role, generation semantics, streams, quotas and batching to executable consumers; preserve existing accepted choices |
| S2-03 | CIC fitted preprocessing identity | PENDING evidence | Record fitted scaler state bound to source, ordered features, training membership, implementation and environment |
| S2-04 | Shared evaluator configuration | ACCEPTED methodology; PENDING executable linkage | Map accepted classifier, metrics and statistical settings to production evaluator consumers |
| S2-05 | Complete benchmark configuration linkage | PENDING consolidation | Identify authoritative settings and consumers across four datasets, seven families and seeds 42-48 |

### Stage boundaries and next action

Runtime activation, GMM current-revision binding, checkpoint publication and
complete pool execution remain Stage 3-4 requirements. Resource/retry freeze
remains a Stage 8 requirement. Their existence does not reopen accepted
training methodology.

Next implementation preparation: define the smallest fitted-state evidence
contract for S2-03, reusing the existing CIC adapter and provenance boundaries.
Present the contract and complete proposed files before applying changes.
This does not authorize heavy computation on a login node.

S2-01 remains explicitly blocked while independent requirements proceed.
No additional RBM confirmatory run or changed stability criterion is selected.

FULL_BENCHMARK_GO=NO


## CIC fitted-state bounded acceptance — job 311667

S2-03: COMPONENT_AND_REAL_DATA_RESTORATION_VERIFIED;
PRODUCTION_PROVENANCE_INTEGRATION_PENDING.

Job 311667 completed on talon07 with exit 0:0 in 51 seconds at revision
e81323282a2662d775f1cee7ea7bd2730ab1184b. Separate capture/restore processes
agreed on fitted-state identity and exact transforms for five training probes.

Evidence: audits/revision_2026/cic_fitted_state_311667.
The record preserves reports, launcher and external artifact hashes.

This advances Stage 2 preprocessing evidence and bounded restoration
verification. It does not close production provenance integration, migrate
existing checkpoints, or establish all-family readiness.

Next: integrate fitted-state evidence into a deliberately versioned consuming
execution contract, preserving existing checkpoint identities and acceptance.

FULL_BENCHMARK_GO=NO


## S2-03 execution integration status at 9b31b43

Source cutoff: 9b31b43a04ce8c15aa0c30d9e4154cb26b9b79bd.

### Verified bounded progress

- Commit 465d8e4 adds the exact fitted-state codec; eight focused checks passed.
- Commit e813232 adds the compute-node capture/restore driver.
- Job 311667 verified real-CIC fitted-state capture and independent-process
  restoration. Acceptance evidence is preserved by commit 87487b7.
- Commit cf529f3 adds the versioned CIC execution envelope and explicit
  fitted-state expectations in shared publication/loading.
- The associated checkpoint regression observation was 153 passed across
  ten modules. This is not a full repository regression.
- Commit 9b31b43 adds the fitted-state-aware CIC execution caller.
- Its scoped suite observation was 30 passed, using training/live-capture
  stand-ins and real envelope/filesystem verification components.
- These suite observations overlap and must not be added as independent tests.

### Implemented contract and limits

The new caller captures the same adapter's fitted state and provenance before
and after the existing selected training procedure. A changed record prevents
execution-envelope preparation.

The versioned envelope links the unchanged inner v1 checkpoint manifest,
captured fitted-state file bytes and fitted-state payload identity.

An explicit expected_fitted_state_sha256 requires the envelope during
publication and loading. Existing legacy calls retain v1 behavior.
No claim is made that all production entry points now require this path.

The caller retains the existing authorization restrictions. General training
execution remains disabled; bounded integration remains CIC seed42 AR/MAF.
Existing checkpoint artifacts and identities are not migrated.

No real model has yet executed through this new caller. Earlier real-CIC
scaler restoration evidence and earlier AR/MAF checkpoints do not establish
that combined execution.

### Remaining S2-03 acceptance

1. Exercise CIC seed42 MAF through the new caller under its unchanged selected
   training recipe, publish the required envelope, and verify restoration in
   a fresh process using externally recorded expected identities.
2. Verify runtime fitted-state/source/environment evidence, checkpoint identity,
   restored validation behavior and rejection of an incompatible expectation.
3. Establish adoption by the relevant production entry points. AR and other
   CIC family consumers retain separate integration/coverage requirements.

The next execution check is MAF first. Its scope is changed-path acceptance,
not convergence, favorable metrics or benchmark-scale synthesis. It does not
authorize reuse of a historical checkpoint as newly trained evidence.
Launcher, resource settings and exact acceptance checks must be reviewed
before submission.

### Connection to the main plan

S2-03 remains IN_PROGRESS. This work also advances bounded Stage 3-4
integration but does not close those stages.

S2-01 RBM sampling decision remains blocked.
S2-02 synthesis recipes, S2-04 evaluator linkage and S2-05 complete benchmark
configuration linkage remain pending within their recorded scopes.
Stages 5-8 and all-family production acceptance remain unresolved.

FULL_BENCHMARK_GO=NO


## MAF CIC execution-envelope acceptance — job 311675

PASS for bounded CICMalDroid2020 / MAF / experimental seed42 execution at
913d86f9c64e6a3a1a0d009f112bcfa889a02ec8.

Job 311675 completed on talon16 in 12m11s with exit 0:0. The selected
two-epoch recipe completed 254 updates through the fitted-state-aware caller.
Required-envelope publication and fresh-process restoration passed.

Best and last states each replayed 1158 validation examples, with expected
and observed NLL both 83.87659391096837. Incompatible fitted-state identity
was rejected. Saved evidence and checkpoint hashes passed verification.

Evidence: audits/revision_2026/maf_cic_execution_311675.

This satisfies the MAF-first execution check in the preceding S2-03 entry.
It does not satisfy AR or other CIC-family consumer adoption, synthesis,
evaluation, clean reproduction or benchmark-wide operational acceptance.
Existing historical checkpoints remain unchanged.

S2-03 remains IN_PROGRESS for remaining consumer adoption and coverage.
Stages 3-4 gain bounded MAF-path evidence; neither stage is complete.
S2-01, S2-02, S2-04 and S2-05 retain their previously recorded status.

Next: reconcile remaining CIC consumer integration requirements against this
accepted shared path before selecting another implementation or compute run.

FULL_BENCHMARK_GO=NO


## MAF CIC envelope synthesis acceptance — job 311680

PASS for bounded synthesis consuming the required fitted-state execution
envelope from job 311675.

Job 311680 completed on talon16 in 44 seconds with exit 0:0 at synthesis
revision e48016b8339cf04a0039484e84649471c7da8476.
Live preprocessing matched training evidence. The restored best checkpoint
replayed all 1158 validation examples at NLL 83.87659391096837.

Five canonical images with conditioning IDs 0-4 passed saved-archive checks.
Same-seed replay and alternate-seed variation passed.

Evidence: audits/revision_2026/maf_cic_envelope_synthesis_311680.

Together with job 311675, this verifies the bounded MAF/CIC seed42
train -> publish -> independent restore -> synthesize path with required
fitted-state evidence. It does not establish complete-pool synthesis,
evaluation, semantic class fidelity or other-family consumer adoption.

S2-03: MAF/CIC seed42 training and synthesis consumers verified;
remaining consumers and evaluation adoption pending.
Stages 3-4 retain their broader unresolved coverage.
FULL_BENCHMARK_GO=NO

Next: inspect production evaluator entry points and map the accepted shared
protocol to executable consumers (S2-04 / Stage 5). Do not launch another
MAF training run or enlarge synthesis solely to repeat this accepted scope.


## Stage 5 input components and classifier implementation selection

Source cutoff: 92b278ce20c63e1550c5739f5cc4514e0f58cf50.

- f1e2e21 consolidates the accepted classifier protocol into classifier_recipe.json.
- 86a79a3 adds bounded canonical real-split views.
- 6f9b9b8 adds the CIC AR/MAF streaming synthetic-pool view.
- 92b278c adds bounded logical real-plus-synthetic training access.
- Latest scoped input-suite observation: 26 passed.
  This includes earlier component checks; counts are not additive.
- These checks use fixtures, not complete real-data evaluator execution.
- CIC synthetic-pool byte validation does not authenticate checkpoint lineage.
- Real views retain a declared provenance digest; the orchestration boundary
  must establish the underlying evidence.
- Combined training rejects real validation/test views.

classifier_recipe.json now records prospective implementation selections for
training randomization, optimizer defaults, initialization, loss aggregation,
selection/stopping, loading and determinism. These are not presented as
recovered historical choices or as execution-verified settings.

Next: implement and verify paired classifier execution consuming the selected
recipe. Then implement selected metrics and evaluation completeness/reporting.
Real-data orchestration, lineage binding and all-family coverage remain open.

S2-04 / Stage 5: IN_PROGRESS.
S2-03 and the other recorded specification/operational gaps remain open.
FULL_BENCHMARK_GO=NO


## Classifier TensorFlow binding acceptance — job 311682

PASS for bounded synthetic-fixture execution at
69a6bc0beacd9e459eb39d6cd4a7312616907b55.
Job 311682 completed on talon07 in 24 seconds with exit 0:0.

Actual architecture, checked Adam controls, initialization replay/variation,
optimizer update and validation-state isolation passed. Both fixture arms
shared initial weights and completed 20 epochs/updates, selecting epoch 19.

Evidence: audits/revision_2026/classifier_binding_311682.

Scope remains component binding. The production paired entry point was not
exercised; no real data or held-out test was evaluated. Actual patience
stopping and nonterminal-best restoration were not demonstrated in this job.
The earlier stand-in control tests remain separate evidence for those cases.

Stage 5 remains IN_PROGRESS. Remaining finite work:
1. Verify the paired entry point with bounded input/lineage orchestration.
2. Implement and validate the accepted metric definitions.
3. Enforce evaluation completeness, paired reporting and missing-run rules.
4. Complete bounded real-data production-path acceptance before scale-out.

Other family execution, S2-03 adoption, reproduction and launch gates retain
their recorded unresolved status. No additional generator training or larger
synthesis run is selected by this acceptance.

FULL_BENCHMARK_GO=NO


## Paired classifier entry-point acceptance — job 311683

PASS for synthetic-fixture execution at
ff6bbb416c281cedfbd7d4a8ca26a5b274058dcf.
Job 311683 completed on talon07 in 11m32s with exit 0:0.

prepare_paired_classifiers exercised the actual input views and classifier
binding. Both arms shared initial weights. Real-only completed 20 updates;
the augmented arm completed 1580 updates under the same 20-epoch maximum.

Selected epochs were 19 and 17 respectively. The augmented arm supplies
actual fixture evidence for nonterminal-best selection/restoration through
the driver's selected-state checks. Both arms reached the epoch limit.
Patience-triggered stopping remains separately covered by stand-in tests.

Evidence: audits/revision_2026/paired_classifier_311683.

This satisfies bounded paired-entry-point fixture acceptance. It does not
establish real-data provenance, generator lineage, held-out test evaluation,
metric completeness, all-family coverage or large-dataset memory acceptance.

Stage 5 remains IN_PROGRESS.
Next: implement and validate accepted metric definitions, followed by
completeness/reporting and bounded real-data orchestration.
No repeat paired fixture or larger generation run is required solely to
re-establish this accepted scope.

FULL_BENCHMARK_GO=NO


## Inception weight identity acceptance — job 311685

PASS for cached no-top ImageNet weight identity.
Job 311685 completed on talon05 in 11 seconds with exit 0:0 at
99a98663508061334fd1c1658fc4c173bc2f9c9e.

The 87910968-byte artifact matched the installed Keras checksum.
Weight SHA-256:
9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a

Evidence: audits/revision_2026/inception_weight_identity_311685.

Metric-component progress:
- 5cb7893: polynomial KID estimator; ten focused tests passed.
- 99a9866: class-balanced reference selection; ten focused tests passed.
- Feature weight artifact identity now has compute-node evidence.

These are separate component observations. Feature extraction has not run.
Next: implement verified-weight InceptionV3 loading and bounded RGB feature
extraction using the accepted resize/preprocessing/inference contract.
FID/cFID, three-scale MS-SSIM, downstream metric/reporting completeness,
real-data orchestration and broader production gates remain unresolved.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Inception RGB feature acceptance — job 311693

PASS for bounded synthetic-fixture feature extraction at
c4d4d8f24c8dde9f98918af0582080a48a3690e8.
Job 311693 completed on talon07 in 23 seconds with exit 0:0.

Verified-weight loading, RGB preprocessing, finite float32 2048-dimensional
features, same-input replay, unchanged model weights, selected-position order,
[16,1] iterator batching and tested invalid-input rejection passed.

Evidence: audits/revision_2026/inception_features_311693.

This closes the bounded extractor fixture check. Primary reference-to-feature
integration and provenance binding remain pending. No primary fidelity result
or malware-semantic representation claim is established.

Stage 5 remains IN_PROGRESS.
Next: connect persisted class-balanced references to verified feature batches
and KID, then finish FID/cFID, three-scale MS-SSIM, downstream metrics and
evaluation completeness/reporting. Reuse accepted component evidence without
repeating generator training or enlarging synthesis solely for these checks.

FULL_BENCHMARK_GO=NO


## Reference-to-Inception/KID integration acceptance — job 311694

PASS for bounded synthetic-fixture integration at
a9946a6700088755afa96f3894bdc1749f670f81.

Job 311694 completed on talon05 in 1m06s with exit 0:0.
Persisted balanced references were consumed by actual verified-weight Inception
extraction and the KID orchestration. Each condition produced ten float32
2048-dimensional feature rows, two per class.

Saved file identities, feature-byte identities, deterministic draw indices
and independent NumPy KID recalculation passed. Fixture KID mean:
0.24403195325683105. All draws used all ten rows; these are not independent
experimental-seed observations.

Evidence: audits/revision_2026/fidelity_integration_311694.

This closes the bounded reference-to-feature/KID fixture check.
Real-data provenance, generator lineage and primary fidelity remain open.
Next: implement and verify FID/cFID mathematics against independent references,
then the remaining selected metrics and evaluation completeness/reporting.
No repeat training or enlarged synthesis is required for this fixture scope.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Saved-feature FID/cFID acceptance — job 311699

PASS for saved 2048-dimensional Inception fixture features.
Job 311699 completed on talon05 in four seconds with exit 0:0 at
483b28940572ce183215d5ad986a1211dd333b20.

Pooled FID and all five class FIDs matched the constant-synthetic analytic
reference. Equal-weight macro cFID and missing-class rejection passed.
No diagonal regularization was applied.

Evidence: audits/revision_2026/fid_features_311699.

Scope: engineering fixture only. A nonzero covariance cross-term at 2048
dimensions, production-scale resources and full fidelity orchestration remain
unverified. No real-data scientific fidelity conclusion is established.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Three-scale MS-SSIM scorer acceptance — job 311889

PASS for bounded synthetic-fixture scorer execution at
5cd3cb933bd007955c8b913fce319f5b45c00a59.

Job 311889 completed on talon07 in 1m05s with exit 0:0.
Selected TensorFlow settings, three-scale geometry, direct-call agreement,
replay, unchanged inputs, identity scores, a constant RGB analytic reference
and nine invalid-input/pair-count rejection cases passed.
Saved report and score-byte identities were verified.

Evidence: audits/revision_2026/ms_ssim_scorer_311889.

The direct-call comparison establishes TensorFlow binding agreement, not
independent general-algorithm validation. The analytic RGB case is separate.
Persisted-pair consumption, complete-class reporting, failed-pair propagation
and complete-pool integration remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Persisted-pair MS-SSIM integration acceptance — job 311890

PASS for bounded synthetic-fixture integration at
0c882596cc37e1180938f0d9f13ad18565f8e2b8.

Job 311890 completed on talon07 in 51 seconds with exit 0:0.
Actual CICPoolView preflight, persisted pair selection and TensorFlow scoring
covered all five classes: 200 unique unordered pairs per class, 1000 total.
Saved selection replay, pair-score coverage, class means and equal-weight macro
checks passed. Fixture macro MS-SSIM was 1.0 for identical images.

Evidence: audits/revision_2026/ms_ssim_integration_311890.

The bounded MS-SSIM scorer and persisted-pair integration checks are complete.
Primary generated-pool evaluation and authenticated lineage remain open.
Failure propagation is covered by stand-in tests, not triggered by this job.
Next: downstream metric definitions, implementation checks and evaluation
completeness/reporting, followed by bounded real-data orchestration.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Trained EvalCNN test-prediction acceptance — job 311891

PASS for bounded synthetic-fixture integration at
1d90df8977a3a4f8ad81c125ec240bcf1a02f541.

Job 311891 completed on talon07 in 2m49s with exit 0:0.
Actual EvalCNN training was followed by a persisted selected-weight/test
expectation and complete ordered prediction collection over 257 fixture
examples in batches [128,128,1].

Saved expectations, prediction archive, selected-weight binding, inference-state
isolation, ordered prediction identity and exact metric replay passed.
The scoped component suite observed 35 passed, including earlier metric tests;
counts are not additive.

Evidence: audits/revision_2026/test_predictions_311891.

This accepts one-classifier engineering integration, not a paired scientific
evaluation. Real-data provenance, generator lineage, selected-weight publication,
fresh-process model restoration and evaluation completeness remain open.
Next: paired result linkage and explicit completeness/reporting checks,
then bounded real-data orchestration.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Stage 5 result validation reconciliation at 7175b27

Source cutoff: 7175b27f0122a21d7c6cce69cd685338dd923ca8.

### Implemented and scoped verification

- 47c88e3 adds paired classifier result linkage; 20 focused tests passed.
- b11bae1 records the prospective revision evaluation completeness contract.
- b1ad819 adds structural status/coverage validation; 29 focused tests passed.
- c760fa5 binds exact result/configuration bytes with strict JSON parsing.
  Its scoped suite observed 62 passed, including the earlier 29 status tests.
- 39b5915 adds downstream metric semantic replay; 11 focused tests passed.
- a3fff32 adds reference/feature-bound KID semantic replay; 20 tests passed.
- 375b182 adds pooled/all-class FID replay using the KID-bound feature sets;
  13 focused tests passed.
- 7175b27 adds MS-SSIM selection, score-identity and aggregation verification;
  20 focused tests passed.

These observations are scoped suites, not a full repository regression.
Overlapping test counts must not be added as independent checks.

### Saved execution evidence replay

The new downstream verifier passed against the saved 257-example EvalCNN
fixture from job 311891.

The KID verifier passed against the saved reference selection and actual
Inception fixture features from job 311694.

The FID verifier passed against job 311699 using those same KID-bound features.

The MS-SSIM verifier passed against job 311890: five classes, 200 pairs/class.
Its supplied score array was derived from the pinned report. This validates
selection/coverage/aggregation consistency, not independent image rescoring.

Standalone metric representations used in these checks were derived from pinned
saved reports. They are not claimed as separately persisted historical files.

### Limits and remaining acceptance

No component currently grants complete_evaluation=true.
Status declarations, artifact hashes and semantic replay are distinct checks.
They do not independently establish source-data authenticity, feature-extraction
lineage, generator lineage or durable publication.

Remaining Stage 5 work:

1. Integrate status, byte binding and semantic checks with explicit cross-result
   identities, coverage and provenance requirements.
2. Publish selected classifier artifacts and verify fresh-process restoration.
3. Exercise actual paired training, selected-weight loading, test prediction and
   result linkage together under one reviewed execution contract.
4. Integrate fidelity/diversity results and preserve incomplete/failure outcomes.
5. Verify seven-seed reporting and benchmark completion denominators.
6. Exercise bounded real-data orchestration with authenticated source,
   preprocessing, generator-checkpoint and synthetic-pool lineage.

Successful fixtures from separate jobs must not be combined into a fictional
complete scientific evaluation.

### Connection to the main plan

S2-04 / Stage 5 remains IN_PROGRESS.
S2-03 adoption and other recorded specification/operational gaps remain open.
Clean reproduction, family-level end-to-end acceptance, resource/retry freeze
and full-matrix launch acceptance remain unresolved.

No additional generator training or expanded synthesis is authorized by this
reconciliation.

FULL_BENCHMARK_GO=NO


## Selected-classifier publication/restoration acceptance — job 312073

PASS for actual EvalCNN synthetic-fixture execution at
5fd21c9a50a232fe6db67d84743bdf8e406ce710.

Job 312073 completed on talon06 in 1m23s with exit 0:0.
Training/publication PID 3449958 and restoration PID 3450177 were distinct.
Selected weights, validated training history, recipe-derived tensor shapes,
external expectation receipt, exclusive publication and anchored loading passed.

Fresh-process assignment reproduced all 257 fixture probabilities exactly.
Saved context links, bundle identities, ordered predictions, metric replay and
incompatible-expectation rejection passed. Optimizer state was not restored.

Evidence: audits/revision_2026/classifier_restoration_312073.

This closes the bounded selected-classifier publication/fresh-restoration
fixture requirement. It does not close paired real-data execution, authenticated
source/generator lineage, general crash durability or evaluation completeness.

Next: integrate paired training, both selected-classifier bundles, restored
test predictions and paired result linkage under one reviewed fixture contract.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Paired classifier publication/restoration acceptance — job 312074

PASS for integrated synthetic-fixture execution at
9ba6c26d8ad93d6fc4dc592c12d306c4970d3751.

Job 312074 completed on talon06 in 8m26s with exit 0:0.
Actual paired training, both selected-classifier publications, externally
recorded expectations, fresh-process restoration, ordered test predictions
and paired-result linkage were exercised together.

Saved bundle identities, training/context links, prediction archives and exact
paired-result replay passed. Each arm covered 257 test examples in batches
[128,128,1]. The fixture macro-F1 difference was -0.19999999999999998.

Evidence: audits/revision_2026/paired_restoration_312074.

This closes the bounded paired classifier persistence-to-result-linkage fixture.
It does not establish real-data utility, authenticated generator lineage,
complete fidelity/diversity integration or complete evaluation.

Next: integrate required-result status, byte binding, metric semantics and
cross-result identity/coverage checks, preserving explicit failure blockers.
Then verify seven-seed reporting and bounded real-data orchestration.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Stage 5 inventory integration reconciliation at 3e2aece

Source cutoff: 3e2aecec7efc05297aa21cc285e7eb271da1e104.

### Implemented and verified

- d095c76 integrates downstream status records and exact artifact bytes with
  arm-metric and paired-result replay; 18 focused tests passed.
- Saved paired execution 312074 passed this downstream integration:
  three downstream groups verified, four fidelity/diversity groups pending,
  complete_evaluation=false. Standalone inventory representations were derived
  from pinned reports, not separately persisted historical metric files.
- 3e2aece integrates KID, pooled FID, class FID and MS-SSIM inventory validation;
  25 focused tests passed.
- Fidelity/diversity integration checks shared synthetic-manifest identity,
  ordered synthetic labels, configuration links and result-specific coverage.
  FID replay consumes the KID-verified feature arrays.
- The combined nine-file scoped suite passed: 189 tests.

These counts overlap earlier component suites and are not additive.
This was not a full repository regression or real-data evaluation.

### Coverage and incomplete outcomes

Downstream coverage counts ordered test examples per arm; paired coverage
counts two comparison arms. Fidelity/diversity coverage counts KID subset
estimates, one pooled FID result, authoritative classes for class FID,
and selected pairs for MS-SSIM.

Pending/failed fidelity/diversity groups remain explicit without requiring
their arrays. FID verification currently requires successful KID evidence.
The downstream integration currently requires both arms' replay inputs even
when a downstream artifact is pending; missing-prediction execution handling
remains open.

### Remaining acceptance boundaries

The two inventory integrations are not yet one complete evaluation orchestrator.
Shared manifest declarations do not authenticate generator lineage, consumed
image bytes, feature extraction or image-score computation.

Older KID/FID and MS-SSIM fixtures used different synthetic manifests.
Their successes cannot be combined into one accepted scientific evaluation.

Remaining work includes cross-result evaluation identity and provenance checks,
shared-execution fidelity/diversity acceptance, partial downstream execution
handling, complete-result publication, seven-seed reporting and bounded
authenticated real-data orchestration.

All completion and full-benchmark authorization gates remain open.
S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Shared-pool fidelity/diversity acceptance — job 312192

PASS for bounded synthetic-fixture integration at
804686c23eac27f111f6a2182d58b4d563601932.

Job 312192 completed on talon06 in 40 seconds with exit 0:0.
The report records its fully qualified hostname, talon06.cm.cluster.

Actual Inception extraction and TensorFlow MS-SSIM scoring exercised one
shared fixture pool. KID and pooled/all-class FID shared selected feature
arrays; MS-SSIM covered all five classes and 1000 persisted selected pairs.
Four fidelity/diversity groups passed exact-byte, configuration, coverage
and saved semantic replay checks. Three downstream groups remain pending.

Evidence: studies/paper01_benchmark/audits/revision_2026/shared_fidelity_312192.
Feature and score arrays remain external with checked hashes.

This closes bounded shared-pool fidelity/diversity fixture execution only.
It does not authenticate source data or generator lineage, establish durable
complete-evaluation publication, or combine separate classifier and fidelity
jobs into a complete scientific evaluation.

Cross-result orchestration/provenance, partial downstream handling,
seven-seed reporting and bounded authenticated real-data acceptance remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Seven-seed reporting arithmetic acceptance at cdb6124

Source: cdb61245f3118e9457c2da4ebcc7a6f6f75551f9.

The focused suite passed 31 tests, including independent SciPy checks of
paired mean, sample SD, standard error and the pointwise 95% Student-t
interval. The reporter requires explicit records for seeds 42-48 and
withholds the interval when fewer than seven declared pairs succeed.

The pinned job 312074 fixture yielded one declared pair and six pending
seeds, with no seven-pair interval. Evidence-backed aggregation, attempt
selection and benchmark-wide completion denominators remain open.


## Partial downstream inventory acceptance at c942dd1

Source: c942dd1c94a08e2bb00d83416eca248cd4555cfe.

The five-file scoped suite passed 121 tests on the Mac and on Talon
(login1, gcs_py310). Either classifier arm can be verified when the
other is pending or failed and has no prediction arrays. The inventory
retains the verified arm and the explicit blockers. Paired success still
requires both verified arms and matching ordered test labels.

The verifier requires predictions from at least one arm. Handling a run
with neither arm available remains an orchestration concern.
Unified evaluation orchestration, cross-result provenance, complete-result
publication and bounded authenticated real-data acceptance remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Unified evaluation result validation at 0836506

Source: 0836506f86c230c7345edd9e78825621d20b83a0.

The 11-file scoped suite passed 269 tests on the Mac and on Talon
(login1, gcs_py310), including 39 tests for the unified validator.
The validator composes downstream and fidelity/diversity verification
under one declared evaluation identity and retains explicit blockers.
It preserves a verified classifier arm if the other is unavailable.
With both arms unavailable, pending or failed downstream records can
coexist with verified fidelity results. Contradictory success or
malformed supplied test identities are rejected.

A matching augmented-pool declaration does not clear the blocker for
evidence that the classifier consumed that pool. Generator lineage,
real-data provenance, complete-result publication and evidence-backed
aggregation remain open. The unified checks used coherent test fixtures.

Next, trace and bind consumed-pool evidence to augmented training and
selected weights, then exercise that linkage in one bounded evaluation.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Augmented classifier pool consumption acceptance — job 313732

PASS for bounded synthetic-fixture evidence of augmented classifier pool
consumption at f2bf9269dc5c48b818e77c1b1743c44ea4b54f2a.

Job 313732 completed on talon07 in 00:09:43 with exit code 0:0.
The saved-evidence check passed. The driver hashed the returned selected
weights during the run, and the saved-evidence check reconciled that digest
across both JSON files.

Execution used evaluator seed 42 and the pinned CIC fixture pool from job
311683. Test-split access was disabled. The selected checkpoint was at
zero-based epoch index 17. Through that checkpoint, 18 epochs accounted for
180,000 synthetic example presentations and 1,422 optimizer updates. Through
the end of training, 20 epochs accounted for 200,000 presentations and 1,580
updates.

Selected weights SHA-256:
9ca712c98dc7a952f3d71e2bcacfd0ebfb66e0536b9ca5acf6f0613cb1bd44ad

Evidence directory:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/pool-consumption-f2bf926-kxLBmm2q

SHA-256:
pool_consumption.json  4b48540e0e033dff034729a2ed20951d5d81fa783b8c41f17cbd2c859c2ca9be
report.json            185794c5b7208bd498cf43e520c6c067fc06584c55e28f2579da9dfa8c98b59d

This closes bounded synthetic-fixture consumption evidence only. Unified
validator integration and real-data and generator lineage authentication
remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Pool-consumption unified validator regression at c2cda79

The 12-file Talon regression at c2cda7995fa23a59cbbcbd99495292cc423a4fe7
passed 284 tests and 3 subtests in 13.22 seconds with
TALON_POOL_CONSUMPTION_REGRESSION=PASS.

The unified validator identifies verified downstream arms from the
downstream-inventory receipt keys. A pending or failed real-only arm cannot
supply the training size used to reject an otherwise verified augmented
result, even when an unverified real-only training report is supplied.
Conflict checks remain active when both training reports are verified.

The pool-consumption blocker remains open pending independent binding of the
augmented training identity to the selected checkpoint.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Paired pool-consumption restoration acceptance — job 313741

PASS for bounded synthetic-fixture acceptance at
1076e64a5d4685331f6ced101e738493f0aa88ad.

Job 313741 completed on talon07 in 00:09:13 with exit code 0:0 and emitted
PAIRED_POOL_CONSUMPTION_ACCEPTANCE_JOB=PASS. Training PID 1314850 published
the selected checkpoint and consumption record; fresh restoration PID 1332339
verified them independently.

The matching augmented consumption record cleared the
augmented_pool_consumption blocker. Omitting the record restored that blocker.
Both cases reported three verified downstream groups.

Evidence directory:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/paired-pool-restoration-1076e64-RsbkPWRg
Verification report: pool_consumption_verification.json

Expectation receipt SHA-256:
0885e68a306820f52feff5cb649edf323f84ea6063a099659581a26ab6f898ba
Pool consumption record SHA-256:
4b48540e0e033dff034729a2ed20951d5d81fa783b8c41f17cbd2c859c2ca9be
Selected weights SHA-256, matching the receipt, restored checkpoint, and
verified downstream result:
9ca712c98dc7a952f3d71e2bcacfd0ebfb66e0536b9ca5acf6f0613cb1bd44ad

Real-data validation and generator-lineage authentication remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## CIC MAF seed-42 two-batch range acceptance — job 313819

PASS for bounded CICMalDroid2020 MAF seed-42 range-worker acceptance at
93d8224f5d470bf70b0da1f97d41b6b74b190a9b.

Job 313819 completed on talon07 in 00:01:33 with exit code 0:0 and emitted
CIC_MAF_RANGE_WORKER=PASS 0 0 2.

The restored checkpoint reproduced validation NLL 83.87659391096837 over
1,158 examples, and live preprocessing matched the training fitted state.

MAF_CIC_TWO_BATCH_SAVED_EVIDENCE=PASS confirmed exact class-0 coverage for
batches 0 and 1, eight samples each, with deterministic sampling seeds 35251
and 35252. The saved image and class-ID contract also passed.

Evidence directory:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-range-93d8224-cIenUSaK

Archive SHA-256:
batch-000  54569575c29433e7efb746aba546ccd196ccb9ed8252599c560f6f8721d14810
batch-001  6617a966f8958f5ee662890cb1d0a74cc7b73cda76a5a08c3e4ea83b503683e9

This accepts only the bounded two-batch MAF range-worker path. Full-pool
acceptance remains open for all 1,250 batches across five classes.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## CIC MAF seed-42 full-pool acceptance — jobs 313821 / 313832

PASS for full CICMalDroid2020 MAF seed-42 pool publication and streaming
verification at 8c58c4e8967da1764e6d7c66397e78befcbcadac.

Array job 313821 completed all five class tasks with exit code 0:0 and all
five worker pass markers. Finalization job 313832 completed on talon06 in
00:00:08 with exit code 0:0.

CIC_MAF_POOL_FINALIZATION=PASS and CIC_MAF_POOL_PREFLIGHT=PASS confirmed
1,250 batches and 10,000 images across five classes, with 2,000 images per
class.

Evidence directory:
~/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-pool-8c58c4e-9Xeks8Mw
Evidence files: finalization_report.json and pool/manifest.json

Manifest SHA-256:
db884e298ea324597bad2d488d03e2c86c205e7a53bbef2a11bd72f06da5651e

This accepts pool publication and streaming verification only.
Generator-lineage authentication and the real-split classifier and evaluator
run remain open. complete_evaluation=false.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## Research continuity charter reference — 2026-11-13

The durable research-context document is
studies/paper01_benchmark/revision_2026/research_continuity_charter.md
(SHA-256 e03873f7cd2c29aa66e28d4a35732725a952d8ea48b9e0fa102d5e1f1c48f79a).

It preserves the research mission, the separate Paper 1–4 publication arcs,
and the reasons for the Revision-2026 engineering audit. It is governance
context, not operational evidence or execution authorization. The current
benchmark specification, accepted decisions, and verified execution evidence
govern operational status; the charter and historical manuscripts do not
override them.

This reference does not change stage statuses, family methodology
authorizations, budgets, runtime activation flags, or completion gates.
complete_evaluation=false. FULL_BENCHMARK_GO=NO.


## GMM CICMalDroid2020 class-0 seed-42 training and synthesis acceptance — jobs 313922 / 313927

PASS for one governed CICMalDroid2020 GMM class-0 seed-42
train -> publish -> independent restore -> sample -> publish-pool path at
training source 6b56e92d906f2f0d6149b3e74ff40bb5277f45ab and synthesis
source 9fe24e8f2adf76d74442d281790405df21d22a99.

GMM_CIC_CLASS0_SEED42_TRAINING=ACCEPTED
GMM_CIC_CLASS0_SEED42_SYNTHESIS=ACCEPTED
GMM_CIC_CLASS0_SEED42_POOL_INTEGRITY=VERIFIED

This entry records accepted execution evidence supplied after the runs. It
reran no job, regenerated no sample, and recomputed no artifact hash. It
supplements the GMM row of the family readiness matrix and P6 only within
the scope stated here.

### Training evidence

- Dataset: cicmaldroid2020; generator: GMM; class: 0; seed: 42.
- Training job 313922 completed with exit code 0:0.
- Training source: 6b56e92d906f2f0d6149b3e74ff40bb5277f45ab.
- Source-freeze SHA-256:
  395751118f1549626841e3fe2167caefd672cf7d1740d0020f0425fc6c2f123a
- Checkpoint metadata SHA-256:
  e43011eccaaace1308c6e2b34fa151d8cdacc682ad832fc0e2eff632d0f5b58f
- Checkpoint state SHA-256:
  9452ebf1c780ceb65f39bebf9cca036cb21a3c88f6cff0eed634ddffdf897769
- Independent governed checkpoint restoration: PASS.

### Synthesis evidence

- Synthesis job 313927 completed with exit code 0:0.
- Synthesis source: 9fe24e8f2adf76d74442d281790405df21d22a99.
- Publisher regressions on Talon: 101 passed.
- Sampling identity SHA-256:
  4d65bda5423e6ae4c6189198787122d2743085d6f6dddb0f0fbb51c41fac51fc
- 2,000 canonical synthetic samples published as 250 ordered batches.
- GMM_CLASS_POOL_PUBLICATION=PASS
- INDEPENDENT_GMM_POOL_VERIFICATION=PASS
- Manifest SHA-256:
  68b72e975927b6d804749ceab821a52b9b8f5656d5ad776ace672fab9aea9291

Pool location on Talon:

`~/trustforge_revision_2026_artifacts/paper01/gmm/cicmaldroid2020/seed42/pools/class0_seed42_synthesis9fe24e8_attempt01/`

### Accepted scope and preserved limitations

- This covers one dataset/class/seed cell.
- Other CICMalDroid2020 classes are not accepted by this evidence.
- Statistical fidelity, diversity, and downstream utility are not established.
- Complete GMM dataset-level evaluation remains pending.
- FULL_BENCHMARK_GO=NO

This entry accepts bounded GMM training, class-0 pool publication and
independent pool verification for the stated cell only. It does not close
benchmark-wide Stage 4 coverage, Stage 5 evaluation, reproduction or launch
gates, and it does not authorize additional GMM classes, seeds, families,
retries or full-benchmark execution.

Stages 4 and 5 remain IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## CIC MAF seed-42 real-data bounded evaluation acceptance — jobs 314063 / 314068

PASS for one bounded CICMalDroid2020 / MAF / seed-42 real-data evaluation at
corrected source 2bee5d1056e074a49d266458df829ba5739a7347.

CIC_MAF_SEED42_EVALUATION_TRAIN=PASS
CIC_MAF_SEED42_EVALUATION_RESTORE=PASS
CIC_MAF_SEED42_BOUNDED_EVALUATION=PASS

Job 314068 completed on talon17 in 00:18:50 with exit code 0:0. Training and
restoration ran as separate Python processes inside one allocation (train PID
2165677, restore PID 2168262); restoration began only after the training
receipt existed and its SHA-256 had been captured successfully.

The predecessor job 314063 failed closed during unified validation because the
driver hashed training_identity_sha256 without the trailing newline byte used
by the accepted validator. Source 2bee5d1 corrects only that digest framing;
the unified validator, recipes, splits, pool, checkpoints and metric
definitions are unchanged. Job 314063 and its outputs are retained unchanged
and are not retroactively accepted.

### Accepted execution evidence

- Dataset: cicmaldroid2020; generator: MAF; experimental seed: 42.
- Classes: 5; real train 8,117; validation 1,158; held-out test 2,323.
- Synthetic pool: 10,000 images, 2,000 per class, 1,250 batches.
- Both classifier arms trained and published; selected states were restored in
  a fresh process and their selected-weight assignments verified.
- Each arm produced test predictions for all 2,323 held-out examples.
- Augmented-arm consumption: 200,000 synthetic examples across 20 epochs,
  stream SHA-256
  105d595099f5659b28498f9acfae5f45f02844f201dcf44a799d78aed2ae26d2.

Run directory:
`~/trustforge_revision_2026_artifacts/paper01/maf/cicmaldroid2020/seed42/bounded_eval-2bee5d1-hni0NB`

Slurm logs:
`~/trustforge_revision_2026_artifacts/paper01/diagnostics/maf-cic-bounded-eval-2bee5d1-hni0NB`

Evidence hashes:
- expectations.json:
  11c5a991c43d96a45cc4bf0774e436149d64f4c859b4acbb2835e7487c840101
- evaluation.json:
  4ce5fca7fb2f7439dcfd57db098a9c17a296823dda5422381a9657c63184d1ab
- restore_report.json:
  9e65a5293b255f2a07f59a4929e09c8372e349caa4729aa5ffb737132a1886b8

### Unified validation scope

- Required verified result groups: 7 of 7.
- Required status blockers: 0; lineage blockers: 0.
- Groups: real_only_downstream, real_plus_synthetic_downstream,
  paired_downstream, pooled_balanced_kid, pooled_balanced_fid, class_fid,
  intra_class_ms_ssim.
- Metric replay and paired replay verified; augmented pool consumption is
  bound to the accepted pool manifest
  db884e298ea324597bad2d488d03e2c86c205e7a53bbef2a11bd72f06da5651e.
- complete_evaluation=false.

### Accepted performance evidence

| Metric | Real-only | Real + synthetic |
|---|---:|---:|
| Macro-F1 | 0.6465410253 | 0.7010525344 |
| Balanced accuracy | 0.6362720813 | 0.6884556007 |
| Macro-AUPRC | 0.7401845206 | 0.8084883252 |
| Brier | 0.4233299249 | 0.3550405532 |
| ECE | 0.1416111567 | 0.0355021160 |

Primary paired endpoint:
augmented_minus_real_only_macro_f1 = 0.05451150907476743

### Preserved boundaries

- This applies only to the evaluated dataset, generator, seed and classifier
  configuration; it does not establish generalization across seeds or cells.
- Generator lineage is DRIVER_VERIFIED; semantic class fidelity remains
  NOT_ESTABLISHED.
- Unified validation records feature_extraction_authentication,
  image_score_recomputation, provenance_validation and publication_validation
  as NOT_PERFORMED. This is bounded evaluation, not exhaustive provenance or
  generator semantic-fidelity authentication.
- Other families, datasets, seeds and full-benchmark execution remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO


## CIC MAF seed-43 real-data bounded evaluation acceptance — job 314240

PASS for one bounded CICMalDroid2020 / MAF / experimental seed-43 real-data
paired evaluation at evaluation source
826e56970141907b34fcab3c9f8a3ad2a7735e81 and generator source
58f965c39aad103dc056f7ac690de90578ed2932.

CIC_MAF_SEED43_EVALUATION_TRAIN=PASS
CIC_MAF_SEED43_EVALUATION_RESTORE=PASS
CIC_MAF_SEED43_BOUNDED_EVALUATION=PASS

Job 314240 completed on talon17 in 00:18:25 with exit code 0:0. Training and
restoration ran in separate processes (PIDs 2265209 and 2267754) inside one
allocation. The restore phase authenticated the training receipt by SHA-256
before evaluating the held-out test split.

### Accepted execution evidence

- Dataset: cicmaldroid2020; family: maf; experimental seed: 43.
- Generator checkpoint:
  57e378c44c870af58e4b8eff38cf1cc976516b0dade249194bb5962c28a724f7
- Synthetic pool manifest:
  95b1d78e90cb8794573b88a195bb1e89fbbfcfcfc7d9562a5210a9971527354f
  (10,000 images, 2,000 per class, 1,250 batches).
- Training receipt, job 314240:
  d040a97aa6a76c5ba6a8a4ed7bbb6b274f2837add91c4dee029a0942af27e7f5
- Shared initial classifier weights; both arms restored and their
  selected-weight assignments verified.
- Test predictions: 2,323 examples per arm with five-class coverage.

Run directory:
`~/trustforge_revision_2026_artifacts/paper01/maf/cicmaldroid2020/seed43/bounded_eval-826e569-2voNVy`

Evidence hashes:
- evaluation.json:
  8d6ac23cd3fb2439295a91682610092fb3d86693b27e3d6786b3889b731c7159
- restore_report.json:
  b5e9dbc243cdd2097d76ae6f9c03e3bde490fd92a0046fa7a4669b2ab0ba8468

### Accepted performance evidence

| Metric | Real-only | Real + synthetic |
|---|---:|---:|
| Macro-F1 | 0.6409108572 | 0.7336864693 |
| Balanced accuracy | 0.6381505545 | 0.7205606907 |
| Macro-AUPRC | 0.7489316286 | 0.8063041171 |
| Brier | 0.4016823170 | 0.3424291324 |
| ECE | 0.0869643200 | 0.0615410572 |

Primary paired endpoint:
augmented_minus_real_only_macro_f1 = 0.09277561208032425

Supporting fidelity/diversity results: pooled class-balanced KID
1.0244120854 (10 draws, 125 per class); pooled balanced FID 598.1498252;
macro class FID 607.4465315; three-scale intra-class MS-SSIM 0.0204661978
(1,000 selected pairs).

### Unified validation scope

- Required verified result groups: 7 of 7.
- Required status blockers: 0; lineage blockers: 0.
- Groups: real_only_downstream, real_plus_synthetic_downstream,
  paired_downstream, pooled_balanced_kid, pooled_balanced_fid, class_fid,
  intra_class_ms_ssim.
- Augmented pool consumption was verified against the seed-43 manifest.
- complete_evaluation=false.

### Cross-seed context

The two accepted bounded seed cells both show a positive paired macro-F1
difference under the unchanged protocol: seed 42 = 0.05451150907476743;
seed 43 = 0.09277561208032425. Two seeds do not complete the seven-seed
study and support no cross-seed statistical-significance claim.

### Preserved boundaries

- Generator lineage is DRIVER_VERIFIED; generator_lineage_authenticated
  remains false in the accepted pool-finalization evidence, and semantic
  class fidelity remains NOT_ESTABLISHED.
- Unified validation records feature_extraction_authentication,
  image_score_recomputation, provenance_validation and publication_validation
  as NOT_PERFORMED.
- This applies only to the evaluated dataset, family, seed and classifier
  configuration. Other families, datasets, seeds and full-benchmark execution
  remain open.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO

## CIC MAF seed-44 generator, pool and real-data bounded evaluation acceptance — jobs 314269 / 314347 / 314349 / 314354 / 314355

PASS for one bounded CICMalDroid2020 / MAF / experimental seed-44 cell spanning
generator training, a two-batch synthesis diagnostic, full-pool production and
real-data paired evaluation. Generator source
8d4a281e04ed2232a134a6da10ff20c56d67316e; synthesis source
8198dc4d42fdb05bc3ebb6fd055b9951ab3d897f; evaluation source
c56a02d627d5506e2ed39badc79c28215ee5849f.

SEED44_GENERATOR_RESTORE=PASS
SEED44_TWO_BATCH_DIAGNOSTIC=PASS
CIC_MAF_POOL_FINALIZATION=PASS
CIC_MAF_SEED44_EVALUATION_TRAIN=PASS
CIC_MAF_SEED44_EVALUATION_RESTORE=PASS
CIC_MAF_SEED44_BOUNDED_EVALUATION=PASS

### Generator identity and restoration

- Job 314269 completed on talon07 in 00:13:26 with exit code 0:0.
- Generation: maf_cic_seed44_execution; selected checkpoint state
  cc58fe65abaa0656e4f09d6d17d83e586f048f50a6a858d01e1f1d7b9191055a.
- Training receipt:
  8e81b888ea0e93b5ae4f420064f10bdc54446a81c564ef90ababc23be192e5f3.
- 254 optimizer updates over two epochs; best validation NLL
  68.69400057356181; fresh-process checkpoint restoration passed inside the
  same allocation.

### Two-batch synthesis diagnostic

- Job 314347 completed on talon17 in 00:01:00 with exit code 0:0.
- Class 0, batches 0 and 1, deterministic sampling seeds 223251 and 223252;
  sixteen images (eight per batch) at canonical 40x40x3 float32 in [0, 1].
- Batch archives:
  94ae65e8c2f4ebc3fb46daf9298649c02d1b34c94a1836b8806ec7595d7f00e4 and
  4c5337ca55b793c022f122408db3abd92d0ef272f84bd391462df6a09a7c0486.

### Full synthetic pool

- Synthesis array 314349 (five class tasks; 00:16:26 to 00:23:46 per task) and
  its afterok finalizer 314354 completed with exit codes 0:0.
- Pool: five classes, 1,250 batches, 10,000 images, 2,000 per class; sampling
  seeds 223251-224500 with no collisions; streaming preflight verified 1,250
  batches and 10,000 images.
- Pool manifest:
  c5fc5cb6aa7732b356b2bb57bfbd28d58f05b771ef50f278456111673fed0cb5.
- Finalization report:
  23463ca4e13bf660ae9550d9f432568f1fd26720f60e3b9172c7cd1d6cadd36c.

Pool root:
`~/trustforge_revision_2026_artifacts/paper01/maf/cicmaldroid2020/seed44/pool_8198dc4-Qm4Rt7`

### Paired evaluation

- Job 314355 completed on talon07 in 00:18:09 with exit code 0:0.
- Training and restoration ran in separate processes (PIDs 1387829 and
  1390171) inside one allocation; the restore phase authenticated the training
  receipt by SHA-256 before evaluating the held-out test split.
- Training receipt, job 314355:
  502785727e350da509787da5dd6d8f2e904750cd9ac0fdd52481ca3fa4765c25.
- Both arms started from shared initial weights
  3862840721ffcecf79d3433ef58fb55bb822425104bc93c2f3ae8d4560404749.
- Both arms trained 20 epochs (stopping outcome maximum_epochs) and selected
  epoch 19: 1,280 optimizer updates real-only, 2,840 augmented; selected
  validation NLL 0.8362294495208688 and 0.6761579209864654.
- Selected classifier states:
  5f8666a570cdabd86dfe3a13bdd040c32900f582ea9adbe6aee2c68cdf09895a and
  017796057ca88ef7e4866004571e36053ab6999eff97bd1a652ec069582e9b6c;
  both restorations verified their selected-weight assignments.
- Augmented consumption: 200,000 synthetic examples across 20 epochs
  (10,000 per epoch) against the accepted seed-44 pool.
- Test predictions: 2,323 held-out examples per arm with five-class coverage.
- The previously reported augmented initial-weight prefix 38628472 was a
  summary transcription error, not an artifact difference; the persisted
  digest equals the real-only digest exactly, and the pairing is enforced by
  three independent accepted checks.

Run directory:
`~/trustforge_revision_2026_artifacts/paper01/maf/cicmaldroid2020/seed44/bounded_eval-c56a02d-Tq7Md4`

Evidence hashes:
- evaluation.json:
  6cf1c1995cd3ae396765631b9029576d141865830b451e5883572904359940cc
- train_report.json:
  757c4a88a844c5a5cd4d5596b84fd297e919c91e08339e6e91b5281dc7883d67
- restore_report.json:
  a68d6f34940c58b08c522aa80bf37efc9b60a53278f08ff6f25ebfee17708deb

### Accepted performance evidence

| Metric | Real-only | Real + synthetic |
|---|---:|---:|
| Macro-F1 | 0.6769984997 | 0.7380814152 |
| Balanced accuracy | 0.6673093749 | 0.7295561666 |
| Macro-AUPRC | 0.7602651263 | 0.8212323940 |
| Brier | 0.4007243583 | 0.3231987426 |
| ECE | 0.1088318296 | 0.0746388359 |

Primary paired endpoint:
augmented_minus_real_only_macro_f1 = 0.06108291546148226

Supporting fidelity/diversity results: pooled class-balanced KID
1.0182850786 (10 draws, 125 per class); pooled balanced FID 594.9966250;
macro class FID 604.2904967; three-scale intra-class MS-SSIM 0.0200221144
(1,000 selected pairs).

### Unified validation scope

- Required verified result groups: 7 of 7.
- Required status blockers: 0; lineage blockers: 0.
- Groups: real_only_downstream, real_plus_synthetic_downstream,
  paired_downstream, pooled_balanced_kid, pooled_balanced_fid, class_fid,
  intra_class_ms_ssim.
- Augmented pool consumption was verified against the seed-44 manifest.
- complete_evaluation=false.

### Cross-seed context

The three accepted bounded seed cells all show a positive paired macro-F1
difference under the unchanged protocol: seed 42 = 0.05451150907476743;
seed 43 = 0.09277561208032425; seed 44 = 0.06108291546148226. Three seeds do
not complete the seven-seed study and support no cross-seed statistical-
significance claim.

### Preserved boundaries

- Generator lineage is DRIVER_VERIFIED; generator_lineage_authenticated
  remains false, semantic class fidelity remains NOT_ESTABLISHED, and
  complete_evaluation and full_benchmark_go remain false.
- Unified validation records feature_extraction_authentication,
  image_score_recomputation, provenance_validation and publication_validation
  as NOT_PERFORMED; successful pool finalization is not independent semantic
  validation.
- This applies only to the evaluated dataset, family, seed and classifier
  configuration. Other families, datasets, seeds and full-benchmark execution
  remain open.
- The accepted seed-42 and seed-43 acceptance entries above are preserved
  unchanged.

S2-04 / Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO

## Research-direction governance clarification — 2026-11-15

Append-only governance entry. It records a direction-control clarification and
introduces no new scientific requirement, mandatory gate, experiment, or
acceptance criterion. No earlier entry is rewritten and no stage is
retroactively closed.

### Recorded continuity failure

The governing documents already existed, but successive execution results
caused the next technically possible experiment to be authorized as if it were
the next required one; each successful MAF seed was followed by authorizing the
numerically next seed. Technical possibility and numerical adjacency were
mistaken for authority. The researcher identified this as a continuity failure
that must be corrected in the durable record.

### Direction control

- Research direction must follow the existing research continuity charter, the
  live benchmark specification, accepted reviewer-driven decisions, and the
  original nine-stage readiness plan with its original exit conditions.
- After each execution result, the applicable requirement or gate must be
  identified before any further action is proposed (charter Section
  "Research-direction control and prevention of execution-driven drift").
- OpenCode remains an implementation executor; execution output does not create
  new scientific direction.

### Accepted bounded evidence

- MAF seeds 42, 43 and 44 are accepted bounded cells under their recorded
  entries.
- Seed 44: generator job 314269; diagnostic 314347; synthesis array 314349;
  finalizer 314354; paired evaluation 314355; evaluation report
  6cf1c1995cd3ae396765631b9029576d141865830b451e5883572904359940cc;
  paired macro-F1 difference 0.06108291546148226; shared initial weights
  3862840721ffcecf79d3433ef58fb55bb822425104bc93c2f3ae8d4560404749.
- Three accepted MAF cells are evidence for those cells only. They do not
  establish cross-seed significance, benchmark-wide readiness, or readiness of
  other generator families or datasets.

### Direction consequence

- Continuation to MAF seed 45 is not automatically justified or authorized.
  It is neither mandated nor forbidden by this entry; it requires a governing
  requirement and explicit researcher authorization like any other execution.
- The next conversation must reconcile actual readiness status and mandatory
  blockers against the original stage exit criteria rather than continuing the
  seed sequence.
- Full benchmark launch remains unauthorized.

### Reference

The research continuity charter was extended on 2026-11-15 with the
direction-control section above; updated charter SHA-256
5399545df1707f075e0381069eea73e2fae97378597330b29e5d73c659f63418
(228 lines). The earlier ledger reference recorded the 2026-11-13 charter
revision and remains historical. A new-conversation navigation aid is provided
at studies/paper01_benchmark/revision_2026/next_conversation_handoff_2026-11-15.md;
it is not a substitute for this ledger or the specification.

Stages remain at their previously recorded statuses. No gate is closed by this
entry.

FULL_BENCHMARK_GO=NO

## S2-02 bounded synthesis-linkage implementation — 2026-11-15

Local implementation record for independent review. This entry records a
bounded Stage 2 documentation/configuration/linkage package. It ran no
training, synthesis, Slurm job, checkpoint publication or synthetic-pool
publication. It does not declare S2-02 or Stage 2 complete.

### Researcher-approved controls transcribed

- GAN: terminal checkpoint; conditional DCGAN; 100-dimensional standard-normal
  latents; class-conditioned generation; existing canonical converter.
- VAE: validation-best checkpoint; standard-normal prior; class-conditioned
  decoder mean; deterministic canonical conversion; no Gaussian observation
  noise; last checkpoint retained for provenance.
- Diffusion: validation-best checkpoint; accepted 1,000-step DDPM and
  reverse-posterior mathematics; existing per-sample noise identity;
  model-call batch size one; last checkpoint retained for provenance.
- GAN/VAE latent identity: `trustforge.paper01.synthesis-latent-stream.v1`
  with explicit `SeedSequence`, `PCG64`, `Generator` and domain/identity
  entropy ordering.
- Initial synthesis batch settings: GAN 16, VAE 32, Diffusion 1. GAN/VAE values
  are provisional; no measured optimum is claimed. Resource verification remains
  Stage 3.

### Bounded implementation evidence

- Recipe:
  `studies/paper01_benchmark/revision_2026/synthesis_linkage_recipe.json`
- Linkage module:
  `src/trustforge/models/paper01_revision_2026/synthesis_linkage.py`
- Focused local tests:
  `tests/test_paper01_synthesis_linkage_revision_2026.py`

The module validates recipe controls, derives the approved GAN/VAE latent
identities before model-call batching, wraps the existing terminal GAN loader
and VAE best-checkpoint loader, reuses the existing Diffusion sampler and
canonical converters, validates generated batches, and validates complete-pool
metadata without generating or publishing a pool.

### Verification boundary

Focused local tests cover configuration, RNG identity, request/checkpoint-role
validation, GAN fixture generation, Diffusion fixture generation, output
validation and complete-pool metadata rejection. Local result:
`60 passed, 1 skipped`. The skipped case is VAE decoder execution, locally
BLOCKED because TensorFlow is unavailable in the local test environment.
The local golden-vector regression passed under NumPy 2.4.2; the recorded
NumPy 2.2.6 baseline is unavailable locally, so pinned-version golden replay
remains UNVERIFIED. No production checkpoint was loaded.

Diffusion durable checkpoint publication/loading remains a Stage 4 dependency.
The current Diffusion checkpoint state remains unpublished and in-memory.

### Status

`S2_02_SCIENTIFIC_POLICY=ACCEPTED`

`S2_02_EXECUTABLE_LINKAGE=LOCAL_IMPLEMENTATION_PENDING_INDEPENDENT_REVIEW`

`S2_02_SYNTHESIS_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Stage 2 remains IN_PROGRESS.

## Stage 2 S2-03/S2-04 researcher acceptance and closure — 2026-11-15

Status: RESEARCHER ACCEPTANCE RECORD. This entry records the researcher's
explicit decision to accept and close S2-03 and S2-04 at their original Stage 2
conditions. The preceding S2-03/S2-04 closure reconciliation was PROPOSED at the
time; this later decision is what formally accepts and closes both requirements.
Earlier entries, statuses and proposal wording are preserved unchanged.

### A. Original Stage 2 closure conditions

- S2-03 — CIC fitted preprocessing identity: "Record fitted scaler state bound
  to source, ordered features, training membership, implementation and
  environment."
- S2-04 — Shared evaluator configuration: "Map accepted classifier, metrics and
  statistical settings to production evaluator consumers."

These are the acceptance conditions assessed as satisfied in the read-only
reconciliation. No additional scientific requirement is introduced here.

### B. S2-03 — researcher acceptance

Disposition: ACCEPTED_AND_CLOSED.

The accepted evidence establishes a recorded StandardScaler fitted state bound
to source and configuration, 139 ordered features, 8,117 ordered real-training
indices, implementation/source provenance, environment identity, independent
fitted-state restoration, and bounded CIC/MAF consumer linkage.

Evidence paths:

- src/trustforge/data/paper01_revision_2026/cic_fitted_state.py
- scripts/check_paper01_cic_fitted_state.py
- src/trustforge/models/paper01_revision_2026/ar_maf_cic_execution_manifest.py
- studies/paper01_benchmark/audits/revision_2026/cic_fitted_state_311667
- studies/paper01_benchmark/audits/revision_2026/maf_cic_execution_311675
- studies/paper01_benchmark/audits/revision_2026/maf_cic_envelope_synthesis_311680

Recorded evidence identities (distinct, not interchangeable):

- Fitted-state capture payload SHA-256:
  fd4ccde26974243f469ad569978fce181a126ae01795957b95a9447220c32fbd.
- MAF execution fitted-state payload SHA-256:
  c177c31e8602e94b06dc965e14dafe8ec3374fd07816b93107df3f5c1fc7f5cf.
- Dataset configuration SHA-256:
  ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b.
- Capture source identity: e81323282a2662d775f1cee7ea7bd2730ab1184b.

The complete historical artifacts were not re-executed during this ledger
update. The original fitted-state provenance condition is satisfied. This
acceptance does not claim that every generator consumes the fitted state or that
all production runs are complete.

### C. S2-04 — researcher acceptance

Disposition: ACCEPTED_AND_CLOSED.

Classifier configuration:

- studies/paper01_benchmark/revision_2026/classifier_recipe.json
- SHA-256 1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982
- Consumers: src/trustforge/evaluation/paper01_classifier_control.py;
  src/trustforge/evaluation/paper01_classifier_tensorflow.py

Metric configuration:

- studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json
- SHA-256 d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85
- Consumers: src/trustforge/evaluation/paper01_downstream_metrics.py;
  src/trustforge/evaluation/paper01_paired_results.py

Statistical reporting:

- src/trustforge/evaluation/paper01_seed_reporting.py
- Accepted rules include experimental seeds 42-48, paired real-only versus
  real-plus-synthetic comparison, held-out real-test evaluation, explicit
  missing/incomplete-run statuses, seven-pair statistical reporting conditions,
  and no inference from incomplete seven-seed results.

Accepted bounded evaluator consumption:

- CIC/MAF seed 42 — job 314068.
- CIC/MAF seed 43 — job 314240.
- CIC/MAF seed 44 — job 314355.

These records demonstrate bounded paired evaluator consumption. They do not
establish full benchmark completion, Stage 5 completion or publication-ready
statistical results.

Known limitation preserved: cross-seed aggregation is implemented and
acceptance-tested but is not yet invoked by a production runner. This is a
later-stage execution/integration concern, not an unresolved original S2-04
linkage condition.

### D. Boundaries

Proven: the original S2-03 provenance record and the original S2-04
configuration-to-consumer mapping, with bounded execution support.

Not proven or not claimed: complete benchmark execution, seven-seed statistical
results, all-family or all-dataset consumer adoption, production provenance
integration for every checkpoint entry point, publication readiness, or any
Stage 3/4/5 completion.

### E. Current five-item Stage 2 register

- S2-01 — RBM production Gibbs depth: BLOCKED. The prospective Gibbs-depth
  scientific decision remains unresolved.
- S2-02 — GAN/VAE/Diffusion synthesis recipes: ACCEPTED_AND_CLOSED; preserved at
  01ab459307c5314d31acbbefdea4676892134560.
- S2-03 — CIC fitted preprocessing identity: ACCEPTED_AND_CLOSED; researcher
  closure decision recorded in this entry.
- S2-04 — Shared evaluator configuration: ACCEPTED_AND_CLOSED; researcher
  closure decision recorded in this entry.
- S2-05 — Complete benchmark configuration linkage: ACCEPTED_AND_CLOSED;
  preserved at 1025f2ec52d9f8350145c40d6d3433b550758388.

S2-01 is the only unresolved requirement in the original five-item Stage 2
register. Stage 3, Stage 4 and Stage 5 statuses remain unchanged. Closure of
configuration requirements does not establish execution readiness.

### F. Stage status

STAGE_2_OVERALL=IN_PROGRESS

FULL_BENCHMARK_GO=NO

SEED45_EXECUTION_AUTHORIZED=NO

## CIC GAN seed-42 bounded evaluation acceptance — job 314607

PASS for one bounded CICMalDroid2020 / GAN / experimental seed-42 real-data
paired evaluation at evaluation source
49d6f5c2b7a31e458b97fe1ebe2b45ffab92ba74 and generator/synthesis source
4324aeecb45a47a72ac0b479d8cd994fe79dc1e0.

Evaluation authorization: P1R2026-GAN-CIC-S42-EVAL-A01

Authorization SHA-256:
100439a0cee3ee550599ae328b73551dd49c708aee7ce3c0b4682f13e1e24fec

Launcher SHA-256:
c7e959d4a2ed8100a0b34aac46c4428b0729a547078dc99c2a77cf0b5c437643

Driver SHA-256:
f1efa4379d3ca09530ff28d64a5fd36506acb36d5eedd166a33abe1510a0e757

```
GAN_EVAL_LAUNCHER_BINDING=PASS
CIC_GAN_SEED42_EVALUATION_PREFLIGHT=PASS
CIC_GAN_SEED42_EVALUATION_RESTORE=PASS
CIC_GAN_SEED42_BOUNDED_EVALUATION=PASS
GAN_CIC_SEED42_EVALUATION_LAUNCHER=PASS
```

Job 314607 completed on talon11 in 00:23:08 with exit code 0:0 and State:
COMPLETED under the approved envelope of 8 CPUs, 32 GB, CPU only, a two-hour
maximum and no automatic retries. Training and restoration ran as separate
processes (PIDs 2641100 and 2643537); restoration began only after the training
receipt existed and its SHA-256 had been captured successfully.

### Evaluation identity

- Paper: Paper 1 Revision-2026.
- Dataset: cicmaldroid2020; generator: GAN; experimental seed: 42; attempt: 1.
- Classes: 0, 1, 2, 3, 4.
- Evaluation source: 49d6f5c2b7a31e458b97fe1ebe2b45ffab92ba74.
- Generator/synthesis source: 4324aeecb45a47a72ac0b479d8cd994fe79dc1e0.

### Accepted synthetic pool

- Five classes; 2,000 images per class; 10,000 synthetic images; 625 production
  batches.
- Pool manifest SHA-256:
  b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1.
- Generator checkpoint SHA-256:
  963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530.
- Generator recipe SHA-256:
  98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9.

### Evaluation controls

- Classifier recipe SHA-256:
  1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982.
- Metric recipe SHA-256:
  d6fca7bf2421f34bafc43621595d7c2c218b17d2e0df891bd78d298549eb3a85.
- Real training samples: 8,117; augmented training samples: 18,117;
  validation samples: 1,158; held-out real test samples: 2,323 per arm.
- Shared initial classifier weights SHA-256:
  4e5a33f021164e0d148bba5f058d0dc01a0cd8ca48ad3f0b7c07f9013e7a19ba.
- Completed augmented epochs: 20; selected augmented epoch index: 19;
  synthetic examples per augmented epoch: 10,000.

### Restoration

- Training PID: 2641100; restoration PID: 2643537.
- Training receipt SHA-256:
  4610ac53f266811c741c0e7db5dee6f35e21bbac077f92406a66063cce030969.
- Real-only selected weights SHA-256:
  33ad5a3c0f9dfabc8ad5f8178fb4a112bfc3bb163064e5b6ad338f6dcab54586.
- Augmented selected weights SHA-256:
  6abf51b8593019aabcbef53e1125461d4f2d058ac2329fd01580ed0641a3e04a.
- Both selected-weight assignments verified; separate training/restoration
  processes verified; receipt identity linkage passed.

### Evaluation artifacts

Artifact root:
`~/trustforge_revision_2026_artifacts/paper01/gan/evaluations/cicmaldroid2020/seed42/runs/attempt01`

- expectations.json:
  4610ac53f266811c741c0e7db5dee6f35e21bbac077f92406a66063cce030969
- train_report.json:
  37c978b1e2859e6012ca65ae1b6cfba50b1b1af442248731837b2b2e540dbbc7
- restore_report.json:
  9ad1e48118551a62155ac25da0d58e092f15e007d2d56bff47eca6a5d28d515e
- evaluation.json:
  1070c889748e612de4ee3d8218b433af0af4445ed25fff86c9804e7a2b402624
- paired_downstream.json:
  f7b33b8ebe3e6049edb27923658390cfb0273138f8cc707d72480c0c9b5d0d42
- pool_consumption.json:
  4bb8dc5d79ccecc41e5a688379a9a69a036c37b98dbcf19a89079ef8a94d59fa

### Accepted performance evidence

| Metric | Real-only | Real+Synthetic |
|---|---:|---:|
| Macro-F1 | 0.646541025275839 | 0.6856195221624064 |
| Balanced accuracy | 0.6362720813251682 | 0.6682172717808181 |
| Macro-AUPRC | 0.7401845205982388 | 0.8171141940261919 |
| Macro recall at 1% FPR | 0.36582116465732895 | 0.5088252777870776 |
| Brier score | 0.423329924895245 | 0.3470150440937509 |
| ECE | 0.1416111566668382 | 0.051022664398865804 |

Primary paired endpoint:
augmented_minus_real_only_macro_f1 = 0.03907849688656739

Observed improvement of 3.90785 percentage points in macro-F1 for this single
bounded seed. This does not establish statistical significance.

### Unified validation scope

- Required verified result groups: 7 of 7.
- Required status blockers: 0; lineage blockers: 0.
- Groups: real_only_downstream, real_plus_synthetic_downstream,
  paired_downstream, pooled_balanced_kid, pooled_balanced_fid, class_fid,
  intra_class_ms_ssim.
- Paired metric replay verified; augmented checkpoint bundle verified;
  synthetic pool consumption verified.
- complete_evaluation=false.

### Preserved boundaries

- This is one successful bounded evaluation, not full benchmark completion,
  independent semantic label-fidelity verification or seven-seed statistical
  aggregation.
- Generator lineage is DRIVER_VERIFIED; semantic class fidelity remains
  NOT_ESTABLISHED.
- Unified validation records feature_extraction_authentication,
  image_score_recomputation, provenance_validation and publication_validation
  as NOT_PERFORMED. These statuses must not be silently promoted to VERIFIED.
- A successful single seed does not establish statistical significance or the
  superiority of GAN over other generator families.
- complete_evaluation=false; full_benchmark_go=false.

### Researcher disposition

The researcher accepted and closed this specific bounded
GAN/CICMalDroid2020 seed-42 evaluation after review of the published evidence.
This is an accepted individual experimental cell. It does not close Stage 5 or
authorize further execution. No additional seed, generator, dataset, repeat,
Slurm submission or full-benchmark run is approved. Stage 2 S2-03 and S2-04
remain closed as previously accepted and are not reopened. Existing S2-01
(BLOCKED) and other stage statuses remain unchanged.

STAGE_2_OVERALL=IN_PROGRESS

FULL_BENCHMARK_GO=NO

## AR CICMalDroid2020 seed-42 bounded production synthesis acceptance — jobs 314793 / 314794

RESEARCHER ACCEPTANCE RECORD. This entry closes one bounded AR
CICMalDroid2020 seed-42 production-synthesis milestone after researcher review
of the reported execution, publication and verifier evidence. It does not close
Stage 4 overall, perform downstream evaluation, or authorize another dataset,
generator, seed or attempt.

### Accepted scope

- Dataset: cicmaldroid2020; family: AR PixelCNN; experimental seed: 42;
  attempt: 01; classes 0,1,2,3,4.
- Synthetic budget: 2,000 images per class; 10,000 images total; 8 images per
  batch; 250 batches per class; 1,250 batches total.
- Canonical output contract: float32 NHWC 40x40x3 in [0,1].
- This is production synthesis only. It is not a classifier-utility,
  semantic-label-fidelity, realism, distributional-quality or seven-seed
  statistical result. No downstream classifier evaluation is claimed or
  authorized.

### Source and scientific identities

- Synthesis source: d7199228b19f9d67d747ca06a31c02230af00fb9.
- Training source: e4473abcdbc53f0de3a01ebe48858f2f3c7b841a.
- Checkpoint generation: ar_cic_seed42.
- Checkpoint SHA-256:
  d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0.
- Dataset configuration SHA-256:
  ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b.
- Recipe SHA-256:
  202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478.

### Slurm execution evidence

- Worker array 314793: 50 expected tasks, array 0-49 with maximum concurrency
  two; 50 COMPLETED with exit code 0:0.
- Worker envelope: 8 CPUs per task, 32 GB per task, 90 minutes per task,
  CPU-only, no automatic retries, requeue disabled.
- Dependent finalizer 314794: COMPLETED, exit code 0:0, elapsed 00:00:31;
  8 CPUs, 32 GB, 30-minute walltime; dependency afterok:314793.
- Accounting and scheduler success are reported execution evidence; they do
  not by themselves establish completed synthesis.

### Authorization provenance

- Authorization ID: P1R2026-AR-CIC-S42-ATTEMPT01-20261118.
- Approved authorization record SHA-256:
  34c7549a8388f5fdec23ebba2698ff379f022d83d066d5faa60b0cd352ff5513.
- The preceding non-active preparation record remained unchanged.
- The active record was created separately for this exact bounded attempt;
  the researcher's explicit approval governs this execution.

### Published pool and manifest

Pool:
`~/trustforge_revision_2026_artifacts/paper01/ar/cicmaldroid2020/seed42/pools/production/attempt01/pool`

Manifest:
`<pool>/manifest.json`

Published manifest SHA-256:
ce2b6c6a332b578ad9f67091cf035f15a57e55e88eb22d9f98ade10ef94835e2.

The accepted finalizer logs report:

- batch_count=1250
- class_counts=[2000,2000,2000,2000,2000]
- images=10000
- pool_samples=10000
- publication_authorized=true

The subsequent existing pool verifier reported the same complete counts and
manifest identity. Finalizer stderr was empty in the supplied read-only
inspection.

### Coverage and verification scope

- Five classes; 2,000 images per class; 10,000 images; 1,250 batches.
- The accepted finalizer and existing verifier report complete coverage and
  manifest identity.
- This entry records execution success from researcher-provided Slurm
  accounting, the pool/manifest from researcher-provided filesystem output,
  and publication/verifier success from the accepted finalizer logs.
- Full independent manifest rehashing or batch-content replay was not
  performed during this documentation review. This entry does not claim a new
  independent verification run; the reported verifier outcome is preserved
  evidence, not a rerun. The absence of a rerun is not a verification failure.

### Researcher acceptance and closure

The researcher accepted the bounded AR CICMalDroid2020 seed-42 production
synthesis result after review of the reported publication and verification
evidence. This closes the bounded production-synthesis milestone only. It does
not accept classifier utility, semantic label fidelity, synthetic realism,
distributional quality or seven-seed statistics, and it does not authorize
downstream evaluation or another seed.

### Status

AR_CIC_SEED42_PRODUCTION_EXECUTION=COMPLETE

AR_CIC_SEED42_POOL_PUBLICATION=COMPLETE

AR_CIC_SEED42_POOL_VERIFIER=PASS_REPORTED

AR_CIC_SEED42_BOUNDED_SYNTHESIS=ACCEPTED

AR_CIC_SEED42_DOWNSTREAM_EVALUATION=NOT_PERFORMED

STAGE_4_OVERALL=IN_PROGRESS

STAGE_5_OVERALL=IN_PROGRESS

RBM_PRODUCTION_SAMPLING=BLOCKED

FULL_BENCHMARK_GO=NO

No other dataset, generator, seed or experimental attempt is authorized by
this entry. AR seed 43 is not authorized. Existing historical entries and
original stage exit criteria are preserved unchanged.

## VAE Stage 3 bounded launcher-integration acceptance — 2026-11-20

RESEARCHER ACCEPTANCE RECORD. The researcher accepted and closed the bounded
VAE Stage 3 launcher-integration milestone after reviewing the implementation
and the reported TensorFlow verification evidence. This is an accepted
implementation milestone only. It does not authorize operational training,
real-data execution, checkpoint publication, synthesis or evaluation.

### Accepted implementation scope

The reviewed implementation package contains four files:

- `src/trustforge/models/paper01_revision_2026/variational_autoencoder_governed_training.py`
  (modified): adds a minimal gate-only authorization helper and uses it in the
  existing governed training function. The helper reads the existing
  `VAE_REAL_DATASET_TRAINING_ENABLED` constant. It does not add a second
  activation flag or any environment, token, development-mode or test-mode
  bypass. Existing gate semantics are preserved.
- `scripts/train_paper01_vae_revision_production.py` (new): Revision-2026 VAE
  production CLI with explicit dataset, experimental-seed, attempt, expected
  source-commit, repository-root and dataset-source inputs. It requires Slurm,
  a clean checkout, the exact expected commit and Revision-2026 module paths.
  It invokes the gate before adapter construction and delegates only to
  `run_governed_vae_training`.
- `slurm/paper01_revision_2026/vae_revision_production.slurm` (new): inert
  production launcher specification. It requires explicit identity and source
  inputs, rejects an incomplete or mismatched checkout, disables automatic
  requeue and invokes only the new CLI. Its resource values are provisional
  implementation settings, not an accepted production resource envelope.
- `tests/test_variational_autoencoder_production_entry_point_revision_2026.py`
  (new): focused entry-point contract tests using fixtures and mocks.

The real-data gate remains disabled:

`VAE_REAL_DATASET_TRAINING_ENABLED = False`

### Accepted verification evidence

The accepted bounded verification was reported from an existing MacStudio
environment and reviewed by the researcher. It is not a new independent rerun
performed during this documentation update.

- Interpreter: `/Users/poisedre/TrustForge/.venv/bin/python`
- Python: 3.10.11
- TensorFlow: 2.20.0
- NumPy: 2.2.6
- pytest: 8.3.3

The interpreter was used read-only with:

`PYTHONPATH=/Users/poisedre/TrustForge-opencode/src:/Users/poisedre/TrustForge-opencode`

Imports were verified to resolve to this checkout, not to the interpreter's
own checkout.

| Module | Result |
| --- | --- |
| `tests/test_variational_autoencoder_production_entry_point_revision_2026.py` | 22 passed |
| `tests/test_paper01_experimental_seed_binding_revision_2026.py` | 36 passed |
| `tests/test_variational_autoencoder_governed_training_revision_2026.py` | 9 passed |
| `tests/test_variational_autoencoder_training_checkpoint_bridge_revision_2026.py` | 7 passed |

Totals:

- 74 passed
- 0 failed
- 0 skipped
- 0 expected failures
- 0 collection errors

The two previously skipped TensorFlow-dependent entry-point tests executed and
passed. These are researcher-reviewed OpenCode-reported test results, not a new
independently rerun verification. The checkpoint-bridge tests construct models
and serialize synthetic in-memory weights; they do not train, read real data or
publish checkpoints.

### Source and implementation identity

- Base source commit: `1d3ed01b06887c4e7fdf73b9eac2f6c65112e340`
- Branch: `engineering/paper01-opencode-stage5`

Reviewed implementation file SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `variational_autoencoder_governed_training.py` | `0ba540319b296b1fd27bd5c263ebea5eda51c1d93f5296b18343af2d0aa49f83` |
| `train_paper01_vae_revision_production.py` | `81938213b1d4190cecf5f57860e7dab91d5322710789ed127b22dcbb62252a98` |
| `vae_revision_production.slurm` | `2e150141c52421e7d38684dca8ee76031db64f9576e6e6d98b3ff264df01646a` |
| `test_variational_autoencoder_production_entry_point_revision_2026.py` | `d1d0bc63eda0868759b9450a430d871dea3815244e203879a4198f3d1d3d9ebf` |

The four files were uncommitted working-tree changes at the recorded base
commit when this entry was prepared. No preservation commit is recorded or
implied by this entry.

### Scientific controls preserved

The accepted implementation does not change:

- conditional convolutional VAE;
- canonical float32 NHWC `40 x 40 x 3` representation;
- latent dimension 100;
- fixed Gaussian observation variance 1.0;
- KL coefficient 1.0;
- no KL annealing;
- Adam optimizer with learning rate 0.001;
- training batch size 64;
- exactly two complete epochs;
- full governed validation after each epoch;
- earliest minimum validation-total-loss selection;
- distinct best and last checkpoint states;
- held-out test isolation from training and selection.

### Acceptance limitations

- No real dataset was accessed.
- No real-data production training was performed.
- No Slurm job was submitted.
- No checkpoint was published.
- No synthetic image pool was generated.
- No downstream classifier evaluation was performed.
- No full benchmark was authorized.

The accepted tests establish bounded integration correctness, not real-data
operational readiness. The production launcher has not been demonstrated
through an authorized Slurm execution. This accepted implementation milestone
therefore does not close Stage 3 overall.

### Stage boundaries

Stage 3 retains execution source/configuration identity, dataset provenance,
launcher integration, activation and fail-closed runtime behavior.

Stage 4 separately retains durable checkpoint publication, independent
checkpoint restoration, checkpoint-backed synthesis, complete class quotas and
pool publication. No Stage 4 requirement is moved into this accepted Stage 3
milestone.

### Status

VAE_STAGE3_LAUNCHER_INTEGRATION=ACCEPTED_AND_CLOSED

VAE_STAGE3_TENSORFLOW_VERIFICATION=PASS_REPORTED

VAE_RUNTIME_TRAINING_ACTIVATION=DISABLED

VAE_REAL_DATA_EXECUTION=NOT_PERFORMED

VAE_CHECKPOINT_PUBLICATION=NOT_PERFORMED

VAE_SYNTHESIS_EXECUTION=NOT_PERFORMED

STAGE_3_OVERALL=IN_PROGRESS

STAGE_4_OVERALL=IN_PROGRESS

RBM_PRODUCTION_SAMPLING=BLOCKED

FULL_BENCHMARK_GO=NO

No other dataset, generator, seed or experimental attempt is authorized by
this entry. This entry does not authorize staging, commit, push, operational
activation, real-data training, Slurm execution, checkpoint publication,
synthesis or evaluation. Existing historical entries and original stage exit
criteria are preserved unchanged.

## Diffusion Stage 3 bounded execution-entry integration acceptance — 2026-11-20

### A. Researcher acceptance

The researcher explicitly accepted and closed the bounded Diffusion Stage 3
execution-entry integration milestone after reviewing the implementation and
the reported 66 passing tests. This closes the bounded execution-entry
integration milestone only. It does not close Stage 3 overall.

### B. Exact implementation scope

The reviewed package contains three new files:

- `scripts/train_paper01_diffusion_revision_production.py`
- `slurm/paper01_revision_2026/diffusion_revision_production.slurm`
- `tests/test_diffusion_production_entry_point_revision_2026.py`

The existing governed Diffusion module was unchanged. No additional
authorization flag was added. No source-level gate was enabled. No training
implementation was duplicated. No model or DDPM mathematics changed. No
checkpoint or synthesis source changed.

### C. Accepted execution-entry controls

The bounded package implements and component-tests:

- explicit dataset, experimental-seed and attempt identity;
- explicit expected source commit;
- clean-checkout and Slurm requirements;
- expected-versus-actual source-commit verification;
- Revision-2026 module-path verification;
- exact `diffusion/models.py` builder binding and callable verification;
- the existing authorization gate checked before module binding, adapter
  construction and model building;
- retention of the governed function's own independent fail-closed gate;
- reuse of the existing dataset adapters and provenance utilities;
- reuse of the governed result's returned `scientific_configuration` as
  consumed execution evidence;
- exclusion of the legacy Diffusion training path;
- no automatic retry, resume or requeue;
- no checkpoint publication or synthesis path.

These are implementation and component-test evidence. They were not validated
through an actual authorized Talon production job.

### D. Verification evidence

The previously reported MacStudio verification used the existing environment:

- Interpreter: `/Users/poisedre/TrustForge/.venv/bin/python`
- Python: 3.10.11
- TensorFlow: 2.20.0
- NumPy: 2.2.6
- pytest: 8.3.3

The interpreter was used with `PYTHONPATH` bound to
`/Users/poisedre/TrustForge-opencode/src:/Users/poisedre/TrustForge-opencode`.

| Test group | Result |
| --- | --- |
| New Diffusion entry-point tests | 30 passed |
| Existing Diffusion regression tests | 36 passed |
| **Total** | **66 passed** |

Totals:

- 0 failed
- 0 skipped
- 0 collection errors

Python compilation and Bash syntax checks were reported as PASS_REPORTED.
These are accepted OpenCode-reported results reviewed by the researcher. No
tests were rerun in this preservation-preparation task.

### E. Procedural test-scope deviation

The approved implementation instructions prohibited model training, including
synthetic-data training. During verification, OpenCode ran the existing
Diffusion regression modules. Those regression tests included tiny synthetic
optimizer-step operations. The optimizer-step operations occurred as part of
the already existing regression tests, not through the new production entry
point.

No real-data training, production training, production Slurm execution or
generator training experiment was reported. This was a limited deviation from
the specified test-execution boundary. It must not be described as complete
compliance with a prohibition on all optimizer updates.

The researcher reviewed this disclosure and accepted the bounded integration
milestone with the deviation documented. This entry does not claim that the
researcher separately authorized those optimizer steps beforehand. No tests
were rerun and no additional work was created to retroactively eliminate the
deviation.

### F. Source and file provenance

- Branch: `engineering/paper01-opencode-stage5`
- Base HEAD: `3691bb57d4c58db66bb31ffe3e246679f8e6c8ee`

Implementation file SHA-256 identities verified during preflight:

| File | SHA-256 |
| --- | --- |
| `train_paper01_diffusion_revision_production.py` | `bad3afa47ecf7acaac97f950b74741d95ed18706b15e95bb5ebdeb19ddbf5e16` |
| `diffusion_revision_production.slurm` | `3cd3f9f97a1dfb3e931f7c0c446be6b1b475a1784b4de3aead7fc4f111e68749` |
| `test_diffusion_production_entry_point_revision_2026.py` | `5f414ebe76715f9d33e625ac985dd697a69186037189e16565f8109a1e6bd248` |

The files remain uncommitted and unpushed at this preservation-preparation
stage. No future Git commit ID is recorded or implied.

### G. Scientific controls preserved

The accepted implementation does not change:

- conditional convolutional U-Net;
- canonical `40 x 40 x 3` input;
- depth 2 and base filters 64;
- time embedding dimension 128;
- linear 1,000-timestep DDPM schedule;
- beta range 0.0001 to 0.02;
- epsilon-prediction MSE;
- Adam optimizer, learning rate 0.0002, beta_1 0.9;
- batch size 64;
- exactly two complete epochs;
- no validation-driven early stopping;
- full training and validation traversal;
- sample-weighted validation MSE;
- earliest minimum-validation-MSE checkpoint selection;
- distinct best and last in-memory states;
- held-out test isolation.

### H. Acceptance limitations

- No real dataset was accessed.
- No real-data production training occurred.
- No production Slurm job was submitted.
- No production checkpoint was prepared.
- No durable checkpoint was published.
- No production synthesis was performed.
- No downstream classifier evaluation occurred.
- No full benchmark was authorized.

These statements are qualified by the documented synthetic optimizer-step
regression-test deviation. This entry does not make the unqualified statement
that no training or optimizer computation occurred. The implementation was
not validated through an actual authorized Diffusion production Slurm
execution.

### I. Stage boundaries

Stage 3 concerns source, configuration, dataset and import identity,
controlled activation, execution entry and failure behavior.

Stage 4 separately concerns durable checkpoint publication, independent
restoration, checkpoint-backed synthesis and complete synthetic pools.

Stage 8 separately concerns production resource and concurrency freeze and
full benchmark launch authorization.

No Stage 4 or Stage 8 requirement is satisfied or closed by this entry.

### J. Final status

DIFFUSION_METHOD_DESIGN=APPROVED

DIFFUSION_TRAINING_IMPLEMENTATION=ACCEPTED

DIFFUSION_STAGE3_ENTRY_INTEGRATION=ACCEPTED_AND_CLOSED

DIFFUSION_STAGE3_ENTRY_TESTS=66_PASS_REPORTED

DIFFUSION_TEST_SCOPE_DEVIATION=DOCUMENTED

DIFFUSION_RUNTIME_TRAINING_ACTIVATION=DISABLED

DIFFUSION_REAL_DATA_EXECUTION=NOT_PERFORMED

DIFFUSION_CHECKPOINT_PUBLICATION=NOT_PERFORMED

DIFFUSION_SYNTHESIS_EXECUTION=NOT_PERFORMED

STAGE_3_OVERALL=IN_PROGRESS

STAGE_4_OVERALL=IN_PROGRESS

RBM_PRODUCTION_SAMPLING=BLOCKED

FULL_BENCHMARK_GO=NO

No other dataset, generator, seed, attempt or experiment is authorized by
this entry. Existing RBM, GAN, VAE, AR, MAF and GMM acceptances are unchanged.
This entry does not authorize staging, commit, push, runtime activation,
real-data training, Slurm execution, checkpoint publication, synthesis,
evaluation or full benchmark launch.

## VAE Stage 3 — CICMalDroid2020 Seed-42 Bounded Real-Data Training Acceptance — Slurm 316716

### Execution authority

- Dataset: `cicmaldroid2020`
- Generator: VAE
- Experimental seed: 42
- Attempt: 01
- Run identity: `commit9eb426b_attempt01`
- Source commit: `9eb426b551f2bd4f07eac6eeac60027e2b9f64ca`
- Execution checkout: `/home/bruno.fonkeng/TrustForge-paper01-vae-stage3-activated`
- Slurm job: 316716
- Compute node: `talon07`

### Observed completion

- Job state: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:02:32`
- Training epochs: 2 expected / 2 completed
- Optimizer updates: 254 expected / 254 completed
- Governed training split: 8,117 examples
- Governed validation split: 1,158 examples
- Governed test split: 2,323 examples
- Expected and actual source commit reported as identical:
  `9eb426b551f2bd4f07eac6eeac60027e2b9f64ca`
- Source status reported as `CLEAN`.
- Imported Revision-2026 VAE modules reported under the activated checkout:
  data integration, governed training, model, observation and training
  runner.
- Configuration identity checks reported as passing for the accepted
  conditional VAE, canonical 40x40x3 input, latent dimension 100,
  observation variance 1.0, KL coefficient 1.0, no KL annealing, Adam
  learning rate 0.001, batch size 64, two complete epochs, full governed
  validation, minimum validation-total-loss selection with earliest-epoch
  tie-breaking, distinct best/last roles, and prohibited test-split usage.

### Resources

- 8 allocated CPUs
- 32 GB requested memory
- 2-hour walltime limit
- Approximately 821 MiB reported batch-step MaxRSS (`840480K`)
- Compute node `talon07` in the `talon` partition
- GPU disabled and unused
- No automatic requeue or retry

### Data provenance

- CIC authoritative source contract:
  `cic_authoritative_source_revision_2026`
- Compute-node full-file SHA-256 source verification method:
  `sha256_full_file_stream_on_compute_node`
- Verification scope: authoritative CSV and feature-column JSON
- Source fingerprints reported:
  - `source_csv`:
    `3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00`
  - `feature_columns_path`:
    `7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f`
- Governed Revision-2026 partition policy: `cic_case3_revision_2026`,
  partition seed 42, train 0.7 (8,117), validation 0.1 (1,158), test 0.2
  (2,323).
- The held-out test split was excluded from training and selection by the
  governed configuration (`test_split_usage=prohibited`).

### Evidence limitations

- Individual per-epoch validation losses were not exposed in stdout.
- The selected best epoch was not separately printed.
- Per-epoch coverage counters were not separately printed.
- The successful governed function return supports its internal completion
  and selection checks, but does not independently expose every intermediate
  quantity.
- Source-file provenance verification does not by itself prove continuous
  source immutability or independently trace every consumed training batch.
- The governed provenance additionally records that it does not authenticate
  checkpoint origin, does not prove training used only verified bytes, and
  does not independently verify partition execution.

### Publication restrictions

- Checkpoint status: `NOT_PREPARED`
- Publication status: `NOT_PUBLISHED`
- No authoritative checkpoint publication
- No synthetic-image generation
- No downstream classifier evaluation

### Log provenance

- `~/p1r1-vae-prod.316716.out`
  - SHA-256:
    `08d99f0955ed8231bd7d0e8d181922a608e5bf960d1354a55dbf2d1ff17f4d67`
- `~/p1r1-vae-prod.316716.err`
  - SHA-256:
    `4f4f6e3b834fc93d674fe8c4c2d4838afbf35027d77d985e57fdbbec0e75d812`

These are OpenCode-reported, command-backed execution observations reviewed
by ChatGPT. They are not described as an independently rerun Talon
verification.

### Status and boundary

- `VAE_STAGE3_CIC_SEED42_TRAINING_ONLY=ACCEPTED_AND_CLOSED`
- `VAE_STAGE4_CHECKPOINT_SYNTHESIS=NOT_ESTABLISHED`
- `STAGE_3_OVERALL=IN_PROGRESS`
- `FULL_BENCHMARK_GO=NO`

This bounded Stage 3 training-only acceptance does not authorize further
seeds, attempts, datasets, generator families, checkpoint publication,
synthesis or evaluation. Historical Stage 2–5 records and original stage exit
conditions remain unchanged. No other dataset, generator, seed, attempt or
experiment is authorized by this entry.

## VAE Stage 4 — Candidate Checkpoint-Publication Integration Acceptance — 2026-11-21

### A. Scope and source identity

- Dataset context: `cicmaldroid2020`.
- Generator: conditional VAE.
- Experimental seed context: 42.
- Accepted Stage 3 training-only job: 316716.
- Stage 3 activation source:
  `9eb426b551f2bd4f07eac6eeac60027e2b9f64ca`.
- Earlier Stage 3 evidence-preservation commit:
  `2ca154225feb89343a81f4924920886dd23ee9e3`.
- Stage 4 implementation commit:
  `99d7f60fe60c4697308024216b49fbe454e8a486`.

Seed 42 is the accepted Stage 3 context and synthetic-fixture identity. No
new real-data training, production checkpoint publication or production
synthesis occurred in this Stage 4 integration.

### B. Accepted implementation

The accepted integration adds:

- a governed training-result to checkpoint-preparation coordinator;
- distinct best and last checkpoint artifact preparation;
- reuse of the existing VAE generation/commit publisher;
- committed-generation verification before a checkpoint directory is exposed;
- synthetic-fixture independent checkpoint restoration;
- the approved observation-version identity
  `gaussian_fixed_variance_1p0_v1`;
- a default-off checkpoint-publication CLI;
- a separate, default-deny authoritative publication authorization;
- preservation of the existing scientific and checkpoint contracts.

The integration is candidate checkpoint-preparation/publication orchestration.
It does not establish authoritative checkpoint publication.

### C. Test evidence

Previously reported and reviewed OpenCode execution evidence:

- Focused coordinator and authorization-boundary module: 16 passed.
- Targeted regression suite excluding two historical pre-activation
  assertions: 201 passed, 2 deselected.
- Unfiltered targeted regression suite: 201 passed, 2 failed.

The two retained historical failures are:

- `test_gate_remains_disabled_in_source`
- `test_disabled_gate_helper_raises`

Both expect the previously disabled VAE training gate, whereas the accepted
Stage 3 activation deliberately enabled that gate. They are not represented
as passing. These results were not rerun during this Git-preservation task.

### D. Remaining limitations

- Authoritative checkpoint publication remains unauthorized.
- Production synthesis remains unauthorized.
- No production checkpoint was created for job 316716.
- Real-data publication and restoration have not been accepted.
- GPFS crash durability remains unverified.
- `crash_durability_verified=False`.
- No production synthetic pool or downstream evaluation was performed.
- Existing loader and synthesis identity limitations remain disclosed.
- This integration does not establish full Stage 4 or full benchmark
  readiness.

### E. Stage dispositions

- `VAE_STAGE3_CIC_SEED42_TRAINING_ONLY=ACCEPTED_AND_CLOSED`
- `VAE_STAGE4_CANDIDATE_PUBLICATION_INTEGRATION=ACCEPTED_AND_CLOSED`
- `VAE_OBSERVATION_VERSION=APPROVED`
- `VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED=NO`
- `VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`
- `STAGE_4_OVERALL=IN_PROGRESS`
- `FULL_BENCHMARK_GO=NO`

No other dataset, generator, seed, attempt or experiment is authorized by
this entry. Existing Stage 2–5 records and original stage exit conditions
remain unchanged.

## VAE Stage 4 — Single-Execution Permit Validator Implementation Acceptance — 2026-11-21

### Scope and disposition

This entry records researcher acceptance of the bounded VAE Stage 4
single-execution checkpoint-publication permit validator implementation.

`VAE_STAGE4_PERMIT_VALIDATOR_IMPLEMENTATION=ACCEPTED_AND_CLOSED`

Implementation commit:

`266a3c785a2458a0b148141328c3230efe025c33`

Parent commit:

`713ce08534c63ab24c2755267f6ca7cd67d49ef1`

The implementation adds only the permit-validation component and its
production CLI integration. It does not authorize authoritative checkpoint
publication, a new training attempt, production synthesis, or a new
experimental execution.

### Accepted custody model

The researcher accepts same-account operational custody under a cooperative,
nonadversarial execution assumption. Owner-only file permissions, canonical
JSON, exact identity binding and hashes protect against accidental
modification, identity confusion and scope substitution.

These controls do not authenticate researcher approval against a malicious
same-UID actor. The SHA-256 integrity identifiers are not cryptographic
proof of researcher authorization, and no OS-account federation, signing
service or external approval platform was introduced.

### Permit identity and approval scope

The validator loads one fixed production permit path and rejects arbitrary
caller-selected permit paths. The permit binds:

- Paper 1 Revision-2026 and the conditional VAE family.
- CICMalDroid2020, seed 42 and attempt 02.
- The exact execution-source commit and `commit<7hex>_attempt02` identity.
- The canonical absolute publication destination.
- Ordered CIC class mapping `["0","1","2","3","4"]`.
- Exactly the `best` and `last` checkpoint roles.
- Observation version `gaussian_fixed_variance_1p0_v1`.
- The accepted checkpoint-v1 contract identity and selection policy.
- Authorization ID, approval-record identity, and authorization scope.
- Best/last publication and committed-generation verification authority only;
  synthesis and evaluation remain false.

The fixed production permit path is:

`~/trustforge_revision_2026_artifacts/paper01/vae/authorizations/vae_checkpoint_publication_permit.json`

The canonical destination pattern is:

`~/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/checkpoints/production/commit<7hex>_attempt02`

Existing, symlinked, repository-internal or otherwise unsafe destinations are
rejected before training.

### Authorization-scope digest

The validator computes a stable authorization-scope SHA-256 over canonical
JSON containing every permit field that defines the approved execution and
its capabilities, excluding only `approval_commit`:

`authorization_scope_sha256 = SHA256(canonical_authorization_scope_bytes)`

This resolves the approval-commit circularity: the approved scope digest can
exist before the approval-ledger commit, and the final permit can reference
that commit without changing the scope digest. The existing complete-permit
SHA-256 remains separate and still covers `approval_commit` for permit-byte
integrity and revalidation.

### Approval-ledger verification

Approval evidence is one explicit, canonical ledger entry line:

> The canonical entry is an HTML comment containing the reserved VAE approval
> marker followed by canonical JSON with exactly two fields:
> `authorization_id` and `authorization_scope_sha256`. Illustrative placeholders
> are not reproduced using the reserved machine-readable marker syntax.

The validator reads the referenced approval ledger blob from the permit's
`approval_commit`, rejects malformed or noncanonical entries, and requires
exactly one entry matching the authorization ID with the exact approved scope
digest. Missing, duplicate, ambiguous, other-ID, or unrelated-text digest
occurrences are rejected. A digest appearing elsewhere in the ledger is not
accepted as approval evidence.

### Verification evidence

Previously reported and reviewed OpenCode execution evidence, not rerun
during this Git-preservation task:

- Focused permit-validation module: 48 passed.
- Targeted regression excluding two stale pre-activation assertions:
  84 passed, 2 deselected.
- Unfiltered targeted regression: 84 passed, 2 failed.

The two retained historical failures are:

- `test_gate_remains_disabled_in_source`
- `test_disabled_gate_helper_raises`

Both expect the obsolete pre-activation disabled training gate; the accepted
Stage 3 activation deliberately enabled that gate. They are not represented
as passing and were not modified.

### Scientific and schema preservation

No VAE model, observation, training, checkpoint-v1 manifest, commit-record,
coordinator, publisher, loader or synthesis-linkage semantics changed. The
accepted scientific settings remain unchanged, including the conditional
convolutional VAE, 40x40x3 input, latent dimension 100, observation variance
1.0, KL coefficient 1.0 without annealing, Adam learning rate 0.001, batch
size 64, exactly two epochs, full governed validation, earliest
minimum-validation-total-loss selection, distinct best/last states, and
best-checkpoint decoder-mean synthesis policy.

### Remaining limitations and status

- Authoritative checkpoint publication remains unauthorized.
- No production checkpoint was published.
- No real permit or execution-specific approval record was created.
- No new training attempt, Slurm execution, synthesis or evaluation occurred.
- GPFS crash durability remains unverified.
- `crash_durability_verified=False`.
- The validator establishes approval consistency, not same-UID adversarial
  authentication.
- This entry does not close Stage 4 overall or establish full benchmark
  readiness.

`VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED=NO`

`VAE_NEW_TRAINING_ATTEMPT_AUTHORIZED=NO`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

No other dataset, generator, seed, attempt or experiment is authorized by
this entry. Existing historical entries and original stage exit conditions
remain unchanged.

## VAE Stage 4 — Publication Activation Acceptance and Execution-Source Freezing — 2026-11-21

### Scope and researcher acceptance

This entry records researcher acceptance of the bounded VAE Stage 4
publication-activation implementation and freezes its execution source for
the proposed CICMalDroid2020 seed-42 attempt 02.

The source-level publication route is enabled. This is not an execution
authorization: actual checkpoint publication still requires the exact
execution-specific permit, a separate approval-ledger commit, a clean frozen
execution source and all existing destination and identity checks.

### Frozen execution source

`VAE_EXECUTION_SOURCE_COMMIT=1aec01c26fab51b87362a47e2cfa797bd428c359`

Frozen commit prefix:

`COMMIT7=1aec01c`

Parent commit:

`79d03e05e899a28c58cf650b42fe0598badb280d`

Proposed run identity:

`commit1aec01c_attempt02`

The later documentation commit is not the execution-source commit.

### Accepted implementation inventory

The frozen implementation commit changed exactly:

- `scripts/train_paper01_vae_revision_production.py`
- `slurm/paper01_revision_2026/vae_revision_production.slurm`
- `tests/test_variational_autoencoder_checkpoint_publication_coordinator_revision_2026.py`
- `tests/test_variational_autoencoder_checkpoint_publication_permit_revision_2026.py`

Accepted SHA-256 values:

- CLI:
  `2b21c205ffbeb0bb2148b2aa22555db6a14d3e93e6766c3783328066871b4a0e`
- Slurm launcher:
  `93435210a68ede1e3036f48a26e30129fa6d84d392b6965c505dba4417b25724`
- Coordinator test module:
  `2a71c9c28c5f20e38c3b1a1f0b0dea05ded292792276d10308f098bca66c8ee3`
- Permit test module:
  `fd78f601c6685bc2de201ceab24d4d1820cf79f794199f18863b0ecb8d8233be`

### Activation behavior

`VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED=True` makes only the
mandatory permit-validation route reachable. The permit remains the sole
publication authority and must bind the exact paper, revision, VAE family,
CICMalDroid2020 dataset, seed 42, attempt 02, run identity, clean execution
source, canonical destination, class mapping, best/last roles, observation
version, checkpoint contract and authorization scope.

The permit and destination are revalidated immediately before publication.
The independent governed training authorization remains enforced. The
launcher defaults publication requests to disabled and keeps synthesis
publication disabled.

### Preserved test evidence

Previously reported and reviewed OpenCode evidence, not rerun during this
Git-preservation task:

- Focused permit-validation suite: 54 passed.
- Targeted regression: 90 passed, 2 deselected.
- Unfiltered targeted regression: 90 passed, 2 failed.
- Launcher argument-capture verification: 15/15 passed.
- Python syntax checks: PASS.
- Bash syntax check: PASS.

The two retained historical failures are:

- `test_gate_remains_disabled_in_source`
- `test_disabled_gate_helper_raises`

Both expect the obsolete pre-activation Stage 3 training-gate condition. They
were not modified and are not represented as passing.

### Talon preflight acceptance and attempt availability

The read-only Talon preflight established:

- Talon home `/home/bruno.fonkeng`.
- Python 3.10.19 in `$HOME/miniforge3/envs/gcs_py310`.
- Slurm 25.05.6.
- Canonical Paper 1 artifact root exists.
- VAE artifact namespace is absent.
- Fixed VAE production permit is absent.
- Proposed attempt-02 execution checkout is absent.
- Stage 3 attempt 01 remains preserved.
- Job 316716 completed with exit code 0:0.
- Scheduler history from 2026-10-01 through 2026-11-21 contains no
  conflicting additional VAE production execution.
- No active user jobs were present.

Attempt 02 is accepted as available within the inspected Talon, scheduler,
artifact and log evidence. The exact source-dependent destination must still
be verified absent after the execution-source commit is frozen.

### Proposed operational paths

Execution checkout:

`/home/bruno.fonkeng/TrustForge-paper01-vae-stage4-attempt02`

Publication destination:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/checkpoints/production/commit1aec01c_attempt02`

Fixed permit path:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/authorizations/vae_checkpoint_publication_permit.json`

These are derived proposals only. No Talon worktree, directory, reservation
marker, permit or checkpoint was created.

### Permit and custody boundary

A future execution requires a separate append-only approval-ledger commit
containing the structured approval entry for the exact authorization-scope
SHA-256. The final permit must reference that later commit and the complete
permit SHA-256 must remain a separate integrity identifier.

No real execution-specific permit or approval marker was issued by this
entry. The accepted same-account cooperative custody model remains an
auditable operational control, not protection against a malicious same-UID
actor.

### Scientific preservation and limitations

No change was made to the conditional VAE architecture, canonical 40x40x3
representation, latent dimension 100, observation variance 1.0, KL
coefficient 1.0, Adam learning rate 0.001, batch size 64, two-epoch budget,
governed validation, earliest minimum-validation-total-loss selection,
distinct best/last states, held-out test isolation, checkpoint schemas or
synthesis policy.

GPFS crash durability remains unverified:

`crash_durability_verified=False`

No real-data training, production checkpoint publication, independent
production restoration, synthesis or downstream evaluation occurred.

### Milestone dispositions

`VAE_STAGE4_OPERATIONAL_PREFLIGHT=ACCEPTED_AND_CLOSED`

`VAE_STAGE4_PUBLICATION_ACTIVATION=ACCEPTED_AND_CLOSED`

`VAE_EXECUTION_SOURCE_COMMIT=FROZEN`

`VAE_CIC_SEED42_ATTEMPT02=PREFLIGHT_AVAILABLE`

`VAE_PUBLICATION_ROUTE_ENABLED=YES`

### Execution restrictions

`VAE_REAL_EXECUTION_PERMIT=NOT_ISSUED`

`VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED=NO`

`VAE_NEW_TRAINING_ATTEMPT_AUTHORIZED=NO`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

The enabled source-level route does not grant actual execution authority.
No other dataset, generator, seed, attempt or experiment is authorized by
this entry. Existing historical entries and original stage exit conditions
remain unchanged.

## VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-02 Execution-Specific Authorization — 2026-11-21

### Researcher approval

The researcher explicitly approved the exact execution-specific authorization
scope and digest recorded below for one bounded checkpoint-producing VAE
execution.

Authorization ID:

`P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION`

Approved authorization-scope SHA-256:

`9218772997ffbb112dd8df20a0058e1e75ac8406bae05338f305075cacf40f93`

Frozen execution-source commit:

`1aec01c26fab51b87362a47e2cfa797bd428c359`

Run identity:

`commit1aec01c_attempt02`

Canonical publication destination:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/checkpoints/production/commit1aec01c_attempt02`

The destination was accepted as available in the prior Talon preflight and
remains subject to the mandatory pre-issuance collision check.

### Approved execution identity

- Dataset: `cicmaldroid2020`
- Generator family: `vae`
- Experimental seed: 42
- Attempt: 2
- Run identity: `commit1aec01c_attempt02`
- Checkpoint roles: `best` and `last`
- Ordered class mapping: `["0", "1", "2", "3", "4"]`
- Observation version: `gaussian_fixed_variance_1p0_v1`
- Checkpoint contract:
  `paper01_revision2026_conditional_vae_checkpoint` schema version `1.0`,
  architecture `paper01_revision2026_conditional_vae_arch_v1`, selection
  policy `min_validation_total_loss_earliest_epoch_v1`
- Authorization scope: checkpoint publication and committed-generation
  verification only; synthesis and evaluation disabled

The accepted scientific configuration remains unchanged: conditional
convolutional VAE, canonical 40x40x3 float32 representation, latent dimension
100, observation variance 1.0, KL coefficient 1.0, Adam learning rate 0.001,
batch size 64, exactly two complete epochs, 254 expected optimizer updates,
full governed validation, earliest minimum-validation-total-loss selection,
distinct best/last states and held-out real-test isolation.

### Structured approval entry

<!-- VAE_CHECKPOINT_PUBLICATION_APPROVAL {"authorization_id":"P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION","authorization_scope_sha256":"9218772997ffbb112dd8df20a0058e1e75ac8406bae05338f305075cacf40f93"} -->

### Documentation erratum

The original historical approval-ledger example reproduced the reserved
machine-readable marker prefix in illustrative documentation, which the strict
accepted parser interpreted as a malformed approval entry. The researcher
authorized this narrow documentation correction. The original text remains
preserved in Git history. No parser, permit schema, scientific configuration,
frozen execution source or approved authorization scope changed.

### Issuance and execution boundary

The researcher authorizes issuance of the exact execution-specific permit
that references this approval commit. The same-account cooperative custody
model remains an auditable operational control, not protection against a
malicious same-UID actor.

Slurm execution release remains separate and held. This entry does not
authorize job submission, training, checkpoint publication, restoration,
synthesis or evaluation. GPFS crash-recovery durability remains unverified:

`crash_durability_verified=False`

### Dispositions

`VAE_EXECUTION_SPECIFIC_AUTHORIZATION_SCOPE=APPROVED`

`VAE_AUTHORIZATION_SCOPE_SHA256=9218772997ffbb112dd8df20a0058e1e75ac8406bae05338f305075cacf40f93`

`VAE_REAL_EXECUTION_PERMIT_ISSUANCE=AUTHORIZED`

`VAE_SLURM_EXECUTION_RELEASE=HELD`

`VAE_NEW_TRAINING_ATTEMPT_AUTHORIZED=NO`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

No other dataset, generator, seed, attempt or experiment is authorized by
this entry. Existing historical entries and original stage exit conditions
remain unchanged.

## VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-02 Train–Checkpoint–Restore Acceptance — 2026-11-21

### Researcher acceptance

The researcher accepted the bounded CICMalDroid2020 VAE seed-42 attempt-02
train–checkpoint–restore execution and closed that milestone.

- Dataset: `cicmaldroid2020`
- Generator: `vae`
- Experimental seed: 42
- Attempt: 02
- Run identity: `commit1aec01c_attempt02`
- Frozen execution-source commit:
  `1aec01c26fab51b87362a47e2cfa797bd428c359`
- Approval-ledger commit:
  `83a297ad6cb486a0a21c8758b99df5609479fe9b`
- Authorization ID:
  `P1R2026-VAE-CIC-S42-A02-CHECKPOINT-PUBLICATION`

### Training and publication

Slurm job `316796`:

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:02:27`
- Node: `talon07`
- MaxRSS: `860356K`
- Completed epochs: 2
- Completed optimizer updates: 254
- Validation total loss: `4508.749372571244`

Training logs:

- `p1r1-vae-prod.316796.out` SHA-256:
  `3f67ecbd0fc33a04ff8b35c62e54ad6d3f4d4457715b5becffc5e1c5387bf898`
- `p1r1-vae-prod.316796.err` SHA-256:
  `d9ef2cfe960a00645ead9d757ad414133cae9a64b79471257c4a3ade02609f8c`

Accepted scientific configuration: conditional convolutional VAE, canonical
40x40x3 float32 input, latent dimension 100, Gaussian observation variance
1.0, KL coefficient 1.0 without annealing, Adam learning rate 0.001, batch
size 64, exactly two complete epochs, full governed validation, earliest
minimum-validation-total-loss selection, distinct best/last states,
test-split usage prohibited, and the governed CIC 70/10/20 partition with
seed 42.

### Checkpoint identities

BEST:

- Generation: `35ca3cd756a88d6b93dcc291f02c4747`
- Manifest SHA-256:
  `f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`
- Commit-record SHA-256:
  `f1fcd7cdc8ac7ee99edf1e794eab726a0219213b68007b3a882e5dbebf282118`

LAST:

- Generation: `0d808f873b163a2d054f87aac34f3043`
- Manifest SHA-256:
  `ba2a5231db0ef90ac9d724525e61cec9fa4bda6c0838740b52312a13cd8816ac`
- Commit-record SHA-256:
  `a98b11cfbe7119944ee13c42eadf95660ab278d713636faeb0e2741575d95f15`

Shared state digests:

- Encoder:
  `c308765d65090a97ca872133ed833d2d6f975c52bb8a031aceb2005a5faa0141`
- Decoder:
  `c7b4f3ff56a13c416f0997e4dd375812cd1db3b6bdadd60d60770a97d06557b0`
- Dataset provenance:
  `e6736e56b7bd8f2e81505d3f8b1811dfc95a0e0514c629d524025a88a03138ae`

The identical model-state bytes are legitimate because the final epoch was
also the selected best epoch. The distinct logical roles and generation
identities remain preserved.

### Independent restoration

Restoration Slurm job `316811`:

- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:00:09`
- Node: `talon07`
- Process ID: `3935002`
- Python: 3.10.19
- CPUs: 8; memory: 32G; GPU disabled

Both committed generations independently verified. BEST governed
restoration, metadata equality, provenance equality and architecture
fingerprint checks passed. The encoder has 16 variables and 3,423,400
parameters; the decoder has 8 variables and 734,659 parameters. All restored
floating-point variables are finite. No optimizer state was loaded, no
training resumed and no synthetic samples were generated.

`VAE_CHECKPOINT_ARTIFACTS_UNCHANGED=YES`

### Preserved evidence

Repository evidence directory:

`studies/paper01_benchmark/audits/revision_2026/vae_stage4_cic_seed42_attempt02_job316796_train_checkpoint_restore`

External restoration evidence directory:

`/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/diagnostics/vae-stage4-cicmaldroid2020-seed42-attempt02-job316796-restoration-20261121`

Evidence inventory SHA-256:

`8743ad8ef634f9af42d51bd80b9e965fa41ceb8767e61c7976a9721ea7cfd568`

README SHA-256:

`9285fec41623554a4d4ffd9ff48ea9df2c7b2dad8b5951a71e02b342ff4e794f`

Checkpoint inventory SHA-256:

`bb6bc01722ac5f8803d8fb4e866c063a3346dcbde60131d6ead67c139283e24f`

Model-state NPZ files, raw datasets, synthetic pools and the live permit were
not copied into Git. The permit remains at its fixed external path with
raw-file SHA-256:

`65c67f73e6b82c2de816589876980ba6f8829736f4d3a39a73566a1aef433d6a`

### Reporting discrepancy

Top-level:

`checkpoint_status=NOT_PREPARED`

`publication_status=NOT_PUBLISHED`

Nested checkpoint-publication result:

`publication_status=PUBLISHED`

The actual commit records, manifests, state files and independent restoration
corroborate the nested publication result. This is a reporting limitation
requiring appropriate treatment by future production reporting consumers; it
does not invalidate the accepted checkpoint execution. The CLI was not
modified and the original logs were not altered.

### Remaining limitations and synthesis boundary

- `crash_durability_verified=False`
- GPFS crash-recovery durability remains unverified.
- Same-account custody does not protect against a malicious same-UID actor.
- Restoration does not independently authenticate checkpoint origin.
- Dataset provenance does not independently prove partition execution or
  continuous source immutability.
- Successful restoration does not establish synthetic-data fidelity or
  downstream utility.
- Production synthesis remains unauthorized.
- The eventual CIC VAE primary pool requires 2,000 samples per class and
  10,000 samples total, subject to the accepted synthesis policy and any
  required bounded synthesis compatibility evidence.
- Independent synthetic-pool publication and verification remain Stage 4
  requirements.

### Dispositions

`VAE_CIC_SEED42_ATTEMPT02_TRAINING=ACCEPTED_AND_CLOSED`

`VAE_CHECKPOINT_PUBLICATION=ACCEPTED_AND_CLOSED`

`VAE_INDEPENDENT_RESTORATION=ACCEPTED_AND_CLOSED`

`VAE_TRAIN_CHECKPOINT_RESTORE_MILESTONE=ACCEPTED_AND_CLOSED`

`VAE_CHECKPOINT_ARTIFACTS_UNCHANGED=YES`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

This closes only the bounded train–checkpoint–restore milestone. It does not
close Stage 4 overall, authorize synthesis or establish full benchmark
readiness. Existing historical entries and original stage exit conditions
remain unchanged.

## VAE Stage 4 — Production Synthesis Integration Milestone A Acceptance — 2026-11-22

### Researcher acceptance

The researcher accepted the bounded VAE production synthesis integration
implementation and closed Milestone A. This acceptance covers implementation
and fixture verification only; it does not authorize real-checkpoint
synthesis, production pool publication or downstream evaluation.

Implementation source commit:

`36165327a9d9eed2a40808db1c51f1301c42bc2d`

Parent:

`df6e6dc3ca81d299d62829fb508dabe5bd8e5b5f`

### Accepted implementation inventory

The implementation commit adds exactly:

- `src/trustforge/models/paper01_revision_2026/vae_cic_pool_publication.py`
- `src/trustforge/evaluation/paper01_vae_pool.py`
- `scripts/publish_paper01_vae_pool.py`
- `slurm/paper01_revision_2026/vae_pool_production.slurm`
- `tests/test_paper01_vae_pool_publication_revision_2026.py`

No existing source, test, scientific configuration or authorization file was
modified.

### Accepted scientific recipe

The approved synthesis recipe remains unchanged:

`98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9`

The VAE path retains the approved best-checkpoint role, conditional decoder-mean
generation, standard-normal prior, latent dimension 100, one-hot requested-class
conditioning, disabled observation noise, model-call batch size 32, quota 2,000
samples per class, deterministic latent stream, and canonical float32 NHWC
40x40x3 output in [0,1].

### Verified implementation behavior

Previously reported and reviewed fixture evidence, not rerun during this
Git-preservation task:

- Focused VAE pool module: 28 passed.
- Relevant regression set: 149 passed.
- Python syntax checks: PASS.
- Slurm launcher Bash syntax check: PASS.
- Complete fixture pool: five classes, 10,000 samples, 315 model-call batches.
- Terminal partial batches: five batches of 16 samples each.
- Independent streaming artifact-byte verification: exercised.
- Invalid, incomplete and corrupted pool cases: rejected.
- Production authorization: default-deny and verified before checkpoint
  loading or generation.

The implementation supports the approved four-dataset architecture through
the governed dataset class-count contract and explicit ordered class mappings.
The CIC-specific schema/module naming does not restrict runtime dataset
acceptance. Non-CIC production execution remains unverified.

### Fixture versus production authority

The 10,000-sample pool was fixture-generated in isolated temporary test
directories from a synthetic checkpoint and test-local gate mock. The
authoritative artifact contract was simulated; it was not a real CIC
production pool.

The real production synthesis gate remains false. Fixture evidence does not
establish real-checkpoint synthesis, authentic production lineage or
downstream evaluator acceptance.

### Remaining requirements

- Real-checkpoint VAE synthesis remains unexecuted.
- Production VAE synthetic-pool publication remains unauthorized.
- Independent verification of a real published VAE pool remains outstanding.
- Downstream evaluator pool admission remains deferred to Milestone B.
- Cross-family and cross-seed operational coverage remains outside this
  milestone.
- GPFS crash durability remains unverified.

### Dispositions

`VAE_MILESTONE_A_IMPLEMENTATION=ACCEPTED_AND_CLOSED`

`VAE_MILESTONE_A_GIT_PRESERVATION=COMPLETED`

`VAE_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`VAE_REAL_CHECKPOINT_SYNTHESIS=NOT_EXECUTED`

`VAE_DOWNSTREAM_EVALUATOR_INTEGRATION=DEFERRED_TO_MILESTONE_B`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

No approval marker or real execution authorization was added by this entry.
Existing historical records and the original checkpoint-publication approval
entry remain unchanged.

## VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-01 Production Synthesis Acceptance — Slurm 316980

Status: researcher-accepted and closed for one bounded production synthesis
execution. This entry records production correctness and provenance only.

### Accepted execution

- Dataset: `cicmaldroid2020`; generator: `vae`; experimental seed: 42;
  attempt: 01; checkpoint role: `best`.
- Slurm job `316980` (`p1r1-vae-pool`), partition `talon`, node `talon06`,
  account `jielun.zhang`.
- Scheduler state `COMPLETED`; exit code `0:0`; elapsed `00:01:15` against a
  `02:00:00` limit; 1 node, 1 task, 8 CPUs, 32G requested; batch MaxRSS
  `649092K`; no requeue or restart.
- Submitted `2026-11-22T22:31:01`, started `2026-11-22T22:31:01`, ended
  `2026-11-22T22:32:16`. Exactly one submission; no retry.

### Lineage

- BEST checkpoint generation: `35ca3cd756a88d6b93dcc291f02c4747`.
- BEST checkpoint manifest SHA-256:
  `f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`.
- Checkpoint run identity: `commit1aec01c_attempt02`.
- Historical checkpoint source commit:
  `1aec01c26fab51b87362a47e2cfa797bd428c359`.
- Frozen synthesis source commit:
  `d4031a9b39565e5382d08e2b28236162498a08ca`.
- Approved synthesis recipe SHA-256:
  `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9`
  (unchanged; recipe authorization fields remain false).
- Production manifest SHA-256:
  `08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6`.
- Latent-stream contract:
  `trustforge.paper01.synthesis-latent-stream.v1`; PCG64; latent dimension
  100; standard normal; NumPy 2.2.6 observed and baseline.

### Publication

- External destination:
  `/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/cicmaldroid2020/seed42/pools/production/checkpoint35ca3cd756a88d6b93dcc291f02c4747_synthesis-d4031a9_attempt01/pool`
- 315 batch artifacts, 63 per class for classes 0–4; 2,000 samples per class;
  10,000 samples total; 316 top-level entries including `manifest.json`.
- Manifest 128,201 bytes; `complete_pool=true`; `status=COMPLETE`;
  `authoritative_artifact=true`; expected and observed totals both 10,000.
- Canonical float32 NHWC output, shape `[40, 40, 3]`, range `[0, 1]`.
- No `FAILURE.json` and no `.manifest.pending`; approximately 192,184,905 pool
  bytes (196 MB). Prior attempt-specific parent directories were created for
  this authorized run; the publisher retained exclusive creation of the leaf.

### Independent verification

- `trustforge.evaluation.paper01_vae_pool.VAEPoolView.preflight()` executed in
  a separate process from the frozen worktree, reading only published
  artifacts and never invoking the generator.
- Result: 315 verified batches, 10,000 verified images, 315 batch records and
  315 `.npy` files, no failure or pending marker, `VERIFIER_PASS`.
- Each record was checked for existence, size, byte SHA-256, shape, dtype,
  finiteness, canonical domain, ordered sample-index coverage and quota.
- A bounded spot check at positions `[0, 31, 1999, 2000, 8000, 9999]` returned
  float32 `(6, 40, 40, 3)` images within `[0.355, 0.534]` and one-hot labels.

### Evidence

- Repository audit directory:
  `studies/paper01_benchmark/audits/revision_2026/vae_stage4_cic_seed42_attempt01_job316980_synthesis_pool`
- Preserved files: `README.md`, `pool_inventory.tsv`, `pool_manifest.json`,
  `submission_command.txt`, `submission_response.txt`, `synthesis.stdout`,
  `synthesis.stderr`, `synthesis_scheduler.sacct.txt`,
  `synthesis_scheduler.scontrol.txt`, `verification.txt`, and `SHA256SUMS`.
- Key digests: `synthesis.stdout`
  `d5c2deba30b7b71b9aa76b50eeb5cbed4f1d3165dec35bb726ca1e2100321fb4`;
  `verification.txt`
  `13e1526e0e202174ac440c6e677d90dfab0469ebc8b44fc319c8cba7d67a4312`.
- External originals:
  `/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/diagnostics/vae-stage4-cicmaldroid2020-seed42-attempt01-synthesis-job316980-20261122`
  plus the pool destination above. Originals were hashed during preservation,
  matched the repository copies and were not modified.
- Not copied into Git: the 315 batch files, the production pool, datasets,
  checkpoint NPZ state and held-out test data.

### Manifest-digest transcription correction

The execution-release command text carried a 62-character manifest digest
ending `...9af53bf812e`, which is not a valid SHA-256 and would have been
rejected before synthesis. Under explicit researcher confirmation, submission
used the accepted 64-character digest
`f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e`, matching
the on-disk BEST manifest. No other input changed; the correction is recorded
in the preserved submission command.

### Accepted versus not yet established

Accepted: production synthesis correctness, artifact completeness, provenance
lineage and independent byte-level integrity verification for this one
CICMalDroid2020 VAE seed-42 attempt.

Not yet established: synthetic fidelity, requested-label fidelity, downstream
classification utility, statistical reliability across seeds, and complete
benchmark results.

### Dispositions

`VAE_CIC_SEED42_SYNTHESIS=ACCEPTED_AND_CLOSED`

`VAE_CIC_SEED42_PRODUCTION_POOL=INDEPENDENTLY_VERIFIED`

`VAE_DOWNSTREAM_EVALUATION_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

This entry adds acceptance evidence only. It does not authorize downstream
evaluation, another seed or dataset, retraining, checkpoint republication or a
full benchmark run.

## VAE Stage 4 — CICMalDroid2020 Seed-42 Attempt-01 Paired Evaluation Acceptance — Slurm 316983

Status: researcher-accepted and closed individual experimental cell. Bounded
seed-level evidence; not a full benchmark result.

### Objective

Compare the accepted EvalCNN under Real-only and Real+Synthetic training for
CICMalDroid2020 with the independently verified 10,000-image VAE seed-42
production pool, evaluated on the governed held-out real test partition.

### Authorized execution

- Single submission: Slurm job `316983` (`p1r1-vae-eval-s42`), partition
  `talon`, node `talon06`, 1 node / 1 task / 8 CPUs / 32G, 2-hour limit.
- State `COMPLETED`, exit `0:0`, elapsed `00:29:43`, no requeue, zero
  restarts, batch MaxRSS `2,781,472K`. No retry; attempt 01 only.
- Evaluation source commit:
  `04d1a3cb929c41b24c8b4e8e8e4a25bc68fe1a4b`.
- Authorization ID `P1R2026-VAE-CIC-S42-EVAL-A01-20261123`, SHA-256
  `e92da8a146d5b775ab79a71cb6efad85c8de8331d8a7ee54b65768023d8435ef`.
- Launcher SHA-256
  `7feaaa72c302b198e86d677ded7c6e8192b80909f542d505f05e21f277f9b1de`;
  driver SHA-256
  `fb0c6b56932be11d888c80c72b2b685902fdb5f23c4445137d35fab8dd7dac65`.
- Training receipt SHA-256
  `6b08c5f73687759e025bf921329e12d85dbc49f45a29a34836f4cc5836c77281`.
- Preflight, train, restore and evaluate phases all reported PASS; `stderr`
  contains only TensorFlow CUDA-absence notices.

### Data and protocol

- Governed partitions: train 8,117; validation 1,158; held-out test 2,323;
  five classes with accepted per-class counts. `test_accessed=false`.
- Synthetic pool manifest SHA-256
  `08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6`;
  315 batches; 2,000 distinct samples/class; 10,000 distinct samples.
- Augmented training presented the 10,000 distinct samples once per epoch over
  20 epochs: 200,000 synthetic presentations and 2,840 augmented updates
  (real-only arm: 1,280 updates). Distinct samples are not presentations.
- Both arms shared initial weights
  `4e5a33f021164e0d148bba5f058d0dc01a0cd8ca48ad3f0b7c07f9013e7a19ba`,
  used EvalCNN, no class weighting, Adam at the accepted learning rate, batch
  128, maximum 20 epochs, patience 5, real-validation selection and
  best-weight restoration. Both selected epoch index 19 and ran to the
  maximum epoch count.
- Selected weights: real-only `33ad5a3c…`; augmented `ed4624ff…`. Both
  classifier bundles were published, hash-matched and restored in a separate
  process (train PID `1978334`, restore PID `1981436`).

### Accepted results

Recomputed exactly from persisted predictions with the accepted metric
recipe; stored and recomputed values agree with difference 0.0.

| Metric | Real-only | Real+Synthetic | Difference |
|---|---:|---:|---:|
| Macro F1 | 0.6465410253 | 0.6844622754 | +0.0379212501 |
| Balanced accuracy | 0.6362720813 | 0.6669344713 | +0.0306623900 |
| Macro AUPRC | 0.7401845206 | 0.7718072623 | +0.0316227417 |
| Brier score | 0.4233299249 | 0.3730527112 | −0.0502772137 |
| ECE | 0.1416111567 | 0.0433544592 | −0.0982566975 |

Per-class F1 preserves the observed heterogeneity:

| Class | Support | Real-only | Real+Synthetic | Delta |
|---|---:|---:|---:|---:|
| 0 | 251 | 0.4564705882 | 0.5381355932 | +0.0816650050 |
| 1 | 420 | 0.5261744966 | 0.6167290886 | +0.0905545920 |
| 2 | 782 | 0.8318284424 | 0.8202368138 | −0.0115916287 |
| 3 | 510 | 0.7016848365 | 0.7134637514 | +0.0117789150 |
| 4 | 360 | 0.7165467626 | 0.7337461300 | +0.0171993674 |

Class 2 declined; the other four improved.

### Synthetic diagnostics

Artifact-authenticated values, not recomputed during the final review:
balanced KID mean `0.577493` (10 draws); pooled balanced FID `382.674`
(625 real vs 625 synthetic features); macro class-conditional FID `400.234`
(complete coverage); macro intra-class 3-scale MS-SSIM `0.937243` (1,000
pairs, 200/class). The pooled FID uses singular empirical covariance matrices
without regularization. Expensive feature extraction and metric replay were
not repeated. Synthetic quality and downstream classification utility remain
distinct dimensions.

### Verification and evidence

- Unified evaluation: 7/7 groups verified, no status or lineage blockers,
  verified pool consumption bound to the augmented selected weights;
  `complete_evaluation=false` is the accepted bounded-cell marker.
- Evidence directory:
  `studies/paper01_benchmark/audits/revision_2026/vae_stage4_cic_seed42_attempt01_job316983_paired_evaluation`
  with `README.md`, `artifact_inventory.tsv`, compact report copies and
  `SHA256SUMS`.
- Primary external run:
  `/home/bruno.fonkeng/trustforge_revision_2026_artifacts/paper01/vae/evaluations/cicmaldroid2020/seed42/runs/attempt01`
- Key digests: `paired_downstream.json` `9016feb2…`, `evaluation.json`
  `7758eba5…`, `train_report.json` `1e64f038…`, `restore_report.json`
  `855b0567…`, `pool_consumption.json` `dedfeef0…`. Prediction arrays,
  classifier states and Inception features remain external with digests in
  the inventory.

### Interpretation boundaries

For this one cell, the augmented arm showed higher macro-F1 (about +0.0379)
and higher balanced accuracy and macro-AUPRC, with lower Brier score and ECE,
on the governed held-out real test partition. One of five classes declined.

This does not establish statistical significance, general VAE superiority,
uniform per-class benefit, universal benefit of synthetic augmentation or
completion of the Revision-2026 benchmark. Cross-seed reliability and
aggregation remain future work under the accepted protocol.

### Dispositions

`VAE_CIC_SEED42_EVALUATION=ACCEPTED_AND_CLOSED`

`VAE_CIC_SEED42_PAIRED_EVALUATION_EVIDENCE=GIT_PRESERVED`

`VAE_DOWNSTREAM_EVALUATION_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

This entry closes one individual experimental cell. It does not authorize
another seed, dataset, generator, retry or full benchmark execution.

## Paper 1 Revision-2026 — Seven-Family Research Readiness Reconciliation — 2026-11-23

Status: researcher-approved read-only reconciliation appended at source
revision `49ed9a880ae912c6a9bdbf4659acd1dc73f1757d`. It reuses accepted
records at their stated scope. It performed no training, synthesis,
checkpoint publication, evaluation, Slurm job, source change or live Talon
re-inspection. Earlier statuses remain preserved; later accepted evidence
supersedes them only as stated below.

### Evidence cutoff and closed milestones

- Source revision: `49ed9a880ae912c6a9bdbf4659acd1dc73f1757d`.
- Stage 5 authenticated aggregation linkage: accepted and preserved at
  `49d01f802f53c8c8776e8cb05a671c65e365b282`. This closes the bounded
  implementation and fixture-verification milestone, not aggregation of a
  production seven-seed matrix.
- Diffusion Stage 4 durable checkpoint bundle: accepted and preserved at
  `2cbd30a0bc4f16ae4d121c33963a70316bbb9137`. This closes the durable bundle
  implementation/loader and fixture verification, not production publication.
- Diffusion Stage 4 production integration: accepted and preserved at
  `49ed9a880ae912c6a9bdbf4659acd1dc73f1757d`; 137 focused/related fixture tests
  reported. This closes the bounded training-to-publication and
  published-checkpoint-to-restoration connections, not actual execution.
- Earlier accepted cells retained: VAE CIC seed 42 (training 316716,
  synthesis 316980, evaluation 316983, macro-F1 +0.0379212501); GAN CIC seed 42
  evaluation 314607 (+0.0390784969); MAF CIC seeds 42–44
  (+0.0545115091 / +0.0927756121 / +0.0610829155); AR CIC seed-42 complete
  synthesis pool (jobs 314793/314794), evaluation not performed; GMM CIC
  class-0 seed-42 training/synthesis (313922/313927).
- The earlier ledger statement that Diffusion checkpoint state is only
  in-memory is superseded for implementation scope by the accepted durable
  bundle; it remains true that no production Diffusion checkpoint was
  published.

### Seven-family readiness register

Statuses are scoped to committed evidence. Fixture tests are not production
execution, and one accepted cell does not establish dataset- or
benchmark-wide coverage.

| Family | 1 Method | 2 Executable source | 3 Launcher/gate | 4 Real training | 5 Durable checkpoint | 6 Restoration | 7 Synthesis/pool | 8 Paired evaluation |
|---|---|---|---|---|---|---|---|---|
| GAN | ACCEPTED | ACCEPTED; activation lives on `42c40fc`; current branch gate disabled | ACCEPTED bounded | ACCEPTED within CIC-42 cell | ACCEPTED | ACCEPTED | ACCEPTED 10k/625 batches | ACCEPTED 314607 |
| VAE | ACCEPTED | ACCEPTED; gate enabled | ACCEPTED | ACCEPTED 316716 | ACCEPTED attempt-02 | ACCEPTED | ACCEPTED 316980, 315 batches | ACCEPTED 316983 |
| Diffusion | ACCEPTED | ACCEPTED implementation/integration; runtime disabled | PARTIAL/PROPOSED; launcher present, gate false | NOT_EXECUTED | NOT_EXECUTED (bundle implemented) | NOT_EXECUTED (fixture only) | NOT_EXECUTED | NOT_EXECUTED |
| AR | ACCEPTED | ACCEPTED for CIC-42 synthesis | ACCEPTED bounded authorization | ACCEPTED within synthesis chain | VERIFIED | VERIFIED | ACCEPTED 10k/1,250 batches | NOT_EXECUTED |
| MAF | ACCEPTED | ACCEPTED | ACCEPTED bounded | ACCEPTED seeds 42–44 | VERIFIED | ACCEPTED | ACCEPTED seeds 42–44 | ACCEPTED seeds 42–44 |
| RBM | ACCEPTED training only | PARTIAL; no production sampling | BLOCKED by S2-01 | UNKNOWN/PARTIAL (Stage-1 integrity only) | PARTIAL | UNKNOWN | BLOCKED | NOT_EXECUTED |
| GMM | ACCEPTED | ACCEPTED class-0 cell | PARTIAL | ACCEPTED class-0 seed 42 | VERIFIED | PASS | PARTIAL; one class only | NOT_EXECUTED |

| Family | 9 Remaining original mandatory requirement | 10 Next authorization boundary |
|---|---|---|
| GAN | Dataset/seed coverage; deliberate activation integration; resource acceptance | Separate execution authorization |
| VAE | Dataset/seed coverage | Separate execution authorization |
| Diffusion | Real training, production publication, fresh-process restoration, synthesis, pool verification | Separate execution authorization after gate decision |
| AR | Paired downstream evaluation; further coverage | Separate execution authorization |
| MAF | Seeds 45–48 not authorized; other datasets | Separate execution authorization |
| RBM | S2-01 prospective production sampling decision | Scientific decision, then execution |
| GMM | All-class dataset pool and evaluation | Separate execution authorization |

Preserved: `RBM_S2_01=BLOCKED`. The closed RBM `INDETERMINATE` assessment is
not reopened and no Gibbs depth is selected.

### Nine-stage readiness

| Stage | Original exit condition | Accepted evidence | Remaining requirement | Status | Close at original scope? |
|---|---|---|---|---|---|
| 1 Preservation | Reconcile state, preserve source/artifacts | Baseline, snapshots, checkpoint inventory | Recheck before mutation | COMPLETE | Yes |
| 2 Specification | Bind datasets, seeds, budgets, evaluator, analysis | Spec §14/15; S2-02–S2-05 accepted; aggregation linkage accepted | S2-01 RBM production sampling decision | IN_PROGRESS | No |
| 3 Execution | Requested→executed identity, gates, failure behavior | Accepted source/entry paths for VAE, GAN, MAF, AR, GMM; Diffusion entry accepted | Diffusion activation; RBM production; remaining distinct paths | IN_PROGRESS | No |
| 4 Checkpoint/synthesis | Durable publication, fresh restoration, counted synthesis per distinct family | VAE, GAN, MAF×3, AR full pools; Diffusion bundle/integration implemented; GMM class 0 | Diffusion production publication/restoration; RBM sampling/synthesis; GMM all classes | IN_PROGRESS | No |
| 5 Evaluation/reporting | Matched paired evaluation, metrics, missing-run, aggregation | Evaluator accepted; VAE/GAN/MAF cells; aggregation + missing-run implementation | AR/GMM/RBM/Diffusion evaluations; production aggregation over real inventory; scale-memory acceptance | IN_PROGRESS | No |
| 6 Reproduction | Clean-environment bounded workflow and traceability | None at required scope | Execute and accept reproduction workflow | PENDING | No |
| 7 Bounded end-to-end | Governed integrated path plus rejection suite | VAE production path exists; Diffusion integration fixture-tested | Accepted missing-output/incomplete/identity-mismatch rejection suite across distinct integrations | PENDING | No |
| 8 Freeze/launch | Frozen revision/config/environment/matrix/resources/retries + GO | None | Freeze and written evidence-based GO/NO-GO | BLOCKED | No |
| 9 Results/release | Reconcile runs, regenerate tables, audit claims | None; no full matrix | Authorized execution then release acceptance | PENDING AFTER EXECUTION | No |

The 196-cell benchmark remains incomplete. A completed bounded cell or a
completed implementation milestone is not benchmark completion.

### Distinct authorization boundaries

- **Individual family training:** accepted methodology, governed data and
  configuration, permitted runtime activation, and explicit researcher
  approval. Training authority does not imply checkpoint publication or
  synthesis. Diffusion additionally needs a gate-activation decision; GAN needs
  a deliberate activation decision; RBM training is authorized but its
  production sampling is not.
- **Full benchmark execution:** S2-01 decision; each distinct family
  integration executed and accepted; Stage 5 production aggregation and
  completion accounting; Stage 6 reproduction; Stage 7 rejection acceptance;
  Stage 8 freeze of revision, configuration, environment, 196-cell matrix,
  resources, concurrency and retries, followed by a written GO/NO-GO.
- **Manuscript release:** authorized run coverage (or an explicitly documented
  partial design), reconciled expected/observed runs with explicit
  PENDING/FAILED/INCOMPLETE statuses, seven-seed aggregation without fabricated
  statistics, regenerated tables/figures, claim-level verification and a
  preserved reproduction release.

### Recommended smallest next action

**Decide S2-01: adopt or decline a qualified fixed-finite-step RBM production
policy.** Parent stage: Stage 2. Original requirement: a separately governed
prospective sampling decision preserving the closed INDETERMINATE assessment.
Existing evidence is insufficient because no independently justified common
transition count or approved production resource ceiling exists, and the
recorded numerical-selection exercise was NO-GO. Minimum acceptance: either
(1) approve the qualified finite-step basis with a prospectively justified
count, explicit initialization/burn-in semantics and credible workload
feasibility, or (2) formally record RBM as incomplete for the current Paper 1
matrix. This is a scientific decision, not implementation or execution, and
requires separate researcher authorization. No new RBM diagnostic is
authorized by this entry.

### Unresolved or contradictory evidence

- Early ledger matrix rows and recipe `status` fields are time-stamped
  snapshots; later explicit researcher acceptance and source activations
  supersede them within their recorded scope.
- Declarative synthesis-recipe authorization flags remain false while separate
  source activations and one-job releases enabled accepted VAE/GAN/MAF/AR
  work; those flags are not a live inventory of executed cells.
- No live Talon artifacts were re-inspected in this reconciliation; execution
  evidence is reported at committed-record scope.

### Preserved restrictions

`DIFFUSION_STAGE4_DURABLE_BUNDLE=ACCEPTED_AND_CLOSED`

`DIFFUSION_STAGE4_PRODUCTION_INTEGRATION=ACCEPTED_AND_CLOSED`

`DIFFUSION_STAGE4_INTEGRATION_PRESERVATION=GIT_VERIFIED`

`STAGE5_AGGREGATION_LINKAGE=ACCEPTED_AND_CLOSED`

`RBM_S2_01=BLOCKED`

`DIFFUSION_REAL_DATASET_TRAINING_ENABLED=False`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`STAGE_4_OVERALL=IN_PROGRESS`

`FULL_BENCHMARK_GO=NO`

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.


## Paper 1 Revision-2026 — S2-01 Prospective Fixed Finite-Step RBM Sampling Policy — Researcher Approval — 2026-11-24

**Authority:** Researcher decision recorded 2026-11-24.
**Scope:** Scientific sampling policy only; implementation and execution remain unauthorized.

### Approved sampling policy

The revised continuous-visible, class-conditional RBM uses a fixed finite-step
generator

`Q_(theta,y,T) = mu_theta K_(theta,y)^T` with `T = 10`.

- Exactly ten complete Gibbs transitions per synthetic sample.
- One complete transition is a class-conditioned hidden-state draw followed by
  a continuous-visible-state draw.
- Each synthetic sample is produced from a separately initialized, independent
  Gibbs chain.
- Governed initialization is preserved: `h_0 = 0`; `v_0 ~ p(v|h=0)` under the
  bounded-exponential visible distribution; internal float64 arithmetic; no
  empirical training, validation or test example initialization.
- The requested class is validated and clamped before the first transition and
  remains fixed throughout sampling.
- Zero separate burn-in; all ten transitions constitute the finite-step
  generation procedure.
- No thinning, continuation-chain sampling, adaptive stopping, or
  dataset-specific Gibbs depth.
- One common transition count applies across all four datasets, all classes and
  seeds 42-48: 882,000 samples (63 classes x 2,000 samples x 7 seeds).

### Computational planning basis

- Internal RBM synthesis planning ceiling: 650 allocated CPU core-hours.
- Researcher-accepted 1.5x planning allowance on the historical linear cost
  estimate.
- Historical linear estimate: 309-331 allocated core-hours; approximately
  464-497 allocated core-hours with the allowance.
- These resource estimates are explicitly unvalidated. Historical diagnostics
  did not measure the complete production synthesis pipeline, and actual
  production feasibility has not been demonstrated. Actual costs may differ
  because of batching, initialization, checkpoint loading, I/O, hashing,
  scheduler behavior and retries.
- The internal ceiling is not a university-issued allocation or a guaranteed
  Talon reservation. Exceeding it requires stopping and separate review rather
  than silently changing the sampling policy or output coverage.

### Scientific interpretation

`T = 10` is a prospectively approved computational operating convention, not a
convergence-certified, mixing-certified, stationary-equivalent, optimal or
shortest-stable depth. This policy does not claim convergence to a stationary
RBM distribution, adequate Gibbs mixing, stationarity or distributional
equivalence, semantic class-label fidelity, image realism or diversity,
beneficial downstream classification utility, or equal computational effort
across generator families. It must not be optimized retrospectively using
downstream classification performance or synthetic-image quality metrics.

### Historical assessment preservation

The original independent confirmatory Gibbs-chain investigation remains closed
and unchanged:

`CONFIRMATORY_ASSESSMENT=INDETERMINATE`

`CONFIRMATORY_INVESTIGATION_CLOSED=YES`

The historical investigation did not establish a shortest stable common Gibbs
depth. This prospective policy does not revise, contradict or retroactively
reinterpret that finding. The November 16 S2-01 archival decision remains part
of the historical record.

### S2-01 reconciliation

The original S2-01 closure condition is satisfied: a separately governed
prospective sampling decision that preserves the closed INDETERMINATE
assessment.

`S2_01_SCIENTIFIC_POLICY=APPROVED`

`RBM_S2_01=CLOSED_SCIENTIFIC_POLICY`

`RBM_PRODUCTION_TRANSITION_COUNT=10`

`RBM_PRODUCTION_GIBBS_DEPTH_SELECTED=YES`

`RBM_PRODUCTION_BURN_IN=0_BY_DESIGN`

Selection is scientific-policy approval, not runtime activation. No sampler,
configuration, checkpoint, synthesis job or experiment is authorized by this
entry.

### Stage 2 reconciliation

The original five-item Stage 2 register is satisfied:

- S2-01 - RBM production Gibbs depth: CLOSED_SCIENTIFIC_POLICY.
- S2-02 - GAN/VAE/Diffusion synthesis recipes: ACCEPTED_AND_CLOSED.
- S2-03 - CIC fitted preprocessing identity: ACCEPTED_AND_CLOSED.
- S2-04 - Shared evaluator configuration: ACCEPTED_AND_CLOSED.
- S2-05 - Complete benchmark configuration linkage: ACCEPTED_AND_CLOSED.

`STAGE_2_ORIGINAL_REQUIREMENTS=ACCEPTED_AND_CLOSED`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

Stage 2 closure is limited to its original documented acceptance conditions. It
does not establish Stage 3-8 readiness, executable configuration, implementation
validation, checkpoint restoration, complete synthesis, paired evaluation,
clean-environment reproduction, full benchmark resource/config/environment/retry
freeze, or publication readiness. No additional Stage 2 requirement is
introduced.

### Preserved authorization boundaries

`RBM_PRODUCTION_SAMPLING_AUTHORIZED=NO`

`RBM_PRODUCTION_SYNTHESIS_AUTHORIZED=NO`

`RBM_SAMPLER_IMPLEMENTATION_AUTHORIZED=NO`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Scientific-policy acceptance is separate from executable configuration,
implementation validation, checkpoint restoration, complete synthesis, paired
evaluation, clean reproduction, full benchmark freeze and publication.

No source, configuration, test, recipe, experimental artifact or earlier ledger
entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — GAN Stage 3 Consolidated Researcher Acceptance — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher reviewed the
recovered GAN execution evidence and accepted the consolidated closure of
three original Stage 3 requirements for the bounded CICMalDroid2020 seed-42
GAN path:

- S3-O3 — GAN controlled activation.
- S3-O4 — GAN synthesis resource evidence.
- S3-O5 — GAN training execution identity.

This entry is a documentation-only reconciliation. It performed no training,
synthesis, evaluation, test, Slurm submission, activation change or source
modification. It does not reopen any accepted checkpoint, restoration,
synthesis or evaluation milestone.

Evidence provenance: the execution, accounting, provenance and hash records
below were recovered from preserved Talon records by the researcher and
provided for review. ChatGPT and the researcher reviewed and accepted them.
OpenCode did not access Talon, rerun the historical execution or independently
recompute these digests during this documentation task.

### S3-O5 — Training execution identity

| Item | Recorded value |
| --- | --- |
| Dataset / seed / attempt | CICMalDroid2020 / 42 / 01 |
| Training Slurm job | 314528 |
| Compute node | talon06 |
| State / exit code | COMPLETED / 0:0 |
| Actual elapsed | 00:27:09 |
| Allocated CPUs / requested memory | 8 / 32 GB |
| Scheduler batch MaxRSS | 3003396K |
| Authorized walltime | 12 hours |
| GPU | disabled and unused |
| Expected and executed source commit | `42c40fce8b7f1005301408444fdcae7ff7f19db5` |
| Scientific baseline commit | `14982b142887095c53e62387749f5e308ce42d5d` |
| Authorization ID | `paper01-gan-cicmaldroid2020-seed42-attempt01-42c40fc` |
| Run identity | `commit42c40fc_attempt01` |

The historical training stdout directly records the execution checkout,
source HEAD, Slurm job ID, dataset, seed, attempt and completed scientific
configuration.

#### Consumed scientific configuration

- Planned generator iterations: 500; completed generator iterations: 500.
- Completed discriminator updates: 1,000.
- Batch size: 256.
- Class-sampling alpha: 0.85.
- Learning rate: 0.0002.
- Adam beta_1: 0.5.
- Latent dimension: 100.
- Class mapping: 0, 1, 2, 3, 4.
- Canonical representation: 40 x 40 x 3.
- Checkpoint selection: terminal governed iteration.

Recorded training exposure: 128,000 total real draws against 8,117 unique
governed training positions, with recorded 100% unique coverage for each of
the five classes.

Governed partition counts: training 8,117; validation 1,158; test 2,323;
partition seed 42.

The dataset provenance records compute-node full-file SHA-256 verification of
the authoritative CSV and feature-column definitions.

Preserved provenance limitations: the provenance record does not independently
prove absence of leakage, continuous source immutability or checkpoint origin.

#### Recovered evidence identities

| Artifact | SHA-256 |
| --- | --- |
| Training stdout | `2129d83341b41f29871fb720e3e21d69fe17ea9d442270cddc441797a7917d24` |
| Training stderr | `73ea76a435ba6bb69f2b55182a95e79607db08898526b0c247f77f2368b2b446` |
| Checkpoint `model_metadata.json` | `af534e4c64e7adb7a3065e4c5c8e3b2f9c8b888734a20041e573cf2f4aa10a04` |
| Exposure evidence | `f944d654d648f3b220e7f3cbe2cf0c8d5600e4d22fbb238cf51a6127a82d053e` |
| Dataset provenance | `e6736e56b7bd8f2e81505d3f8b1811dfc95a0e0514c629d524025a88a03138ae` |
| Accepted checkpoint identity | `963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530` |

The exposure and dataset provenance hashes match the corresponding references
in the checkpoint metadata.

### S3-O4 — Synthesis resource evidence

| Item | Recorded value |
| --- | --- |
| Dataset / seed / attempt | CICMalDroid2020 / 42 / 01 |
| Synthesis Slurm job | 314574 |
| Source commit | `4324aeecb45a47a72ac0b479d8cd994fe79dc1e0` |
| Authorization ID | `P1R2026-GAN-CIC-S42-A01-20261116` |
| Run identity | `commit4324aee_attempt01` |
| Compute node | talon06 |
| State / exit code | COMPLETED / 0:0 |
| Actual elapsed | 00:01:54 |
| Allocated CPUs / requested memory | 8 / 32 GB |
| Scheduler batch MaxRSS | 595100K |
| `seff` reported memory | 581.15 MB |
| `seff` reported memory efficiency | 1.77% |
| `seff` reported CPU efficiency | 11.73% |
| Authorized walltime | 12 hours |
| GPU | disabled and unused |
| Automatic retries permitted | zero |
| Model-call batch size | 16 |
| Batches per class | 125 |
| Verified batches | 625 |
| Verified images | 10,000 |
| Samples per class | 2,000 |
| Pool execution status | COMPLETE |

| Artifact | SHA-256 |
| --- | --- |
| Synthesis authorization | `4b2a77aba6f57402b015f35dc98e9e11d129b40c9ac1272f9196674152a2af96` |
| Synthesis recipe | `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9` |
| Pool manifest | `b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1` |

The recovered synthesis execution record, completion log, manifest hash,
authorization and Slurm accounting support the accepted bounded resource
evidence.

Scheduler-reported memory utilization (`MaxRSS`, `seff`) is distinguished from
independent process-level measurement; no independent process RSS measurement
is recorded. The efficiency figures are utilization context, not a claim of
optimal resources, measured optimum or benchmark-wide resource feasibility.

### Historical execution lineage

| Phase | Slurm job | Outcome |
| --- | --- | --- |
| GAN training | 314528 | COMPLETED |
| Initial restoration | 314534 | FAILED |
| Import diagnostic | 314535 | COMPLETED |
| Restoration retry | 314537 | COMPLETED |
| Production synthesis | 314574 | COMPLETED |
| Paired evaluation | 314607 | COMPLETED |

The failed initial restoration at job 314534 is preserved as historical
evidence. The three source revisions are distinct and not interchangeable:

- Training: `42c40fc`.
- Synthesis: `4324aee`.
- Evaluation: `49d6f5c`.

No previously accepted checkpoint, restoration, synthesis or evaluation
milestone is reopened by this entry.

### S3-O3 — Controlled activation decision

The researcher accepts the existing isolated GAN activation mechanism as
satisfying the original bounded controlled-activation requirement.

Preserved:

- Historical activation commit `42c40fc`.
- Execution-specific authorization.
- Exact source-commit binding.
- Default-deny behavior on the current development branch.

The historical gate-enabled state is supported by the activated source
revision and the successful governed execution; the recovered stdout does not
independently print the gate value.

No activation merge or cherry-pick is required for this closure. GAN is not
enabled on the current branch. Future experimental execution remains subject
to separate researcher authorization and the original Stage 8 freeze and
launch requirements.

### Scientific interpretation boundaries

The recovered progress logs and terminal diagnostics are descriptive execution
evidence. They do not establish GAN convergence, absence of mode collapse,
semantic class fidelity, synthetic-data realism, benchmark-wide superiority or
general resource optimality. The terminal diagnostics did not influence
checkpoint selection. No scientific policy is changed by this entry.

### Status and preserved boundaries

`S3_O3_GAN_CONTROLLED_ACTIVATION=ACCEPTED_AND_CLOSED`

`S3_O4_GAN_SYNTHESIS_RESOURCE_EVIDENCE=ACCEPTED_AND_CLOSED`

`S3_O5_GAN_TRAINING_EXECUTION_IDENTITY=ACCEPTED_AND_CLOSED`

`GAN_STAGE3_BOUNDED_REQUIREMENTS=ACCEPTED_AND_CLOSED`

`GAN_ISOLATED_ACTIVATION_APPROACH=RETAINED`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=IN_PROGRESS`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Stage 3 overall remains IN_PROGRESS. The remaining original Stage 3 closure
items are:

- S3-O2 — RBM governed production entry and activation.
- S3-O1 — Diffusion controlled activation and applicable operational evidence.

Neither is begun, assessed or closed by this entry. No other family, dataset,
seed, attempt or stage status is changed.

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — RBM Stage 3 Implementation and Historical Provenance Acceptance — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher accepted the
existing RBM production sampler and the governed RBM production entry at
fixture-verification scope, accepted preservation of both historical RBM
checkpoints, closed the historical provenance recovery with documented
limitations, and accepted planning of a bounded operational verification.

This entry does not authorize operational execution, an active authorization
record, runtime activation, checkpoint access, dataset access, sampling,
training, synthesis, evaluation or any Slurm submission. It does not close
S3-O2 overall.

Evidence provenance: the historical Talon execution, scheduler and artifact
records below were recovered by the researcher through read-only inspection
and provided for review. OpenCode did not access Talon, rerun historical
execution or recompute the historical artifact hashes during this
documentation task.

### Accepted implementation milestones

- Existing production sampler implementation commit:
  `6acc68258bf189a350c67eac1c24fd6226558152`.
- Governed production entry implementation commit:
  `22f43ca74b61433264ba0ff22f50017c079bf896`.

`RBM_STAGE3_SAMPLER_IMPLEMENTATION=ACCEPTED_FIXTURE_SCOPE`

`RBM_STAGE3_ENTRY_IMPLEMENTATION=ACCEPTED_FIXTURE_SCOPE`

The entry implementation was reviewed by ChatGPT and accepted by the
researcher at fixture scope. Verification evidence:

- 60 focused entry-integration tests passed in 3.11 seconds.
- Python compilation PASS.
- Slurm launcher shell syntax PASS.
- Existing scientific sampler and checkpoint implementation unchanged.
- Authorization precedes checkpoint artifact inspection, hashing, loading and
  sampling.
- Correct total allocated CPU arithmetic.
- Explicit distinction between per-job resource limits and cumulative workload
  planning.
- Production activation remains disabled.

The approved ten-transition sampling policy and the closed historical
INDETERMINATE investigation remain unchanged. Implementation acceptance is not
operational readiness.

### Historical CICMalDroid2020 checkpoint

| Item | Recorded value |
| --- | --- |
| Dataset / seed | CICMalDroid2020 / 42 |
| Associated training job | 306843 |
| Node | talon06 |
| Scheduler state | FAILED |
| Exit code | 1:0 |
| Elapsed | 00:01:23 |
| Allocated CPUs / requested memory | 8 / 32 GB |
| Recorded training examples | 8,117 |
| Class counts | 877, 1470, 2732, 1782, 1256 |
| Maximum optimizer updates | 64 |
| Selected checkpoint step | 64 |
| Checkpoint selection | final governed update |

| Artifact | SHA-256 |
| --- | --- |
| Checkpoint state | `10f64703515290ea1493b731bb8e941f7b200c6ad18a61acfc8a1be8bd17cd72` |
| Checkpoint metadata | `63b36cfe8f2d0dbc7838f3b6a7223ac2bfeb905e5211d50b1631925cccfc28a5` |
| Training stdout | `d44d82f289495882e536213fd4ecd46f1f3bd99ef69cb0941eb877358db2faaf` |
| Training stderr | `9b29ed942e71f8560fc9918476767637a39efef305bd42d1da367a5e0fc0e34e` |

Preserved historical reporting exception:

`AttributeError: 'RBMTrainingResult' object has no attribute 'completed_steps'`

The previously accepted checkpoint-integrity disposition is preserved. The
researcher does not reopen historical training because of this reporting
exception. The scheduler outcome remains `FAILED`; it is not rewritten as
`COMPLETED`. No exact executed training source commit is invented.

### Historical MaleX checkpoint

| Item | Recorded value |
| --- | --- |
| Dataset / seed | MaleX / 42 |
| Training job | 307935 |
| Node | talon16 |
| Scheduler state | COMPLETED |
| Exit code | 0:0 |
| Elapsed | 01:30:31 |
| Allocated CPUs / requested memory | 8 / 32 GB |
| Recorded training examples | 731,049 |
| Class counts | benign 125,794; malware 605,255 |
| Maximum optimizer updates | 5,712 |
| Completed optimizer updates | 5,712 |
| Selected checkpoint step | 5,712 |
| Checkpoint selection | final governed update |
| MaleX staging | PASS |

Training and launcher completion are recorded at the stated historical scope.

| Artifact | SHA-256 |
| --- | --- |
| Checkpoint state | `18f8ddf4f1d13bd8a9c660c968019a22421f097a40068c7a7be31f29052231b8` |
| Checkpoint metadata | `55721f7414b09de5720d1003ef5f29420aa06307079cb5a36c73b8043e3c4231` |
| Training stdout | `6e1f62648e8a7bbff7a79667f55782249b6277170c99b9ba3b61e4d2fb43ae4a` |
| Training stderr | `ffede51db22b54a3d24fa2eae50eb330f4bb73224f20affc167048b3b32ed66b` |

### Recorded common training configuration

Both historical checkpoints record:

- Visible dimension: 4800.
- Hidden units: 128.
- Batch size: 256.
- CD training depth: 1.
- Learning rate: 0.0005.
- Class-balance exponent: 0.85.
- Optimizer: manual SGD.
- Parameter dtype: float64.

CD-1 is the historical training gradient-estimation depth only. It is not the
approved production sampling policy. Production sampling remains exactly ten
complete Gibbs transitions with zero separate burn-in.

Recorded reconstruction diagnostics are descriptive execution evidence. They
do not establish generative quality, convergence, mixing sufficiency or
stationary-distribution equivalence.

### Evidence limitations

The historical checkpoint state digests are distinguished from metadata and
log hashes computed during evidence recovery. For both historical checkpoints:

- Exact executed training source commit: UNKNOWN.
- Consumed preprocessing fingerprint: UNKNOWN.
- Historical execution evidence is accepted only at its demonstrated scope.
- The historical provenance investigation is closed with limitations.
- No benchmark-wide provenance waiver is granted.
- No additional historical training is authorized.

These accepted limitations do not remove the original traceability
requirements for future benchmark execution.

`RBM_HISTORICAL_TRAINING_EVIDENCE=ACCEPTED_BOUNDED_SCOPE`

`RBM_HISTORICAL_PROVENANCE_INVESTIGATION=CLOSED_WITH_LIMITATIONS`

`RBM_HISTORICAL_TRAINING_SOURCE=UNKNOWN`

`RBM_HISTORICAL_PREPROCESSING_FINGERPRINT=UNKNOWN`

### Status and preserved boundaries

`RBM_PRODUCTION_SAMPLING_AUTHORIZED=NO`

`S3_O2_RBM=IN_PROGRESS`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`GAN_STAGE3_BOUNDED_REQUIREMENTS=ACCEPTED_AND_CLOSED`

`STAGE_3_OVERALL=IN_PROGRESS`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Bounded operational verification remains planned but unauthorized. No runtime
activation, active authorization record or Slurm submission is created by this
entry. No other family, dataset, seed, attempt or stage status is changed.

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — RBM Stage 3 Operational Acceptance and Closure — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher accepted the
bounded RBM operational verification and closed S3-O2 at its original Stage 3
scope following ChatGPT review. The approved scientific policy, production
sampler, governed execution entry, historical provenance limitations and
successful operational evidence are preserved.

This entry records the completed RBM Stage 3 work package. It does not close
Stage 3 overall, authorize Stage 4 execution or authorize the full benchmark.
No additional experimental execution is authorized.

Evidence provenance: the Talon execution, scheduler, artifact and terminal
hash records below were supplied by the researcher from preserved evidence.
OpenCode did not access Talon, restore the checkpoint, rerun sampling or
independently recompute these hashes during this documentation task.

### Previously accepted RBM foundation

- Scientific ten-transition sampling policy: approved 2026-11-24, unchanged.
- Production sampler implementation:
  `6acc68258bf189a350c67eac1c24fd6226558152`.
- Governed production entry:
  `22f43ca74b61433264ba0ff22f50017c079bf896`.
- Historical provenance acceptance:
  `b81ea28b642657dce58d16d3b47db2f8e91dbc89`.

Historical checkpoint acceptance and its provenance limitations are preserved
without reopening the historical investigation.

### Attempt-01 operational failure

| Item | Recorded value |
| --- | --- |
| Slurm job | 317176 |
| Dataset / seed / attempt | MaleX / 42 / 01 |
| Requested class | 0 — benign |
| Requested sample indices | 0–15 |
| Requested samples | 16 |
| Activation source | `94af0348b84539886e8e62bc7e1c998f1b8b8d99` |
| Node | talon16 |
| Slurm version | 25.05.6 |
| Scheduler state | FAILED |
| Exit code | 1:0 |
| Elapsed | 00:00:40 |
| Allocated CPUs | 8 |
| Allocated memory | 32 GB |
| Walltime limit | 600 seconds |
| Requeue | 0 |
| Restarts | 0 |
| Scheduler-reported batch MaxRSS | 768 KB |

Exact failure:

`RBMProductionAuthorizationError: Slurm allocation lacks TRES`

The failure occurred during resource authorization, before checkpoint
restoration or sampling. The actual controller record provided:

`ReqTRES=cpu=8,mem=32G,node=1,billing=8`

`AllocTRES=cpu=8,mem=32G,node=1,billing=8`

No standalone `TRES` field was present. Actual allocation matched the
authorized resource envelope. The fail-closed boundary prevented sampling
under an unrecognized scheduler representation.

`RBM_ATTEMPT_01=CLOSED_FAILED`

The attempt-01 authorization was retired at 07:50:02 UTC on 2026-11-24. The
failed activation revision remains preserved and unchanged.

Evidence directory:

`~/trustforge_revision_2026_artifacts/paper01/rbm/operational_verification/malex_seed42_attempt01/`

| Artifact | SHA-256 |
| --- | --- |
| `slurm-317176.out` | `87b8fdec0897109a61ff81902e0bd3a005ac9f9537b1974b31d7baefdeec5ca7` |
| `slurm-317176.err` | `de8085bf186065c9ebe842c38b7355ca45a35f730c533815709c28354421391d` |
| `authorization.json` | `8784310abdd76c312c5f4b090b2fc35b92e6662472129e579a7e38b0c4068f11` |
| `authorization_retirement.txt` | `046bf967f81ad55333c7bc09de04699890d6bd4775b17319b9450ea0629fe804` |
| `scontrol_job.txt` | `7745295291db69cb390e0b631be2e5bd429f14f64ffbf044b03367240086e3ae` |
| `sacct_resources.txt` | `4d84e3407b75721778f44e60b3e8906c65d4f823cbe44049e9475c58ba2c3ad5` |
| `slurm_version.txt` | `c0654e25716a40eff478d8852c09f79af82bef30c69d811a9a6a9fc54037bafa` |

Retired authorization:

`~/trustforge_revision_2026_artifacts/paper01/rbm/authorizations/retired/malex_seed42_attempt01.317176.json`

All seven historical evidence hashes were subsequently reverified by the
researcher during attempt-02 preflight and terminal reconciliation.

### Targeted compatibility correction

Accepted corrective commit:

`bfd9c68db58139a996aa85708069852a838e7992`

Commit message:

`fix(paper01): validate RBM allocated Slurm resources on Talon`

Corrective scope:

- Recognize actual `AllocTRES` allocation evidence.
- Preserve applicable legacy `TRES` compatibility.
- Exclude `ReqTRES` as allocation evidence.
- Reject contradictory CPU, node and memory evidence.
- Reject GPU evidence in either allocated representation.
- Preserve authorization-before-checkpoint ordering.
- Preserve the disabled development activation gate.

Verification:

- 79 focused integration tests passed in 3.14 seconds.
- Python compilation PASS.
- Git diff integrity PASS.
- Two authorized files modified.
- Scientific sampler, checkpoint implementation, CLI and launcher unchanged.

| Corrected artifact | SHA-256 |
| --- | --- |
| Production entry | `95cebf0c92f69394ad51d31a6a3ca327d9828da88eef6f5316b460bb64b37d08` |
| Integration tests | `f00c10865a303909e65ff380b4ce2b0dd6b9cd8884b9b42ead87943f870e2f2b` |

The correction is accepted at fixture scope and was successfully verified
through the subsequent actual Talon execution.

### Successful attempt-02 operational verification

| Item | Recorded value |
| --- | --- |
| Slurm job | 317232 |
| Dataset / seed / attempt | MaleX / 42 / 02 |
| Run identity | `commita3de7e8_attempt02` |
| Authorization ID | `P1R2026-RBM-MALEX-S42-ATTEMPT02-A3DE7E8` |
| Activation source | `a3de7e8074cf1eeb319f34b50d40dd815b634e71` |
| Activation parent | `bfd9c68db58139a996aa85708069852a838e7992` |
| Requested / resolved source commit | Exact match |
| Bound Revision-2026 module paths | 6 |
| Node | talon16 |
| Slurm version | 25.05.6 |
| Scheduler state | COMPLETED |
| Exit code | 0:0 |
| Elapsed | 00:00:22 |
| Application status | SAMPLING_COMPLETE |
| Requested class | 0 — benign |
| Requested sample range | [0,16) |
| Consumed sample indices | Exactly 0–15 |
| Samples returned | 16 |
| Gibbs transitions per sample | 10 |
| Separate burn-in | 0 |
| Chain semantics | `independent_chain_per_sample` |
| Class conditioning | `fixed_requested_class` |
| Initialization | `h0_zero_v0_bounded_exponential_p_v_given_h0` |
| Internal dtype | float64 |
| Canonical dtype | float32 |
| Canonical shape | [40,40,3] |
| Pool publication | NOT_PERFORMED |
| Downstream evaluation | NOT_PERFORMED |

Checkpoint and policy identities:

| Artifact | SHA-256 |
| --- | --- |
| Checkpoint state | `18f8ddf4f1d13bd8a9c660c968019a22421f097a40068c7a7be31f29052231b8` |
| Checkpoint metadata | `55721f7414b09de5720d1003ef5f29420aa06307079cb5a36c73b8043e3c4231` |
| Consumed scientific policy digest | `676f1aca169fadf9dd30d734e33267395675ce18cc1978554dbe3939ea4d9ffa` |

The checkpoint and policy identities match the approved execution
specification. The corrected allocation validator accepted the observed Talon
`AllocTRES` representation. The existing governed checkpoint was restored and
the accepted sampling policy executed successfully.

The 16-sample result is bounded operational verification, not a complete
benchmark synthetic pool.

### Operational resource evidence

| Item | Recorded value |
| --- | --- |
| Partition | talon |
| Nodes | 1 |
| Tasks | 1 |
| CPUs per task | 8 |
| Total allocated CPUs | 8 |
| Allocated memory | 32 GB |
| Walltime limit | 600 seconds |
| Elapsed execution | 22 seconds |
| GPU allocation | None recorded |
| Requeue | 0 |
| Restarts | 0 |
| Scheduler-reported batch MaxRSS | 512 KB |

Observed controller allocation:

`AllocTRES=cpu=8,mem=32G,node=1,billing=8`

The resource envelope matched the execution authorization.

The recorded individual-job maximum was 1.3333333333333333 authorized CPU
core-hours, based on authorized CPUs multiplied by the walltime limit. This is
not measured CPU consumption.

The scheduler-reported `MaxRSS` values for both attempts appear implausibly
low and are retained as reported, not accepted as reliable
memory-consumption estimates.

Preserved:

`measured_resource_consumption=NOT_MEASURED`

`cumulative_rbm_planning_core_hours=650`

`cumulative_workload_enforcement=NOT_PERFORMED`

No full-scale synthesis feasibility or cumulative resource compliance is
claimed. The CUDA initialization diagnostic did not prevent successful
CPU-only execution. Static launcher diagnostic strings are not authoritative
runtime activation or allocation evidence.

### Attempt-02 terminal evidence and retirement

Evidence directory:

`~/trustforge_revision_2026_artifacts/paper01/rbm/operational_verification/malex_seed42_attempt02/`

| Artifact | SHA-256 |
| --- | --- |
| `slurm-317232.out` | `3c6837b09400c8c84993381754f38cec267ab52b7ff35f7c8406ee58fff9a521` |
| `slurm-317232.err` | `c0ca95ba52f5927e1c86b1abef363d48a9ff94971e4fd7b7c1121906a9a6a29c` |
| `authorization.json` | `3a3d3d8c9221b1c4ef44028c8b2255f5de9a11aa9437849f3da42c82eb318e3f` |
| `authorization_retirement.txt` | `e167c92ed16335c78822546544d15dbb7755f83167ed90e2b8607fd97fac1f39` |
| `scontrol_job.txt` | `e1016ab30dc90b86b8d422fc29022197c28079343a11b27ca65305ae4628cd08` |
| `sacct_resources.txt` | `770d6acf8594125fae66d458dc31b31dc4b1c2885517a94682195a59ea02640f` |
| `slurm_version.txt` | `c0654e25716a40eff478d8852c09f79af82bef30c69d811a9a6a9fc54037bafa` |
| `execution_result.json` | `55ada5beaa0231fdfae220490cad7bfa0c16ef230969273e1f34de703f8847fb` |

Retired authorization:

`~/trustforge_revision_2026_artifacts/paper01/rbm/authorizations/retired/malex_seed42_attempt02.317232.json`

Retirement time: 2026-11-24, 15:11:19 UTC.

Terminal reconciliation established:

`EXECUTION_RECORD_EXTRACTION=PASS`

`EXECUTION_IDENTITY_VERIFICATION=PASS`

`SCIENTIFIC_POLICY_CONSUMPTION=PASS`

`SAMPLE_COUNT=16`

`TRANSITION_COUNT=10`

`POOL_PUBLICATION=NOT_PERFORMED`

`ATTEMPT_01_PRESERVED=YES`

`ATTEMPT_02=CLOSED_COMPLETED`

`AUTHORIZATION_RETIRED=YES`

`RBM_OPERATIONAL_VERIFICATION=PASS`

`ADDITIONAL_SUBMISSIONS_AUTHORIZED=NO`

Both active authorization paths are absent. Both execution attempts and their
respective source identities remain preserved.

### Stage 3 closure rationale

The original bounded RBM execution requirements have been satisfied through:

- Accepted governed entry and launcher integration.
- Verified execution-specific activation.
- Exact requested/resolved source and import identities.
- Successful resource authorization.
- Existing checkpoint restoration.
- Consumed scientific policy and sample identity verification.
- Successful compute-node execution.
- Recorded operational resource evidence.
- Preserved failed-attempt evidence and demonstrated corrective execution.
- Retired execution authorizations.

The isolated activation approach is retained. The development RBM gate remains
disabled. No further RBM execution is required for this bounded Stage 3
milestone.

The successful 16-sample execution is not a complete benchmark synthetic pool
and does not establish Stage 4 coverage.

### Scientific and evidence limitations

Preserved:

- Historical executed training source commits: UNKNOWN.
- Historical consumed preprocessing fingerprints: UNKNOWN.
- Historical provenance investigation: CLOSED_WITH_LIMITATIONS.
- No benchmark-wide provenance waiver.
- No reliable measured memory-consumption result.
- No cumulative 650 CPU core-hour feasibility determination.
- No synthetic pool publication.
- No requested-label fidelity assessment.
- No convergence or mixing certification.
- No downstream utility result.

The historical provenance limitations and Stage 4–8 obligations remain
applicable. The researcher supplied the Talon evidence and terminal hashes;
this entry does not claim independent Talon access or independent
recomputation.

### Consolidated statuses

`RBM_ATTEMPT_01=CLOSED_FAILED`

`RBM_ATTEMPT_01_AUTHORIZATION_RETIRED=YES`

`RBM_SLURM_COMPATIBILITY_CORRECTION=ACCEPTED_AND_OPERATIONALLY_VERIFIED`

`RBM_ATTEMPT_02=CLOSED_COMPLETED`

`RBM_ATTEMPT_02_AUTHORIZATION_RETIRED=YES`

`RBM_OPERATIONAL_VERIFICATION=ACCEPTED_BOUNDED_SCOPE`

`RBM_STAGE3_BOUNDED_REQUIREMENTS=ACCEPTED_AND_CLOSED`

`RBM_HISTORICAL_PROVENANCE_INVESTIGATION=CLOSED_WITH_LIMITATIONS`

`RBM_HISTORICAL_TRAINING_SOURCE=UNKNOWN`

`RBM_HISTORICAL_PREPROCESSING_FINGERPRINT=UNKNOWN`

`RBM_PRODUCTION_SAMPLING_AUTHORIZED=NO`

`S3_O2_RBM=ACCEPTED_AND_CLOSED`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`GAN_STAGE3_BOUNDED_REQUIREMENTS=ACCEPTED_AND_CLOSED`

`STAGE_3_OVERALL=IN_PROGRESS`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

S3-O1 — Diffusion controlled activation and applicable operational
verification — remains the outstanding item in the original Stage 3 closure
register. Stage 3 overall is not closed.

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — Diffusion Stage 3 Operational Acceptance and Consolidated Stage 3 Closure — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher accepted the
successful governed Diffusion execution and closed S3-O1 at its original
bounded scope. The researcher further determined that all five outstanding
Stage 3 requirements in the original closure register have satisfied their
bounded acceptance criteria and that Stage 3 is complete at the original
agreed scope.

This entry introduces no new scientific requirement and reopens no accepted
investigation. It does not authorize Stage 4 execution or the full benchmark.

Evidence provenance: the Talon execution, scheduler, checkpoint, terminal
artifact and hash records below were supplied by the researcher and reviewed
by ChatGPT. OpenCode did not access Talon, execute training, restore any
checkpoint or independently recompute remote artifact hashes during this
documentation task. ChatGPT inspected the exact executed source to establish
the zero-based epoch convention. The researcher performed terminal
preservation and file-integrity verification.

### Diffusion attempt-01 failure preservation

| Item | Recorded value |
| --- | --- |
| Dataset / seed / attempt | CICMalDroid2020 / 42 / 01 |
| Slurm job | 317239 |
| Execution source | `4b6aa8537c98eb686b325c5878599eb45ceb800f` |
| Compute node | talon16 |
| Scheduler state | FAILED |
| Exit code | 1:0 |
| Elapsed | 9 seconds |
| Failure boundary | model-builder import |
| Immediate failure | Keras optree `ImportError` |
| Training reached | NO |
| Checkpoint publication reached | NO |
| Automatic retries | 0 |
| Execution authority | EXHAUSTED |
| Disposition | CLOSED_FAILED |

Historical evidence directory:

`~/trustforge_revision_2026_artifacts/paper01/diffusion/operational_verification/cicmaldroid2020_seed42_attempt01/`

Execution specification SHA-256:

`7307f607ffd40f88800ec0cb332c0a1fd53d88fac147ef5a8d309b9dbcffbf91`

The failed attempt remains preserved. Its scheduler outcome is not rewritten.

### Dependency diagnosis and compatibility correction

Recorded behavior:

- `optree` was installed and directly importable.
- `pip check` reported no broken requirements.
- The original production import sequence reproduced the failure on login2.
- Explicit `optree` preload before loading the production entry resolved the
  reproduced failure.
- The precise underlying Keras/native-library mechanism remains UNKNOWN.

Controlled comparison artifact SHA-256:

`f24c21e541e92e10700e3355fd977521382139b252f63cb0e7e1b70602c0f878`

Accepted corrective commit:

`d58c660701c3e5e4cf28ea7e5141ffb0c28914f8`

Commit message:

`fix(paper01): preload optree for governed Diffusion entry`

The correction adds explicit early `optree` initialization in the governed
production entry before TensorFlow/Keras-dependent production imports. No
scientific methodology, architecture, dataset preprocessing, training
mathematics, checkpoint implementation, launcher or package environment was
changed.

Verification:

- Three focused tests passed.
- 31 tests deliberately deselected.
- Runtime: 4.80 seconds.
- Python compilation: PASS.
- Git diff integrity: PASS.
- No model construction or optimizer steps.

Corrected production entry SHA-256:

`c17ea751c96f47280b4f3ccc50283d74c1b6a32c43fd041c7d8d9c49f04d1e2b`

Focused regression test module SHA-256:

`32a3876d3cda536f5bf94f4cb4fd45f9b204e1055f41f4382017b96e06ee4a43`

Corrected Talon login-node import verification passed without manual optree
preloading. Evidence directory:

`~/trustforge_revision_2026_artifacts/paper01/diffusion/operational_verification/import_correction_d58c660/`

Verification artifact SHA-256:

`bdacba46bf159d3122f25bdf9cec8b86d5d864f9f24fd5d5a2f2ecd5d2588a65`

The subsequent successful compute-node training execution establishes
operational effectiveness of the correction.

### Successful attempt-02 execution

| Item | Recorded value |
| --- | --- |
| Dataset / seed / attempt | CICMalDroid2020 / 42 / 02 |
| Slurm job | 317249 |
| Run identity | `commitb5d8178_attempt02` |
| Activation source | `b5d8178fd894e6da46181809bf0b986b2b5f9712` |
| Activation parent | `d58c660701c3e5e4cf28ea7e5141ffb0c28914f8` |
| Requested / resolved source commits | Exact match |
| Source status | CLEAN |
| Bound Revision-2026 modules | 11 |
| Exact model builder | `diffusion/models.py` in the activated checkout |
| Compute node | talon08 |
| Slurm version | 25.05.6 |
| Scheduler state | COMPLETED |
| Exit code | 0:0 |
| Elapsed | 00:13:03 |
| Authorized submissions | 1 |
| Actual submissions | 1 |
| Automatic retries | 0 |
| Requeue | 0 |
| Restarts | 0 |
| Runtime activation | Isolated source-level gate |
| Development Diffusion gate | False |
| Development RBM gate | False |

The exact activation diff contains only the approved `False` to `True` training
gate change. The isolated activation branch is retained without merging into
development.

### Consumed scientific configuration and provenance

| Item | Recorded value |
| --- | --- |
| Dataset ID | cicmaldroid2020 |
| Classes | 5 |
| Class mapping | `["0", "1", "2", "3", "4"]` |
| Training examples | 8,117 |
| Validation examples | 1,158 |
| Held-out test examples | 2,323 |
| Partition policy | `cic_case3_revision_2026` |
| Partition seed | 42 |
| Partition fractions | 0.70 / 0.10 / 0.20 |
| Canonical observation shape | [40, 40, 3] |
| Architecture | Conditional convolutional U-Net |
| Depth | 2 |
| Base filters | 64 |
| Time embedding dimension | 128 |
| DDPM schedule | Linear |
| Timesteps | 1,000 |
| Beta start / end | 0.0001 / 0.02 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam beta_1 | 0.9 |
| Objective | Epsilon-prediction MSE |
| Completed epochs | 2 |
| Completed optimizer updates | 254 |
| Early stopping | None |
| Checkpoint selection | Minimum validation MSE |
| Tie-break | Earliest epoch |
| Held-out test usage | Prohibited |

Authoritative source and configuration hashes:

| Artifact | SHA-256 |
| --- | --- |
| Source CSV | `3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00` |
| Feature-column JSON | `7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f` |

Compute-node source verification method:

`sha256_full_file_stream_on_compute_node`

Source contract:

`cic_authoritative_source_revision_2026`

Preserved provenance limitations:

- `does_not_authenticate_checkpoint_origin`
- `does_not_prove_continuous_source_immutability`
- `does_not_prove_training_used_only_verified_bytes`
- `does_not_independently_verify_partition_execution`

Preserved fitted-scaler limitation: the Diffusion adapter refits
`StandardScaler` on the governed training partition but does not record a
separate consumed fitted-state digest. This bounded acceptance is not a waiver
of future benchmark-wide reproducibility obligations.

### Training and checkpoint publication

| Item | Recorded value |
| --- | --- |
| Completed epochs | 2 |
| Completed updates | 254 |
| Best validation MSE | 0.034428963337949314 |
| Best epoch index | 1 |
| Last epoch index | 1 |
| Epoch convention | Zero-based |
| Best epoch ordinal | 2 |
| Last epoch ordinal | 2 |
| Checkpoint status | PREPARED |
| Publication status | PUBLISHED |
| Primary checkpoint role | best |
| Pending publication marker | Absent |

Interpretation: the training runner uses `range(epochs)` and checkpoint-state
preparation assigns the `last` role to `APPROVED_EPOCHS - 1`. The stored epoch
index `1` therefore denotes the second and final epoch. Both best and last
refer to the same final epoch; identical weights and validation MSE are
expected. No checkpoint-selection defect is established. No corrective
training or attempt 03 is required.

Checkpoint directory:

`~/trustforge_revision_2026_artifacts/paper01/diffusion/cicmaldroid2020/seed42/checkpoints/production/commitb5d8178_attempt02/`

| Item | Recorded value |
| --- | --- |
| Manifest schema | `trustforge.paper01.diffusion-checkpoint-bundle.v1` |
| Manifest bytes | 15,527 |
| Best tensor files | 46 |
| Last tensor files | 46 |
| Tensor-file integrity checks | 92/92 PASS |
| Reported disk consumption | 20M |
| Manifest SHA-256 | `8cbeb041d58d71dc8d2828a949afe78c8d75d30a357315b36cb9331d9608b0fe` |
| Reported best/last aggregate state SHA-256 | `6a94707ab23322fdd145bd40f864fb0bbee956a707404b9377a2cfcdfebfc459` |

File-level integrity verification is distinguished from aggregate array-state
digest recomputation and independent restoration. Neither aggregate digest
recomputation nor independent model restoration was performed.

### Operational resource evidence

| Item | Recorded value |
| --- | --- |
| Partition | talon |
| Nodes | 1 |
| Tasks | 1 |
| CPUs per task | 8 |
| Allocated CPUs | 8 |
| Allocated memory | 32 GB |
| GPU allocation | None recorded |
| Walltime limit | 8 hours |
| Actual elapsed time | 783 seconds |
| Maximum authorized allocated CPU core-hours | 64 |
| Actual allocated CPU core-hours | 1.74 |
| Slurm-reported TotalCPU | 57:09.525 |
| Derived reported CPU efficiency | Approximately 54.75% |
| Scheduler-reported batch MaxRSS | 512K |
| Actual peak memory consumption | UNKNOWN |

Allocated core-hours are not measured CPU consumption. The 512K MaxRSS is
implausibly low and is retained only as reported. The terminal `scontrol` query
could no longer retrieve the completed job; the earlier controller allocation
record and terminal `sacct` accounting establish the resource allocation.

The resource envelope was operationally sufficient for this bounded execution
only. No full benchmark resource feasibility claim is established.

### Terminal evidence preservation

Attempt-02 evidence directory:

`~/trustforge_revision_2026_artifacts/paper01/diffusion/operational_verification/cicmaldroid2020_seed42_attempt02/`

| Artifact | SHA-256 |
| --- | --- |
| `execution_specification.json` | `cd269af24e78c6b73588867f0b58f42eab40e8dad106635a558593b546683acd` |
| `submission_attempted.txt` | `ae06ef3609b1045d4eb1796c05de2608c858798402d6212cba537ae32c1567a4` |
| `submission.stdout` | `797a95ccbf86099028f7827be809221d1873b8c6a1cce9dbe775a919904c30a2` |
| `submission.stderr` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `job_id.txt` | `797a95ccbf86099028f7827be809221d1873b8c6a1cce9dbe775a919904c30a2` |
| `slurm-317249.out` | `1286e12f10cf2f5caa7147227a5cc23adc96f3168c0e92b591d2ddda3fab32f3` |
| `slurm-317249.err` | `f8eff4c409c7152a35ede51fffb2bd0380f26589e5470f59027c637b52b4d247` |
| `sacct_resources.txt` | `5df7c1f7656c0b37557f7fdd75815239934adf7621b1d1100c5b28265aace610` |
| `slurm_version.txt` | `c0654e25716a40eff478d8852c09f79af82bef30c69d811a9a6a9fc54037bafa` |
| `execution_result.json` | `a420e4b14676a7882784cd6162a7377bb1fb0db1f60f58c479225c8c16ca731d` |
| `checkpoint_integrity_report.json` | `aa8db8c28ea3792ebeffbf9894af5857dde70632722edaacd3fb9e7dddbf99e2` |
| `execution_closure.txt` | `4493b7865e4f4e8ccb9b9be35fbd9de59b4bbdc77002a246e216d38d2f0207af` |
| Checkpoint `manifest.json` | `8cbeb041d58d71dc8d2828a949afe78c8d75d30a357315b36cb9331d9608b0fe` |

All thirteen identities were successfully reverified by the researcher.

Closure time: 2026-11-24 16:59:52 UTC.

Attempt-02 disposition: `CLOSED_COMPLETED`.

Execution authority: `EXHAUSTED`. Runtime grant present: `NO`. Additional
submissions authorized: `NO`.

All fifteen historical attempt-01 and compatibility verification artifacts
were also reverified.

### Original S3-O1 requirement reconciliation

| Requirement | Accepted bounded disposition | Preserved limitation |
| --- | --- | --- |
| R1 requested/resolved/consumed identity | ACCEPTED — exact source, dataset, seed, attempt, run identity and consumed configuration | None at bounded scope |
| R2 source and import binding | ACCEPTED — clean checkout, 11 module paths, exact `diffusion/models.py` builder | None at bounded scope |
| R3 dataset and preprocessing provenance | ACCEPTED — authoritative source hashes, Case-3 partition and source contract | Fitted-scaler digest not separately captured; provenance limitations preserved |
| R4 launcher integration | ACCEPTED — actual launcher execution with explicit identity and resource envelope | None at bounded scope |
| R5 controlled activation | ACCEPTED — isolated one-line source gate, no merge to development | Global gate has no execution-specific grant; operational discipline enforced |
| R6 configuration, failure, fallback, retry and resume | ACCEPTED — gate ordering, no early stopping, no retry/resume, actual fail-closed attempt-01 and successful attempt-02 | No dataset/seed-wide failure matrix |
| R7 operational resource evidence | ACCEPTED — actual allocation, elapsed and accounting | Peak memory UNKNOWN; 64 core-hour ceiling not a feasibility result |

`S3_O1_DIFFUSION=ACCEPTED_AND_CLOSED`

### Original Stage 3 exit-criteria reconciliation

The governing Stage 3 exit condition is:

> Confirm requested → resolved → consumed source/config/data identity, gates,
> retries, and failure behavior.

The Stage 3 register additionally requires accepted source/entry paths for each
distinct family integration rather than full dataset or full seed coverage.
No original Stage 3 passage requires the complete 196-cell benchmark.

The previously outstanding Stage 3 register items were:

- Diffusion controlled activation and operational evidence.
- RBM governed production entry and activation.
- GAN controlled activation.
- GAN synthesis resource evidence.
- GAN training execution identity.

They are now represented by S3-O1 through S3-O5. Earlier accepted ledger
entries establish S3-O2 through S3-O5. No genuine original Stage 3 requirement
outside this five-item register remains unresolved. The 2026-11-23 phrase
"remaining distinct paths" referred to the per-family outstanding paths now
covered by the accepted family records below. Dataset/seed coverage,
restoration, synthesis, evaluation and aggregation remain later-stage
obligations.

`STAGE_1_OVERALL=COMPLETE`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

### Consolidated five-item closure register

`S3_O1_DIFFUSION=ACCEPTED_AND_CLOSED`

`S3_O2_RBM=ACCEPTED_AND_CLOSED`

`S3_O3_GAN_CONTROLLED_ACTIVATION=ACCEPTED_AND_CLOSED`

`S3_O4_GAN_SYNTHESIS_RESOURCE_EVIDENCE=ACCEPTED_AND_CLOSED`

`S3_O5_GAN_TRAINING_EXECUTION_IDENTITY=ACCEPTED_AND_CLOSED`

### Consolidated seven-family bounded Stage 3 readiness

| Family | Demonstrated bounded scope | Not demonstrated at Stage 3 closure |
| --- | --- | --- |
| GAN | CICMalDroid2020 seed-42 activation, checkpoint, synthesis and paired evaluation cell | Other datasets/seeds |
| VAE | CICMalDroid2020 seed-42 train, checkpoint, restore, synthesis and paired evaluation | Other datasets/seeds |
| Diffusion | CICMalDroid2020 seed-42 training and durable best/last publication | Fresh-process restoration, synthesis, pool |
| Autoregressive | CICMalDroid2020 seed-42 training/checkpoint and complete synthesis pool | Paired downstream evaluation |
| MAF | CICMalDroid2020 seeds 42–44 train, checkpoint, full pool and paired evaluation | Seeds 45–48 and other datasets |
| RBM | MaleX seed-42 16-sample bounded production sampling | Full synthetic pool and all-class coverage |
| GMM | CICMalDroid2020 class-0 seed-42 train, checkpoint, synthesis and pool verification | Other classes and downstream evaluation |

These scopes are not interchangeable. Stage 3 completion does not imply
identical end-to-end coverage across families.

### Preserved Stage 4–9 obligations

Stage 3 completion does not establish complete production readiness.

- Diffusion durable publication is demonstrated; independent restoration and
  counted production synthesis remain outstanding.
- RBM bounded sampling is demonstrated; complete synthetic pool publication is
  not established.
- GMM class-level evidence is not all-class completion.
- Remaining downstream evaluation and production aggregation obligations are
  preserved.
- Clean-environment reproducibility and integrated workflow validation remain
  subject to their original gates.
- Full benchmark authorization remains withheld.
- No Stage 4 experimental execution is authorized by this entry.

### Consolidated statuses

`DIFFUSION_ATTEMPT_01=CLOSED_FAILED`

`DIFFUSION_ATTEMPT_01_EXECUTION_AUTHORITY=EXHAUSTED`

`DIFFUSION_COMPATIBILITY_CORRECTION=ACCEPTED_AND_OPERATIONALLY_VERIFIED`

`DIFFUSION_ATTEMPT_02=CLOSED_COMPLETED`

`DIFFUSION_ATTEMPT_02_EXECUTION_AUTHORITY=EXHAUSTED`

`DIFFUSION_TRAINING_COMPLETION=ACCEPTED_BOUNDED_SCOPE`

`DIFFUSION_CHECKPOINT_PUBLICATION=ACCEPTED`

`DIFFUSION_TENSOR_FILE_INTEGRITY=92_OF_92_MATCHED`

`DIFFUSION_EPOCH_INDEXING=ZERO_BASED`

`DIFFUSION_PEAK_MEMORY_CONSUMPTION=UNKNOWN`

`DIFFUSION_FRESH_PROCESS_RESTORATION=NOT_PERFORMED`

`DIFFUSION_SYNTHESIS_AUTHORIZED=NO`

`DIFFUSION_ADDITIONAL_TRAINING_REQUIRED_FOR_STAGE3=NO`

`S3_O1_DIFFUSION=ACCEPTED_AND_CLOSED`

`S3_O2_RBM=ACCEPTED_AND_CLOSED`

`S3_O3_GAN_CONTROLLED_ACTIVATION=ACCEPTED_AND_CLOSED`

`S3_O4_GAN_SYNTHESIS_RESOURCE_EVIDENCE=ACCEPTED_AND_CLOSED`

`S3_O5_GAN_TRAINING_EXECUTION_IDENTITY=ACCEPTED_AND_CLOSED`

`STAGE_1_OVERALL=COMPLETE`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — S4-0 Completed Family Integrations and Finite Stage 4 Closure Register — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher accepted the
documentation-only closure of the previously accepted representative Stage 4
integrations for GAN, VAE, Autoregressive and MAF, and accepted the finite
Stage 4 closure register S4-0 through S4-A1.

This entry is append-only and does not authorize implementation, training,
checkpoint restoration, synthesis, evaluation or Slurm execution. The ordering
of the remaining GMM, RBM and Diffusion work packages is provisional.

Evidence provenance: the execution, artifact, scheduler and hash records below
were retrieved from existing authoritative committed ledger entries. OpenCode
did not access Talon, restore checkpoints, execute experiments or independently
recompute remote artifact hashes during this documentation task.

### Governing Stage 4 exit condition

Research Continuity Charter §6:

> **Checkpoint/synthesize:** Durable, integrity-checked publication; fresh
> restoration; identity-safe, fully counted synthesis for each distinct family
> integration.

Ledger Stage 4 additionally retains:

> Complete checkpoint and synthesis integration — Verify durable restoration
> and correct labeled sampling for every distinct family path.

and:

> Stage 4 separately retains durable checkpoint publication, independent
> checkpoint restoration, checkpoint-backed synthesis, complete class quotas and
> pool publication.

Representative integration scope is preserved. The 196-cell benchmark is not a
precondition for Stage 4 closure.

### S4-0 GAN representative integration — CICMalDroid2020 seed 42

| Item | Accepted evidence |
| --- | --- |
| Training job | 314528 |
| Training activation source | `42c40fce8b7f1005301408444fdcae7ff7f19db5` |
| Checkpoint run identity | `commit42c40fc_attempt01` |
| Checkpoint identity | `963b45ba5c467fe935e13a34bfb3abae52fe52636eebfd796787c04526de6530` |
| Initial restoration | Job 314534 — FAILED, preserved |
| Import diagnostic | Job 314535 — COMPLETED |
| Restoration retry | Job 314537 — COMPLETED |
| Production synthesis | Job 314574 — COMPLETED |
| Synthesis source | `4324aeecb45a47a72ac0b479d8cd994fe79dc1e0` |
| Synthesis recipe | `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9` |
| Pool manifest | `b4a640411b8120c412e51d2aa68b9e110ad2bfcc0215a1d8b20e6d7776dbceb1` |
| Pool content | 10,000 images; 5 classes × 2,000; 625 batches |
| Paired evaluation | Job 314607 |

The historical failed restoration at 314534, its diagnostic at 314535 and the
successful retry at 314537 are preserved. GAN restoration is accepted in its
own evidence format; the absence of a VAE-style standalone restoration report
is not a failure of the accepted requirement.

Accepted pool verification: the evaluation driver authenticated the pool
manifest, checkpoint identity, 625 ordered batches and pool consumption.
`synthesis_record_authentication` remained `NOT_ARTIFACT_VERIFIED`, and
semantic class fidelity remained `NOT_ESTABLISHED`.

`S4_GAN=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

### S4-0 VAE representative integration — CICMalDroid2020 seed 42

| Phase | Accepted evidence |
| --- | --- |
| Stage 3 training-only | Job 316716 — no production checkpoint created |
| Stage 4 training + publication | Job 316796; source `1aec01c26fab51b87362a47e2cfa797bd428c359` |
| BEST generation | `35ca3cd756a88d6b93dcc291f02c4747` |
| BEST manifest | `f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e` |
| Independent restoration | Job 316811 |
| Production synthesis | Job 316980; source `d4031a9b39565e5382d08e2b28236162498a08ca` |
| Synthesis recipe | `98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9` |
| Pool manifest | `08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6` |
| Pool content | 10,000 images; 5 classes × 2,000; 315 batches |
| Independent pool verification | PASS — 315 batches / 10,000 images |
| Paired evaluation | Job 316983 |

Job 316716 is preserved as a Stage 3 training-only execution. The Stage 4
checkpoint lineage is job 316796 and the published BEST generation above.

`S4_VAE=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

### S4-0 Autoregressive representative integration — CICMalDroid2020 seed 42

| Item | Accepted evidence |
| --- | --- |
| Training source | `e4473abcdbc53f0de3a01ebe48858f2f3c7b841a` |
| Checkpoint SHA-256 | `d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0` |
| Synthesis source | `d7199228b19f9d67d747ca06a31c02230af00fb9` |
| Dataset configuration SHA-256 | `ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b` |
| Recipe SHA-256 | `202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478` |
| Worker array | 314793 — 50/50 tasks COMPLETED |
| Finalizer | 314794 — COMPLETED |
| Pool manifest | `ce2b6c6a332b578ad9f67091cf035f15a57e55e88eb22d9f98ade10ef94835e2` |
| Pool content | 10,000 images; 5 classes × 2,000; 1,250 batches |

Fresh best-checkpoint restoration was accepted in the AR/MAF synthesis chain
(jobs 311064/311065) and the checkpoint-backed AR production synthesis chain
above. Downstream evaluation remains a Stage 5 obligation and is not a Stage 4
blocker.

`S4_AR=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

### S4-0 MAF representative integration — CICMalDroid2020 seeds 42–44

| Seed | Accepted evidence |
| --- | --- |
| 42 | Training/checkpoint path source `913d86f9c64e6a3a1a0d009f112bcfa889a02ec8`; synthesis source `8c58c4e8967da1764e6d7c66397e78befcbcadac`; array 313821; finalizer 313832; manifest `db884e298ea324597bad2d488d03e2c86c205e7a53bbef2a11bd72f06da5651e`; 10,000 / 2,000 per class / 1,250 batches |
| 43 | Generator source `58f965c39aad103dc056f7ac690de90578ed2932`; checkpoint `57e378c44c870af58e4b8eff38cf1cc976516b0dade249194bb5962c28a724f7`; pool manifest `95b1d78e90cb8794573b88a195bb1e89fbbfcfcfc7d9562a5210a9971527354f`; 10,000 / 2,000 per class / 1,250 batches; evaluation job 314240 |
| 44 | Generator source `8d4a281e04ed2232a134a6da10ff20c56d67316e`; checkpoint state `cc58fe65abaa0656e4f09d6d17d83e586f048f50a6a858d01e1f1d7b9191055a`; synthesis source `8198dc4d42fdb05bc3ebb6fd055b9951ab3d897f`; array 314349; finalizer 314354; manifest `c5fc5cb6aa7732b356b2bb57bfbd28d58f05b771ef50f278456111673fed0cb5`; 10,000 / 2,000 per class / 1,250 batches; evaluation 314355 |

Seed 43 worker/finalizer identities were not recorded in the accepted
evaluation entry and are marked UNKNOWN rather than inferred. Seed 44 and
seed 42 full-pool verifications were accepted at their recorded scopes.

Preserved limitations include generator-lineage authentication limits,
`NOT_ESTABLISHED` semantic class fidelity where recorded, and unperformed
feature-extraction/provenance/publication validation where the accepted
entries state so. Seed 45 is not authorized.

`S4_MAF=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

### S4-0 closure

`S4_0_COMPLETED_FAMILY_INTEGRATIONS=ACCEPTED_AND_CLOSED`

`S4_GAN=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_VAE=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_AR=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_MAF=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

### Finite outstanding Stage 4 register

| ID | Requirement | Existing accepted evidence | Remaining gap | Minimum necessary action | Authorization |
| --- | --- | --- | --- | --- | --- |
| S4-G1 | GMM representative all-class checkpoint-backed pool | CIC class-0 seed-42 training, checkpoint, restoration, synthesis and pool; class-specific methodology and pool implementation | Class-general production pool integration; classes 1–4 checkpoint availability UNKNOWN; complete five-class accounting and verification | Reconcile class 1–4 checkpoint availability and reuse eligibility before any retraining; generalize class pool integration; produce and verify five class pools totalling 10,000 images | Separate implementation/training/synthesis authorization |
| S4-R1 | RBM representative complete MaleX seed-42 pool | Historical checkpoint; bounded restoration and 16-sample production sampling; approved T=10 policy; production entry | Multi-class range orchestration; durable pool publication; 4,000-sample coverage; resource planning/accounting | Implement or reuse range and pool publication for both MaleX classes; execute one seed-42 pool; verify 2 × 2,000 | Separate implementation/execution authorization; 650 core-hour planning ceiling |
| S4-D1 | Diffusion representative CIC seed-42 checkpoint-backed synthesis | Job 317249 training and durable publication; 92/92 tensor files; loader, sampler, recipe and linkage | Independent fresh-process restoration; production synthesis entry/launcher; 10,000-sample counted synthesis; independent pool verification | Reuse checkpoint; implement production synthesis/pool path; restore, synthesize five classes × 2,000, publish and verify | Separate implementation/execution authorization |
| S4-A1 | Consolidated Stage 4 exit | S4-0 plus above | Final reconciliation of all seven families | Documentation-only after S4-G1, S4-R1, S4-D1 | Researcher documentation approval |

S4-G1 checkpoint availability must be reconciled before training is proposed;
classes 1–4 are not declared nonexistent. S4-R1 must not reopen Gibbs-depth,
mixing or convergence questions. S4-D1 must not propose additional training and
must preserve zero-based epoch indexing and accepted best/last equality.

### Subsequent-stage boundaries

Stage 5–9 obligations remain separate: downstream evaluation, production
aggregation, scale-memory acceptance, clean-environment reproduction,
integrated rejection-suite verification, source/environment/resource freeze,
full benchmark authorization and manuscript release. They are not moved into
Stage 4.

### Status

`STAGE_1_OVERALL=COMPLETE`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`S4_0_COMPLETED_FAMILY_INTEGRATIONS=ACCEPTED_AND_CLOSED`

`S4_GAN=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_VAE=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_AR=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_MAF=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_G1_GMM=OUTSTANDING`

`S4_R1_RBM=OUTSTANDING`

`S4_D1_DIFFUSION=OUTSTANDING`

`S4_A1_CONSOLIDATED_EXIT=PENDING`

`STAGE_4_OVERALL=IN_PROGRESS`

`NEW_SCIENTIFIC_DECISIONS_REQUIRED_FOR_STAGE4=NONE_IDENTIFIED`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — S4-G1 GMM Representative All-Class Integration Closure — 2026-11-24

RESEARCHER ACCEPTANCE RECORD. On 2026-11-24 the researcher accepted the
completed representative GMM CICMalDroid2020 seed-42 all-class checkpoint-backed
synthetic pool integration and prepared this S4-G1 closure entry for
preservation.

The scope is one representative dataset/seed integration, not full GMM
benchmark completion. No new scientific requirement is introduced.

Evidence provenance: the Talon execution, scheduler, artifact and terminal hash
records below were supplied by the researcher from preserved execution results
and reviewed by ChatGPT. OpenCode did not access Talon, restore checkpoints,
execute synthesis or independently recompute remote artifact bytes during this
documentation task.

### Governing Stage 4 exit condition

Research Continuity Charter §6:

> **Checkpoint/synthesize:** Durable, integrity-checked publication; fresh
> restoration; identity-safe, fully counted synthesis for each distinct family
> integration.

Ledger Stage 4 additionally retains durable checkpoint publication,
independent checkpoint restoration, checkpoint-backed synthesis, complete class
quotas and pool publication. Representative integration scope applies.

### S4-G1 requirement and closure basis

S4-G1 required one representative complete GMM CICMalDroid2020 seed-42
checkpoint-backed pool covering all five classes and 2,000 samples per class.

The completed work comprises:

1. Reconciliation of historical checkpoint availability.
2. Class-general pool publication and verification implementation.
3. Acquisition of missing class 1–4 checkpoints.
4. Fresh-process restoration of those four checkpoints.
5. Durable publication and independent verification of four new class pools.
6. Reuse and compatibility verification of the historical class-0 pool.
7. Five-class verification totalling 10,000 samples.
8. Preservation and closure of both execution authorizations.

The scoped inventory found only class 0 in the inspected governed namespace.
That inventory does not prove global nonexistence of historical checkpoints
elsewhere.

### Implementation evidence

Accepted implementation commit:

`6cc0d71e977d4187f0986544496c793aa38b583b`

Commit message:

`feat(paper01): generalize GMM class pool publication and verification`

Changed files (3; 401 insertions, 25 deletions):

- `src/trustforge/models/paper01_revision_2026/gaussian_mixture_class_pool.py`
- `scripts/publish_paper01_gmm_class_pool.py`
- `tests/test_gmm_production_class_pool_revision_2026.py`

Focused fixture tests: 30 passed, reported duration 4.94 seconds.

Class-general dataset/class/experimental-seed identity handling and complete
dataset-pool verification were accepted. The existing class-pool schema,
sampling methodology, 2,000-sample class budget and batch size eight were
preserved. No new scientific decisions were required.

The existing v4 training source-freeze document was reused successfully at the
new execution revision:

`395751118f1549626841e3fe2167caefd672cf7d1740d0020f0425fc6c2f123a`

### Execution and artifact locations

Talon execution checkout:

`$HOME/TrustForge-paper01-gmm-cic-seed42-stage4`

Clean detached source revision:

`6cc0d71e977d4187f0986544496c793aa38b583b`

Artifact cell root:

`$HOME/trustforge_revision_2026_artifacts/paper01/gmm/cicmaldroid2020/seed42`

Training evidence directory relative to the artifact cell:

`operational_verification/stage4_classes1_to_4_commit6cc0d71_attempt01/`

Synthesis evidence directory relative to the artifact cell:

`operational_verification/stage4_synthesis_classes1_to_4_commit6cc0d71_attempt01/`

Existing class-0 checkpoint:

`production/class0_seed42_commit6b56e92_attempt02/`

New class 1–4 checkpoints:

`production/class{class_id}_seed42_commit6cc0d71_attempt01/`

Existing class-0 pool:

`pools/class0_seed42_synthesis9fe24e8_attempt01/`

New class 1–4 pools:

`pools/class{class_id}_seed42_synthesis6cc0d71_attempt01/`

### Training acceptance evidence

Historical class-0 training job: 313922.
Historical class-0 training source:
`6b56e92d906f2f0d6149b3e74ff40bb5277f45ab`.

New class 1–4 training source:
`6cc0d71e977d4187f0986544496c793aa38b583b`.

| Class | Training job | Train count | Validation count | Maximum updates | Validation interval | Selected step | Selected validation NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 313922 | 877 | 125 | 100 | 20 | 100 | -7533.081822699489 |
| 1 | 317311 | 1470 | 210 | 100 | 20 | 100 | -8322.46310957256 |
| 2 | 317312 | 2732 | 390 | 110 | 22 | 110 | -9754.151403607893 |
| 3 | 317313 | 1782 | 254 | 100 | 20 | 100 | -6279.135953301703 |
| 4 | 317314 | 1256 | 179 | 100 | 20 | 100 | -6718.665788788249 |

All four new jobs completed with exit code 0:0, zero requeue and zero restarts.

All four new checkpoint metadata/state pairs passed exact identity,
state-integrity, governed budget and checkpoint-selection verification.

Total real training coverage: 8,117. Total real validation coverage: 1,158.

Class-dependent training budget execution is recorded without claiming
convergence or changing the governed budget. All five checkpoints use five
components, dimension 4,800, float64 parameters and the governed censored
diagonal Gaussian mixture observation model.

### Accepted checkpoint hashes

| Class | Metadata SHA-256 | State SHA-256 |
|---|---|---|
| 0 | `e43011eccaaace1308c6e2b34fa151d8cdacc682ad832fc0e2eff632d0f5b58f` | `9452ebf1c780ceb65f39bebf9cca036cb21a3c88f6cff0eed634ddffdf897769` |
| 1 | `bab0b2dd4609be1c2dd88d7ac3a5c3b31265a6466d8f37d886dcd3f3f4ecf54a` | `99c4dad68492d929344824e7d3d5eaa22ce7fa4d3fc9f020e788730096811f1d` |
| 2 | `d3072d225bd87268c28411931c292200f0470b49860f5c504f0cc23a94cae589` | `90d2ec52f69b2459c691d8b0710b521a367b1821786594425816069bdbe1c6ea` |
| 3 | `6feffcbf5827dbe8edf99be5c725aad9a85e5e53d9cba5cc4a9c97e799f157f7` | `b13a5a864f21723d68f8a73f4f9dd0d3eb1b6f9149e7d5927958d47317c4b4d4` |
| 4 | `4c80258e81da66a961dfd353e0bef3cd8b4e67d79084fb0ab5d4c81eaf420356` | `8fa66f0018a088e4fac112135805a5e9ee5d5fd84ef5533dbd75ec5af9ca0bbe` |

All five state files are 384,800 bytes.

### Restoration and synthesis acceptance

Historical class-0 synthesis job: 313927.
Historical class-0 synthesis source:
`9fe24e8f2adf76d74442d281790405df21d22a99`.

New synthesis source:
`6cc0d71e977d4187f0986544496c793aa38b583b`.

| Class | Synthesis job | Manifest SHA-256 |
|---|---:|---|
| 0 | 313927 | `68b72e975927b6d804749ceab821a52b9b8f5656d5ad776ace672fab9aea9291` |
| 1 | 317315 | `528ae8180c0231fd4c62be7994529832a875104e6f951b178ecbd3e484648280` |
| 2 | 317316 | `729c1231da7e15d2aee63d1025b54142d2af69a51a68e727d4539091dcb3cbb9` |
| 3 | 317317 | `fa945c01edd0eb8aff6210787a7935fbe58f77607227c8e920e3e84fc2bcc655` |
| 4 | 317318 | `5b42fc77f7c99326e9922486e0f804acd42b22b4ade860336f7e8a88dc3e7735` |

All four new jobs completed with exit code 0:0 and zero restarts. Each new job
performed independent fresh-process checkpoint restoration, production
synthesis and durable publication, separate-process pool verification, and
exact checkpoint metadata/state hash preservation.

All five manifests were subsequently independently verified through
`verify_gmm_dataset_pools()` using their explicit accepted paths.

Final verified coverage:

- Dataset: cicmaldroid2020
- Family: gmm
- Experimental seed: 42
- Classes: 0, 1, 2, 3, 4
- Samples per class: 2,000
- Batch size: 8
- Batches per class: 250
- Total batches: 1,250
- Newly generated samples: 8,000
- Historical class-0 samples reused: 2,000
- Total verified samples: 10,000
- Canonical representation: NHWC float32, 40 × 40 × 3, finite values in [0,1]
- Schema: `trustforge.paper01.gmm-class-pool.v1`

The class-0 artifacts were preserved and accepted under the generalized
verifier without regeneration.

### Resource accounting

New training allocation: 8 CPUs, 32 GB, one-hour walltime per job.
New restoration/synthesis allocation: 4 CPUs, 16 GB, 30-minute walltime per
job. Maximum observed concurrency: two for both work packages.

| Class | Training elapsed seconds | Training node | Synthesis elapsed seconds | Synthesis node |
|---|---:|---|---:|---|
| 1 | 38 | talon16 | 78 | talon06 |
| 2 | 94 | talon07 | 21 | talon07 |
| 3 | 39 | talon08 | 22 | talon06 |
| 4 | 33 | talon16 | 20 | talon07 |

Training allocated CPU core-seconds: 1,632.
Training allocated CPU core-hours: 0.4533333333333333.
Training authorized ceiling: 32 core-hours.

Synthesis allocated CPU core-seconds: 564.
Synthesis allocated CPU core-hours: 0.15666666666666668.
Synthesis authorized ceiling: 8 core-hours.

Combined actual allocated CPU core-hours: 0.61.
Combined authorized ceiling: 40 core-hours.

Historical class-0 execution and login-node verification are excluded from
these combined totals.

Training MaxRSS evidence: class 1 640K (unreliable); class 2 1,087,200K;
class 3 512K (unreliable); class 4 520K (unreliable).

Synthesis MaxRSS evidence: class 1 839,892K; class 2 590,620K; class 3 589,828K;
class 4 590,288K.

Full-benchmark resource requirements must not be inferred from this single
representative dataset/seed.

### Terminal evidence preservation

Training authorization ID:

`P1R2026-GMM-CIC-S42-C1TO4-A01-20261124`

Training closure time: `2026-11-24T18:53:04Z`.

Training evidence:

| Artifact | SHA-256 |
|---|---|
| Execution specification | `dee6da9966374c60c9f69dddae35f59362d96f7cae5966f06879761274923e71` |
| Checkpoint integrity report | `b6e69cd57f13d245bb43981efa94f280d302060aa4345e430ee907dc549c5792` |
| Terminal accounting | `4eeb1163009df3c700f1298ba087662260946b3711dbdc59ba53958ac94387b9` |
| Execution closure | `de91597cae832482166dff029208cc4091494a948fa27e501a5315fcf68da0b9` |

Training preservation verified 20 submission evidence hashes, 16
checkpoint/log hashes and 13 terminal evidence hashes.

Synthesis authorization ID:

`P1R2026-GMM-CIC-S42-SYNTH-C1TO4-A01-20261124`

Synthesis closure time: `2026-11-24T19:00:26Z`.

Synthesis evidence:

| Artifact | SHA-256 |
|---|---|
| Execution specification | `d1fd95c9712bc58dda0f87fde6082a1b43f99295bc723e34d4c3e36e798fc687` |
| External Slurm wrapper | `a9814caae550da5d30d3ce4f15f612857eab690bda79c391b40414272eb03b88` |
| Complete dataset pool report | `c05d9902da870952d0acb163491d9b931ac19eaa95482704dc76f7956e35dea8` |
| Terminal accounting | `0561a545a5381cfd97b66145179257a1aedcbf84d0461bf84e19d5b7b5a83efc` |
| Execution closure | `1a76972370470fd0c899bfdfc9ff159fd5b225ebcc065ebdeb8f2770395d03d0` |

Synthesis preservation verified 20 submission evidence hashes and 28 terminal
evidence hashes.

Both execution authorizations are `CLOSED_COMPLETED` with submission authority
`EXHAUSTED`.

The historical synthesis execution closure artifact retains
`RESEARCHER_ACCEPTANCE=PENDING_REVIEW`; this entry records the later researcher
acceptance recommendation without altering that external artifact.

### Preserved limitations

- Representative CICMalDroid2020 seed-42 integration only, not full GMM
  benchmark completion.
- Semantic class fidelity `NOT_ESTABLISHED`.
- Downstream evaluation `NOT_PERFORMED` for the complete GMM pool.
- Training convergence `NOT_ESTABLISHED`.
- Actual peak training memory unreliable for classes 1, 3 and 4.
- Historical class-0 source revisions retained.
- External synthesis wrapper is operational evidence, not a newly committed
  general-purpose launcher.
- Execution specification is documentary authorization, not a
  cryptographically enforced runtime grant.
- Independent artifact verification does not authenticate every possible
  historical execution claim.
- No additional scientific requirements introduced.
- No class-0 retraining or regeneration performed.
- No additional GMM seed authorized.

### Updated finite register

These statuses are the proposed closure state, effective upon researcher
acceptance and Git preservation of this documentation entry.

`STAGE_1_OVERALL=COMPLETE`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`S4_0_COMPLETED_FAMILY_INTEGRATIONS=ACCEPTED_AND_CLOSED`

`S4_GAN=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_VAE=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_AR=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_MAF=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_G1_GMM=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_R1_RBM=OUTSTANDING`

`S4_D1_DIFFUSION=OUTSTANDING`

`S4_A1_CONSOLIDATED_EXIT=PENDING`

`STAGE_4_OVERALL=IN_PROGRESS`

`NEW_SCIENTIFIC_DECISIONS_REQUIRED_FOR_STAGE4=NONE_IDENTIFIED`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Remaining Stage 4 work:

- S4-R1: MaleX seed-42 RBM complete pool, two classes × 2,000, approved T=10
  sampler policy, reuse existing checkpoint, preserve the 650 allocated CPU
  core-hour cumulative planning ceiling.
- S4-D1: CICMalDroid2020 seed-42 Diffusion fresh restoration and complete
  10,000-sample pool, reuse accepted published checkpoint.
- S4-A1: documentation-only consolidated Stage 4 exit after both remaining
  integrations close.

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.

## Paper 1 Revision-2026 — S4-R1 RBM Representative Complete-Pool Integration Closure

RESEARCHER ACCEPTANCE RECORD. The researcher accepted the completed
representative MaleX seed-42 RBM checkpoint-backed all-class synthetic pool and
authorized terminal evidence preservation and retirement of both execution
grants. Researcher acceptance was granted before terminal closure. Terminal
closure completed on 2026-11-25 UTC, corresponding to 2026-11-24 local Talon
time.

The scope is one representative dataset/seed integration, not full RBM
benchmark completion. No new scientific requirement is introduced, and the
accepted Gibbs-depth decision is not reopened.

Evidence provenance: the Talon execution, scheduler, artifact, terminal and
retirement hash records below were supplied by the researcher from preserved
results and reviewed by ChatGPT. OpenCode did not access Talon, restore
checkpoints, execute sampling, verify artifact bytes or modify external
artifacts during this documentation task.

### Governing Stage 4 exit condition

Research Continuity Charter §6:

> **Checkpoint/synthesize:** Durable, integrity-checked publication; fresh
> restoration; identity-safe, fully counted synthesis for each distinct family
> integration.

Ledger Stage 4 additionally retains durable checkpoint publication,
independent checkpoint restoration, checkpoint-backed synthesis, complete class
quotas and pool publication. Representative integration scope applies.

### S4-R1 requirement and closure basis

S4-R1 required one representative complete MaleX seed-42 checkpoint-backed
pool covering both authoritative classes with exactly 2,000 samples per class.

The completed work comprises:

1. Accepted class-pool publication and independent-verification implementation.
2. Isolated production activation of the governed RBM entry point.
3. Reuse of the existing accepted shared MaleX seed-42 checkpoint.
4. Sequential authorized synthesis for class 0 and class 1.
5. Durable exclusive publication of two complete class pools.
6. Separate-process independent dataset-pool verification.
7. Researcher acceptance and terminal evidence preservation.
8. Retirement of both execution grants.

### Accepted implementation evidence

Implementation commit:

`23f64da71c2d8b3091c6e52e0ba7a2f6479a5dd1`

Parent:

`ccceb6074de3cefb8cc4d6bc749d8e780b43f420`

Commit message:

`feat(paper01): publish and verify governed RBM class pools`

Changed files (3; 1,736 insertions, 4 deletions):

- `src/trustforge/models/paper01_revision_2026/restricted_boltzmann_class_pool.py`
- `scripts/run_paper01_rbm_production_sampling.py`
- `tests/test_rbm_class_pool_revision_2026.py`

Accepted test evidence: 39 focused tests passed and 276 combined RBM
regression tests passed. Tests were not rerun for this documentation-only
operation.

Class-pool publication module SHA-256:

`da6dc1efcb8ebaa722028bebf59ef90e820328f8ce44618c6d1549e360667412`

Production CLI SHA-256:

`6f5565cb3584ec42e8f466016bc54537c0cce2e504d2f878d939abbec9c955df`

Fixture tests SHA-256:

`ddf1a8df40481d083b25928abc048fadef30aa7fd6ad648c06eba4244f543e6e`

The implementation provides exclusive durable publication, independent batch
verification, authoritative class identity checks, sampling-policy identity
reconstruction and complete dataset-pool coverage verification.

### Isolated production activation

Activation branch:

`engineering/paper01-rbm-malex-seed42-stage4-activation`

Activation commit:

`34adec1306bff87f82c565c82b4594912cb8cca1`

Parent: `23f64da71c2d8b3091c6e52e0ba7a2f6479a5dd1`.

The activation changes only:

`RBM_PRODUCTION_SAMPLING_ENABLED = False`

to:

`RBM_PRODUCTION_SAMPLING_ENABLED = True`

Production activation remains isolated from the development branch.

Activated entry SHA-256:

`f736183649e81aec52b033ba69987f5e1a816f4c29ef4ade250b9c784efa2159`

Talon checkout:

`$HOME/TrustForge-paper01-rbm-malex-seed42-stage4`

Corrected external launcher SHA-256:

`8218e2e5a3960358c49030d3c68a482add53721910e4283568a5c283a94b91ca`

The external launcher correction repaired two malformed checksum-line newline
escapes and did not change the accepted sampling methodology. The external
launcher remains operational evidence, not a committed source artifact.

### Checkpoint and scientific policy

- Dataset: `malex`
- Experimental seed: 42
- Class mapping: class 0 `benign`; class 1 `malware`

Existing checkpoint directory:

`$HOME/trustforge_revision_2026_artifacts/paper01/rbm/malex/seed42/`

Checkpoint metadata SHA-256:

`55721f7414b09de5720d1003ef5f29420aa06307079cb5a36c73b8043e3c4231`

Checkpoint state SHA-256:

`18f8ddf4f1d13bd8a9c660c968019a22421f097a40068c7a7be31f29052231b8`

Accepted sampling-policy digest:

`676f1aca169fadf9dd30d734e33267395675ce18cc1978554dbe3939ea4d9ffa`

Policy:

- Exactly ten complete Gibbs transitions per independent sample.
- Separate burn-in: zero.
- Fixed requested-class conditioning.
- Independent chain per sample.
- Governed initialization: `h0_zero_v0_bounded_exponential_p_v_given_h0`.
- Internal arithmetic dtype: float64.
- Published dtype: float32.
- Canonical image shape: 40 × 40 × 3.
- Finite canonical values in [0,1].
- Exactly 2,000 samples per class.

No retraining or additional checkpoint acquisition was performed. The
accepted Gibbs-depth decision remains closed and is not reopened.

### Production execution evidence

Both jobs used the same accepted shared checkpoint, sampling policy and
activation revision.

| Parameter | Attempt 03 | Attempt 04 |
|---|---|---|
| Job | 317368 | 317371 |
| Class | 0 benign | 1 malware |
| State | COMPLETED | COMPLETED |
| Exit code | 0:0 | 0:0 |
| Elapsed seconds | 496 | 474 |
| Allocated CPUs | 8 | 8 |
| Allocated memory | 32 GiB | 32 GiB |
| Walltime limit | 3,600 seconds | 3,600 seconds |
| Node | talon07 | talon07 |
| Total CPU seconds | 515.722 | 513.621 |
| Reported batch MaxRSS | 532768K | 516428K |
| Samples | 2,000 | 2,000 |
| Batches | 250 | 250 |

Attempt-03 run identity:

`commit34adec1_attempt03`

Attempt-04 run identity:

`commit34adec1_attempt04`

Authorizations:

`P1R2026-RBM-MALEX-S42-STAGE4-C0-A03`

`P1R2026-RBM-MALEX-S42-STAGE4-C1-A04`

Each job used one node, one task, eight CPUs, CPU-only execution and a
one-hour walltime limit. Jobs were submitted sequentially. Attempt 04 was
released only after independent attempt-03 verification, researcher
acceptance and retirement of the attempt-03 grant. No additional experimental
job was submitted.

### Durable pool publication and verification

Artifact cell:

`$HOME/trustforge_revision_2026_artifacts/paper01/rbm/malex/seed42`

Class-0 pool:

`pools/class0_seed42_stage4_attempt03/`

Manifest SHA-256:

`46ee31796db6714fcdd9de63198604c325ab908298d7ba7907dca0a46ff847fe`

Class-1 pool:

`pools/class1_seed42_stage4_attempt04/`

Manifest SHA-256:

`0d5801131ea2a48b5f9776dd9d05c741132b6428ec0246a33c1910cad3a4d00c`

Manifest schema:

`trustforge.paper01.rbm-class-pool.v1`

Final independently verified evidence:

- Two authoritative classes.
- 2,000 samples per class.
- 250 batches per class.
- Eight samples per batch.
- 4,000 total images.
- 500 total batches.
- Complete ordered sample-index coverage.
- Shared checkpoint metadata/state identity.
- Shared sampling-policy identity.
- Both class execution lineages verified.
- The previously accepted class-0 manifest unchanged.

Returned terminal markers:

`INDEPENDENT_DATASET_POOL_VERIFICATION=PASS`

`BOTH_CLASS_EXECUTION_LINEAGES=PASS`

`ACCEPTED_CLASS0_MANIFEST_UNCHANGED=PASS`

`VERIFIED_CLASS_COUNT=2`

`VERIFIED_SAMPLE_COUNT=4000`

`VERIFIED_BATCH_COUNT=500`

`ADDITIONAL_SAMPLING_PERFORMED=NO`

The independent verifier was executed in a separate process on Talon.
Verification establishes structural correctness and recorded-lineage
consistency; it does not authenticate every historical execution claim or
establish semantic class fidelity.

### Resource accounting

Attempt 03 allocated CPU core-hours:

`1.1022222222222222`

Attempt 04 allocated CPU core-hours:

`1.0533333333333332`

Combined allocated CPU core-hours:

`2.1555555555555554`

Combined authorized ceiling: 16 allocated CPU core-hours.

Known historical production-sampling expenditure:

- Job 317176: 40 seconds × 8 CPUs.
- Job 317232: 22 seconds × 8 CPUs.
- Combined historical allocation: 0.13777777777777778 core-hours.

Known cumulative production expenditure:

`2.2933333333333334`

Accepted cumulative RBM planning ceiling: 650 allocated CPU core-hours.

Historical accounting is not exhaustive, and the cumulative planning ceiling
is not enforced by the runtime production entry. The combined representative
execution provides resource-planning evidence, not a validated full-benchmark
runtime guarantee.

A linear projection to 882,000 samples is approximately 475.3 allocated CPU
core-hours. This is an estimate based on observed representative throughput,
not an established full-benchmark requirement.

### Terminal evidence preservation

Evidence root relative to the artifact cell:

`operational_verification/stage4_complete_pool_attempt01/`

Attempt-03 terminal evidence:

`attempt03_class0/terminal_acceptance/`

Attempt-04 terminal evidence:

`attempt04_class1/terminal_acceptance/`

Both retired grants are under:

`$HOME/trustforge_revision_2026_artifacts/paper01/rbm/authorizations/retired/`

Attempt-03 retired filename:

`malex_seed42_attempt03.317368.json`

SHA-256:

`0bbe71e28d68cac8fadcad9fd5c468eac789ff4b4638f654f5e9892bd40ee25c`

Attempt-04 retired filename:

`malex_seed42_attempt04.317371.json`

SHA-256:

`184cda4165627ad0d46e44a56a62b0bde2cd1929a8ac9d3ea3f474fa81a96640`

Attempt-03 retirement record SHA-256:

`25eff83948b2f3c3a794793ae81e729ca9990fca5674aa027d36075dc412a00f`

Attempt-04 retirement record SHA-256:

`46812af11be453b0545a7ff1efcf2f65d289c82da602ee9dcff22ee6a08b6cfd`

Attempt-04 terminal evidence checksum inventory SHA-256:

`fe2f2dd87d3057f460b72bac02ac144ef411f870539b17b6927daa1c3ccf958d`

Corrected closure driver SHA-256:

`b82dd9b3a631b4b411f38ad8a88269af960e77539542a875e11cf0cae502c34c`

The successful corrected closure preflight began at `2026-11-25T02:26:21Z`.

The first closure driver stopped before retirement because a historical
completed job was no longer available through the live job-specific queue
query. A bounded correction replaced that query with a user-level queue check
while preserving historical accounting validation. The corrected operation
completed successfully; this administrative interruption was not an
experimental failure.

Final terminal markers:

`ATTEMPT04_AUTHORIZATION_RETIRED=YES`

`BOTH_ACTIVE_GRANTS_ABSENT=YES`

`S4_R1_B=ACCEPTED_AND_CLOSED`

`ADDITIONAL_SAMPLING_PERFORMED=NO`

`SLURM_SUBMISSIONS=0`

`REPOSITORY_MODIFIED=NO`

`FULL_BENCHMARK_GO=NO`

Both execution authorities are exhausted. No further RBM execution is
authorized.

### Preserved limitations

- Representative MaleX seed-42 integration only, not full RBM benchmark
  completion.
- Historical dataset-source provenance
  `NOT_ESTABLISHED_BY_RBM_CHECKPOINT_METADATA`.
- Semantic class fidelity `NOT_ESTABLISHED`.
- Gibbs-chain convergence and mixing `NOT_ESTABLISHED`.
- Downstream evaluation `NOT_PERFORMED` for this complete RBM pool.
- Historical accounting is not exhaustive.
- Full-benchmark resource feasibility `NOT_ESTABLISHED`.
- External artifact evidence is researcher-supplied and ChatGPT-reviewed.
- Structural verification does not independently authenticate every
  historical execution claim.
- No new scientific policy or requirement introduced.
- No additional seed, dataset, retraining or execution authorized.

### Updated finite Stage 4 register

These statuses are the proposed documentation closure state, effective upon
researcher approval and Git preservation of this entry.

`STAGE_1_OVERALL=COMPLETE`

`STAGE_2_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`STAGE_3_OVERALL=COMPLETE_AT_ORIGINAL_SCOPE`

`S4_0_COMPLETED_FAMILY_INTEGRATIONS=ACCEPTED_AND_CLOSED`

`S4_GAN=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_VAE=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_AR=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_MAF=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_G1_GMM=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_R1_A=ACCEPTED_AND_PRESERVED`

`S4_R1_B=ACCEPTED_AND_CLOSED`

`S4_R1_C=PREPARED_PENDING_GIT_PRESERVATION`

`S4_R1_RBM=ACCEPTED_AND_CLOSED_BOUNDED_SCOPE`

`S4_D1_DIFFUSION=OUTSTANDING`

`S4_A1_CONSOLIDATED_EXIT=PENDING`

`STAGE_4_OVERALL=IN_PROGRESS`

`NEW_SCIENTIFIC_DECISIONS_REQUIRED_FOR_STAGE4=NONE_IDENTIFIED`

`ADDITIONAL_EXPERIMENTAL_EXECUTION_AUTHORIZED=NO`

`FULL_BENCHMARK_GO=NO`

Remaining Stage 4 work is exclusively the original S4-D1 Diffusion
representative synthesis integration, followed by the S4-A1 consolidated
documentation exit.

No source, configuration, test, recipe, experimental artifact or earlier
ledger entry was modified. This reconciliation adds documentation only.
