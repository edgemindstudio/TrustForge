# Paper 1 Revision-2026 — New-Conversation Handoff (2026-11-15)

**Purpose:** Navigation and continuity aid for a fresh ChatGPT conversation.
**Status:** Not a replacement for the continuity charter, the benchmark
specification, or the production-readiness ledger. If this document disagrees
with those records, the governing documents and verified execution evidence win.

**Evidence labels used below:** VERIFIED (checked in Git or on Talon during
preparation), ACCEPTED (recorded researcher/scientific-lead decision),
REPORTED (stated in a handoff or summary, not independently re-verified here),
BLOCKED (a required dependency or decision is unresolved), PROPOSED (not yet
accepted), UNKNOWN (not established by the inspected records).

---

## 1. Why Paper 1 was revised (ACCEPTED context)

The original Paper 1, *A Unified Comparative Benchmark for Synthetic Malware
Images*, was submitted and received substantive reviewer criticism. The
revision must address the accepted reviewer concerns, strengthen experimental
coverage and statistical reporting, improve methodological transparency, and
establish a reproducible unified benchmark. Reviewer concerns recorded in the
charter include: only two small-image datasets and three seeds; small gains
needing calibrated statistical interpretation; incomplete downstream-classifier
details; incomplete reporting of promised diagnostics; insufficient rationale
for the image representation and the fixed synthetic budget; and limitations to
generator-family comparability.

Reviewer-response records: the GLOBECOM 2026 review texts themselves are
**UNKNOWN** inside this repository; their substance is **ACCEPTED** context as
summarized in `research_continuity_charter.md` Section 1. No consolidated
reviewer-response matrix was found under `studies/paper01_benchmark/`
(searched during preparation). Do not invent one.

Why the engineering audit happened (ACCEPTED context): shared Papers 2–4 code
plus hardcoded paths, stale imports, configuration precedence, historical
defaults, outdated files, artifact ambiguity and cross-paper modifications
meant an apparently correct command could execute the wrong scientific
protocol. Engineering correctness supports the research; it is not the
research objective.

## 2. Principal research objective (ACCEPTED)

A unified, reproducible

`TRAIN → CHECKPOINT → RESTORE → SYNTHESIZE → EVALUATE → AGGREGATE`

framework for comparing synthetic malware-image generators under controlled
conditions, with auditable identity from manuscript claim to artifact.

## 3. Accepted Revision-2026 experimental design (ACCEPTED)

- Datasets (4): USTC-TFC2016, CICMalDroid2020, MalNet-Image, MaleX.
- Families (7): GAN, VAE, Diffusion, AR, MAF, RBM, GMM.
- Seeds (7): 42–48.
- Primary synthetic budget: 2,000 images per class per experimental seed
  (not a uniform training budget).
- Canonical representation: float32 NHWC 40×40×3 in [0,1].
- Matrix: 4 × 7 × 7 = 196 dataset–family–seed cells plus class-specific work.
- Fixed governed partitions; training-only fitted preprocessing;
  validation-only selection; held-out real test isolation; paired real-only
  versus real-plus-synthetic classifiers; seed-aware, estimation-focused
  aggregation.

Historical manuscript results (two datasets, three seeds) remain historical and
must not be merged with Revision-2026 results as equivalent experiments.

## 4. Nine-stage readiness structure and documented status

Source: ledger "Stage plan" table (documented 2026-11-13 snapshot) plus later
appended acceptance entries.

| Stage | Documented status |
|---|---|
| 1. Establish and preserve state | COMPLETE (ACCEPTED) |
| 2. Freeze experimental specification | IN_PROGRESS — specification is DRAFT/PARTIAL CONSOLIDATION |
| 3. Trace production execution | IN_PROGRESS |
| 4. Checkpoint and synthesis integration | IN_PROGRESS — CIC MAF seed-42 pool accepted at that time; later appended entries add seed-43 and seed-44 accepted bounded pools |
| 5. Validate evaluation and reporting | IN_PROGRESS — real-data evaluation accepted for three MAF cells; benchmark-wide evaluation remains open |
| 6. Professional reproduction workflow | PENDING |
| 7. End-to-end acceptance | PENDING |
| 8. Freeze and launch | PENDING/BLOCKED — no accepted revision/configuration/resource freeze or GO decision |
| 9. Verify results and release | PENDING AFTER AUTHORIZED EXECUTION |

**Important:** stage statuses in the table are a time-stamped snapshot. Later
appended entries add accepted evidence but do not retroactively rewrite the
table. Current gate closure must be reconciled from the ledger as a whole and
from verified evidence; the exact present-day closure of Stage 4/5 across all
families and datasets is **UNKNOWN** without that reconciliation.

## 5. Engineering concerns motivating the audit (ACCEPTED context)

Hardcoded paths and branch pins; stale imports and duplicated implementations;
configuration precedence and silent defaults; historical file/cache reuse;
artifact ambiguity (which checkpoint/pool a number came from); cross-paper code
evolution mutating Paper 1 behavior; scheduler success being mistaken for
scientific correctness. The ledger's evidence vocabulary
(REPORTED / INSPECTED / EXECUTION_VERIFIED / BLOCKED / COMPLETE / PENDING) is
the intended discipline.

## 6. What the completed MAF 42–44 work proves — and does not

ACCEPTED bounded evidence (three cells, one dataset, one family):

| Seed | Generator job | Pool / synthesis jobs | Evaluation job | Paired Δmacro-F1 |
|---|---|---|---|---|
| 42 | 311675 | pool 313821 / finalizer 313832 | 314068 | +0.05451150907476743 |
| 43 | 314075 | pool 314222 / finalizer 314227 | 314240 | +0.09277561208032425 |
| 44 | 314269 | diag 314347, array 314349, finalizer 314354 | 314355 | +0.06108291546148226 |

Proves (ACCEPTED, within scope): the CIC MAF execution path can train, publish
a checkpoint, restore it in a fresh process, synthesize a verified
10,000-image pool (2,000/class), and run a paired real-only versus augmented
evaluation with seven verified result groups for those seeds.

Does not prove (ACCEPTED limitations): cross-seed statistical significance;
benchmark-wide readiness; readiness of the other six families or the other
three datasets; semantic class fidelity (`NOT_ESTABLISHED`); generator lineage
authentication at the unified-validator layer (`generator_lineage_authenticated
= false`); complete evaluation (`complete_evaluation=false`); full-benchmark
authorization (`FULL_BENCHMARK_GO=NO`).

## 7. Latest accepted Git identities at preparation time

- Repository: `edgemindstudio/TrustForge` (VERIFIED).
- OpenCode branch: `engineering/paper01-opencode-stage5` (VERIFIED).
- Last independently verified commit on that branch before this documentation
  commit: `886bb539fd9b12d81f34db30f14d052123a12d42` (VERIFIED; parent
  `c56a02d627d5506e2ed39badc79c28215ee5849f`).
- Central Talon/GitHub production-readiness branch:
  `engineering/paper01-production-readiness` at
  `6b56e92d906f2f0d6149b3e74ff40bb5277f45ab` (VERIFIED via GitHub and Talon).
  It is **divergent** from the Stage-5 branch and is intentionally preserved;
  no automatic synchronization was performed.
- This handoff file is added by a documentation-only commit; always re-read
  current `git log -1` rather than trusting the SHA above.

## 8. External Talon artifact roots (VERIFIED at preparation time)

Root: `~/trustforge_revision_2026_artifacts/paper01`

- `maf/cicmaldroid2020/seed42/…` — accepted seed-42 generator/evaluation evidence.
- `maf/cicmaldroid2020/seed43/…` — accepted seed-43 generator, pool
  (`pool_f15f434-Yb5lcI`), diagnostic and evaluation evidence.
- `maf/cicmaldroid2020/seed44/generator_exec-8d4a281-8agFZz` — generator job
  314269; checkpoint `cc58fe65…1055a`; receipt `8e81b888…e5f3`.
- `maf/cicmaldroid2020/seed44/synth_diag-8198dc4-Xw3Kp9` — diagnostic job 314347.
- `maf/cicmaldroid2020/seed44/pool_8198dc4-Qm4Rt7` — pool manifest
  `c5fc5cb6…0cb5`; finalization report `23463ca4…d36c`.
- `maf/cicmaldroid2020/seed44/bounded_eval-c56a02d-Tq7Md4` — evaluation job
  314355; `evaluation.json` `6cf1c199…40cc`.
- `diagnostics/` — launcher scripts and Slurm logs for those jobs.

These external artifacts are **not tracked by Git** and are **not** backed up to
GitHub (VERIFIED by inspection). They exist only under the Talon filesystem
paths above.

## 9. Known limitations and unresolved readiness requirements

Reproduce these from the ledger's "Current evidence and gap register" and
later entries; the list below is a pointer, not a fabricated inventory.

- Stage 2 (specification freeze) IN_PROGRESS: remaining sections (exact seed
  acceptance, family training/selection policies, downstream classifier policy,
  metric definitions and references, estimands/multiplicity, retry and
  aggregation rules, final executable linkage). (ACCEPTED/IN_PROGRESS)
- Source-binding and runtime-activation gaps recorded as BLOCKED in the gap
  register (e.g. GMM production source binding, RBM production Gibbs depth).
  Their present-day status is **UNKNOWN** without re-verification.
- Benchmark-wide checkpoint/pool coverage for the other families, datasets and
  seeds is not established (PENDING/UNKNOWN).
- Stage 6 clean-environment reproduction and reviewer traceability map: PENDING.
- Stage 7 bounded end-to-end acceptance including rejection paths: PENDING.
- Stage 8 freeze (source, config, environment, matrix, resources, retries,
  GO/NO-GO): PENDING/BLOCKED.
- Stage 9 result reconciliation, tables/figures, claim audit, release: PENDING
  after authorized execution.
- Feature-weight identity, retry/resource limits, KID/FID reference identity
  and final shared-policy executable linkage: recorded as remaining decisions.

## 10. Four distinct decisions — do not collapse them

1. Accepted methodology (paper/revision scientific contract): largely ACCEPTED,
   with Stage-2 specification work still IN_PROGRESS.
2. Operational training readiness per family/dataset path: partially
   established for CIC MAF; other paths UNKNOWN/INCOMPLETE.
3. Full benchmark launch (`FULL_BENCHMARK_GO`): **NO** (ACCEPTED).
4. Post-run aggregation and manuscript release: not started for Revision-2026
   (PENDING).

A bounded success at one level never implies the next level.

## 11. Start by inspecting current state, not this snapshot

1. GitHub: branch tips for `engineering/paper01-opencode-stage5` and
   `engineering/paper01-production-readiness`.
2. Mac: `~/TrustForge-opencode` — branch, HEAD, `git status`, unpublished work.
3. Talon: central `~/TrustForge` (expect pre-existing untracked backup/zip files;
   inventory them, do not delete), isolated worktrees under
   `~/trustforge_revision_2026_execution/paper01/…`, Slurm queue/accounting.
4. Read the ledger's latest entries and the specification's current section
   statuses in full.
5. Report any disagreement between documents and observed evidence instead of
   resolving it by assumption.

## 12. Required first decision in the next conversation (PROPOSED process)

Before authorizing any execution, determine exactly what remains to be done
before a defensible GO decision for the unified 196-cell benchmark: which
original stage exit conditions are satisfied, which are BLOCKED or PENDING,
which accepted evidence can be reused, and which smallest action closes the
nearest mandatory requirement. State the answer with citations to the ledger and
specification.

## 13. Explicit prohibition

Do not authorize MAF seed 45, another MAF seed, or another generator family
merely because it is next in numerical or list order. Seed 45 is **not**
authorized (`SEED45_EXECUTION_AUTHORIZED=NO`) and is not implied by the
acceptance of seeds 42–44. Any new execution requires a governing requirement,
a stated acceptance condition, and explicit researcher authorization.

---

**Current governance status at preparation time:** `FULL_BENCHMARK_GO=NO`;
`SEED42/43/44_MAF_EVALUATION=ACCEPTED_BOUNDED`; `SEED45_EXECUTION_AUTHORIZED=NO`.
