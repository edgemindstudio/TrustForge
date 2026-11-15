# TrustForge — Research Continuity and Governance Charter

**Recorded:** 2026-11-13
**Purpose:** Durable context for future ChatGPT conversations, OpenCode handoffs, and research/engineering reviews.
**Status:** Researcher-described mission and working agreement; not a substitute for live repository evidence or an authorization to execute changes.

## 1. Why this effort began

The original GenCyberSynth Paper 1, *A Unified Comparative Benchmark for Synthetic Malware Images*, was not selected for IEEE GLOBECOM 2026. The three reviews recognized the value of a reproducible, leakage-conscious, seed-aware, seven-family, real-only-versus-augmented benchmark but raised substantive concerns: only two small-image datasets, three seeds, small gains requiring calibrated statistical interpretation, insufficient downstream-classifier details, incomplete reporting of promised diagnostic metrics, the rationale for image representation and the fixed synthetic budget, and limitations to generator-family comparability. The reviews also raised optional directions such as budget ablation, resolution sensitivity, and stronger pretrained generators. Not every suggestion is a mandatory Paper 1 launch condition.

In response, the researcher decided to **add two datasets** and revise Paper 1 substantially. Meanwhile, development of Papers 2–4 had evolved shared code and experimentation. Returning to Paper 1 exposed risks from hardcoded paths, stale imports, configuration precedence, historical defaults, outdated implementation files, caches, and checkpoints: an apparently current command could execute older behavior. Plausible output, a passing job, or a current-looking file did not prove that the intended scientific protocol actually ran.

The ensuing engineering audit was **not a cosmetic cleanup, an invitation to redesign everything, or an end in itself**. Its purpose is to create a scientifically defensible, operationally trustworthy baseline in which the researcher can identify and prove precisely which data, preprocessing, code, settings, model, checkpoint, samples, evaluator, and statistics produced each result, while later paper updates cannot silently rewrite earlier papers' experimental meaning.

## 2. Central scientific mission

Dissertation theme: **Trustworthy probabilistic and generative machine learning for synthetic cybersecurity data, malware-image classification, and risk-aware security decision-making.** The central questions concern whether synthetic data are faithful, useful, calibrated, robust, and safe to rely upon under specified tasks and conditions, not simply whether a generator produces images or one classification metric improves.

### Core publication arc

- **Paper 1 — benchmark foundation:** Controlled comparison of seven generator families; reproducible train → checkpoint → restore → synthesize → evaluate → aggregate pipeline; fixed data partitions, class identities, model policies, real-only comparators, held-out real testing, diagnostic metrics, uncertainty-aware interpretation, explicit failed/missing runs.
- **Paper 2 — external conditional-fidelity audit:** Does a sample requested as class *c* externally resemble class *c* to an independent real-only auditor? Separate requested-label fidelity from internal conditioning loss and downstream utility.
- **Paper 3 — utility, budgets and failure regimes:** When does augmentation help or hurt under class imbalance, class selection, generation budgets, and evaluator choices?
- **Paper 4 — selective synthetic-data policies:** Which samples should be retained, filtered, ranked, or allocated under utility and budget constraints?

Further dissertation research may study calibration and robustness under shift, abstention and cost-sensitive decisions, uncertainty-aware generation, scalable HPC reproducibility, artifact-backed research methods, a synthesis journal article, and broader probabilistic cybersecurity problems. These are **not automatic launch requirements for Paper 1**. Keep evidence/interfaces available for future use without requiring future studies to be completed now.

Original dissertation proposal and later paper-stack planning are intellectual context. Historical manuscripts retain the experimental identity of the work they actually report. **The live Revision-2026 specification and accepted decisions, verified against the repository, control current Paper 1 execution**; an older proposal or manuscript must not silently override them.

## 3. Paper 1 Revision-2026 target

- **Datasets (4):** USTC-TFC2016, CICMalDroid2020, MalNet-Image, MaleX.
- **Generator families (7):** GAN, VAE, Diffusion, AR, MAF, RBM, GMM.
- **Seeds (7):** 42, 43, 44, 45, 46, 47, 48.
- **Primary synthetic budget:** 2,000 samples **per class per experimental seed**. This does **not** define a uniform training budget.
- **Canonical input:** float32 NHWC 40 × 40 × 3 in [0,1], preserving governed source-to-canonical transforms and actual information content (e.g., grayscale replication does not create new color features).
- **Matrix:** 4 × 7 × 7 = **196 dataset–family–seed cells**, plus class-specific work where required.
- **Scientific discipline:** fixed governed training/validation/test membership; dataset-specific class mappings; training-only fitted preprocessing; validation-only selection; held-out real test isolation; separately recorded real-only and real+synthetic outcomes; no silent substitutions.

The earlier Paper 1 manuscript describes two datasets and three seeds; it is **historical**, not an output of this 196-cell revised benchmark. Old results and paper-specific artifacts must remain preserved and labeled.

## 4. Five nonnegotiable invariants

1. **Experimental identity:** Each run declares its paper/revision, dataset/source identity, class mapping, partition, preprocessing, model/family policy, exact seed, training controls, synthesis budget, evaluator, and expected output coverage.
2. **Executed identity:** There is auditable evidence of what code and resolved settings the running process actually consumed; imports, hardcoded values, environment changes, caches, resume/fallback behavior, and data paths cannot silently redirect it.
3. **Artifact identity:** A checkpoint, synthetic archive/pool, evaluator result, and table are explicitly linked to originating source, settings, dataset, run identity, content integrity, selected training state, and required sample counts. Independent restoration and wrong-identity/incomplete-output rejection must be demonstrated within the accepted gate's scope.
4. **Paper isolation:** Each paper has its own frozen experimental contract, source/environment reference, inputs and output namespace. Shared infrastructure may evolve, but later changes in Paper 2/3/4 cannot silently change the reproducible behavior or artifacts for Paper 1, and vice versa. Avoid duplicating entire codebases merely to create isolation; prefer explicit versioning, immutable manifests, pinned dependencies, and validation.
5. **Reproduction and defensible reporting:** A researcher can trace manuscript claim → accepted protocol/configuration → executable source/environment → actual run → checkpoint → samples → evaluation → statistical aggregation → table/figure. Access restrictions and compute needs are documented honestly. Report failures, incomplete runs, uncertainties, and limitations rather than inventing clean denominators or overstating negligible differences.

A passing unit test, scheduler success, existing artifact, or successful bounded fixture proves **only its observed scope**. The audit must not collapse methodological acceptance, operational activation, completed training, checkpoint publication, restored sampling, evaluation completion, full benchmark GO, and final release into one status.

## 5. Review-driven scientific standards

Address reviewer concerns through genuine controls and defensible claims: four datasets with correctly described representations; seven seeds and paired uncertainty analysis; full specification and verified consumption of downstream-classifier controls; complete, traceable promised diagnostics; appropriate explanation of the 2,000/class budget and family-specific training policies; cybersecurity-specific motivation; and no unqualified claims of statistically meaningful gains or generator superiority. Additional ablation or pretrained-generator studies require a specific scientific decision and should not be smuggled in as unapproved engineering gates. The purpose is to reduce avoidable review weaknesses, **not to promise that peer review will be free of criticism**.

## 6. Stage plan and completion rules

1. **Preserve:** Reconcile Git/Talon, protect historical artifacts, capture inventory.
2. **Specify:** Bind scientific protocol and dataset/model/evaluator settings to executable configuration.
3. **Execute:** Confirm requested → resolved → consumed source/config/data identity, gates, retries, and failure behavior.
4. **Checkpoint/synthesize:** Durable, integrity-checked publication; fresh restoration; identity-safe, fully counted synthesis for each distinct family integration.
5. **Evaluate/report:** Matched real-only and augmented evaluation, verified pool consumption, bounded memory, required metrics, exact reference identity/coverage, missing-run treatment, and seed-aware aggregation.
6. **Reproduce:** Tested clean-environment installation, authorized data prep, small example, benchmark run, table recreation, and traceability.
7. **Bounded end to end:** Exercise shared production orchestration and each distinct integration, including rejection paths.
8. **Freeze/launch:** Freeze source, configuration, environment, dataset/matrix, resources, concurrency and retry rules; record a written evidence-based GO/NO-GO.
9. **Release:** Following authorized execution, reconcile expected/completed runs, generate tables/figures, verify manuscript claims, and preserve reproducible release.

**Stopping rule:** Reuse accepted evidence at the level it proves; close a gate once its original exit condition is satisfied; never keep inventing hardening work to delay the scientific experiment. Prioritize blockers to independently permitted training separately from blockers to synthesis, evaluator completion, full benchmark GO, and publication.

## 7. Three distinct authorization decisions

- **Per-family training:** Scientifically accepted methodology is necessary but not sufficient; require current governed data, executable settings, permitted runtime activation, and explicit researcher approval. Training-only authority does not automatically allow benchmark checkpoint publication or synthesis.
- **Full benchmark launch:** Requires accepted applicable readiness gates, full integration and frozen execution controls; one family or one dataset success cannot imply 196-cell readiness.
- **Post-run result/release acceptance:** Requires reconciled output coverage, validated aggregation, regenerated tables/figures, and claim-level scientific review. Results may legitimately be negative or inconclusive.

## 8. Researcher–ChatGPT–OpenCode–Talon working agreement

ChatGPT acts as scientific/engineering lead: establish research rationale and experimental policy; audit evidence read-only; assess blockers; prepare precise action prompts/commands; independently review returned diffs, tests and logs. OpenCode is an **implementation executor**, not an independent scientific authority. The researcher approves proposed edits and personally authorizes commands, commits, pushes, merges, activation changes, checkpoint publication and Talon jobs. Do not independently mutate repositories or submit jobs. Use the academic GitHub identity for academic TrustForge work.

Give **one complete copy-pasteable action block at a time**; explicitly say whether it belongs in Mac Terminal, OpenCode or Talon shell. State parent stage/gate, purpose, effect, expected acceptance evidence, and which decision it enables. Inspect the returned output before issuing another action. Avoid broad cleanups, global replacements, speculative refactors, rerunning accepted experiments, or changing historical frozen outputs. Prefer bounded tests that directly close a mandatory requirement.

## 9. Recorded handoff snapshot — NOT live authority

As recorded on **2026-11-13**, the previous conversation reported **7/7 methodological training authorizations**, but **not 7/7 verified operational training readiness**. The full benchmark remained **`FULL_BENCHMARK_GO=NO`**. GAN/VAE/Diffusion activation and GMM current-revision runner binding were prominent *training-path verification* questions; RBM Gibbs-depth decision blocked **production sampling**, not automatically training. CIC MAF seed-42 had a verified full pool of **10,000 images** (five classes × 2,000), with Talon array 313821 and finalizer 313832, and manifest SHA256 `db884e298ea324597bad2d488d03e2c86c205e7a53bbef2a11bd72f06da5651e`. This did not authenticate full generator lineage or complete real-data evaluation.

Repository: `edgemindstudio/TrustForge`; central branch `engineering/paper01-production-readiness`; OpenCode branch `engineering/paper01-opencode-stage5`; Mac worktree `~/TrustForge-opencode`; Talon checkout `~/TrustForge`; external Talon artifact root `~/trustforge_revision_2026_artifacts/paper01`.

The GitHub-only read-only comparison in the new conversation found central `c9baf964b51078982b5deea53d4887f35c082606` and Stage-5 branch **11 commits ahead, 0 behind**, eligible for fast-forward *at that remote snapshot*. Local/Talon worktrees, any unpushed ledger reconciliation, and current operational gates remain **UNKNOWN until checked**. The previous chat said the next agreed step was review of a ledger-only reconciliation diff before any merge, push, or new Talon job.

**Never reuse this snapshot as proof of today's repository/cluster state.** Start every new session by reading current ledger and specification on relevant branches, verifying HEAD and worktree state, reconciling recent OpenCode and Talon execution evidence, and separating accepted facts from unsupported assumptions.

## 10. Authoritative records to inspect

- `studies/paper01_benchmark/revision_2026/production_readiness_ledger.md` — original stage exit criteria, shared readiness matrix, finite acceptance sequence, later evidence and current decisions.
- `studies/paper01_benchmark/revision_2026/benchmark_specification.md` — accepted and draft scientific/executable details; check section statuses and later authority reconciliation rather than trusting an old heading alone.
- Family-specific authorization, dataset source contracts, executable manifests, frozen configs, and referenced evidence directories at the **specific source revision**.
- Original Paper 1 submission and GLOBECOM 2026 reviews; later Paper 1–4 manuscripts; original dissertation proposal; broader publication roadmap. These establish scholarly motivation and historical scope but **do not replace current execution contracts**.

## 11. A required new-chat opening procedure

1. Read this charter to understand **why** the project exists; do not make the researcher explain the hardcoding/stale-code/paper-isolation problem again.
2. Read the **live** ledger and specification, including original exit criteria and later appended closures. Treat old statuses as time-stamped snapshots and reconcile contradictions.
3. Inspect GitHub branch tips, current local worktree and any OpenCode/Talon transcripts. Distinguish proposed commands, actual edits, test outcomes, commits, pushes, and Slurm jobs.
4. Produce a seven-family register with distinct states for methodology, launcher/activation, real training, checkpoint, synthesis, evaluation, and authorization; record exact mandatory blockers.
5. Produce a stage-by-stage critical path to **training all seven families**, while tracking the separate critical path to full benchmark launch and paper release.
6. Identify **one smallest mandatory, permitted next action** and its acceptance condition. Obtain user execution/approval under the working agreement.
7. Update the **central ledger** after material milestones, preserving original criteria, accepted evidence, gaps, and scope; stop when the gate closes.

**Governing test:** Can another researcher prove what scientific experiment was intended, what truly executed, what artifacts were produced, and how each paper's findings remain reproducible and isolated as the wider dissertation evolves?

## Research-direction control and prevention of execution-driven drift — 2026-11-15

**Status:** Governance clarification, recorded after a continuity failure. It adds no experimental requirement, changes no accepted result, and authorizes no execution.

**Observed failure:** Although this charter, the benchmark specification and the
ledger already existed, successive execution results caused the next technically
possible experiment to be authorized as if it were the next required one. In
particular, each successful MAF seed was followed by authorizing the numerically
next seed. Technical possibility and numerical adjacency are not authority.

### A. Why Revision-2026 exists

Paper 1 was submitted and received substantive reviewer criticism. The revision
must address accepted reviewer concerns, strengthen experimental coverage and
statistical reporting, improve methodological transparency, and establish a
reproducible unified benchmark.

The subsequent engineering audit was initiated because hardcoded paths, stale
imports, configuration precedence, historical defaults, outdated files, artifact
ambiguity, and cross-paper modifications could cause apparently correct commands
to execute the wrong scientific protocol.

Engineering correctness is necessary to support the scientific research; it is
not the research objective itself. A passing job, a clean test run, or a tidy
repository does not by itself advance the science.

### B. Central Paper 1 contribution

The principal methodological contribution is a unified and reproducible

TRAIN → CHECKPOINT → RESTORE → SYNTHESIZE → EVALUATE → AGGREGATE

framework for comparing synthetic malware-image generators under controlled
conditions. The accepted design is:

- Four datasets.
- Seven generator families.
- Seven experimental seeds, 42–48.
- Fixed primary budget of 2,000 synthetic images per class per experimental seed.
- Governed preprocessing and label mappings.
- Leakage-safe real-data partitions.
- Paired real-only and augmented classifier comparisons.
- Held-out real testing.
- Seed-aware statistical aggregation.
- Reproducible execution and artifact provenance.

Generator checkpoints, pool production, and successful individual seeds are
supporting evidence, not substitutes for the unified benchmark. A bounded cell
result is not a benchmark result.

### C. Governance hierarchy

The existing authority rules in this charter apply unchanged. In particular:

- Reviewer feedback and accepted researcher decisions establish the motivation
  and scientific revision objectives.
- The live accepted benchmark specification governs intended experiments.
- The readiness ledger governs completed evidence, outstanding requirements, and
  gate status.
- Verified repository, runtime, and artifact evidence establishes actual
  execution state.
- OpenCode is an implementation executor, not an independent source of
  scientific direction.

Do not give a newly generated summary greater authority than the original
governing documents. A chat-generated report, however detailed, is evidence
about a moment, not a replacement for the charter, specification or ledger.

### D. Mandatory next-action rule

Whenever OpenCode returns a result, ChatGPT must first determine:

1. Which existing requirement or gate authorized the work?
2. What evidence was actually produced?
3. Whether the original acceptance condition has been satisfied.
4. Which agreed requirement genuinely remains unresolved.
5. Whether the next action is mandatory for the research mission or merely
   technically possible.
6. Whether the current gate should now be closed.
7. What smallest authorized action follows from the governing plan.

Do not select the next experimental seed merely because the previous seed
succeeded. Do not create new readiness stages, repeat accepted validations, or
introduce speculative hardening without an identified scientific or operational
blocker. When no governing requirement calls for further execution, the correct
action is to reconcile evidence and stop.

### E. Readiness versus production

Keep these categories explicitly distinct:

- Engineering defect correction.
- Production-readiness verification.
- Individual bounded experimental execution.
- Full benchmark launch authorization.
- Post-run aggregation and manuscript release.

A representative bounded run may help establish readiness, but the evidence
required for each distinct generator/dataset path must be judged according to the
existing readiness criteria.

Do not assume that all 196 planned experiment cells require individually
negotiated scientific approval. Equally, do not assume that three completed MAF
cells prove full benchmark readiness. The ledger's readiness matrix and finite
acceptance sequence define what evidence each path needs.

### F. Continuity enforcement

Each new ChatGPT conversation must read this charter, the readiness ledger, the
benchmark specification, and current branch evidence before suggesting execution.
It must explicitly state the original mission, current governing stage,
applicable original exit conditions, accepted evidence, outstanding mandatory
blockers, and the next justified action.

If the documents and observed execution evidence disagree, report the
disagreement rather than inventing a resolution. Preserve accepted evidence at
the scope it actually proves, and treat every historical status heading as a
time-stamped snapshot rather than current authority.
