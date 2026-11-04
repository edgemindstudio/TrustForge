# Paper 1 Revision-2026 — Operational Integration Contract

Status: DRAFT / NO PRODUCTION EXECUTION AUTHORIZATION

Source commit: 42ac1fa794b6a2aeb7894881cc0cdb61ff77fa38

## 1. Locked scientific scope

- Datasets: USTC-TFC2016, CICMalDroid2020, MalNet-Image, MaleX v1.
- Generator families: GAN, VAE, Diffusion, AR, MAF, RBM, GMM.
- Experimental seed count: seven.
- Exact seed enumeration: PENDING EXPLICIT LOCK.
- Synthetic budget: 2,000 images per class per experimental seed.
- Canonical representation: NHWC 40 x 40 x 3, float32, [0,1].
- Dataset partition seed and experimental seed are distinct.
- Target generator/dataset/seed combinations: 196.

## 2. Scientific authorization

All seven generator families are methodologically training-authorized.

This authorization does not imply:
- runtime gate activation;
- successful real-data production execution;
- durable checkpoint publication;
- checkpoint-backed production synthesis;
- revision evaluator acceptance;
- full benchmark acceptance.

Existing methodology must not be reopened without contradictory evidence.

## 3. Accepted operational precedents

GMM:
- Real MaleX class-0 seed-42 training pilot accepted.
- Selected checkpoint persisted and independently restored.
- Pilot is not an all-class or all-seed production result.

RBM:
- Real MaleX seed-42 governed training and checkpoint evidence accepted.
- Revision-specific compute-node Slurm runner exists.
- Reuse its execution conventions, not its model-specific budget.

## 4. Required integration work

AR:
- Bind revision training and class-conditioned sampling to dataset identity.
- Verify selected durable weights and independent restoration.

MAF:
- Orchestrate accepted conditional training steps.
- Establish selected checkpoint publication and restoration.

GMM:
- Generalize accepted training/persistence invocation to governed classes
  and experiments without overwriting accepted pilot evidence.

GAN:
- Establish revision production Slurm invocation and output binding.
- Preserve disabled execution gates pending controlled authorization.
- Reuse accepted checkpoint-v3 contracts.

RBM:
- Reuse the existing revision compute-node runner.
- Preserve accepted checkpoints and source hashes.

VAE:
- Establish real-data compute-node invocation.
- Reuse existing checkpoint contracts.
- Preserve separate execution, publication and synthesis restrictions.

Diffusion:
- Establish real-data compute-node invocation.
- Implement/verify independently reloadable durable checkpoint handling.
- Preserve separate execution, publication and synthesis restrictions.

## 5. Mandatory common run contract

Each production invocation must bind:
- authoritative dataset ID and class mapping;
- experimental seed and frozen source revision;
- governed training/validation partitions;
- family-specific approved scientific controls;
- output namespace and checkpoint identity;
- collision-safe artifact publication;
- complete execution and failure evidence.

Historical checkpoints cannot be silently substituted.
Missing checkpoint weights cannot trigger random-weight synthesis.
Test data cannot participate in generator training or checkpoint selection.

## 6. Evaluation and scale-out

Before full scale-out:
- verify independent checkpoint restoration;
- verify 2,000 samples per requested class;
- verify class labels, provenance and sample counts;
- verify revision evaluation and held-out test isolation;
- complete a bounded end-to-end pilot for each family;
- verify storage quota or obtain an approved storage allocation;
- finalize exact seven-seed enumeration.

Historical M7.4 evidence is not Revision-2026 execution evidence.

## 7. Operational status

Scientific methodology: ACCEPTED 7/7.
Dataset validation: PREVIOUSLY PASSED.
Real-data production integration: PARTIAL.
GAN/VAE/Diffusion runtime activation: DISABLED.
Checkpoint-backed synthesis for full matrix: NOT VERIFIED.
Revision evaluation: NOT VERIFIED.
Full 196-combination production scale-out: NO-GO.

## 8. Change governance

This file is a planning contract, not authorization to execute.
No runtime guard may be bypassed.
No historical artifact may be overwritten.
Each implementation change requires bounded regression acceptance.
