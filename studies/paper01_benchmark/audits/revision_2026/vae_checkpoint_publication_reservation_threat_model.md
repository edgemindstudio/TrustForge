# VAE Checkpoint Publication — Reservation Threat Model

Status: Engineering contract under review.
Scope: TrustForge Paper 1 revision-2026.
Environment: UND Talon, CPU-only, GPFS-backed storage.

## 1. Security objective

Generation reservation must exclusively establish a checkpoint
generation namespace without overwriting existing research artifacts.

A reserved generation is not a committed checkpoint.

## 2. Cooperative publication assumption

The current implementation assumes that processes with filesystem
authority over the private publication namespace do not maliciously
replace generation-directory entries during reservation.

Independent cooperating publishers may compete for a generation ID.
Exclusive mkdir must reject an existing entry.

This assumption is not a security guarantee against malicious
same-UID processes.

## 3. Required private-parent validation

Before generation creation, the reservation operation must:

- Validate the canonical generation ID.
- Validate the opened generation-parent descriptor.
- Require a directory owned by the effective UID.
- Reject group and other POSIX permission bits.
- Fail before creating the generation if validation fails.

These requirements do not establish comprehensive ACL isolation,
privileged-actor exclusion, or immutable directory entries.

## 4. Directory creation and descriptor semantics

The implementation uses descriptor-relative mkdir followed by
descriptor-relative directory opening with O_NOFOLLOW.

O_NOFOLLOW rejects symlink substitution but does not prevent
replacement with another real directory.

The opened descriptor identifies the reopened directory inode.
It does not prove that inode was originally created by mkdir.

## 5. Unresolved adversarial threats

The following are explicitly unresolved:

- Real-directory substitution after mkdir.
- Empty private same-UID directory substitution.
- Malicious same-UID namespace manipulation.
- Privileged namespace manipulation.
- Filesystem ACL enforcement beyond inspected POSIX mode.
- Proof of atomic creation-to-open directory identity.

The strict XFAIL regressions documenting substitution must remain.

## 6. Failure and recovery rules

A failed reservation must not automatically delete a directory
that may have been created.

An uncertain filesystem operation outcome must not be interpreted
as proof that an operation did not occur.

A reserved generation must not be treated as a committed checkpoint.

Directory-entry synchronization and full transaction crash
durability remain separate acceptance requirements.

## 7. Publication restrictions

The cooperative threat model permits continued development
and disposable synthetic testing only.

It does not authorize authoritative checkpoint publication,
real-data VAE training, or scientific acceptance.

## 8. Evidence baseline

Full VAE regression execution:

670 passed, 2 xfailed in 96.93 seconds.

Both XFAIL cases concern directory-substitution vulnerabilities.

The passing tests provide regression evidence, not independent
proof of complete security or GPFS crash durability.

## 9. Acceptance requirements

Before authoritative checkpoint publication:

- Define and approve the final publication threat model.
- Establish and test the complete publication transaction.
- Validate non-overwriting commit behavior on supported GPFS.
- Verify failure classification and recovery behavior.
- Complete security and scientific acceptance reviews.
- Explicitly authorize the applicable publication workflow.

Current acceptance:

VAE_CHECKPOINT_IMPLEMENTATION_ACCEPTED=NO
VAE_REAL_DATASET_TRAINING_MAY_BEGIN=NO
