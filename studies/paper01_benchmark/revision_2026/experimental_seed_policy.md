# Paper 1 Revision-2026 — Experimental Seed Policy

Status: Proposed canonical policy; pending research acceptance.

## Experimental model seeds

`[42, 43, 44, 45, 46, 47, 48]`

Exactly seven distinct experimental seeds are proposed for
every approved dataset–generator-family combination.

## Separation of identities

The governed dataset partition seed is 42 and is independent
of experimental model seeds.

Experimental seeds control model-training repetitions and
associated governed synthesis identities.

Evaluator randomization must be specified separately while
preserving paired Real-only versus Real+Synthetic comparisons.

## Scientific boundaries

- Four approved datasets.
- Seven approved generator families.
- Seven experimental seeds.
- Primary synthesis budget: 2,000 samples per class.
- Historical three-seed configurations remain unchanged.
- Seed selection is fixed before full benchmark execution.
- Distinct seeds do not imply independent dataset samples.
- Missing or failed runs must be reported rather than omitted.

## Authorization

This document does not activate training, authorize checkpoint
publication, or permit full benchmark scale-out.
