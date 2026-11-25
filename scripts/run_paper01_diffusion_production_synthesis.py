#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 Diffusion production synthesis entry.

One invocation validates one explicit dataset/experimental-seed/attempt/
class/sample-range, the synthesis recipe and checkpoint identities, an
execution-specific authorization record and the actual Slurm allocation, then
restores the published validation-best checkpoint in a fresh process. In
synthesis mode it generates exactly the authorized range and publishes it
durably; in restoration mode it verifies restoration only.

Compute-node only. The production gate remains disabled in source. This entry
point never trains, selects checkpoints, retries, requeues or submits jobs.
"""

from __future__ import annotations

# Talon's TensorFlow 2.20.0 / Keras 3.11.3 can fail tree initialization when
# optree is first imported later during model-builder binding. Preload optree
# before any TensorFlow/Keras-dependent production import.
import optree  # noqa: F401

import argparse
import json
from pathlib import Path


def _load_module():
    from trustforge.models.paper01_revision_2026 import (
        diffusion_production_synthesis as synthesis,
    )

    return synthesis


def make_parser():
    """Build the explicit-identity production argument parser."""
    module = _load_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=module.MODES,
    )
    parser.add_argument(
        "--dataset-id", required=True,
        choices=tuple(sorted(module.DATASET_CLASS_COUNTS)),
    )
    parser.add_argument("--experimental-seed", required=True, type=int)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--expected-authorization-id", required=True)
    parser.add_argument("--expected-synthesis-source-commit", required=True)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256", required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-state-sha256", required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-training-source-commit", required=True,
    )
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--sample-start", type=int)
    parser.add_argument("--sample-stop", type=int)
    parser.add_argument("--output-destination", type=Path)
    return parser


def _require_synthesis_arguments(parser, args):
    missing = [
        option
        for option, value in (
            ("--class-id", args.class_id),
            ("--sample-start", args.sample_start),
            ("--sample-stop", args.sample_stop),
            ("--output-destination", args.output_destination),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "--mode synthesis requires " + ", ".join(missing)
        )


def _reject_restoration_arguments(parser, args):
    unexpected = [
        option
        for option, value in (
            ("--class-id", args.class_id),
            ("--sample-start", args.sample_start),
            ("--sample-stop", args.sample_stop),
            ("--output-destination", args.output_destination),
        )
        if value is not None
    ]
    if unexpected:
        parser.error(
            "--mode restoration must not bind " + ", ".join(unexpected)
        )


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    module = _load_module()

    common = {
        "dataset_id": args.dataset_id,
        "experimental_seed": args.experimental_seed,
        "attempt": args.attempt,
        "expected_authorization_id": args.expected_authorization_id,
        "expected_synthesis_source_commit": (
            args.expected_synthesis_source_commit
        ),
        "repository_root": args.repository_root,
        "checkpoint_directory": args.checkpoint_dir,
        "expected_checkpoint_manifest_sha256": (
            args.expected_checkpoint_manifest_sha256
        ),
        "expected_checkpoint_state_sha256": (
            args.expected_checkpoint_state_sha256
        ),
        "expected_checkpoint_training_source_commit": (
            args.expected_checkpoint_training_source_commit
        ),
        "expected_recipe_sha256": args.expected_recipe_sha256,
    }

    if args.mode == "synthesis":
        _require_synthesis_arguments(parser, args)
        request = module.DiffusionProductionSynthesisRequest(
            **common,
            class_id=args.class_id,
            sample_start=args.sample_start,
            sample_stop=args.sample_stop,
            output_destination=args.output_destination,
        )
        result = module.execute_diffusion_production_synthesis(request)
    else:
        _reject_restoration_arguments(parser, args)
        request = module.DiffusionRestorationVerificationRequest(**common)
        result = module.verify_diffusion_restoration(request)

    print(
        "DIFFUSION_PRODUCTION_SYNTHESIS_RESULT "
        + json.dumps(result, sort_keys=True, default=str),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
