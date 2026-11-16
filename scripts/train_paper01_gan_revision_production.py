#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 GAN production training entry point.

One invocation plans and runs exactly one dataset/experimental-seed checkpoint
through the existing governed GAN runner. The checkpoint destination is
derived from the accepted source revision and declared numeric attempt; the
operator cannot supply an arbitrary path.

Compute-node only. Requires a clean committed source revision. Does not submit
Slurm jobs, authorize execution, publish synthetic data or select a new
scientific rule.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from trustforge.models.paper01_revision_2026 import (
    gan_governed_runner,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.gan_class_mapping import (
    resolve_gan_class_mapping,
)
from trustforge.models.paper01_revision_2026.gan_governed_runner import (
    prepare_gan_run,
)
from trustforge.models.paper01_revision_2026.gan_production_binding import (
    build_gan_run_identity,
    create_gan_production_checkpoint_directory,
    resolve_gan_production_checkpoint_directory,
)
from trustforge.models.paper01_revision_2026.gan_production_authorization import (
    require_gan_production_scope,
)
from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.provenance.git import capture_git_state


DATASET_IDS = tuple(sorted(DATASET_CLASS_COUNTS))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id", required=True, choices=DATASET_IDS,
    )
    parser.add_argument(
        "--experimental-seed", required=True, type=int,
    )
    parser.add_argument(
        "--attempt", required=True, type=int,
    )
    parser.add_argument(
        "--expected-source-commit", required=True,
    )
    parser.add_argument(
        "--repository-root", type=Path, default=Path.cwd(),
    )
    parser.add_argument("--ustc-source-dir", type=Path)
    parser.add_argument("--cic-source-csv", type=Path)
    parser.add_argument("--cic-feature-columns", type=Path)
    parser.add_argument("--malnet-image-root", type=Path)
    parser.add_argument("--malnet-preparation-dir", type=Path)
    parser.add_argument("--malnet-mapping-path", type=Path)
    parser.add_argument("--malex-label-archive", type=Path)
    parser.add_argument("--malex-benign-root", type=Path)
    parser.add_argument("--malex-malware-root", type=Path)
    return parser


def require_environment(args):
    """Require a Slurm allocation and one clean accepted source revision."""
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError(
            "GAN production training requires a Slurm job"
        )

    state = capture_git_state(args.repository_root)

    if state.dirty:
        raise RuntimeError(
            "GAN production training requires a clean working tree"
        )

    if state.commit != args.expected_source_commit:
        raise RuntimeError(
            "source revision does not match the expected commit"
        )

    return state


def require_sources(args, names):
    values = [getattr(args, name) for name in names]

    if any(value is None for value in values):
        raise ValueError(
            "missing required dataset source arguments: "
            + ", ".join(names)
        )

    for path in values:
        if not path.exists() or path.is_symlink():
            raise ValueError(
                f"missing or symlinked dataset source: {path}"
            )

    return values


def build_adapter(args):
    """Construct the governed revision-2026 adapter for one dataset."""
    from trustforge.data.paper01_revision_2026 import (
        CICMalDroidRevision2026,
        MalNetImageRevision2026,
        MaleXRevision2026,
        USTCRevision2026,
    )

    if args.dataset_id == "ustc_tfc2016":
        (source,) = require_sources(args, ("ustc_source_dir",))
        return USTCRevision2026(source)

    if args.dataset_id == "cicmaldroid2020":
        source, columns = require_sources(
            args, ("cic_source_csv", "cic_feature_columns"),
        )
        return CICMalDroidRevision2026(
            source, columns, validate_hashes=True,
        )

    if args.dataset_id == "malnet_image":
        image_root, preparation, mapping = require_sources(
            args,
            (
                "malnet_image_root",
                "malnet_preparation_dir",
                "malnet_mapping_path",
            ),
        )
        return MalNetImageRevision2026(
            image_root,
            preparation,
            mapping,
            validate_hashes=True,
            validate_production_contract=True,
        )

    label_archive, benign_root, malware_root = require_sources(
        args,
        (
            "malex_label_archive",
            "malex_benign_root",
            "malex_malware_root",
        ),
    )
    return MaleXRevision2026(
        label_archive,
        benign_root,
        malware_root,
        validate_hashes=True,
        validate_production_contract=True,
    )


def run(args) -> dict:
    """Plan and execute one authorized GAN production attempt."""
    state = require_environment(args)

    run_identity = build_gan_run_identity(
        expected_source_commit=state.commit,
        attempt=args.attempt,
    )

    require_governed_experimental_seed(args.experimental_seed)

    # Authorization is checked before any destination directory is created.
    gan_governed_runner.require_gan_production_authorization()

    checkpoint_directory = resolve_gan_production_checkpoint_directory(
        dataset_id=args.dataset_id,
        experimental_seed=args.experimental_seed,
        run_identity=run_identity,
        expected_source_commit=state.commit,
    )

    require_gan_production_scope(
        dataset_id=args.dataset_id,
        experimental_seed=args.experimental_seed,
        checkpoint_directory=checkpoint_directory,
        repository_root=args.repository_root,
    )

    dataset = build_adapter(args)

    class_mapping = resolve_gan_class_mapping(
        dataset_id=args.dataset_id,
        adapter=dataset,
    )

    checkpoint_directory = (
        create_gan_production_checkpoint_directory(
            dataset_id=args.dataset_id,
            experimental_seed=args.experimental_seed,
            run_identity=run_identity,
            expected_source_commit=state.commit,
        )
    )

    preparation = prepare_gan_run(
        dataset=dataset,
        dataset_id=args.dataset_id,
        experimental_seed=args.experimental_seed,
        class_mapping=class_mapping,
        checkpoint_directory=checkpoint_directory,
    )

    result = gan_governed_runner.execute_gan_production_run(
        preparation,
        dataset=dataset,
    )

    summary = {
        "dataset_id": args.dataset_id,
        "experimental_seed": args.experimental_seed,
        "run_identity": run_identity,
        "source_commit": state.commit,
        "class_mapping": list(preparation.plan.class_mapping),
        "batch_size": preparation.plan.batch_size,
        "planned_generator_iterations": (
            preparation.plan.planned_generator_iterations
        ),
        "checkpoint_directory": str(checkpoint_directory),
        "completed_generator_iterations": result[
            "completed_generator_iterations"
        ],
        "completed_discriminator_updates": result[
            "completed_discriminator_updates"
        ],
    }
    print(
        "GAN_PRODUCTION_RESULT "
        + json.dumps(summary, sort_keys=True, allow_nan=False),
        flush=True,
    )

    return summary


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
