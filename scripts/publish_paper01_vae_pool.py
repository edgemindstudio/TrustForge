#!/usr/bin/env python3
"""Publish one complete governed VAE synthetic pool from a verified BEST checkpoint.

Production execution is default-deny. The source-level synthesis gate must be
separately activated before this entry point can verify or load a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.vae_cic_pool_publication import (
    publish_vae_pool,
    require_vae_production_synthesis_authorization,
)


def _class_mapping(value):
    labels = tuple(
        item.strip()
        for item in value.split(",")
    )
    if not labels or any(not label for label in labels):
        raise argparse.ArgumentTypeError(
            "class mapping must be a nonempty comma-separated list"
        )
    return labels


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--expected-synthesis-source-commit",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-publication-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--dataset-id",
        required=True,
        choices=tuple(sorted(DATASET_CLASS_COUNTS)),
    )
    parser.add_argument("--experimental-seed", required=True, type=int)
    parser.add_argument(
        "--class-mapping",
        required=True,
        type=_class_mapping,
    )
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def run(args):
    require_vae_production_synthesis_authorization()

    manifest_path = publish_vae_pool(
        args.destination,
        publication_root=args.checkpoint_publication_root,
        generation_id=args.generation_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
        dataset_id=args.dataset_id,
        experimental_seed=args.experimental_seed,
        class_mapping=args.class_mapping,
        repository_root=args.repository_root,
        expected_synthesis_source_commit=(
            args.expected_synthesis_source_commit
        ),
    )

    from trustforge.evaluation.paper01_vae_pool import VAEPoolView

    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    view = VAEPoolView(
        manifest_path,
        manifest_sha256=manifest_sha256,
        family="vae",
        experimental_seed=args.experimental_seed,
        recipe_sha256=manifest["recipe_sha256"],
        checkpoint_sha256=manifest["checkpoint_manifest_sha256"],
    )
    preflight = view.preflight()
    summary = {
        "status": "COMPLETE",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "dataset_id": args.dataset_id,
        "family": "vae",
        "experimental_seed": args.experimental_seed,
        "class_mapping": list(args.class_mapping),
        "checkpoint_generation_id": args.generation_id,
        "checkpoint_manifest_sha256": args.expected_manifest_sha256,
        "synthesis_source_commit": args.expected_synthesis_source_commit,
        "verified_batches": preflight["verified_batches"],
        "verified_images": preflight["verified_images"],
    }
    print(
        "VAE_POOL_PUBLICATION_COMPLETE "
        + json.dumps(summary, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return summary


def main(argv=None):
    args = make_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
