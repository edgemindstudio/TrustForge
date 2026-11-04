#!/usr/bin/env python3
"""Isolated Paper 1 Revision-2026 GMM pilot launcher."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


DATASET_IDS = (
    "ustc_tfc2016",
    "cicmaldroid2020",
    "malnet_image",
    "malex",
)

PILOT_SEED = 42
LABEL_BATCH_SIZE = 8192


def exact_class_positions(
    adapter, *, dataset_id, split, class_id, class_count
):
    """Collect exact class positions without loading images."""
    size = int(adapter.split_size(split))

    if size <= 0:
        raise RuntimeError(f"{split} split is empty")

    matched = []

    for start in range(0, size, LABEL_BATCH_SIZE):
        stop = min(start + LABEL_BATCH_SIZE, size)

        positions = np.arange(
            start, stop, dtype=np.int64
        )

        if dataset_id in (
            "ustc_tfc2016",
            "cicmaldroid2020",
        ):
            raw_labels = np.asarray(adapter.labels(split))
            if raw_labels.shape != (size,):
                raise RuntimeError(
                    f"{split} full labels have invalid shape"
                )
            raw_batch = raw_labels[start:stop]

        elif dataset_id in (
            "malnet_image",
            "malex",
        ):
            raw_batch = np.asarray(
                adapter.labels(split, positions)
            )
        else:
            raise ValueError(
                f"unsupported governed dataset: {dataset_id}"
            )

        if raw_batch.shape != (stop - start,):
            raise RuntimeError(
                f"{split} labels have invalid shape"
            )

        if (
            raw_batch.dtype.kind not in ("i", "u")
            or np.any(raw_batch < 0)
            or np.any(raw_batch >= class_count)
        ):
            raise RuntimeError(
                f"{split} contains invalid class IDs"
            )

        batch = raw_batch.astype(
            np.int32, copy=False
        )

        if batch.shape != (stop - start,):
            raise RuntimeError(
                f"{split} labels have invalid shape"
            )

        if np.any(batch < 0) or np.any(
            batch >= class_count
        ):
            raise RuntimeError(
                f"{split} contains invalid class IDs"
            )

        selected = positions[
            batch == class_id
        ]

        if selected.size:
            matched.append(selected)

    if not matched:
        raise RuntimeError(
            f"{split} has no class-{class_id} examples"
        )

    return np.concatenate(matched)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_IDS,
    )

    parser.add_argument(
        "--class-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--checkpoint-directory",
        required=True,
        type=Path,
    )

    parser.add_argument(
        '--malex-extracted-root',
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.seed != PILOT_SEED:
        raise ValueError(
            "initial governed pilot permits seed 42 only"
        )

    if args.class_id < 0:
        raise ValueError(
            "class ID must be nonnegative"
        )

    job_id = os.environ.get(
        "SLURM_JOB_ID", ""
    ).strip()

    if not job_id:
        raise RuntimeError(
            "GMM pilot training requires a Slurm allocation"
        )

    repo = Path.home() / "TrustForge"

    sys.path.insert(
        0, str(repo / "scripts")
    )

    from train_paper01_rbm_revision_checkpoint import (
        _build_adapter,
        _class_count,
        _ordered_class_mapping,
    )

    from trustforge.models.paper01_revision_2026.gan_adapter_identity import (
        require_gan_adapter_identity,
    )

    from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
        require_gan_dataset_provenance,
    )

    from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
        train_class_model,
    )

    from trustforge.models.paper01_revision_2026.gaussian_mixture_checkpoint import (
        persist_gmm_checkpoint,
        load_gmm_checkpoint,
    )

    if args.dataset != 'malex' and args.malex_extracted_root is not None:
        raise ValueError(
            '--malex-extracted-root is valid only for malex'
        )

    adapter = _build_adapter(
        dataset_id=args.dataset,
        repo_root=repo,
        malex_extracted_root=args.malex_extracted_root,
    )

    count = _class_count(
        dataset_id=args.dataset,
        adapter=adapter,
    )

    if args.class_id >= count:
        raise ValueError(
            "class ID exceeds governed class count"
        )

    mapping = _ordered_class_mapping(
        dataset_id=args.dataset,
        adapter=adapter,
        class_count=count,
    )

    identity = require_gan_adapter_identity(
        dataset_id=args.dataset,
        adapter=adapter,
    )

    provenance = require_gan_dataset_provenance(
        dataset_id=args.dataset,
        adapter=adapter,
    )

    # Dataset-position enumeration must be exact.
    train_positions = exact_class_positions(
        adapter,
        dataset_id=args.dataset,
        split="train",
        class_id=args.class_id,
        class_count=count,
    )

    validation_positions = exact_class_positions(
        adapter,
        dataset_id=args.dataset,
        split="validation",
        class_id=args.class_id,
        class_count=count,
    )

    root = (
        Path.home()
        / "trustforge_revision_2026_artifacts"
        / "paper01"
        / "gmm"
    )

    checkpoint = (
        root
        / args.dataset
        / f"seed{args.seed}"
        / f"class{args.class_id}"
    )

    if (
        args.checkpoint_directory.expanduser().resolve()
        != checkpoint.resolve()
    ):
        raise ValueError(
            "checkpoint path violates governed pilot layout"
        )

    if checkpoint.exists():
        raise FileExistsError(
            "pilot checkpoint destination already exists"
        )

    if not checkpoint.parent.is_dir():
        raise FileNotFoundError(
            "pilot checkpoint parent must already exist"
        )

    print(
        "GMM_PILOT_PREFLIGHT "
        + json.dumps(
            {
                "dataset_id": args.dataset,
                "class_id": args.class_id,
                "class_name": mapping[args.class_id],
                "experimental_seed": args.seed,
                "adapter_type": identity,
                "source_contract": provenance[
                    "source_contract"
                ],
                "train_class_count": int(
                    train_positions.size
                ),
                "validation_class_count": int(
                    validation_positions.size
                ),
                "slurm_job_id": job_id,
                "checkpoint_directory": str(checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Reserve a new destination before model optimization.
    checkpoint.mkdir(exist_ok=False)

    result = train_class_model(
        adapter=adapter,
        class_id=args.class_id,
        experimental_seed=args.seed,
        train_class_positions=train_positions,
        validation_class_positions=validation_positions,
    )

    published = persist_gmm_checkpoint(
        checkpoint,
        dataset_id=args.dataset,
        class_id=args.class_id,
        experimental_seed=args.seed,
        training_result=result,
        validation_class_count=int(
            validation_positions.size
        ),
    )

    restored = load_gmm_checkpoint(
        checkpoint,
        requested_dataset_id=args.dataset,
        requested_class_id=args.class_id,
        requested_experimental_seed=args.seed,
    )

    if (
        published.metadata_path
        != restored.metadata_path
    ):
        raise RuntimeError(
            "published checkpoint restoration mismatch"
        )

    print(
        "GMM_PILOT_COMPLETE "
        + json.dumps(
            {
                "dataset_id": args.dataset,
                "class_id": args.class_id,
                "experimental_seed": args.seed,
                "selected_step": result.selected_step,
                "selected_validation_nll":
                    result.selected_validation_nll,
                "optimizer_steps_completed":
                    result.optimizer_steps_completed,
                "validation_class_count": int(
                    validation_positions.size
                ),
                "checkpoint_directory": str(checkpoint),
                "restoration": "PASS",
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
