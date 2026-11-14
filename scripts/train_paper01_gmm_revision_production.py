#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 class-specific GMM training.

One invocation trains one dataset/class/experimental-seed checkpoint.
Compute-node only. Requires an explicit clean committed source revision and
the matching current source-freeze file. Never replaces an existing
checkpoint directory. This runner does not authorize Slurm submission or
production scale-out.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from trustforge.models.paper01_revision_2026.gaussian_mixture_source_binding import (
    require_governed_experimental_seed,
    verify_gmm_source_binding,
)

CLASS_COUNTS = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
    "malnet_image": 47,
    "malex": 2,
}


def require_environment(args):
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError("GMM production training requires a Slurm job")
    return verify_gmm_source_binding(
        repository_root=Path.cwd(),
        source_freeze_path=args.source_freeze,
        expected_source_commit=args.expected_source_commit,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True,
                        choices=tuple(CLASS_COUNTS))
    parser.add_argument("--class-id", required=True, type=int)
    parser.add_argument("--experimental-seed", required=True, type=int)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--source-freeze", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
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


def require_sources(args, names: tuple[str, ...]) -> list[Path]:
    values = [getattr(args, name) for name in names]
    if any(value is None for value in values):
        raise ValueError(
            "missing required dataset source arguments: " + ", ".join(names)
        )
    for path in values:
        if not path.exists() or path.is_symlink():
            raise ValueError(f"missing or symlinked dataset source: {path}")
    return values


def build_adapter(args):
    from trustforge.data.paper01_revision_2026 import (
        USTCRevision2026,
        CICMalDroidRevision2026,
        MalNetImageRevision2026,
        MaleXRevision2026,
    )

    if args.dataset_id == "ustc_tfc2016":
        (source,) = require_sources(args, ("ustc_source_dir",))
        return USTCRevision2026(source)

    if args.dataset_id == "cicmaldroid2020":
        source, columns = require_sources(
            args, ("cic_source_csv", "cic_feature_columns")
        )
        return CICMalDroidRevision2026(
            source, columns, validate_hashes=True
        )

    if args.dataset_id == "malnet_image":
        image_root, prep, mapping = require_sources(
            args, (
                "malnet_image_root",
                "malnet_preparation_dir",
                "malnet_mapping_path",
            )
        )
        return MalNetImageRevision2026(
            image_root, prep, mapping,
            validate_hashes=True,
            validate_production_contract=True,
        )

    label_archive, benign_root, malware_root = require_sources(
        args, (
            "malex_label_archive",
            "malex_benign_root",
            "malex_malware_root",
        )
    )
    return MaleXRevision2026(
        label_archive, benign_root, malware_root,
        validate_hashes=True,
        validate_production_contract=True,
    )


def class_positions(adapter, split: str, class_id: int) -> np.ndarray:
    """Return zero-based split-local positions for one governed class."""
    size = adapter.split_size(split)

    if type(size) is not int or size <= 0:
        raise RuntimeError(f"{split} split size must be positive")

    adapter_name = type(adapter).__name__

    whole_split_adapters = {
        "USTCRevision2026",
        "CICMalDroidRevision2026",
    }
    positioned_adapters = {
        "MalNetImageRevision2026",
        "MaleXRevision2026",
    }

    def validate_labels(raw, expected_count):
        labels = np.asarray(raw)

        if labels.ndim != 1 or labels.size != expected_count:
            raise RuntimeError(
                f"{split} labels and requested positions disagree"
            )

        if not np.issubdtype(labels.dtype, np.integer):
            raise RuntimeError(
                f"{split} labels must be integer class IDs"
            )

        return labels

    if adapter_name in whole_split_adapters:
        labels = validate_labels(adapter.labels(split), size)
        positions = np.flatnonzero(labels == class_id).astype(np.int64)

    elif adapter_name in positioned_adapters:
        chunks = []
        chunk_size = 65536

        for begin in range(0, size, chunk_size):
            end = min(begin + chunk_size, size)

            logical_positions = np.arange(
                begin, end, dtype=np.int64
            )

            labels = validate_labels(
                adapter.labels(split, logical_positions),
                end - begin,
            )

            matches = np.flatnonzero(labels == class_id)

            if matches.size:
                chunks.append(matches.astype(np.int64) + begin)

        positions = (
            np.concatenate(chunks)
            if chunks
            else np.empty(0, dtype=np.int64)
        )

    else:
        raise TypeError(
            f"unsupported GMM dataset adapter: {adapter_name}"
        )

    if positions.size == 0:
        raise RuntimeError(
            f"requested class absent from {split}"
        )

    return positions


def require_fresh_destination(args, expected_source_commit) -> Path:
    """Require an unused, identity-bound production checkpoint directory."""
    root = (
        Path.home()
        / "trustforge_revision_2026_artifacts"
        / "paper01"
        / "gmm"
    ).resolve(strict=True)

    requested = args.checkpoint_dir.expanduser().absolute()

    if ".." in requested.parts:
        raise ValueError("checkpoint path must not contain parent traversal")

    for candidate in (requested, *requested.parents):
        if candidate.is_symlink():
            raise ValueError("checkpoint path contains a symlink")

    expected_parent = (
        root
        / args.dataset_id
        / f"seed{args.experimental_seed}"
        / "production"
    )

    if requested.parent != expected_parent:
        raise ValueError(
            "checkpoint directory does not match dataset/seed "
            "production namespace"
        )

    prefix = (
        f"class{args.class_id}_seed{args.experimental_seed}_"
        f"commit{expected_source_commit[:7]}_attempt"
    )

    if not requested.name.startswith(prefix):
        raise ValueError(
            "checkpoint directory does not match class/seed/commit identity"
        )

    attempt = requested.name[len(prefix):]

    if len(attempt) < 2 or not attempt.isascii() or not attempt.isdigit():
        raise ValueError("checkpoint attempt must be a numeric identifier")

    if requested.exists():
        raise FileExistsError(
            f"checkpoint destination exists: {requested}"
        )

    if not requested.parent.is_dir():
        raise ValueError(
            "production checkpoint parent directory must already exist"
        )

    return requested


def run(args) -> None:
    source_binding = require_environment(args)

    if type(args.class_id) is not int or not (
        0 <= args.class_id < CLASS_COUNTS[args.dataset_id]
    ):
        raise ValueError("invalid class ID for dataset")

    seed = require_governed_experimental_seed(args.experimental_seed)

    destination = require_fresh_destination(
        args, source_binding["expected_source_commit"],
    )
    adapter = build_adapter(args)

    source_paths = {
        name: str(getattr(args, name).expanduser().resolve())
        for name in (
            "ustc_source_dir", "cic_source_csv", "cic_feature_columns",
            "malnet_image_root", "malnet_preparation_dir",
            "malnet_mapping_path", "malex_label_archive",
            "malex_benign_root", "malex_malware_root",
        )
        if getattr(args, name) is not None
    }
    execution_evidence = {
        **source_binding,
        "dataset_adapter": type(adapter).__name__,
        "dataset_source_paths": source_paths,
    }

    train_positions = class_positions(adapter, "train", args.class_id)
    validation_positions = class_positions(
        adapter, "validation", args.class_id
    )

    print("GMM_PRODUCTION_PREFLIGHT", json.dumps({
        "dataset_id": args.dataset_id,
        "class_id": args.class_id,
        "experimental_seed": seed,
        "train_class_count": int(train_positions.size),
        "validation_class_count": int(validation_positions.size),
        "checkpoint_directory": str(destination),
        "source_head": source_binding["expected_source_commit"],
        "source_freeze_sha256": source_binding["source_freeze_sha256"],
    }), flush=True)

    from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
        train_class_model,
    )
    from trustforge.models.paper01_revision_2026.gaussian_mixture_checkpoint import (
        persist_gmm_checkpoint,
        load_gmm_checkpoint,
    )

    # Atomic reservation: never overwrite or reuse an existing directory.
    destination.mkdir(mode=0o700, exist_ok=False)

    training_result = train_class_model(
        adapter=adapter,
        class_id=args.class_id,
        experimental_seed=seed,
        train_class_positions=train_positions,
        validation_class_positions=validation_positions,
    )

    persist_gmm_checkpoint(
        destination,
        dataset_id=args.dataset_id,
        class_id=args.class_id,
        experimental_seed=seed,
        training_result=training_result,
        validation_class_count=int(validation_positions.size),
        execution_evidence=execution_evidence,
    )

    # Independent read from the newly published artifact pair.
    restored = load_gmm_checkpoint(
        destination,
        requested_dataset_id=args.dataset_id,
        requested_class_id=args.class_id,
        requested_experimental_seed=seed,
    )
    if restored is None:
        raise RuntimeError("GMM checkpoint restoration returned no result")

    print("GMM_PRODUCTION_COMPLETE", json.dumps({
        "dataset_id": args.dataset_id,
        "class_id": args.class_id,
        "experimental_seed": seed,
        "source_head": source_binding["expected_source_commit"],
        "source_freeze_sha256": source_binding["source_freeze_sha256"],
        "selected_step": int(training_result.selected_step),
        "selected_validation_nll": float(
            training_result.selected_validation_nll
        ),
        "checkpoint_directory": str(destination),
    }), flush=True)


def main() -> None:
    args = make_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
