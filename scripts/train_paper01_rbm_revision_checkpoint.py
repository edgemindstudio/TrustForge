#!/usr/bin/env python3
"""Train one governed Paper-1 revision-2026 RBM checkpoint.

This runner is intentionally compute-node-only.  It does not introduce a
reduced pilot-training budget: training is delegated unchanged to
train_shared_conditional_rbm(), which applies the governed revision budget.

Produced checkpoints may subsequently be identified as revision-specific
chain-audit evidence.  Historical RBM artifacts are never read.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from trustforge.data.paper01_revision_2026 import (
    CICMalDroidRevision2026,
    MaleXRevision2026,
    MalNetImageRevision2026,
    USTCRevision2026,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    persist_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    train_shared_conditional_rbm,
)


DATASET_IDS = (
    "ustc_tfc2016",
    "cicmaldroid2020",
    "malnet_image",
    "malex",
)

LABEL_SCAN_BATCH_SIZE = 8192

DEFAULT_EXPERIMENTAL_SEED = 42


def _require_slurm_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError(
            "revision RBM checkpoint training must run inside a Slurm allocation"
        )


def _repo_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()

    if not (root / ".git").is_dir():
        raise ValueError(
            f"not a Git repository root: {root}"
        )

    return root


def _build_adapter(
    *,
    dataset_id: str,
    repo_root: Path,
    malex_extracted_root: Path | None = None,
):
    home = Path.home()

    if dataset_id == "ustc_tfc2016":
        return USTCRevision2026(
            home
            / "gencys"
            / "data"
            / "USTC-TFC2016_malware"
        )

    if dataset_id == "cicmaldroid2020":
        return CICMalDroidRevision2026(
            home
            / "gencys"
            / "data"
            / "CICMalDroid2020_raw"
            / "feature_vectors_syscalls_frequency_5_Cat.csv",
            home
            / "gencys"
            / "data"
            / "CICMalDroid2020_stage3"
            / "feature_columns.json",
        )

    if dataset_id == "malnet_image":
        return MalNetImageRevision2026(
            home
            / "gencys"
            / "data"
            / "MalNet-Image_raw"
            / "malnet-images",
            home
            / "gencys"
            / "data"
            / "MalNet-Image_revision_2026"
            / "type-1.0-prepared",
            repo_root
            / "studies"
            / "paper01_benchmark"
            / "audits"
            / "revision_2026"
            / "malnet_image_type_mapping.json",
        )

    if dataset_id == "malex":
        extracted = (
            malex_extracted_root
            if malex_extracted_root is not None
            else (
                home
                / "gencys"
                / "data"
                / "MaleX_raw"
                / "extracted"
            )
        )

        return MaleXRevision2026(
            extracted
            / "labels_zipped"
            / "MaleXv1-m864669-b179725_10062017.hdf5",
            extracted
            / "ben_bigram_imgs_zipped"
            / "byteplot_imgs_RxR"
            / "256",
            extracted
            / "mal_byteplot_imgs_zipped"
            / "byteplot_imgs_RxR"
            / "256",
        )

    raise ValueError(
        f"unsupported revision dataset: {dataset_id}"
    )


def _class_count(
    *,
    dataset_id: str,
    adapter,
) -> int:
    if dataset_id == "ustc_tfc2016":
        return 9

    if dataset_id == "cicmaldroid2020":
        return 5

    if dataset_id in (
        "malnet_image",
        "malex",
    ):
        return int(
            adapter.num_classes
        )

    raise AssertionError(
        dataset_id
    )


def _ordered_class_mapping(
    *,
    dataset_id: str,
    adapter,
    class_count: int,
) -> tuple[str, ...]:
    # USTC and CIC revision source contracts do not establish authoritative
    # human-readable class names.  Numeric strings therefore remain explicit
    # computational identifiers rather than inferred semantics.
    if dataset_id in (
        "ustc_tfc2016",
        "cicmaldroid2020",
    ):
        return tuple(
            str(class_id)
            for class_id in range(
                class_count
            )
        )

    if dataset_id in (
        "malnet_image",
        "malex",
    ):
        mapping = tuple(
            str(
                adapter.id_to_class[
                    class_id
                ]
            )
            for class_id in range(
                class_count
            )
        )

        if len(
            set(
                mapping
            )
        ) != class_count:
            raise RuntimeError(
                "revision adapter class mapping is not one-to-one"
            )

        return mapping

    raise AssertionError(
        dataset_id
    )


def _training_labels(
    *,
    dataset_id: str,
    adapter,
) -> np.ndarray:
    training_count = int(
        adapter.split_size(
            "train"
        )
    )

    if training_count <= 0:
        raise RuntimeError(
            "revision training split is empty"
        )

    if dataset_id in (
        "ustc_tfc2016",
        "cicmaldroid2020",
    ):
        labels = np.asarray(
            adapter.labels(
                "train"
            ),
            dtype=np.int32,
        )

        if labels.shape != (
            training_count,
        ):
            raise RuntimeError(
                "revision full training-label vector has invalid shape"
            )

        return labels

    labels = np.empty(
        training_count,
        dtype=np.int32,
    )

    for start in range(
        0,
        training_count,
        LABEL_SCAN_BATCH_SIZE,
    ):
        stop = min(
            start
            + LABEL_SCAN_BATCH_SIZE,
            training_count,
        )

        positions = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        batch_labels = np.asarray(
            adapter.labels(
                "train",
                positions,
            ),
            dtype=np.int32,
        )

        if batch_labels.shape != (
            stop - start,
        ):
            raise RuntimeError(
                "revision position-addressed training labels have invalid shape"
            )

        labels[
            start:stop
        ] = batch_labels

    return labels


def _train_class_positions(
    *,
    labels: np.ndarray,
    class_count: int,
) -> dict[int, np.ndarray]:
    if labels.ndim != 1:
        raise ValueError(
            "training labels must be one-dimensional"
        )

    expected = np.arange(
        class_count,
        dtype=np.int32,
    )

    observed = np.unique(
        labels
    )

    if not np.array_equal(
        observed,
        expected,
    ):
        raise RuntimeError(
            "training class universe is not exact contiguous zero-based IDs: "
            f"{observed.tolist()}"
        )

    result = {
        class_id: np.flatnonzero(
            labels
            == class_id
        ).astype(
            np.int64,
            copy=False,
        )
        for class_id in range(
            class_count
        )
    }

    total = sum(
        int(
            positions.shape[
                0
            ]
        )
        for positions in result.values()
    )

    if total != labels.shape[
        0
    ]:
        raise RuntimeError(
            "training class-position inventory does not cover full training split"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one governed Paper-1 revision-2026 RBM checkpoint "
            "inside a Slurm compute allocation."
        )
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--dataset-id",
        choices=DATASET_IDS,
        required=True,
    )

    parser.add_argument(
        "--experimental-seed",
        type=int,
        default=DEFAULT_EXPERIMENTAL_SEED,
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--malex-extracted-root",
        default=None,
        help=(
            "Optional MaleX extracted-data root. "
            "When omitted, the governed historical source location under "
            "~/gencys/data/MaleX_raw/extracted is used."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(
        argv
    )

    _require_slurm_compute_node()

    repo_root = _repo_root(
        args.repo_root
    )

    experimental_seed = int(
        args.experimental_seed
    )

    if experimental_seed < 0:
        raise ValueError(
            "experimental seed must be non-negative"
        )

    checkpoint_dir = Path(
        args.checkpoint_dir
    ).expanduser().resolve()

    malex_extracted_root = None

    if args.malex_extracted_root is not None:
        if args.dataset_id != "malex":
            raise ValueError(
                "--malex-extracted-root is valid only with dataset-id=malex"
            )

        malex_extracted_root = Path(
            args.malex_extracted_root
        ).expanduser().resolve()

        if not malex_extracted_root.is_dir():
            raise FileNotFoundError(
                "MaleX extracted-data root does not exist or is not a directory: "
                f"{malex_extracted_root}"
            )

    if checkpoint_dir.exists():
        raise FileExistsError(
            "refusing to reuse or overwrite revision RBM checkpoint directory: "
            f"{checkpoint_dir}"
        )

    adapter = _build_adapter(
        dataset_id=args.dataset_id,
        repo_root=repo_root,
        malex_extracted_root=malex_extracted_root,
    )

    class_count = _class_count(
        dataset_id=args.dataset_id,
        adapter=adapter,
    )

    class_mapping = _ordered_class_mapping(
        dataset_id=args.dataset_id,
        adapter=adapter,
        class_count=class_count,
    )

    labels = _training_labels(
        dataset_id=args.dataset_id,
        adapter=adapter,
    )

    class_positions = _train_class_positions(
        labels=labels,
        class_count=class_count,
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"dataset_id={args.dataset_id}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"experimental_seed={experimental_seed}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"training_examples={labels.shape[0]}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        "class_counts="
        + ",".join(
            f"{class_id}:{class_positions[class_id].shape[0]}"
            for class_id in range(
                class_count
            )
        )
    )

    print(
        "[P1-R1 RBM TRAIN] "
        "training_budget=GOVERNED_UNMODIFIED"
    )

    training_result = train_shared_conditional_rbm(
        adapter=adapter,
        dataset_id=args.dataset_id,
        experimental_seed=experimental_seed,
        train_class_positions=class_positions,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    persisted = persist_rbm_checkpoint(
        checkpoint_dir,
        dataset_id=args.dataset_id,
        experimental_seed=experimental_seed,
        class_mapping=class_mapping,
        training_result=training_result,
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"optimizer_steps_completed="
        f"{training_result.optimizer_steps_completed}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"checkpoint_dir={checkpoint_dir}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"state_path={persisted.state_path}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        f"metadata_path={persisted.metadata_path}"
    )

    print(
        "[P1-R1 RBM TRAIN] "
        "revision_checkpoint_training=COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
