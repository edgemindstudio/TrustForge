"""Focused tests for the revision-2026 RBM checkpoint trainer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


RUNNER_PATH = Path(
    "scripts/train_paper01_rbm_revision_checkpoint.py"
).resolve()

SPEC = importlib.util.spec_from_file_location(
    "paper01_rbm_revision_checkpoint_trainer",
    RUNNER_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        "unable to load RBM revision checkpoint trainer"
    )

runner = importlib.util.module_from_spec(
    SPEC
)

sys.modules[
    SPEC.name
] = runner

SPEC.loader.exec_module(
    runner
)


class SplitWideAdapter:
    def __init__(
        self,
        labels,
    ):
        self._labels = np.asarray(
            labels,
            dtype=np.int32,
        )

    def split_size(
        self,
        split,
    ):
        assert split == "train"

        return int(
            self._labels.shape[
                0
            ]
        )

    def labels(
        self,
        split,
    ):
        assert split == "train"

        return self._labels.copy()


class PositionAddressedAdapter:
    def __init__(
        self,
        labels,
        id_to_class,
    ):
        self._labels = np.asarray(
            labels,
            dtype=np.int32,
        )

        self.num_classes = len(
            id_to_class
        )

        self.id_to_class = dict(
            id_to_class
        )

    def split_size(
        self,
        split,
    ):
        assert split == "train"

        return int(
            self._labels.shape[
                0
            ]
        )

    def labels(
        self,
        split,
        logical_positions,
    ):
        assert split == "train"

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        return self._labels[
            positions
        ]


def test_supported_dataset_ids_are_exact():
    assert runner.DATASET_IDS == (
        "ustc_tfc2016",
        "cicmaldroid2020",
        "malnet_image",
        "malex",
    )


def test_default_experimental_seed_preserves_historical_seed42():
    assert runner.DEFAULT_EXPERIMENTAL_SEED == 42


def test_parser_malex_extracted_root_defaults_to_none():
    args = runner.build_parser().parse_args(
        [
            "--dataset-id",
            "malex",
            "--checkpoint-dir",
            "/tmp/checkpoint",
        ]
    )

    assert args.malex_extracted_root is None


def test_parser_accepts_malex_extracted_root_override():
    args = runner.build_parser().parse_args(
        [
            "--dataset-id",
            "malex",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--malex-extracted-root",
            "/tmp/staged-malex",
        ]
    )

    assert args.malex_extracted_root == "/tmp/staged-malex"


def test_build_adapter_uses_explicit_malex_extracted_root(
    monkeypatch,
    tmp_path,
):
    staged_root = tmp_path / "staged-malex"
    captured = {}

    def fake_malex_revision_2026(
        label_archive,
        benign_byteplot_root,
        malware_byteplot_root,
    ):
        captured["label_archive"] = Path(label_archive)
        captured["benign_byteplot_root"] = Path(benign_byteplot_root)
        captured["malware_byteplot_root"] = Path(malware_byteplot_root)
        return object()

    monkeypatch.setattr(
        runner,
        "MaleXRevision2026",
        fake_malex_revision_2026,
    )

    runner._build_adapter(
        dataset_id="malex",
        repo_root=tmp_path,
        malex_extracted_root=staged_root,
    )

    assert captured["label_archive"] == (
        staged_root
        / "labels_zipped"
        / "MaleXv1-m864669-b179725_10062017.hdf5"
    )
    assert captured["benign_byteplot_root"] == (
        staged_root
        / "ben_bigram_imgs_zipped"
        / "byteplot_imgs_RxR"
        / "256"
    )
    assert captured["malware_byteplot_root"] == (
        staged_root
        / "mal_byteplot_imgs_zipped"
        / "byteplot_imgs_RxR"
        / "256"
    )


def test_slurm_compute_node_is_required(
    monkeypatch,
):
    monkeypatch.delenv(
        "SLURM_JOB_ID",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Slurm allocation",
    ):
        runner._require_slurm_compute_node()


def test_slurm_guard_accepts_job_identity(
    monkeypatch,
):
    monkeypatch.setenv(
        "SLURM_JOB_ID",
        "12345",
    )

    runner._require_slurm_compute_node()


def test_repo_root_must_be_git_repository(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="Git repository",
    ):
        runner._repo_root(
            str(
                tmp_path
            )
        )


def test_ustc_and_cic_use_numeric_string_class_mapping():
    dummy = object()

    assert runner._ordered_class_mapping(
        dataset_id="ustc_tfc2016",
        adapter=dummy,
        class_count=3,
    ) == (
        "0",
        "1",
        "2",
    )

    assert runner._ordered_class_mapping(
        dataset_id="cicmaldroid2020",
        adapter=dummy,
        class_count=2,
    ) == (
        "0",
        "1",
    )


def test_malnet_and_malex_use_adapter_semantic_mapping():
    adapter = PositionAddressedAdapter(
        labels=[
            0,
            1,
        ],
        id_to_class={
            0: "alpha",
            1: "beta",
        },
    )

    assert runner._ordered_class_mapping(
        dataset_id="malnet_image",
        adapter=adapter,
        class_count=2,
    ) == (
        "alpha",
        "beta",
    )

    assert runner._ordered_class_mapping(
        dataset_id="malex",
        adapter=adapter,
        class_count=2,
    ) == (
        "alpha",
        "beta",
    )


def test_semantic_class_mapping_must_be_one_to_one():
    adapter = PositionAddressedAdapter(
        labels=[
            0,
            1,
        ],
        id_to_class={
            0: "same",
            1: "same",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="one-to-one",
    ):
        runner._ordered_class_mapping(
            dataset_id="malnet_image",
            adapter=adapter,
            class_count=2,
        )


@pytest.mark.parametrize(
    "dataset_id",
    (
        "ustc_tfc2016",
        "cicmaldroid2020",
    ),
)
def test_split_wide_training_labels(
    dataset_id,
):
    adapter = SplitWideAdapter(
        [
            0,
            1,
            1,
            0,
        ]
    )

    observed = runner._training_labels(
        dataset_id=dataset_id,
        adapter=adapter,
    )

    np.testing.assert_array_equal(
        observed,
        np.array(
            [
                0,
                1,
                1,
                0,
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.parametrize(
    "dataset_id",
    (
        "malnet_image",
        "malex",
    ),
)
def test_position_addressed_training_labels_cover_full_split(
    dataset_id,
):
    source = np.asarray(
        [
            0,
            1,
            0,
            1,
            1,
            0,
        ],
        dtype=np.int32,
    )

    adapter = PositionAddressedAdapter(
        labels=source,
        id_to_class={
            0: "a",
            1: "b",
        },
    )

    observed = runner._training_labels(
        dataset_id=dataset_id,
        adapter=adapter,
    )

    np.testing.assert_array_equal(
        observed,
        source,
    )


def test_training_labels_reject_empty_split():
    adapter = SplitWideAdapter(
        []
    )

    with pytest.raises(
        RuntimeError,
        match="empty",
    ):
        runner._training_labels(
            dataset_id="ustc_tfc2016",
            adapter=adapter,
        )


def test_train_class_positions_cover_exact_training_split():
    labels = np.asarray(
        [
            0,
            1,
            1,
            2,
            0,
            2,
        ],
        dtype=np.int32,
    )

    positions = runner._train_class_positions(
        labels=labels,
        class_count=3,
    )

    np.testing.assert_array_equal(
        positions[
            0
        ],
        np.array(
            [
                0,
                4,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        positions[
            1
        ],
        np.array(
            [
                1,
                2,
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        positions[
            2
        ],
        np.array(
            [
                3,
                5,
            ],
            dtype=np.int64,
        ),
    )


def test_train_class_positions_require_contiguous_zero_based_classes():
    labels = np.asarray(
        [
            0,
            2,
        ],
        dtype=np.int32,
    )

    with pytest.raises(
        RuntimeError,
        match="contiguous zero-based",
    ):
        runner._train_class_positions(
            labels=labels,
            class_count=3,
        )


def test_class_count_contracts():
    dummy = type(
        "Dummy",
        (),
        {
            "num_classes": 47,
        },
    )()

    assert runner._class_count(
        dataset_id="ustc_tfc2016",
        adapter=dummy,
    ) == 9

    assert runner._class_count(
        dataset_id="cicmaldroid2020",
        adapter=dummy,
    ) == 5

    assert runner._class_count(
        dataset_id="malnet_image",
        adapter=dummy,
    ) == 47


def test_source_contains_no_reduced_training_budget():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    ).lower()

    forbidden = (
        "reduced_budget",
        "pilot_max_updates",
        "max_updates=",
        "train_per_class_cap",
        "epoch_limit",
    )

    for token in forbidden:
        assert token not in source


def test_source_delegates_training_to_governed_training_function():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    )

    assert "train_shared_conditional_rbm(" in source
    assert "persist_rbm_checkpoint(" in source


def test_source_does_not_reference_historical_rbm_checkpoint_files():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    ).lower()

    forbidden = (
        "rbm_best",
        "rbm_last",
        "weights.h5",
        "restrictedboltzmann/",
    )

    for token in forbidden:
        assert token not in source
