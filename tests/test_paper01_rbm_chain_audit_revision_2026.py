"""Focused tests for the Paper 1 revision-2026 RBM chain-audit runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    persist_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    train_shared_conditional_rbm,
)
from trustforge.provenance import canonical_json_bytes


RUNNER_PATH = Path(
    "scripts/run_paper01_rbm_chain_audit.py"
).resolve()

SPEC = importlib.util.spec_from_file_location(
    "paper01_rbm_chain_audit_runner",
    RUNNER_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        "unable to load RBM chain-audit runner"
    )

runner = importlib.util.module_from_spec(
    SPEC
)

sys.modules[SPEC.name] = runner

SPEC.loader.exec_module(
    runner
)


DATASET_ID = "tiny-rbm-chain-audit-test"
EXPERIMENTAL_SEED = 42
CLASS_MAPPING = (
    "class_0",
    "class_1",
)


class TinyAdapter:
    def __init__(self):
        self.train_images = np.stack(
            [
                np.full((40, 40, 3), 0.20, dtype=np.float32),
                np.full((40, 40, 3), 0.30, dtype=np.float32),
                np.full((40, 40, 3), 0.70, dtype=np.float32),
                np.full((40, 40, 3), 0.80, dtype=np.float32),
            ],
            axis=0,
        )

        self.train_labels = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.validation_images = np.stack(
            [
                np.full((40, 40, 3), 0.25, dtype=np.float32),
                np.full((40, 40, 3), 0.75, dtype=np.float32),
            ],
            axis=0,
        )

        self.validation_labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def split_size(
        self,
        split,
    ):
        if split == "train":
            return 4

        if split == "validation":
            return 2

        raise AssertionError(
            split
        )

    def canonical_batch(
        self,
        split,
        logical_positions,
    ):
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        if split == "train":
            return (
                self.train_images[
                    positions
                ],
                self.train_labels[
                    positions
                ],
            )

        if split == "validation":
            return (
                self.validation_images[
                    positions
                ],
                self.validation_labels[
                    positions
                ],
            )

        raise AssertionError(
            split
        )


@pytest.fixture(scope="module")
def training_result():
    return train_shared_conditional_rbm(
        adapter=TinyAdapter(),
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        train_class_positions={
            0: np.array(
                [0, 1],
                dtype=np.int64,
            ),
            1: np.array(
                [2, 3],
                dtype=np.int64,
            ),
        },
    )


@pytest.fixture
def checkpoint(
    tmp_path,
    training_result,
):
    return persist_rbm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_mapping=CLASS_MAPPING,
        training_result=training_result,
    )


def test_candidate_depths_are_exact_governed_set():
    assert runner.AUDIT_DEPTHS == (
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    )


def test_selection_status_remains_unselected():
    assert runner.SELECTION_STATUS == "UNSELECTED"


def test_compute_node_is_required(
    monkeypatch,
):
    monkeypatch.delenv(
        "SLURM_JOB_ID",
        raising=False,
    )

    monkeypatch.delenv(
        "SLURM_JOB_NODELIST",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Slurm compute allocation",
    ):
        runner._require_compute_node()


def test_compute_identity_is_recorded(
    monkeypatch,
):
    monkeypatch.setenv(
        "SLURM_JOB_ID",
        "12345",
    )

    monkeypatch.setenv(
        "SLURM_JOB_NODELIST",
        "node001",
    )

    observed = runner._require_compute_node()

    assert observed == {
        "slurm_job_id": "12345",
        "slurm_job_nodelist": "node001",
    }


def test_class_ids_are_validated_and_unique():
    assert runner._parse_class_ids(
        (
            "0",
            "1",
        ),
        class_count=2,
    ) == (
        0,
        1,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        runner._parse_class_ids(
            (
                "0",
                "0",
            ),
            class_count=2,
        )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        runner._parse_class_ids(
            (
                "2",
            ),
            class_count=2,
        )


def test_prefix_coupled_trajectory_captures_exact_depths(
    checkpoint,
):
    record = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=7,
    )

    assert tuple(
        item.depth
        for item in record.depths
    ) == runner.AUDIT_DEPTHS

    assert len(
        record.visible_states
    ) == len(
        runner.AUDIT_DEPTHS
    )

    assert record.class_id == 0

    assert record.replicate_index == 0

    assert record.sample_index == 7


def test_trajectory_preserves_class_identity(
    checkpoint,
):
    record = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=1,
        replicate_index=0,
        sample_index=11,
    )

    assert record.class_id == 1

    assert all(
        depth.finite_state
        for depth in record.depths
    )


def test_trajectory_states_are_float64_finite_and_bounded(
    checkpoint,
):
    record = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=13,
    )

    for visible in record.visible_states:
        assert visible.dtype == np.float64

        assert visible.shape == (
            1,
            4800,
        )

        assert np.isfinite(
            visible
        ).all()

        assert np.all(
            visible >= 0.0
        )

        assert np.all(
            visible <= 1.0
        )


def test_same_identity_reproduces_full_prefix_trajectory(
    checkpoint,
):
    first = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=17,
    )

    second = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=17,
    )

    for left, right in zip(
        first.visible_states,
        second.visible_states,
    ):
        np.testing.assert_array_equal(
            left,
            right,
        )


def test_independent_sample_identity_changes_trajectory(
    checkpoint,
):
    first = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=21,
    )

    second = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=1,
        sample_index=22,
    )

    assert any(
        not np.array_equal(
            left,
            right,
        )
        for left, right in zip(
            first.visible_states,
            second.visible_states,
        )
    )


def test_aggregate_depths_uses_cross_trajectory_coordinate_variance(
    checkpoint,
):
    trajectories = [
        runner.run_trajectory(
            checkpoint=checkpoint,
            class_id=0,
            replicate_index=0,
            sample_index=31,
        ),
        runner.run_trajectory(
            checkpoint=checkpoint,
            class_id=0,
            replicate_index=1,
            sample_index=32,
        ),
    ]

    aggregate = runner._aggregate_depths(
        trajectories
    )

    assert len(
        aggregate
    ) == len(
        runner.AUDIT_DEPTHS
    )

    first_depth = aggregate[
        0
    ]

    matrix = np.concatenate(
        [
            trajectory.visible_states[
                0
            ]
            for trajectory in trajectories
        ],
        axis=0,
    )

    expected_variance = np.var(
        matrix,
        axis=0,
        dtype=np.float64,
    )

    assert first_depth[
        "visible_coordinate_variance_mean"
    ] == pytest.approx(
        float(
            np.mean(
                expected_variance,
                dtype=np.float64,
            )
        )
    )

    assert first_depth[
        "visible_coordinate_variance_min"
    ] == pytest.approx(
        float(
            np.min(
                expected_variance
            )
        )
    )

    assert first_depth[
        "visible_coordinate_variance_max"
    ] == pytest.approx(
        float(
            np.max(
                expected_variance
            )
        )
    )


def test_adjacent_depth_diagnostics_are_absent_only_at_first_depth(
    checkpoint,
):
    record = runner.run_trajectory(
        checkpoint=checkpoint,
        class_id=0,
        replicate_index=0,
        sample_index=41,
    )

    first = record.depths[
        0
    ]

    assert first.previous_depth_displacement is None
    assert first.visible_mean_change_from_previous_depth is None
    assert (
        first.visible_variance_mean_change_from_previous_depth
        is None
    )

    for item in record.depths[
        1:
    ]:
        assert item.previous_depth_displacement is not None
        assert item.visible_mean_change_from_previous_depth is not None
        assert (
            item.visible_variance_mean_change_from_previous_depth
            is not None
        )


def test_output_path_must_remain_inside_repository(
    tmp_path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    outside = tmp_path / "outside.json"

    with pytest.raises(
        ValueError,
        match="inside repository",
    ):
        runner._validate_output_path(
            repo_root=repo_root,
            output_path=outside,
        )


def test_output_path_must_be_json(
    tmp_path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(
        ValueError,
        match=".json",
    ):
        runner._validate_output_path(
            repo_root=repo_root,
            output_path=Path(
                "audit.txt"
            ),
        )


def test_existing_evidence_is_never_overwritten(
    tmp_path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    target = repo_root / "audit.json"
    target.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="overwrite",
    ):
        runner._validate_output_path(
            repo_root=repo_root,
            output_path=target,
        )


def test_canonical_json_writer_uses_exact_one_trailing_newline(
    tmp_path,
):
    target = tmp_path / "audit.json"

    payload = {
        "b": 2,
        "a": 1,
    }

    runner._write_canonical_json(
        target,
        payload,
    )

    raw = target.read_bytes()

    assert raw == (
        canonical_json_bytes(
            payload
        )
        + b"\n"
    )

    assert not raw.endswith(
        b"\n\n"
    )


def test_canonical_json_writer_refuses_existing_file(
    tmp_path,
):
    target = tmp_path / "audit.json"

    target.write_bytes(
        b"existing\n"
    )

    with pytest.raises(
        FileExistsError,
    ):
        runner._write_canonical_json(
            target,
            {
                "a": 1,
            },
        )


def test_runner_has_no_chain_length_selection_function():
    names = set(
        vars(
            runner
        )
    )

    forbidden = {
        "select_chain_length",
        "choose_chain_length",
        "production_chain_length",
        "select_burn_in",
        "choose_burn_in",
    }

    assert names.isdisjoint(
        forbidden
    )


def test_runner_source_does_not_use_final_benchmark_metrics():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    ).lower()

    forbidden = (
        "fid",
        "kid",
        "ms-ssim",
        "classifier accuracy",
        "downstream accuracy",
    )

    for token in forbidden:
        assert token not in source


def test_runner_source_does_not_reference_historical_checkpoint_names():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    ).lower()

    forbidden = (
        "rbm_best.weights.h5",
        "rbm_last.weights.h5",
        "rbm_epoch_",
    )

    for token in forbidden:
        assert token not in source


def test_runner_source_does_not_reuse_cd_k():
    source = RUNNER_PATH.read_text(
        encoding="utf-8"
    )

    assert "CD_K" not in source
    assert "restricted_boltzmann_cd" not in source


def test_governed_tiny_training_replay_is_exact():
    """Identical governed training identities reproduce final state."""

    positions = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([2, 3], dtype=np.int64),
    }

    def train_once():
        return train_shared_conditional_rbm(
            adapter=TinyAdapter(),
            dataset_id=DATASET_ID,
            experimental_seed=EXPERIMENTAL_SEED,
            train_class_positions={
                key: value.copy()
                for key, value in positions.items()
            },
        )

    first = train_once()
    second = train_once()

    assert (
        first.optimizer_steps_completed
        == second.optimizer_steps_completed
        == first.budget.max_updates
    )

    assert first.budget == second.budget
    first_spec = first.traversal_spec
    second_spec = second.traversal_spec

    assert first_spec.dataset_id == second_spec.dataset_id
    assert (
        first_spec.total_training_examples
        == second_spec.total_training_examples
    )

    for field in (
        "class_ids",
        "class_counts",
        "class_probabilities",
    ):
        np.testing.assert_array_equal(
            getattr(first_spec, field),
            getattr(second_spec, field),
        )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()

    assert set(first_state) == set(second_state) == {
        "W", "a", "b", "U"
    }

    for key in ("W", "a", "b", "U"):
        np.testing.assert_array_equal(
            first_state[key],
            second_state[key],
        )
        assert np.isfinite(first_state[key]).all()

    assert [
        observation.step
        for observation in first.validation_history
    ] == [
        observation.step
        for observation in second.validation_history
    ]

    assert len(first.validation_history) == 3

    for left, right in zip(
        first.validation_history,
        second.validation_history,
    ):
        assert (
            left.diagnostics.reconstruction_mse
            == right.diagnostics.reconstruction_mse
        )


def test_diagnostic_optimizer_wrapper_preserves_training(
    monkeypatch,
):
    """Observing CD gradients must not change governed training."""

    import trustforge.models.paper01_revision_2026.restricted_boltzmann_training as training_module

    positions = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([2, 3], dtype=np.int64),
    }

    def train_once():
        return training_module.train_shared_conditional_rbm(
            adapter=TinyAdapter(),
            dataset_id=DATASET_ID,
            experimental_seed=EXPERIMENTAL_SEED,
            train_class_positions={
                key: value.copy()
                for key, value in positions.items()
            },
        )

    baseline = train_once()

    original_optimizer = training_module.apply_manual_sgd_update
    records = []

    def observing_optimizer(*, model, cd_result):
        gradient = np.array(
            cd_result.delta_U,
            dtype=np.float64,
            copy=True,
        )
        gradient.setflags(write=False)

        result = original_optimizer(
            model=model,
            cd_result=cd_result,
        )

        records.append({
            "gradient": gradient,
            "norm": float(np.linalg.norm(gradient)),
            "optimizer_norm": result.delta_U_norm,
        })

        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            training_module,
            "apply_manual_sgd_update",
            observing_optimizer,
        )
        observed = train_once()

    assert len(records) == observed.budget.max_updates

    for record in records:
        assert record["gradient"].shape == (2, 128)
        assert record["gradient"].dtype == np.float64
        assert not record["gradient"].flags.writeable
        assert np.isfinite(record["gradient"]).all()
        assert record["norm"] == record["optimizer_norm"]

    assert (
        baseline.optimizer_steps_completed
        == observed.optimizer_steps_completed
    )

    assert baseline.budget == observed.budget

    baseline_state = baseline.model.state_dict()
    observed_state = observed.model.state_dict()

    for name in ("W", "a", "b", "U"):
        np.testing.assert_array_equal(
            baseline_state[name],
            observed_state[name],
        )

    assert [
        (item.step, item.diagnostics.reconstruction_mse)
        for item in baseline.validation_history
    ] == [
        (item.step, item.diagnostics.reconstruction_mse)
        for item in observed.validation_history
    ]
