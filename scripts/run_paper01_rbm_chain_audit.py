#!/usr/bin/env python3
"""Paper 1 revision-2026 RBM sampling-chain length audit runner.

This runner gathers revision-specific evidence at the governed candidate
transition depths:

    1, 2, 5, 10, 20, 50, 100

It intentionally does NOT select a production chain length.  Selection is a
separate governed decision after evidence review.

The runner consumes only revision-2026 RBM checkpoints and uses the governed
sampling primitives.  Historical Bernoulli-visible checkpoints are not
accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np

from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    load_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_sampling import (
    gibbs_transition,
    initialize_chain,
)
from trustforge.provenance import canonical_json_bytes


SCHEMA_VERSION = "trustforge.paper01.revision2026.rbm_chain_audit.v1"

AUDIT_DEPTHS = (
    1,
    2,
    5,
    10,
    20,
    50,
    100,
)

DEFAULT_REPLICATES = 16

DISPLACEMENT_METRIC = "mean_absolute_visible_coordinate_difference"

SELECTION_STATUS = "UNSELECTED"


@dataclass(frozen=True)
class DepthDiagnostics:
    """Diagnostics for one governed trajectory at one candidate depth."""

    depth: int
    finite_state: bool
    visible_min: float
    visible_max: float
    visible_mean: float
    visible_variance_mean: float
    visible_variance_min: float
    visible_variance_max: float
    hidden_activation_mean: float
    initialization_displacement: float
    previous_depth_displacement: float | None
    visible_mean_change_from_previous_depth: float | None
    visible_variance_mean_change_from_previous_depth: float | None


@dataclass(frozen=True)
class TrajectoryRecord:
    """One independent governed prefix-coupled audit trajectory."""

    class_id: int
    replicate_index: int
    sample_index: int
    depths: tuple[DepthDiagnostics, ...]
    visible_states: tuple[np.ndarray, ...]


def _exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an exact integer, not bool"
        )

    try:
        result = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if result != value:
        raise TypeError(
            f"{name} must be an exact integer"
        )

    if result < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return result


def _git_head(
    repo_root: Path,
) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(
                repo_root
            ),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return completed.stdout.strip()


def _require_compute_node() -> dict[str, str]:
    """Fail closed unless this process is running inside a Slurm allocation."""

    job_id = os.environ.get(
        "SLURM_JOB_ID",
        ""
    ).strip()

    node_list = os.environ.get(
        "SLURM_JOB_NODELIST",
        ""
    ).strip()

    if not job_id:
        raise RuntimeError(
            "RBM chain audit must execute inside a Slurm compute allocation"
        )

    return {
        "slurm_job_id": job_id,
        "slurm_job_nodelist": node_list,
    }


def _parse_class_ids(
    raw_values: Sequence[str],
    *,
    class_count: int,
) -> tuple[int, ...]:
    if not raw_values:
        raise ValueError(
            "at least one --class-id is required"
        )

    parsed: list[int] = []

    for raw in raw_values:
        try:
            class_id = int(
                raw
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid class id: {raw}"
            ) from exc

        if str(
            class_id
        ) != raw.strip():
            raise ValueError(
                f"class id must be a canonical integer: {raw}"
            )

        if class_id < 0 or class_id >= class_count:
            raise ValueError(
                f"class id {class_id} lies outside [0,{class_count - 1}]"
            )

        parsed.append(
            class_id
        )

    if len(
        set(
            parsed
        )
    ) != len(
        parsed
    ):
        raise ValueError(
            "duplicate class IDs are not permitted"
        )

    return tuple(
        parsed
    )


def _mean_absolute_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if left.shape != right.shape:
        raise ValueError(
            "displacement operands must have identical shape"
        )

    value = float(
        np.mean(
            np.abs(
                left
                - right
            ),
            dtype=np.float64,
        )
    )

    if not np.isfinite(
        value
    ):
        raise RuntimeError(
            "displacement diagnostic is non-finite"
        )

    return value


def _visible_variance_summary(
    visible: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    """Return mean/min/max coordinate-variance diagnostics.

    A single trajectory state contains one visible vector, so coordinate
    variance is summarized over the 4,800 coordinates of that state.
    """

    coordinate_variance = np.var(
        visible,
        axis=1,
        dtype=np.float64,
    )

    mean_value = float(
        np.mean(
            coordinate_variance,
            dtype=np.float64,
        )
    )

    minimum = float(
        np.min(
            coordinate_variance
        )
    )

    maximum = float(
        np.max(
            coordinate_variance
        )
    )

    for name, value in (
        (
            "visible variance mean",
            mean_value,
        ),
        (
            "visible variance minimum",
            minimum,
        ),
        (
            "visible variance maximum",
            maximum,
        ),
    ):
        if not np.isfinite(
            value
        ):
            raise RuntimeError(
                f"{name} is non-finite"
            )

    return (
        mean_value,
        minimum,
        maximum,
    )


def _diagnostics(
    *,
    depth: int,
    initial_visible: np.ndarray,
    previous_visible: np.ndarray | None,
    previous_mean: float | None,
    previous_variance_mean: float | None,
    visible: np.ndarray,
    hidden: np.ndarray,
) -> DepthDiagnostics:
    finite_state = bool(
        np.isfinite(
            visible
        ).all()
        and np.isfinite(
            hidden
        ).all()
    )

    if not finite_state:
        raise RuntimeError(
            "RBM chain audit encountered non-finite state"
        )

    visible_min = float(
        np.min(
            visible
        )
    )

    visible_max = float(
        np.max(
            visible
        )
    )

    if visible_min < 0.0 or visible_max > 1.0:
        raise RuntimeError(
            "RBM chain audit encountered out-of-range visible state"
        )

    visible_mean = float(
        np.mean(
            visible,
            dtype=np.float64,
        )
    )

    (
        variance_mean,
        variance_min,
        variance_max,
    ) = _visible_variance_summary(
        visible
    )

    hidden_activation_mean = float(
        np.mean(
            hidden,
            dtype=np.float64,
        )
    )

    initialization_displacement = _mean_absolute_difference(
        visible,
        initial_visible,
    )

    if previous_visible is None:
        previous_depth_displacement = None
        visible_mean_change = None
        variance_change = None
    else:
        previous_depth_displacement = _mean_absolute_difference(
            visible,
            previous_visible,
        )

        visible_mean_change = float(
            abs(
                visible_mean
                - float(
                    previous_mean
                )
            )
        )

        variance_change = float(
            abs(
                variance_mean
                - float(
                    previous_variance_mean
                )
            )
        )

    for name, value in (
        (
            "visible mean",
            visible_mean,
        ),
        (
            "hidden activation mean",
            hidden_activation_mean,
        ),
        (
            "initialization displacement",
            initialization_displacement,
        ),
    ):
        if not np.isfinite(
            value
        ):
            raise RuntimeError(
                f"{name} is non-finite"
            )

    return DepthDiagnostics(
        depth=depth,
        finite_state=finite_state,
        visible_min=visible_min,
        visible_max=visible_max,
        visible_mean=visible_mean,
        visible_variance_mean=variance_mean,
        visible_variance_min=variance_min,
        visible_variance_max=variance_max,
        hidden_activation_mean=hidden_activation_mean,
        initialization_displacement=initialization_displacement,
        previous_depth_displacement=previous_depth_displacement,
        visible_mean_change_from_previous_depth=visible_mean_change,
        visible_variance_mean_change_from_previous_depth=variance_change,
    )


def run_trajectory(
    *,
    checkpoint,
    class_id: int,
    replicate_index: int,
    sample_index: int,
) -> TrajectoryRecord:
    """Run one prefix-coupled trajectory and capture governed candidate depths."""

    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=class_id,
        sample_index=sample_index,
    )

    if state.class_id != class_id:
        raise RuntimeError(
            "RBM chain initialization changed requested class identity"
        )

    initial_visible = state.visible.copy()

    previous_candidate_visible: np.ndarray | None = None
    previous_candidate_mean: float | None = None
    previous_candidate_variance_mean: float | None = None

    records: list[
        DepthDiagnostics
    ] = []

    visible_states: list[
        np.ndarray
    ] = []

    depth_set = set(
        AUDIT_DEPTHS
    )

    for transition in range(
        1,
        max(
            AUDIT_DEPTHS
        )
        + 1,
    ):
        state = gibbs_transition(
            checkpoint=checkpoint,
            state=state,
            sample_index=sample_index,
        )

        if state.class_id != class_id:
            raise RuntimeError(
                "RBM chain audit detected class switching"
            )

        if state.transition_count != transition:
            raise RuntimeError(
                "RBM chain transition counter is inconsistent"
            )

        if transition not in depth_set:
            continue

        record = _diagnostics(
            depth=transition,
            initial_visible=initial_visible,
            previous_visible=previous_candidate_visible,
            previous_mean=previous_candidate_mean,
            previous_variance_mean=previous_candidate_variance_mean,
            visible=state.visible,
            hidden=state.hidden,
        )

        records.append(
            record
        )

        visible_states.append(
            state.visible.copy()
        )

        previous_candidate_visible = state.visible.copy()
        previous_candidate_mean = record.visible_mean
        previous_candidate_variance_mean = record.visible_variance_mean

    if tuple(
        record.depth
        for record in records
    ) != AUDIT_DEPTHS:
        raise RuntimeError(
            "RBM chain audit failed to capture exact governed candidate depths"
        )

    if len(
        visible_states
    ) != len(
        AUDIT_DEPTHS
    ):
        raise RuntimeError(
            "RBM chain audit visible-state capture is incomplete"
        )

    return TrajectoryRecord(
        class_id=class_id,
        replicate_index=replicate_index,
        sample_index=sample_index,
        depths=tuple(
            records
        ),
        visible_states=tuple(
            visible_states
        ),
    )


def _aggregate_depths(
    trajectories: Sequence[
        TrajectoryRecord
    ],
) -> list[
    dict[
        str,
        object,
    ]
]:
    aggregates: list[
        dict[
            str,
            object,
        ]
    ] = []

    for depth_index, depth in enumerate(
        AUDIT_DEPTHS
    ):
        depth_records = [
            trajectory.depths[
                depth_index
            ]
            for trajectory in trajectories
        ]

        visible_means = np.asarray(
            [
                record.visible_mean
                for record in depth_records
            ],
            dtype=np.float64,
        )

        depth_visible = np.concatenate(
            [
                trajectory.visible_states[
                    depth_index
                ]
                for trajectory in trajectories
            ],
            axis=0,
        )

        if depth_visible.shape != (
            len(
                trajectories
            ),
            4800,
        ):
            raise RuntimeError(
                "RBM chain-audit aggregate visible matrix has invalid shape"
            )

        coordinate_variances = np.var(
            depth_visible,
            axis=0,
            dtype=np.float64,
        )

        if coordinate_variances.shape != (
            4800,
        ):
            raise RuntimeError(
                "RBM chain-audit coordinate variance has invalid shape"
            )

        if not np.isfinite(
            coordinate_variances
        ).all():
            raise RuntimeError(
                "RBM chain-audit coordinate variance is non-finite"
            )

        coordinate_variance_mean = float(
            np.mean(
                coordinate_variances,
                dtype=np.float64,
            )
        )

        coordinate_variance_min = float(
            np.min(
                coordinate_variances
            )
        )

        coordinate_variance_max = float(
            np.max(
                coordinate_variances
            )
        )

        hidden_means = np.asarray(
            [
                record.hidden_activation_mean
                for record in depth_records
            ],
            dtype=np.float64,
        )

        initialization_displacements = np.asarray(
            [
                record.initialization_displacement
                for record in depth_records
            ],
            dtype=np.float64,
        )

        previous_displacements = [
            record.previous_depth_displacement
            for record in depth_records
            if record.previous_depth_displacement is not None
        ]

        mean_changes = [
            record.visible_mean_change_from_previous_depth
            for record in depth_records
            if record.visible_mean_change_from_previous_depth is not None
        ]

        variance_changes = [
            record.visible_variance_mean_change_from_previous_depth
            for record in depth_records
            if record.visible_variance_mean_change_from_previous_depth
            is not None
        ]

        aggregate = {
            "depth": depth,
            "trajectory_count": len(
                depth_records
            ),
            "finite_state_all": all(
                record.finite_state
                for record in depth_records
            ),
            "visible_min": float(
                min(
                    record.visible_min
                    for record in depth_records
                )
            ),
            "visible_max": float(
                max(
                    record.visible_max
                    for record in depth_records
                )
            ),
            "visible_mean_mean": float(
                np.mean(
                    visible_means,
                    dtype=np.float64,
                )
            ),
            "visible_mean_std": float(
                np.std(
                    visible_means,
                    dtype=np.float64,
                )
            ),
            "visible_coordinate_variance_mean": coordinate_variance_mean,
            "visible_coordinate_variance_min": coordinate_variance_min,
            "visible_coordinate_variance_max": coordinate_variance_max,
            "hidden_activation_mean_mean": float(
                np.mean(
                    hidden_means,
                    dtype=np.float64,
                )
            ),
            "hidden_activation_mean_std": float(
                np.std(
                    hidden_means,
                    dtype=np.float64,
                )
            ),
            "initialization_displacement_mean": float(
                np.mean(
                    initialization_displacements,
                    dtype=np.float64,
                )
            ),
            "initialization_displacement_std": float(
                np.std(
                    initialization_displacements,
                    dtype=np.float64,
                )
            ),
            "previous_depth_displacement_mean": (
                None
                if not previous_displacements
                else float(
                    np.mean(
                        np.asarray(
                            previous_displacements,
                            dtype=np.float64,
                        ),
                        dtype=np.float64,
                    )
                )
            ),
            "absolute_visible_mean_change_mean": (
                None
                if not mean_changes
                else float(
                    np.mean(
                        np.asarray(
                            mean_changes,
                            dtype=np.float64,
                        ),
                        dtype=np.float64,
                    )
                )
            ),
            "absolute_visible_variance_change_mean": (
                None
                if not variance_changes
                else float(
                    np.mean(
                        np.asarray(
                            variance_changes,
                            dtype=np.float64,
                        ),
                        dtype=np.float64,
                    )
                )
            ),
        }

        aggregates.append(
            aggregate
        )

    return aggregates


def _validate_output_path(
    *,
    repo_root: Path,
    output_path: Path,
) -> Path:
    root = repo_root.resolve()

    target = (
        output_path
        if output_path.is_absolute()
        else root
        / output_path
    ).resolve()

    try:
        target.relative_to(
            root
        )
    except ValueError as exc:
        raise ValueError(
            "RBM chain-audit evidence output must remain inside repository"
        ) from exc

    if target.suffix != ".json":
        raise ValueError(
            "RBM chain-audit evidence output must be a .json file"
        )

    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite existing RBM chain-audit evidence: {target}"
        )

    return target


def _write_canonical_json(
    path: Path,
    payload: dict[
        str,
        object,
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoded = (
        canonical_json_bytes(
            payload
        )
        + b"\n"
    )

    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL,
        0o600,
    )

    try:
        with os.fdopen(
            descriptor,
            "wb",
        ) as handle:
            handle.write(
                encoded
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )
    except Exception:
        try:
            path.unlink(
                missing_ok=True
            )
        finally:
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Paper 1 revision-2026 RBM prefix-coupled "
            "sampling-chain audit on a Slurm compute node."
        )
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )

    parser.add_argument(
        "--class-id",
        action="append",
        required=True,
        help=(
            "Governed class ID to audit. Repeat this option for multiple classes."
        ),
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
    )

    parser.add_argument(
        "--sample-index-offset",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    return parser


def main(
    argv: Sequence[
        str
    ] | None = None,
) -> int:
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    compute_identity = _require_compute_node()

    repo_root = Path(
        args.repo_root
    ).expanduser().resolve()

    if not (
        repo_root
        / ".git"
    ).is_dir():
        parser.error(
            f"not a Git repository root: {repo_root}"
        )

    replicates = _exact_nonnegative_integer(
        args.replicates,
        name="replicates",
    )

    if replicates < 2:
        raise ValueError(
            "RBM chain audit requires at least two independent replicates"
        )

    sample_index_offset = _exact_nonnegative_integer(
        args.sample_index_offset,
        name="sample_index_offset",
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    ).expanduser().resolve()

    checkpoint = load_rbm_checkpoint(
        checkpoint_dir
    )

    class_ids = _parse_class_ids(
        args.class_id,
        class_count=checkpoint.model.class_count,
    )

    trajectories: list[
        TrajectoryRecord
    ] = []

    for class_id in class_ids:
        for replicate_index in range(
            replicates
        ):
            sample_index = (
                sample_index_offset
                + class_id
                * replicates
                + replicate_index
            )

            trajectories.append(
                run_trajectory(
                    checkpoint=checkpoint,
                    class_id=class_id,
                    replicate_index=replicate_index,
                    sample_index=sample_index,
                )
            )

    payload: dict[
        str,
        object,
    ] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "paper01_revision_2026",
        "model_family": "restricted_boltzmann",
        "audit_type": "sampling_chain_length",
        "repository_head": _git_head(
            repo_root
        ),
        "dataset_id": checkpoint.metadata[
            "dataset_id"
        ],
        "experimental_seed": checkpoint.metadata[
            "experimental_seed"
        ],
        "checkpoint_state_sha256": checkpoint.metadata[
            "state_sha256"
        ],
        "checkpoint_selection_rule": checkpoint.metadata[
            "checkpoint_selection_rule"
        ],
        "class_ids": list(
            class_ids
        ),
        "replicates_per_class": replicates,
        "sample_index_offset": sample_index_offset,
        "candidate_depths": list(
            AUDIT_DEPTHS
        ),
        "prefix_coupling": True,
        "independent_replicates": True,
        "class_clamped": True,
        "initialization": {
            "hidden_state": "zero_vector",
            "visible_parameter": "a",
            "visible_distribution": "bounded_exponential_p_v_given_h_zero",
            "internal_dtype": "float64",
            "empirical_data_initialization": False,
        },
        "displacement_metric": DISPLACEMENT_METRIC,
        "selection_status": SELECTION_STATUS,
        "production_chain_length": None,
        "production_burn_in": None,
        "final_benchmark_metrics_used_for_selection": False,
        "compute_identity": compute_identity,
        "trajectory_count": len(
            trajectories
        ),
        "trajectories": [
            {
                "class_id": trajectory.class_id,
                "replicate_index": trajectory.replicate_index,
                "sample_index": trajectory.sample_index,
                "depths": [
                    asdict(
                        record
                    )
                    for record in trajectory.depths
                ],
            }
            for trajectory in trajectories
        ],
        "aggregate_by_depth": _aggregate_depths(
            trajectories
        ),
    }

    output_path = _validate_output_path(
        repo_root=repo_root,
        output_path=Path(
            args.output_json
        ),
    )

    _write_canonical_json(
        output_path,
        payload,
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"repository_head={payload['repository_head']}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"dataset_id={payload['dataset_id']}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"experimental_seed={payload['experimental_seed']}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"class_ids={list(class_ids)}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"replicates_per_class={replicates}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"candidate_depths={list(AUDIT_DEPTHS)}"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        "production_chain_length=UNSELECTED"
    )

    print(
        "[P1-R1 RBM CHAIN AUDIT] "
        f"evidence={output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
