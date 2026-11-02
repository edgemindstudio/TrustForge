"""GAN provenance execution-boundary integration tests.

No TensorFlow import, image loading, or training is permitted.
"""

import ast
from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_adapter_identity,
    gan_governed_runner,
)


class LightweightMaleX:
    num_classes = 2
    id_to_class = {0: "benign", 1: "malware"}
    class_to_id = {"benign": 0, "malware": 1}

    def split_size(self, split):
        assert split == "train"
        return 1000


def make_preparation(monkeypatch, tmp_path):
    monkeypatch.setitem(
        gan_adapter_identity.DATASET_ADAPTER_TYPES,
        "malex",
        LightweightMaleX,
    )
    dataset = LightweightMaleX()
    preparation = gan_governed_runner.prepare_gan_run(
        dataset=dataset,
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        checkpoint_directory=tmp_path,
    )
    return preparation, dataset


def test_unauthorized_execution_does_not_call_provenance(
    monkeypatch, tmp_path
):
    preparation, dataset = make_preparation(
        monkeypatch, tmp_path
    )

    calls = []

    def unexpected_provenance(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Unauthorized execution reached provenance")

    monkeypatch.setattr(
        gan_governed_runner,
        "require_gan_dataset_provenance",
        unexpected_provenance,
    )

    assert (
        gan_governed_runner.GAN_FEASIBILITY_TRAINING_ENABLED
        is False
    )

    with pytest.raises(PermissionError, match="not authorized"):
        gan_governed_runner.execute_gan_feasibility_run(
            preparation, dataset=dataset
        )

    assert calls == []


def test_provenance_failure_blocks_tensorflow_boundary(
    monkeypatch, tmp_path
):
    preparation, dataset = make_preparation(
        monkeypatch, tmp_path
    )

    calls = []

    def reject_provenance(*, dataset_id, adapter):
        calls.append((dataset_id, adapter))
        raise ValueError("provenance deliberately rejected")

    monkeypatch.setattr(
        gan_governed_runner,
        "require_gan_dataset_provenance",
        reject_provenance,
    )

    # Temporarily enable only the isolated negative test.
    # No optimizer, model, or training code can be reached.
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_FEASIBILITY_TRAINING_ENABLED",
        True,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "isolated-negative-test")

    with pytest.raises(
        ValueError, match="provenance deliberately rejected"
    ):
        gan_governed_runner.execute_gan_feasibility_run(
            preparation, dataset=dataset
        )

    assert calls == [("malex", dataset)]


def test_source_order_preserves_all_execution_boundaries():
    source_path = Path(
        gan_governed_runner.__file__
    )
    tree = ast.parse(
        source_path.read_text(encoding="utf-8")
    )

    execute = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_gan_feasibility_run"
    )

    calls = []
    tf_imports = []

    for node in ast.walk(execute):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(
                    (node.func.id, node.lineno)
                )
        elif isinstance(node, ast.Import):
            if any(
                alias.name == "tensorflow"
                for alias in node.names
            ):
                tf_imports.append(node.lineno)

    assert len(tf_imports) == 1

    def only_line(name):
        matches = [
            line for call_name, line in calls
            if call_name == name
        ]
        assert len(matches) == 1, (name, matches)
        return matches[0]

    identity_line = only_line(
        "require_gan_adapter_identity"
    )
    provenance_line = only_line(
        "require_gan_dataset_provenance"
    )
    mapping_line = only_line(
        "resolve_gan_class_mapping"
    )

    assert identity_line < provenance_line
    assert mapping_line < provenance_line
    assert provenance_line < tf_imports[0]

    # Authorization and allocation guards remain before
    # all provenance work and TensorFlow initialization.
    lines = source_path.read_text(
        encoding="utf-8"
    ).splitlines()

    authorization_line = next(
        i for i, line in enumerate(lines, 1)
        if "if not GAN_FEASIBILITY_TRAINING_ENABLED:" in line
    )
    slurm_line = next(
        i for i, line in enumerate(lines, 1)
        if 'if not os.environ.get("SLURM_JOB_ID"):' in line
    )

    assert authorization_line < slurm_line
    assert slurm_line < provenance_line


def test_original_authorization_flags_remain_disabled():
    assert (
        gan_governed_runner.GAN_FEASIBILITY_TRAINING_ENABLED
        is False
    )
    assert (
        gan_governed_runner.GAN_PRODUCTION_TRAINING_ENABLED
        is False
    )
