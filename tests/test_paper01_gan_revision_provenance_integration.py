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
    source_path = Path(gan_governed_runner.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    core = functions["_execute_gan_authorized_run"]

    calls = []
    tf_imports = []

    for node in ast.walk(core):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append((node.func.id, node.lineno))
        elif isinstance(node, ast.Import):
            if any(
                alias.name == "tensorflow"
                for alias in node.names
            ):
                tf_imports.append(node.lineno)

    assert len(tf_imports) == 1

    def only_line(name):
        matches = [
            line
            for call_name, line in calls
            if call_name == name
        ]
        assert len(matches) == 1, (name, matches)
        return matches[0]

    identity_line = only_line("require_gan_adapter_identity")
    mapping_line = only_line("resolve_gan_class_mapping")
    provenance_line = only_line("require_gan_dataset_provenance")
    build_line = only_line("build_gan_v3_provenance")

    lines = source.splitlines()

    def guarded_line(fragment):
        matches = [
            i
            for i in range(core.lineno, core.end_lineno + 1)
            if fragment in lines[i - 1]
        ]
        assert len(matches) == 1, (fragment, matches)
        return matches[0]

    feasibility_line = guarded_line(
        "if not GAN_FEASIBILITY_TRAINING_ENABLED:"
    )
    production_line = guarded_line(
        "if not GAN_PRODUCTION_TRAINING_ENABLED:"
    )
    slurm_line = guarded_line(
        'if not os.environ.get("SLURM_JOB_ID"):'
    )

    assert feasibility_line < slurm_line
    assert production_line < slurm_line
    assert slurm_line < identity_line
    assert slurm_line < mapping_line
    assert identity_line < provenance_line
    assert mapping_line < provenance_line
    assert provenance_line < build_line
    assert build_line < tf_imports[0]

    for name, expected_mode in (
        ("execute_gan_feasibility_run", "feasibility"),
        ("execute_gan_production_run", "production"),
    ):
        wrapper = functions[name]
        statement = wrapper.body[-1]
        assert isinstance(statement, ast.Return)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == "_execute_gan_authorized_run"

        modes = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "mode"
        ]
        assert len(modes) == 1
        assert isinstance(modes[0], ast.Constant)
        assert modes[0].value == expected_mode

        # The shared guarded execution is invoked exactly once, from the
        # final return statement.
        execution_calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_execute_gan_authorized_run"
        ]
        assert execution_calls == [call]

        if expected_mode == "production":
            # Production preparation validation must precede execution.
            guards = [
                node for node in wrapper.body
                if isinstance(node, ast.If)
            ]
            assert len(guards) == 1
            guard = guards[0]
            assert guard.lineno < statement.lineno
            assert isinstance(guard.test, ast.UnaryOp)
            assert isinstance(guard.test.op, ast.Not)
            assert isinstance(guard.test.operand, ast.Call)
            assert isinstance(guard.test.operand.func, ast.Name)
            assert guard.test.operand.func.id == "isinstance"
            assert [
                ast.unparse(argument)
                for argument in guard.test.operand.args
            ] == ["preparation", "GANRunPreparation"]
            assert len(guard.body) == 1
            assert isinstance(guard.body[0], ast.Raise)
            assert isinstance(guard.body[0].exc, ast.Call)
            assert isinstance(guard.body[0].exc.func, ast.Name)
            assert guard.body[0].exc.func.id == "TypeError"

            # Governed experimental-seed validation must precede execution.
            seed_calls = [
                node
                for node in wrapper.body
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id
                == "require_governed_experimental_seed"
            ]
            assert len(seed_calls) == 1
            seed_call = seed_calls[0].value
            assert seed_call.lineno < statement.lineno
            assert [
                ast.unparse(argument)
                for argument in seed_call.args
            ] == ["preparation.plan.experimental_seed"]


def test_original_authorization_flags_remain_disabled():
    assert (
        gan_governed_runner.GAN_FEASIBILITY_TRAINING_ENABLED
        is False
    )
    assert (
        gan_governed_runner.GAN_PRODUCTION_TRAINING_ENABLED
        is False
    )
