"""Gated runner tests: no TensorFlow or training required."""

from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026.gan_adapter_identity import (
    DATASET_ADAPTER_TYPES,
    require_gan_adapter_identity,
)
from trustforge.data.paper01_revision_2026 import (
    CICMalDroidRevision2026,
    MaleXRevision2026,
    MalNetImageRevision2026,
    USTCRevision2026,
)

from trustforge.models.paper01_revision_2026.gan_governed_runner import (
    GAN_FEASIBILITY_TRAINING_ENABLED,
    GAN_PRODUCTION_TRAINING_ENABLED,
    GANRunPreparation,
    execute_gan_feasibility_run,
    prepare_gan_run,
)


@pytest.fixture(autouse=True)
def allow_lightweight_runner_fixture(monkeypatch):
    monkeypatch.setitem(
        DATASET_ADAPTER_TYPES, "malex", DatasetFixture
    )


class DatasetFixture:
    num_classes = 2
    id_to_class = {0: "benign", 1: "malware"}
    class_to_id = {"benign": 0, "malware": 1}

    def split_size(self, split):
        assert split == "train"
        return 1000


def prepare(tmp_path):
    return prepare_gan_run(
        dataset=DatasetFixture(),
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        checkpoint_directory=tmp_path,
    )


def test_feasibility_gate_is_disabled():
    assert GAN_FEASIBILITY_TRAINING_ENABLED is False
    assert GAN_PRODUCTION_TRAINING_ENABLED is False


def test_preparation_is_read_only(tmp_path):
    result = prepare(tmp_path)
    assert isinstance(result, GANRunPreparation)
    assert result.plan.planned_generator_iterations == 500
    assert result.plan.planned_discriminator_updates == 1000
    assert list(tmp_path.iterdir()) == []


def test_execution_rejected_before_tensorflow(tmp_path):
    result = prepare(tmp_path)
    with pytest.raises(PermissionError, match="not authorized"):
        execute_gan_feasibility_run(result, dataset=DatasetFixture())


def test_existing_metadata_rejected(tmp_path):
    (tmp_path / "model_metadata.json").write_text("{}")
    with pytest.raises(FileExistsError):
        prepare(tmp_path)


def test_existing_state_rejected(tmp_path):
    (tmp_path / "model_state.ckpt.index").write_text("occupied")
    with pytest.raises(FileExistsError):
        prepare(tmp_path)


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(ValueError):
        prepare_gan_run(
            dataset=DatasetFixture(),
            dataset_id="malex",
            experimental_seed=42,
            class_mapping=("benign", "malware"),
            checkpoint_directory=tmp_path / "not-created",
        )


def test_invalid_class_mapping_rejected(tmp_path):
    with pytest.raises(ValueError):
        prepare_gan_run(
            dataset=DatasetFixture(),
            dataset_id="malex",
            experimental_seed=42,
            class_mapping=("malware",),
            checkpoint_directory=tmp_path,
        )


def test_reordered_caller_mapping_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        prepare_gan_run(
            dataset=DatasetFixture(),
            dataset_id="malex",
            experimental_seed=42,
            class_mapping=("malware", "benign"),
            checkpoint_directory=tmp_path,
        )


def test_authoritative_malex_mapping():
    from trustforge.models.paper01_revision_2026.gan_class_mapping import (
        resolve_gan_class_mapping,
    )

    assert resolve_gan_class_mapping(
        dataset_id="malex", adapter=DatasetFixture()
    ) == ("benign", "malware")


def test_inconsistent_reverse_mapping_rejected():
    from trustforge.models.paper01_revision_2026.gan_class_mapping import (
        resolve_gan_class_mapping,
    )

    dataset = DatasetFixture()
    dataset.class_to_id = {"benign": 1, "malware": 0}

    with pytest.raises(ValueError):
        resolve_gan_class_mapping(
            dataset_id="malex", adapter=dataset
        )


def test_numeric_identifiers_remain_explicit():
    from trustforge.models.paper01_revision_2026.gan_class_mapping import (
        resolve_gan_class_mapping,
    )

    assert resolve_gan_class_mapping(
        dataset_id="ustc_tfc2016",
        adapter=object(),
    ) == tuple(str(i) for i in range(9))

    assert resolve_gan_class_mapping(
        dataset_id="cicmaldroid2020",
        adapter=object(),
    ) == tuple(str(i) for i in range(5))


def test_malnet_mapping_uses_integer_id_order():
    from trustforge.models.paper01_revision_2026.gan_class_mapping import (
        resolve_gan_class_mapping,
    )

    class Adapter:
        num_classes = 47
        id_to_class = {
            index: f"family_{index}" for index in reversed(range(47))
        }
        class_to_id = {
            f"family_{index}": index for index in range(47)
        }

    result = resolve_gan_class_mapping(
        dataset_id="malnet_image",
        adapter=Adapter(),
    )
    assert result[0] == "family_0"
    assert result[-1] == "family_46"


def test_real_adapter_registry_has_exact_governed_types():
    expected = {
        "ustc_tfc2016": USTCRevision2026,
        "cicmaldroid2020": CICMalDroidRevision2026,
        "malnet_image": MalNetImageRevision2026,
        "malex": MaleXRevision2026,
    }

    # The autouse fixture patches MaleX for runner unit tests.
    # Verify all four actual bindings independently.
    actual = dict(DATASET_ADAPTER_TYPES)
    actual["malex"] = MaleXRevision2026

    assert actual == expected


@pytest.mark.parametrize(
    "dataset_id,adapter_type",
    [
        ("ustc_tfc2016", USTCRevision2026),
        ("cicmaldroid2020", CICMalDroidRevision2026),
        ("malnet_image", MalNetImageRevision2026),
    ],
)
def test_correct_real_adapter_type_accepted(dataset_id, adapter_type):
    adapter = object.__new__(adapter_type)
    assert require_gan_adapter_identity(
        dataset_id=dataset_id,
        adapter=adapter,
    ) == adapter_type.__name__


def test_wrong_real_adapter_type_rejected():
    adapter = object.__new__(USTCRevision2026)

    with pytest.raises(TypeError, match="identity mismatch"):
        require_gan_adapter_identity(
            dataset_id="cicmaldroid2020",
            adapter=adapter,
        )


def test_unknown_dataset_identity_rejected():
    with pytest.raises(ValueError):
        require_gan_adapter_identity(
            dataset_id="unknown_dataset",
            adapter=object(),
        )


def test_plain_object_cannot_impersonate_adapter():
    with pytest.raises(TypeError, match="identity mismatch"):
        require_gan_adapter_identity(
            dataset_id="ustc_tfc2016",
            adapter=object(),
        )
