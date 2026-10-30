"""Integrated v3 publication tests with real scientific verification."""

from pathlib import Path
import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_checkpoint_v3_persistence as persistence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_bundle import (
    verify_gan_checkpoint_v3_bundle,
)
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)
from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    MALEX_SOURCE_FINGERPRINTS,
    MALEX_PARTITION_POLICY,
    MALEX_LIMITATIONS,
)


class FakeIteration:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class FakeOptimizer:
    def __init__(self, value):
        self.iterations = FakeIteration(value)


class FakeModel:
    def __init__(self):
        class Input:
            shape = (None, 2)

        self.inputs = [object(), Input()]


class FakeCheckpoint:
    def write(self, prefix):
        prefix = Path(prefix)
        (prefix.parent / (prefix.name + ".index")).write_bytes(
            b"fixture-index"
        )
        (prefix.parent / (
            prefix.name + ".data-00000-of-00001"
        )).write_bytes(b"fixture-shard")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        persistence,
        "_checkpoint_objects",
        lambda *args: FakeCheckpoint(),
    )

    plan = plan_gan_run(
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        train_size=731049,
    )

    sizes = (125794, 605255)
    total = plan.nominal_real_exposures
    draws = (total // 2, total - total // 2)

    rows = []
    for index, size in enumerate(sizes):
        count = draws[index]
        unique = min(size, count)
        rows.append({
            "class_id": index,
            "train_size": size,
            "draws": count,
            "unique_positions": unique,
            "repeated_draws": count - unique,
            "nominal_passes": count / size,
            "unique_coverage": unique / size,
        })

    exposure, _ = build_gan_exposure_evidence(
        plan=plan,
        exposure_report={
            "batches": plan.planned_generator_iterations,
            "total_real_draws": total,
            "total_unique_positions": sum(
                row["unique_positions"] for row in rows
            ),
            "per_class": rows,
        },
    )

    provenance = {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "malex",
        "source_contract": "governed_adapter_constructor",
        "source_fingerprints": dict(MALEX_SOURCE_FINGERPRINTS),
        "partition_policy": dict(MALEX_PARTITION_POLICY),
        "verification_method":
            "production_adapter_integrity_checks_on_compute_node",
        "verification_scope":
            "label_archive_identity_sequences_and_partition_fingerprints",
        "verification_limitations": list(MALEX_LIMITATIONS),
    }

    arguments = {
        "dataset_id": "malex",
        "experimental_seed": 42,
        "class_mapping": ["benign", "malware"],
        "planned_generator_iterations":
            plan.planned_generator_iterations,
        "exposure_evidence": exposure,
        "dataset_provenance": provenance,
        "generator": FakeModel(),
        "discriminator": FakeModel(),
        "generator_optimizer":
            FakeOptimizer(plan.planned_generator_iterations),
        "discriminator_optimizer":
            FakeOptimizer(plan.planned_discriminator_updates),
    }
    return tmp_path, arguments


def publish(prepared):
    root, arguments = prepared
    metadata = persistence.persist_terminal_gan_checkpoint_v3(
        root, **arguments
    )
    return root, metadata


def test_integrated_publication(prepared):
    root, metadata = publish(prepared)
    verified = verify_gan_checkpoint_v3_bundle(root, metadata)
    assert verified["dataset_provenance"]["dataset_id"] == "malex"
    assert verified["exposure_evidence"]["dataset_id"] == "malex"


@pytest.mark.parametrize(
    "filename",
    [
        "model_state.ckpt.index",
        "model_state.ckpt.data-00000-of-00001",
        "gan_exposure_evidence.json",
        "gan_dataset_provenance.json",
    ],
)
def test_modified_artifact_rejected(prepared, filename):
    root, metadata = publish(prepared)
    path = root / filename
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_invalid_exposure_prevents_publication(prepared):
    root, arguments = prepared
    arguments["exposure_evidence"]["total_real_draws"] = 1
    with pytest.raises(ValueError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **arguments
        )
    assert list(root.iterdir()) == []


def test_invalid_provenance_prevents_publication(prepared):
    root, arguments = prepared
    arguments["dataset_provenance"]["verification_limitations"] = []
    with pytest.raises(ValueError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **arguments
        )
    assert list(root.iterdir()) == []


def test_metadata_published_last(prepared):
    root, metadata = publish(prepared)
    assert (root / "model_metadata.json").is_file()
    assert not (root / ".gan_checkpoint_publication.lock").exists()
    assert metadata["schema_version"] == "3.0"
