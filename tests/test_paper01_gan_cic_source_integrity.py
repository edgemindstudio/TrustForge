"""Independent CIC GAN source-integrity boundary tests."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/trustforge/models/paper01_revision_2026"
    / "gan_dataset_provenance.py"
)

spec = importlib.util.spec_from_file_location(
    "isolated_cic_gan_provenance",
    MODULE_PATH,
)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def adapter_for(csv, features, **changes):
    values = dict(
        validate_hashes=True,
        partition_seed=42,
        train_fraction=0.7,
        validation_fraction=0.1,
        test_fraction=0.2,
        source_csv=csv,
        feature_columns_path=features,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def synthetic_sources(tmp_path):
    csv = tmp_path / "source.csv"
    features = tmp_path / "features.json"
    csv.write_bytes(b"synthetic CIC rows")
    features.write_bytes(b"synthetic feature contract")
    return csv, features


def test_authoritative_hash_constants_are_present():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert (
        "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00"
        in source
    )
    assert (
        "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f"
        in source
    )


def test_missing_source_rejected(synthetic_sources):
    csv, features = synthetic_sources
    csv.unlink()
    with pytest.raises(FileNotFoundError):
        provenance.require_gan_dataset_provenance(
            dataset_id="cicmaldroid2020",
            adapter=adapter_for(csv, features),
        )


@pytest.mark.parametrize(
    "attribute",
    ["source_csv", "feature_columns_path"],
)
def test_non_path_rejected(synthetic_sources, attribute):
    csv, features = synthetic_sources
    with pytest.raises(ValueError, match="requires Path"):
        provenance.require_gan_dataset_provenance(
            dataset_id="cicmaldroid2020",
            adapter=adapter_for(
                csv, features, **{attribute: str(csv)}
            ),
        )


@pytest.mark.parametrize(
    "attribute",
    ["source_csv", "feature_columns_path"],
)
def test_modified_source_rejected(synthetic_sources, attribute):
    csv, features = synthetic_sources
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        provenance.require_gan_dataset_provenance(
            dataset_id="cicmaldroid2020",
            adapter=adapter_for(csv, features),
        )


def test_integrity_flag_cannot_be_bypassed(synthetic_sources):
    csv, features = synthetic_sources
    with pytest.raises(ValueError, match="validate_hashes"):
        provenance.require_gan_dataset_provenance(
            dataset_id="cicmaldroid2020",
            adapter=adapter_for(
                csv, features, validate_hashes=False
            ),
        )


def test_hashing_algorithm_uses_actual_file_bytes(synthetic_sources):
    csv, features = synthetic_sources

    original = provenance._sha256_file(csv)
    assert original == sha256(csv.read_bytes()).hexdigest()

    csv.write_bytes(csv.read_bytes() + b" changed")
    changed = provenance._sha256_file(csv)

    assert changed != original
