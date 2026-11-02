"""MaleX source-order provenance regression tests."""

from hashlib import sha256

import numpy as np
import pytest

from trustforge.data.paper01_revision_2026 import malex


def sequence_hash(values):
    digest = sha256()
    for value in values:
        digest.update(bytes(value))
        digest.update(b"\n")
    return digest.hexdigest()


@pytest.fixture
def identity_adapter(monkeypatch):
    benign = np.asarray(
        [f"{value:064x}".encode("ascii") for value in (1, 2, 3)],
        dtype="S64",
    )
    malware = np.asarray(
        [f"{value:064x}".encode("ascii") for value in (4, 5, 6, 7)],
        dtype="S64",
    )

    monkeypatch.setattr(malex, "MALEX_BENIGN_SOURCE_COUNT", 3)
    monkeypatch.setattr(malex, "MALEX_MALWARE_SOURCE_COUNT", 4)
    monkeypatch.setattr(
        malex, "MALEX_EXPECTED_BENIGN_SEQUENCE_SHA256",
        sequence_hash(benign),
    )
    monkeypatch.setattr(
        malex, "MALEX_EXPECTED_MALWARE_SEQUENCE_SHA256",
        sequence_hash(malware),
    )

    adapter = object.__new__(malex.MaleXRevision2026)
    adapter.validate_hashes = True
    adapter.validate_production_contract = True
    adapter._benign_source = benign
    adapter._malware_source = malware

    return adapter


def test_matching_source_sequences_pass(identity_adapter):
    identity_adapter._validate_source_sequences()


@pytest.mark.parametrize("name", ["_benign_source", "_malware_source"])
def test_reordered_source_sequence_rejected(identity_adapter, name):
    values = getattr(identity_adapter, name).copy()
    values[[0, 1]] = values[[1, 0]]
    setattr(identity_adapter, name, values)

    with pytest.raises(ValueError, match="sequence SHA-256 mismatch"):
        identity_adapter._validate_source_sequences()


@pytest.mark.parametrize("name", ["_benign_source", "_malware_source"])
def test_modified_source_identity_rejected(identity_adapter, name):
    values = getattr(identity_adapter, name).copy()
    values[0] = b"f" * 64
    setattr(identity_adapter, name, values)

    with pytest.raises(ValueError, match="sequence SHA-256 mismatch"):
        identity_adapter._validate_source_sequences()


def test_wrong_source_count_rejected(identity_adapter):
    identity_adapter._benign_source = (
        identity_adapter._benign_source[:-1]
    )

    with pytest.raises(ValueError, match="count mismatch"):
        identity_adapter._validate_source_sequences()


def test_wrong_source_dtype_rejected(identity_adapter):
    identity_adapter._benign_source = (
        identity_adapter._benign_source.astype("S65")
    )

    with pytest.raises(TypeError, match="dtype mismatch"):
        identity_adapter._validate_source_sequences()


@pytest.mark.parametrize(
    "hashes,production",
    [(False, False), (False, True), (True, False)],
)
def test_nonproduction_validation_skipped(
    identity_adapter, hashes, production
):
    identity_adapter.validate_hashes = hashes
    identity_adapter.validate_production_contract = production
    identity_adapter._benign_source = None
    identity_adapter._malware_source = None

    identity_adapter._validate_source_sequences()
