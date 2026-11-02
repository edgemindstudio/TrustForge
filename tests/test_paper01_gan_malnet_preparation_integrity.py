"""Isolated MalNet baseline-verification tests."""

import hashlib
import json

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_malnet_preparation_integrity as integrity,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "type-1.0-prepared"
    root.mkdir()

    baseline_file = tmp_path / "baseline.json"
    baseline_splits = {}
    manifest_splits = {}

    for split, rows in integrity.SPLIT_COUNTS.items():
        artifacts = {}
        for field, filename in (
            ("prepared_text_sha256", f"{split}.txt"),
            ("offsets_sha256", f"{split}_offsets.npy"),
        ):
            data = f"fixture:{filename}".encode("utf-8")
            (root / filename).write_bytes(data)
            artifacts[field] = digest(data)

        baseline_splits[split] = {
            "rows": rows,
            **artifacts,
        }
        manifest_splits[split] = {
            "provider_rows": rows,
            "prepared_rows": rows,
            "prepared_text_file": f"{split}.txt",
            "offsets_file": f"{split}_offsets.npy",
            **artifacts,
        }

    baseline = {
        "dataset_id": "malnet_image",
        "preparation": "type-1.0-prepared",
        "provider_archive_sha256_declared":
            integrity.PROVIDER_ARCHIVE_SHA256,
        "splits": baseline_splits,
    }
    manifest = {
        "dataset": "MalNet-Image",
        "task": "type-level classification",
        "provider_archive_sha256":
            integrity.PROVIDER_ARCHIVE_SHA256,
        "splits": manifest_splits,
    }

    write_json(baseline_file, baseline)
    write_json(root / "manifest.json", manifest)

    monkeypatch.setattr(
        integrity, "BASELINE_SHA256",
        digest(baseline_file.read_bytes()),
    )
    monkeypatch.setattr(
        integrity, "MANIFEST_SHA256",
        digest((root / "manifest.json").read_bytes()),
    )

    return root, baseline_file


def verify(prepared):
    root, baseline = prepared
    return integrity.verify_malnet_preparation_integrity(
        preparation_dir=root,
        baseline_path=baseline,
    )


def test_valid_fixture(prepared):
    result = verify(prepared)
    assert len(result["verified_artifacts"]) == 6
    assert result["original_image_bytes_verified"] is False


def test_baseline_tamper_rejected(prepared):
    _, baseline = prepared
    baseline.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="baseline SHA-256"):
        verify(prepared)


def test_manifest_tamper_rejected(prepared):
    root, _ = prepared
    (root / "manifest.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify(prepared)


@pytest.mark.parametrize(
    "filename",
    [
        "train.txt",
        "validation_offsets.npy",
        "test.txt",
    ],
)
def test_prepared_file_tamper_rejected(prepared, filename):
    root, _ = prepared
    with (root / filename).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="artifact SHA-256"):
        verify(prepared)


def test_missing_file_rejected(prepared):
    root, _ = prepared
    (root / "test_offsets.npy").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify(prepared)


def test_symlink_rejected(prepared):
    root, _ = prepared
    original = root / "train.txt"
    destination = root / "copied.txt"
    destination.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(destination)

    with pytest.raises(ValueError, match="missing or unsafe"):
        verify(prepared)


def test_manifest_filename_traversal_rejected(prepared, monkeypatch):
    root, _ = prepared
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["prepared_text_file"] = "../train.txt"
    write_json(path, manifest)

    monkeypatch.setattr(
        integrity, "MANIFEST_SHA256",
        digest(path.read_bytes()),
    )

    with pytest.raises(ValueError, match="unsafe artifact filename"):
        verify(prepared)


def test_wrong_split_count_rejected(prepared, monkeypatch):
    root, _ = prepared
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["train"]["prepared_rows"] = 1
    write_json(path, manifest)

    monkeypatch.setattr(
        integrity, "MANIFEST_SHA256",
        digest(path.read_bytes()),
    )

    with pytest.raises(ValueError, match="row-count mismatch"):
        verify(prepared)


def test_baseline_symlink_rejected(prepared):
    root, baseline = prepared
    target = baseline.with_name("target.json")
    target.write_bytes(baseline.read_bytes())
    baseline.unlink()
    baseline.symlink_to(target)

    with pytest.raises(ValueError, match="missing or unsafe"):
        verify((root, baseline))
