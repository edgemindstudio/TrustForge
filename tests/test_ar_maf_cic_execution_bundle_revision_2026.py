"""Filesystem enforcement checks using synthetic execution-envelope fixtures."""

import pytest

from test_ar_maf_cic_execution_manifest_revision_2026 import fixture
from trustforge.models.paper01_revision_2026 import (
    ar_maf_checkpoint_bundle as bundle,
    ar_maf_checkpoint_publication as publication,
    ar_maf_cic_execution_manifest as execution,
)


def request(arguments, *, fitted=True):
    keys = (
        "recipe_raw", "expected_identity",
        "expected_source_commit", "expected_source_files",
    )
    result = {key: arguments[key] for key in keys}
    if fitted:
        result["expected_fitted_state_sha256"] = (
            arguments["expected_fitted_state_sha256"]
        )
    return result


def publish(parent, name, arguments, manifest_raw, *, fitted=True):
    return publication.publish_checkpoint_bundle(
        parent, name,
        manifest_raw=manifest_raw,
        best_raw=arguments["best_raw"],
        last_raw=arguments["last_raw"],
        **request(arguments, fitted=fitted),
    )


def test_execution_publication_and_anchored_read(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    raw = execution.prepare_cic_execution_manifest(**arguments)
    published = publish(tmp_path, "execution", arguments, raw)
    loaded = bundle.read_checkpoint_bundle(
        tmp_path / "execution", **request(arguments),
    )
    assert published["manifest"] == loaded["manifest"]
    assert loaded["best_raw"] == arguments["best_raw"]
    assert loaded["fitted_state_raw"] == arguments["fitted_state_raw"]


def test_new_path_rejects_legacy_before_destination_creation(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError):
        publish(
            tmp_path, "rejected", arguments,
            arguments["checkpoint_manifest_raw"],
        )
    assert not (tmp_path / "rejected").exists()


def test_legacy_path_rejects_execution_envelope(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    raw = execution.prepare_cic_execution_manifest(**arguments)
    with pytest.raises((ValueError, KeyError)):
        publish(tmp_path, "rejected", arguments, raw, fitted=False)
    assert not (tmp_path / "rejected").exists()


def test_legacy_publication_and_read_still_work(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    publish(
        tmp_path, "legacy", arguments,
        arguments["checkpoint_manifest_raw"], fitted=False,
    )
    loaded = bundle.read_checkpoint_bundle(
        tmp_path / "legacy", **request(arguments, fitted=False),
    )
    assert loaded["best_raw"] == arguments["best_raw"]
    with pytest.raises(ValueError):
        bundle.read_checkpoint_bundle(
            tmp_path / "legacy", **request(arguments),
        )


def test_wrong_requested_fitted_identity_rejected_on_read(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    raw = execution.prepare_cic_execution_manifest(**arguments)
    publish(tmp_path, "execution", arguments, raw)
    requested = request(arguments)
    requested["expected_fitted_state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="fitted-state identity mismatch"):
        bundle.read_checkpoint_bundle(tmp_path / "execution", **requested)


def test_changed_state_rejected_on_read(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    raw = execution.prepare_cic_execution_manifest(**arguments)
    publish(tmp_path, "execution", arguments, raw)
    (tmp_path / "execution" / "best_state.npz").write_bytes(b"corrupted")
    with pytest.raises(Exception):
        bundle.read_checkpoint_bundle(
            tmp_path / "execution", **request(arguments),
        )


def test_existing_generation_is_not_overwritten(fixture, tmp_path):
    arguments = fixture[0]
    tmp_path.chmod(0o700)
    raw = execution.prepare_cic_execution_manifest(**arguments)
    publish(tmp_path, "execution", arguments, raw)
    original = (tmp_path / "execution" / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish(tmp_path, "execution", arguments, raw)
    assert (tmp_path / "execution" / "manifest.json").read_bytes() == original
