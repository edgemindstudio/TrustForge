import hashlib
from pathlib import Path
import runpy
import pytest

import trustforge.evaluation.paper01_classifier_bundle as bundle
from trustforge.evaluation.paper01_classifier_manifest import (
    prepare_classifier_manifest, canonical_manifest,
)

HELPERS = runpy.run_path(str(
    Path(__file__).with_name("test_paper01_classifier_manifest.py")
))
RECIPE = HELPERS["RECIPE"]


def package():
    context, training, state = HELPERS["fixture"]()
    manifest = prepare_classifier_manifest(
        context=context, training_report=training,
        state_raw=state, recipe_raw=RECIPE,
    )
    raw = canonical_manifest(manifest)
    return {
        "manifest_raw": raw,
        "state_raw": state,
        "recipe_raw": RECIPE,
        "expected_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_context": context,
        "expected_selected_weights_sha256": training["selected_weights_sha256"],
    }


def private_parent(tmp_path):
    parent = tmp_path / "publications"
    parent.mkdir(mode=0o700)
    return parent


def publish(parent, prepared):
    return bundle.publish_classifier_bundle(parent, "generation", **prepared)


def read(directory, prepared):
    return bundle.read_classifier_bundle(
        directory,
        **{
            key: value for key, value in prepared.items()
            if key not in ("manifest_raw", "state_raw")
        },
    )


def test_publish_and_read_verified_bytes(tmp_path):
    parent, prepared = private_parent(tmp_path), package()
    result = publish(parent, prepared)
    assert result["publication_status"] == "WRITTEN_VERIFIED_AND_SYNCED"
    loaded = read(parent / "generation", prepared)
    assert loaded["state_raw"] == prepared["state_raw"]
    assert loaded["manifest_raw"] == prepared["manifest_raw"]
    assert set(p.name for p in (parent / "generation").iterdir()) == bundle.MEMBERS


def test_collision_does_not_overwrite(tmp_path):
    parent, prepared = private_parent(tmp_path), package()
    publish(parent, prepared)
    original = (parent / "generation/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish(parent, prepared)
    assert (parent / "generation/manifest.json").read_bytes() == original


def test_nonprivate_parent_rejected_before_generation_creation(tmp_path):
    parent, prepared = private_parent(tmp_path), package()
    parent.chmod(0o755)
    with pytest.raises(ValueError):
        publish(parent, prepared)
    assert not (parent / "generation").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", "a\\b"])
def test_invalid_generation_names_rejected(tmp_path, name):
    parent, prepared = private_parent(tmp_path), package()
    with pytest.raises(ValueError):
        bundle.publish_classifier_bundle(parent, name, **prepared)
    assert list(parent.iterdir()) == []


def test_wrong_expectation_rejected_before_writing(tmp_path):
    parent, prepared = private_parent(tmp_path), package()
    prepared["expected_selected_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        publish(parent, prepared)
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("fault", ["missing", "extra", "changed"])
def test_incomplete_or_changed_bundle_rejected(tmp_path, fault):
    parent, prepared = private_parent(tmp_path), package()
    publish(parent, prepared)
    directory = parent / "generation"
    if fault == "missing":
        (directory / "manifest.json").unlink()
    elif fault == "extra":
        (directory / "unexpected").write_bytes(b"extra")
    else:
        (directory / "selected_state.npz").write_bytes(
            prepared["state_raw"] + b"extra"
        )
    with pytest.raises(ValueError):
        read(directory, prepared)


@pytest.mark.parametrize("member", ["manifest.json", "selected_state.npz"])
def test_symlinked_member_rejected(tmp_path, member):
    parent, prepared = private_parent(tmp_path), package()
    publish(parent, prepared)
    directory = parent / "generation"
    member_path = directory / member
    external = tmp_path / "external"
    external.write_bytes(member_path.read_bytes())
    member_path.unlink()
    member_path.symlink_to(external)
    with pytest.raises(ValueError):
        read(directory, prepared)


def test_symlinked_bundle_directory_rejected(tmp_path):
    parent, prepared = private_parent(tmp_path), package()
    publish(parent, prepared)
    alias = tmp_path / "alias"
    alias.symlink_to(parent / "generation", target_is_directory=True)
    with pytest.raises(ValueError):
        read(alias, prepared)


def test_state_write_failure_retains_artifact_without_manifest(tmp_path, monkeypatch):
    parent, prepared = private_parent(tmp_path), package()
    original = bundle.write_exclusive_vae_artifact

    def fail_after_write(fd, name, raw):
        result = original(fd, name, raw)
        if name == "selected_state.npz":
            raise OSError("fixture state synchronization failure")
        return result

    monkeypatch.setattr(bundle, "write_exclusive_vae_artifact", fail_after_write)
    with pytest.raises(OSError):
        publish(parent, prepared)
    directory = parent / "generation"
    assert (directory / "selected_state.npz").is_file()
    assert not (directory / "manifest.json").exists()


def test_manifest_phase_failure_is_indeterminate_and_retained(tmp_path, monkeypatch):
    parent, prepared = private_parent(tmp_path), package()
    original = bundle.write_exclusive_vae_artifact

    def fail_after_write(fd, name, raw):
        result = original(fd, name, raw)
        if name == "manifest.json":
            raise OSError("fixture manifest synchronization failure")
        return result

    monkeypatch.setattr(bundle, "write_exclusive_vae_artifact", fail_after_write)
    with pytest.raises(bundle.IndeterminateClassifierPublicationError):
        publish(parent, prepared)
    directory = parent / "generation"
    assert (directory / "selected_state.npz").is_file()
    assert (directory / "manifest.json").is_file()
    # A readable bundle does not resolve the failed publication outcome.
    assert read(directory, prepared)["state_raw"] == prepared["state_raw"]
