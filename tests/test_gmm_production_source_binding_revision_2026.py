"""Tests for explicit GMM production source binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_source_binding import (
    GOVERNED_EXPERIMENTAL_SEEDS,
    HISTORICAL_GIT_BASELINE,
    IMPLEMENTATION_PARENT_REVISION,
    REQUIRED_SOURCE_FILES,
    SOURCE_FREEZE_SCHEMA,
    require_governed_experimental_seed,
    verify_gmm_source_binding,
)


def _git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    ).strip()


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "GMM Fixture")
    _git(root, "config", "user.email", "gmm@example.invalid")
    for name in REQUIRED_SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("fixture source: " + name + "\n").encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def _manifest(root, *, schema=SOURCE_FREEZE_SCHEMA):
    files = {}
    for name in REQUIRED_SOURCE_FILES:
        raw = (root / name).read_bytes()
        files[name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return {
        "schema": schema,
        "family": "gmm",
        "historical_git_baseline": HISTORICAL_GIT_BASELINE,
        "implementation_parent_revision": IMPLEMENTATION_PARENT_REVISION,
        "execution_authorization": {
            "mode": "external_expected_source_commit_required",
            "final_execution_authorized_revision": None,
        },
        "seed_policy": {
            "experimental_seeds": list(GOVERNED_EXPERIMENTAL_SEEDS),
        },
        "files": files,
        "status": "IMPLEMENTATION_PREPARED_EXECUTION_NOT_AUTHORIZED",
    }


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "source-freeze.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_accepted_explicit_source_commit_is_bound(tmp_path):
    root, commit = _repo(tmp_path)
    freeze = _write_manifest(tmp_path, _manifest(root))
    evidence = verify_gmm_source_binding(
        repository_root=root,
        source_freeze_path=freeze,
        expected_source_commit=commit,
    )
    assert evidence["expected_source_commit"] == commit
    assert evidence["historical_git_baseline"] == HISTORICAL_GIT_BASELINE
    assert evidence["source_freeze_sha256"] == hashlib.sha256(
        freeze.read_bytes()
    ).hexdigest()
    assert set(evidence["source_files_sha256"]) == set(REQUIRED_SOURCE_FILES)


def test_historical_or_wrong_commit_is_rejected(tmp_path):
    root, _ = _repo(tmp_path)
    freeze = _write_manifest(tmp_path, _manifest(root))
    with pytest.raises(ValueError, match="historical source baseline"):
        verify_gmm_source_binding(
            repository_root=root,
            source_freeze_path=freeze,
            expected_source_commit=HISTORICAL_GIT_BASELINE,
        )
    with pytest.raises(ValueError, match="does not match HEAD"):
        verify_gmm_source_binding(
            repository_root=root,
            source_freeze_path=freeze,
            expected_source_commit="0" * 40,
        )


def test_frozen_source_hash_mismatch_is_rejected(tmp_path):
    root, commit = _repo(tmp_path)
    manifest = _manifest(root)
    name = REQUIRED_SOURCE_FILES[0]
    manifest["files"][name]["sha256"] = "0" * 64
    freeze = _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_gmm_source_binding(
            repository_root=root,
            source_freeze_path=freeze,
            expected_source_commit=commit,
        )


def test_dirty_source_is_rejected(tmp_path):
    root, commit = _repo(tmp_path)
    freeze = _write_manifest(tmp_path, _manifest(root))
    (root / "src/untracked.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worktree is not clean"):
        verify_gmm_source_binding(
            repository_root=root,
            source_freeze_path=freeze,
            expected_source_commit=commit,
        )


def test_old_source_freeze_schema_cannot_fall_back(tmp_path):
    root, commit = _repo(tmp_path)
    freeze = _write_manifest(
        tmp_path,
        _manifest(root, schema="trustforge.paper01.gmm.source-freeze.v3"),
    )
    with pytest.raises(ValueError, match="source-freeze schema"):
        verify_gmm_source_binding(
            repository_root=root,
            source_freeze_path=freeze,
            expected_source_commit=commit,
        )


@pytest.mark.parametrize("seed", [41, 49, -1, True, 42.0])
def test_unsupported_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="42-48"):
        require_governed_experimental_seed(seed)


@pytest.mark.parametrize("seed", list(GOVERNED_EXPERIMENTAL_SEEDS))
def test_governed_seeds_are_accepted(seed):
    assert require_governed_experimental_seed(seed) == seed


@pytest.mark.parametrize(
    "name,digest",
    (
        (
            "gmm_production_source_freeze.json",
            "077925a5196752095a50b78eccdcd6575c447a0cb8b8a8c9e6bca527b5b8e4e4",
        ),
        (
            "gmm_production_source_freeze_v2.json",
            "1cc982e7ec8f95a173b144e847832072daca043be030bc1f9f5e000b418b4a5e",
        ),
        (
            "gmm_production_source_freeze_v3.json",
            "69bac9dd209abc502f8cce3f617840f8d1ea8096b9813ba2ec3ea57a7b68920b",
        ),
    ),
)
def test_historical_source_freezes_remain_intact(name, digest):
    path = (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026"
        / name
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_current_v4_freeze_matches_working_tree():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root
            / "studies/paper01_benchmark/revision_2026/"
            "gmm_production_source_freeze_v4.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema"] == SOURCE_FREEZE_SCHEMA
    assert manifest["family"] == "gmm"
    assert manifest["historical_git_baseline"] == HISTORICAL_GIT_BASELINE
    assert manifest["implementation_parent_revision"] == (
        IMPLEMENTATION_PARENT_REVISION
    )
    assert manifest["execution_authorization"] == {
        "mode": "external_expected_source_commit_required",
        "final_execution_authorized_revision": None,
    }
    assert manifest["seed_policy"] == {
        "experimental_seeds": list(GOVERNED_EXPERIMENTAL_SEEDS),
    }
    assert set(manifest["files"]) == set(REQUIRED_SOURCE_FILES)
    for name in REQUIRED_SOURCE_FILES:
        raw = (root / name).read_bytes()
        assert manifest["files"][name] == {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
