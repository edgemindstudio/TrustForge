"""Bind GMM production execution to an explicit clean source revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


SCHEMA = "trustforge.paper01.gmm.execution-binding.v1"
SOURCE_FREEZE_SCHEMA = "trustforge.paper01.gmm.source-freeze.v4"
HISTORICAL_GIT_BASELINE = "42ac1fa794b6a2aeb7894881cc0cdb61ff77fa38"
IMPLEMENTATION_PARENT_REVISION = (
    "4bdcc20fb4a1886062239d416629f173b3522245"
)
GOVERNED_EXPERIMENTAL_SEEDS = tuple(range(42, 49))
REQUIRED_SOURCE_FILES = (
    "scripts/train_paper01_gmm_revision_production.py",
    "slurm/paper01_revision_2026/gmm_revision_production.slurm",
    "src/trustforge/models/paper01_revision_2026/gaussian_mixture_checkpoint.py",
    "src/trustforge/models/paper01_revision_2026/gaussian_mixture_source_binding.py",
    "src/trustforge/models/paper01_revision_2026/gaussian_mixture_training.py",
    "src/trustforge/models/paper01_revision_2026/gaussian_mixture_training_budget.py",
    "studies/paper01_benchmark/revision_2026/operational_integration_contract.md",
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_governed_experimental_seed(value):
    """Return one accepted GMM experimental seed or fail closed."""
    if type(value) is not int or value not in GOVERNED_EXPERIMENTAL_SEEDS:
        raise ValueError(
            "GMM experimental seed must be one of 42-48"
        )
    return value


def _git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        stderr=subprocess.PIPE,
        text=True,
    )


def _digest(value, *, name):
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _commit(value):
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise ValueError("expected source commit must be 40 lowercase hex")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_gmm_source_binding(
    *, repository_root, source_freeze_path, expected_source_commit,
):
    """Verify the clean committed source and exact frozen file contents.

    The final execution-authorized commit is supplied externally after the
    implementation is committed. It is deliberately not embedded in the
    source-freeze file because a commit cannot contain its own final hash.
    """
    root = Path(repository_root).expanduser().resolve()
    expected = _commit(expected_source_commit)
    if expected == HISTORICAL_GIT_BASELINE:
        raise ValueError(
            "historical source baseline is not current execution authority"
        )

    head = _git(root, "rev-parse", "HEAD").strip()
    if head != expected:
        raise ValueError("expected source commit does not match HEAD")
    # Reject any tracked modification globally, and any untracked source or
    # configuration inside the consumed project trees. Scheduler output in
    # the repository root is not source.
    tracked_status = _git(
        root, "status", "--porcelain", "--untracked-files=no",
    )
    untracked_source = _git(
        root, "ls-files", "--others", "--exclude-standard", "--",
        "scripts", "src", "slurm", "common", "studies",
    )
    if tracked_status.strip() or untracked_source.strip():
        raise ValueError("GMM production source worktree is not clean")

    freeze = Path(source_freeze_path).expanduser()
    if freeze.is_symlink() or not freeze.is_file():
        raise ValueError("GMM source-freeze file unavailable")
    freeze_raw = freeze.read_bytes()
    try:
        manifest = json.loads(freeze_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid GMM source-freeze JSON") from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SOURCE_FREEZE_SCHEMA
        or manifest.get("family") != "gmm"
    ):
        raise ValueError("current GMM source-freeze schema required")
    if manifest.get("historical_git_baseline") != HISTORICAL_GIT_BASELINE:
        raise ValueError("historical source baseline mismatch")
    if (
        manifest.get("implementation_parent_revision")
        != IMPLEMENTATION_PARENT_REVISION
    ):
        raise ValueError("implementation parent revision mismatch")
    if manifest.get("execution_authorization") != {
        "mode": "external_expected_source_commit_required",
        "final_execution_authorized_revision": None,
    }:
        raise ValueError("GMM execution authorization binding mismatch")
    if (
        manifest.get("seed_policy")
        != {"experimental_seeds": list(GOVERNED_EXPERIMENTAL_SEEDS)}
    ):
        raise ValueError("GMM governed seed policy mismatch")

    files = manifest.get("files")
    if type(files) is not dict or set(files) != set(REQUIRED_SOURCE_FILES):
        raise ValueError("GMM source-freeze file inventory mismatch")
    source_hashes = {}
    for name in REQUIRED_SOURCE_FILES:
        entry = files[name]
        if (
            type(entry) is not dict
            or set(entry) != {"bytes", "sha256"}
            or type(entry["bytes"]) is not int
            or entry["bytes"] <= 0
        ):
            raise ValueError("invalid GMM source-freeze file entry: " + name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen GMM source unavailable: " + name)
        raw_size = path.stat().st_size
        if raw_size != entry["bytes"]:
            raise ValueError("frozen GMM source size mismatch: " + name)
        digest = _sha256_file(path)
        if digest != _digest(
            entry["sha256"], name="frozen source sha256",
        ):
            raise ValueError("frozen GMM source hash mismatch: " + name)
        source_hashes[name] = digest

    return {
        "schema": SCHEMA,
        "source_freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "expected_source_commit": expected,
        "historical_git_baseline": HISTORICAL_GIT_BASELINE,
        "implementation_parent_revision": IMPLEMENTATION_PARENT_REVISION,
        "source_files_sha256": source_hashes,
        "governed_experimental_seeds": list(GOVERNED_EXPERIMENTAL_SEEDS),
    }


__all__ = [
    "GOVERNED_EXPERIMENTAL_SEEDS",
    "HISTORICAL_GIT_BASELINE",
    "IMPLEMENTATION_PARENT_REVISION",
    "REQUIRED_SOURCE_FILES",
    "SCHEMA",
    "SOURCE_FREEZE_SCHEMA",
    "require_governed_experimental_seed",
    "verify_gmm_source_binding",
]
