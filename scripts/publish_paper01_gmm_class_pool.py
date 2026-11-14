#!/usr/bin/env python3
"""Publish one validated GMM class pool from an accepted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _git(root, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments],
        text=True,
    )


def _require_source(root, expected):
    if _git(root, "rev-parse", "HEAD").strip() != expected:
        raise RuntimeError("expected synthesis source commit does not match HEAD")
    tracked = _git(root, "status", "--porcelain", "--untracked-files=no")
    untracked = _git(
        root, "ls-files", "--others", "--exclude-standard", "--",
        "scripts", "src", "slurm", "common", "studies",
    )
    if tracked.strip() or untracked.strip():
        raise RuntimeError("synthesis source worktree is not clean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--expected-synthesis-source-commit", required=True)
    parser.add_argument("--expected-training-source-commit", required=True)
    parser.add_argument(
        "--expected-training-source-freeze-sha256", required=True,
    )
    args = parser.parse_args()

    root = args.repository_root.expanduser().resolve()
    if root != Path.cwd().resolve():
        raise RuntimeError("execution checkout mismatch")
    checkpoint_dir = args.checkpoint_dir.expanduser()
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("accepted checkpoint directory unavailable")
    _require_source(root, args.expected_synthesis_source_commit)

    from trustforge.models.paper01_revision_2026.gaussian_mixture_class_pool import (
        sample_and_publish_gmm_class_pool,
    )

    report = sample_and_publish_gmm_class_pool(
        checkpoint_dir,
        args.destination,
        expected_training_source_commit=args.expected_training_source_commit,
        expected_training_source_freeze_sha256=(
            args.expected_training_source_freeze_sha256
        ),
        synthesis_source_commit=args.expected_synthesis_source_commit,
    )
    print("GMM_CLASS_POOL_PUBLICATION=PASS")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
