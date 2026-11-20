#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 VAE production entry point.

One invocation validates one explicit dataset/experimental-seed/attempt
identity and delegates to the existing governed VAE training boundary.

Compute-node only. Requires a clean committed source revision and a Slurm
allocation. The real-data gate remains disabled. This entry point does not
construct a dataset adapter before authorization, prepare checkpoints,
publish artifacts, synthesize images or evaluate classifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.provenance.git import capture_git_state


SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REVISION_PACKAGE = (
    Path("src") / "trustforge" / "models" / "paper01_revision_2026"
)

SOURCE_ARGUMENTS = {
    "ustc_tfc2016": ("ustc_source_dir",),
    "cicmaldroid2020": (
        "cic_source_csv",
        "cic_feature_columns",
    ),
    "malnet_image": (
        "malnet_image_root",
        "malnet_preparation_dir",
        "malnet_mapping_path",
    ),
    "malex": (
        "malex_label_archive",
        "malex_benign_root",
        "malex_malware_root",
    ),
}


@dataclass(frozen=True)
class ProductionEnvironment:
    """Verified execution identity without dataset access."""

    actual_source_commit: str
    repository_root: Path
    module_paths: dict[str, str]


def make_parser():
    """Build the explicit-identity production argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-id",
        required=True,
        choices=tuple(sorted(DATASET_CLASS_COUNTS)),
    )
    parser.add_argument(
        "--experimental-seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--attempt",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--expected-source-commit",
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        required=True,
        type=Path,
    )

    parser.add_argument("--ustc-source-dir", type=Path)
    parser.add_argument("--cic-source-csv", type=Path)
    parser.add_argument("--cic-feature-columns", type=Path)
    parser.add_argument("--malnet-image-root", type=Path)
    parser.add_argument("--malnet-preparation-dir", type=Path)
    parser.add_argument("--malnet-mapping-path", type=Path)
    parser.add_argument("--malex-label-archive", type=Path)
    parser.add_argument("--malex-benign-root", type=Path)
    parser.add_argument("--malex-malware-root", type=Path)

    return parser


def _load_governed_module():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_governed_training as governed,
    )

    return governed


def _load_adapter_types():
    from trustforge.data.paper01_revision_2026 import (
        CICMalDroidRevision2026,
        MaleXRevision2026,
        MalNetImageRevision2026,
        USTCRevision2026,
    )

    return {
        "ustc_tfc2016": USTCRevision2026,
        "cicmaldroid2020": CICMalDroidRevision2026,
        "malnet_image": MalNetImageRevision2026,
        "malex": MaleXRevision2026,
    }


def _load_revision_modules():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_data_integration,
        variational_autoencoder_governed_training,
        variational_autoencoder_model,
        variational_autoencoder_observation,
        variational_autoencoder_training_runner,
    )

    return {
        "variational_autoencoder_data_integration":
            variational_autoencoder_data_integration,
        "variational_autoencoder_governed_training":
            variational_autoencoder_governed_training,
        "variational_autoencoder_model":
            variational_autoencoder_model,
        "variational_autoencoder_observation":
            variational_autoencoder_observation,
        "variational_autoencoder_training_runner":
            variational_autoencoder_training_runner,
    }


def _load_vae_constants():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_governed_training as governed,
        variational_autoencoder_model as model,
        variational_autoencoder_observation as observation,
    )

    return {
        "epochs": governed.APPROVED_EPOCHS,
        "batch_size": governed.APPROVED_BATCH_SIZE,
        "learning_rate": governed.APPROVED_LEARNING_RATE,
        "image_shape": model.IMAGE_SHAPE,
        "latent_dim": model.LATENT_DIM,
        "observation_variance": observation.OBSERVATION_VARIANCE,
        "kl_weight": observation.KL_WEIGHT,
    }


def effective_configuration(*, constants=None):
    """Return the settings actually consumed by the governed VAE path."""
    if constants is None:
        constants = _load_vae_constants()

    return {
        "architecture": "conditional_convolutional_vae",
        "canonical_image_shape": list(constants["image_shape"]),
        "canonical_input_domain": [0.0, 1.0],
        "internal_model_domain": [-1.0, 1.0],
        "latent_dimension": constants["latent_dim"],
        "observation_variance": constants["observation_variance"],
        "kl_coefficient": constants["kl_weight"],
        "kl_annealing": False,
        "optimizer": "adam",
        "learning_rate": constants["learning_rate"],
        "batch_size": constants["batch_size"],
        "epochs": constants["epochs"],
        "validation":
            "full_governed_validation_after_each_epoch",
        "selection_metric": "validation_total_loss",
        "selection_direction": "minimize",
        "selection_tie_break": "earliest_epoch",
        "checkpoint_roles": ["best", "last"],
        "test_split_usage": "prohibited",
    }


def validate_request(args):
    """Validate explicit identity inputs without touching dataset bytes."""
    dataset_id = args.dataset_id

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset ID")

    seed = require_governed_experimental_seed(
        args.experimental_seed
    )

    attempt = args.attempt
    if type(attempt) is not int or attempt <= 0:
        raise ValueError("attempt must be a positive integer")

    commit = args.expected_source_commit
    if (
        not isinstance(commit, str)
        or SOURCE_COMMIT_PATTERN.fullmatch(commit) is None
    ):
        raise ValueError(
            "expected source commit must be a full lowercase SHA-1"
        )

    required_sources = SOURCE_ARGUMENTS[dataset_id]
    missing = [
        name
        for name in required_sources
        if getattr(args, name, None) is None
    ]
    if missing:
        raise ValueError(
            "missing required dataset source arguments: "
            + ", ".join(missing)
        )

    return {
        "dataset_id": dataset_id,
        "experimental_seed": seed,
        "attempt": attempt,
        "expected_source_commit": commit,
        "run_identity": (
            f"commit{commit[:7]}_attempt{attempt:02d}"
        ),
    }


def require_revision_module_binding(*, repository_root, modules=None):
    """Require imported VAE modules to resolve inside this checkout."""
    package = (Path(repository_root) / REVISION_PACKAGE).resolve()

    if modules is None:
        modules = _load_revision_modules()

    if not modules:
        raise RuntimeError(
            "no Revision-2026 modules were bound"
        )

    paths = {}

    for name, module in modules.items():
        path = Path(
            getattr(module, "__file__", "")
        ).resolve()

        try:
            path.relative_to(package)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} is not bound to the Revision-2026 "
                f"module path: {path}"
            ) from exc

        if not path.is_file():
            raise RuntimeError(
                f"Revision-2026 module is missing: {path}"
            )

        paths[name] = str(path)

    return paths


def require_environment(args):
    """Require Slurm, a clean checkout and the expected source commit."""
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError(
            "VAE production training requires a Slurm allocation"
        )

    state = capture_git_state(args.repository_root)

    if state.dirty:
        raise RuntimeError(
            "VAE production training requires a clean "
            "working tree and index"
        )

    if state.commit != args.expected_source_commit:
        raise RuntimeError(
            "source revision does not match the expected commit"
        )

    module_paths = require_revision_module_binding(
        repository_root=state.repo_root,
    )

    return ProductionEnvironment(
        actual_source_commit=state.commit,
        repository_root=state.repo_root,
        module_paths=module_paths,
    )


def require_sources(args, names):
    """Require explicit, non-symlinked dataset source paths."""
    values = [getattr(args, name) for name in names]

    if any(value is None for value in values):
        raise ValueError(
            "missing required dataset source arguments: "
            + ", ".join(names)
        )

    paths = []

    for value in values:
        path = Path(value).expanduser()

        if not path.exists() or path.is_symlink():
            raise ValueError(
                f"missing or symlinked dataset source: {path}"
            )

        paths.append(path.resolve())

    return paths


def build_adapter(args):
    """Construct the accepted Revision-2026 adapter for one dataset."""
    types = _load_adapter_types()
    dataset_id = args.dataset_id

    if dataset_id == "ustc_tfc2016":
        (source,) = require_sources(
            args,
            ("ustc_source_dir",),
        )
        return types["ustc_tfc2016"](source)

    if dataset_id == "cicmaldroid2020":
        source, columns = require_sources(
            args,
            ("cic_source_csv", "cic_feature_columns"),
        )
        return types["cicmaldroid2020"](
            source,
            columns,
            validate_hashes=True,
        )

    if dataset_id == "malnet_image":
        image_root, preparation, mapping = require_sources(
            args,
            (
                "malnet_image_root",
                "malnet_preparation_dir",
                "malnet_mapping_path",
            ),
        )
        return types["malnet_image"](
            image_root,
            preparation,
            mapping,
            validate_hashes=True,
            validate_production_contract=True,
        )

    if dataset_id == "malex":
        label_archive, benign_root, malware_root = (
            require_sources(
                args,
                (
                    "malex_label_archive",
                    "malex_benign_root",
                    "malex_malware_root",
                ),
            )
        )
        return types["malex"](
            label_archive,
            benign_root,
            malware_root,
            validate_hashes=True,
            validate_production_contract=True,
        )

    raise ValueError("unsupported governed dataset ID")


def run(args):
    """Validate identity, enforce the gate, then delegate training."""
    request = validate_request(args)
    environment = require_environment(args)
    governed = _load_governed_module()

    # The gate must run before adapter construction or dataset access.
    governed.require_vae_production_authorization()

    adapter = build_adapter(args)

    result = governed.run_governed_vae_training(
        adapter=adapter,
        dataset_id=request["dataset_id"],
        experimental_seed=request["experimental_seed"],
    )

    trained = result["training_result"]

    summary = {
        "dataset_id": request["dataset_id"],
        "experimental_seed": request["experimental_seed"],
        "attempt": request["attempt"],
        "run_identity": request["run_identity"],
        "expected_source_commit":
            request["expected_source_commit"],
        "actual_source_commit":
            environment.actual_source_commit,
        "repository_root": str(environment.repository_root),
        "source_status": "CLEAN",
        "revision_module_paths": environment.module_paths,
        "effective_configuration": effective_configuration(),
        "checkpoint_status": result["checkpoint_status"],
        "publication_status": result["publication_status"],
        "dataset_provenance": result["dataset_provenance"],
        "training_result": {
            "epochs_completed": trained["epochs_completed"],
            "completed_updates": trained["completed_updates"],
        },
    }

    print(
        "VAE_REVISION_PRODUCTION_RESULT "
        + json.dumps(summary, sort_keys=True, default=str),
        flush=True,
    )

    return summary


def main(argv=None):
    args = make_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
