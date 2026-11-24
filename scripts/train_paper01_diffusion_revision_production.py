#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 Diffusion production entry point.

One invocation validates one explicit dataset/experimental-seed/attempt
identity and delegates to the existing governed Diffusion training boundary.

Compute-node only. Requires a clean committed source revision and a Slurm
allocation. The real-data gate remains disabled. After successful governed
training this entry point prepares and durably publishes the accepted best/last
checkpoint bundle. Publication and execution require separate researcher
authorization; production synthesis and downstream evaluation are not part of
this entry point.
"""

from __future__ import annotations

# Talon's TensorFlow 2.20.0 / Keras 3.11.3 can fail tree initialization when
# optree is first imported later during model-builder binding. Preload optree
# before any TensorFlow/Keras-dependent production import.
import optree  # noqa: F401

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
ACCEPTED_MODEL_BUILDER = Path("diffusion") / "models.py"

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
class ExecutionContext:
    """Verified execution identity without dataset access."""

    actual_source_commit: str
    repository_root: Path


@dataclass(frozen=True)
class ModuleBinding:
    """Verified Revision-2026 and model-builder import identity."""

    module_paths: dict[str, str]
    model_builder_path: str


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
    parser.add_argument(
        "--checkpoint-root",
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
        diffusion_governed_training as governed,
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
        diffusion_ddpm_math,
        diffusion_governed_training,
        diffusion_training_runner,
        diffusion_training_step,
        experimental_seed_policy,
        gan_adapter_identity,
        gan_checkpoint_manifest,
        gan_dataset_provenance,
        gan_v3_provenance_adapter,
        variational_autoencoder_data_integration,
        variational_autoencoder_traversal,
    )

    return {
        "diffusion_ddpm_math": diffusion_ddpm_math,
        "diffusion_governed_training": diffusion_governed_training,
        "diffusion_training_runner": diffusion_training_runner,
        "diffusion_training_step": diffusion_training_step,
        "experimental_seed_policy": experimental_seed_policy,
        "gan_adapter_identity": gan_adapter_identity,
        "gan_checkpoint_manifest": gan_checkpoint_manifest,
        "gan_dataset_provenance": gan_dataset_provenance,
        "gan_v3_provenance_adapter": gan_v3_provenance_adapter,
        "variational_autoencoder_data_integration":
            variational_autoencoder_data_integration,
        "variational_autoencoder_traversal":
            variational_autoencoder_traversal,
    }


def _load_model_builder_module():
    from diffusion import models as model_builder

    return model_builder


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


def require_environment(args):
    """Require Slurm, a clean checkout and the expected source commit."""
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError(
            "Diffusion production training requires a Slurm allocation"
        )

    state = capture_git_state(args.repository_root)

    if state.dirty:
        raise RuntimeError(
            "Diffusion production training requires a clean "
            "working tree and index"
        )

    if state.commit != args.expected_source_commit:
        raise RuntimeError(
            "source revision does not match the expected commit"
        )

    return ExecutionContext(
        actual_source_commit=state.commit,
        repository_root=state.repo_root,
    )


def require_diffusion_production_authorization(governed):
    """Reuse the existing Diffusion real-data gate before adapter creation."""
    if governed.DIFFUSION_REAL_DATASET_TRAINING_ENABLED is not True:
        raise PermissionError(
            "Diffusion real-dataset training is not authorized"
        )


def require_module_binding(
    *,
    repository_root,
    modules=None,
    model_builder=None,
):
    """Require accepted Revision-2026 and Diffusion model-builder imports."""
    root = Path(repository_root).resolve()
    package = (root / REVISION_PACKAGE).resolve()
    accepted_builder = (root / ACCEPTED_MODEL_BUILDER).resolve()

    if modules is None:
        modules = _load_revision_modules()

    if model_builder is None:
        model_builder = _load_model_builder_module()

    if not modules:
        raise RuntimeError(
            "no Revision-2026 modules were bound"
        )

    if not accepted_builder.is_file():
        raise RuntimeError(
            f"accepted Diffusion model builder is missing: "
            f"{accepted_builder}"
        )

    paths = {}

    for name, module in modules.items():
        path = Path(
            getattr(module, "__file__", "")
        ).resolve()

        if not path.is_relative_to(package):
            raise RuntimeError(
                f"{name} is not bound to the Revision-2026 "
                f"module path: {path}"
            )

        if not path.is_file():
            raise RuntimeError(
                f"Revision-2026 module is missing: {path}"
            )

        paths[name] = str(path)

    builder_path = Path(
        getattr(model_builder, "__file__", "")
    ).resolve()

    if builder_path != accepted_builder:
        raise RuntimeError(
            "Diffusion model builder is not the accepted source: "
            f"{builder_path}"
        )

    if not callable(
        getattr(model_builder, "build_diffusion_model", None)
    ):
        raise RuntimeError(
            "Diffusion model builder lacks build_diffusion_model"
        )

    return ModuleBinding(
        module_paths=paths,
        model_builder_path=str(builder_path),
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


def _create_private_parents(path):
    """Create missing attempt-namespace parents with restrictive modes."""
    path = Path(path)
    if path.exists():
        return
    _create_private_parents(path.parent)
    path.mkdir(mode=0o700)


def publish_governed_checkpoint(
    *,
    result,
    request,
    environment,
    adapter,
    checkpoint_root,
):
    """Prepare and durably publish the accepted Diffusion best/last states."""
    from trustforge.models.paper01_revision_2026.ar_maf_dataset_preflight import (
        adapter_class_mapping,
    )
    from trustforge.models.paper01_revision_2026.diffusion_checkpoint_bundle import (
        publish_diffusion_checkpoint_bundle,
    )
    from trustforge.models.paper01_revision_2026.diffusion_checkpoint_state import (
        prepare_diffusion_checkpoint_states,
    )

    if (
        not isinstance(result, dict)
        or result.get("checkpoint_status") != "NOT_PREPARED"
        or result.get("publication_status") != "NOT_PUBLISHED"
    ):
        raise ValueError("training result is not eligible for publication")
    if (
        result.get("dataset_id") != request["dataset_id"]
        or result.get("experimental_seed")
        != request["experimental_seed"]
    ):
        raise ValueError("training result identity differs from the request")

    mapping = adapter_class_mapping(
        dataset_id=request["dataset_id"], adapter=adapter,
    )
    prepared = prepare_diffusion_checkpoint_states(
        governed_result=result, class_mapping=mapping,
    )
    root = (
        Path(checkpoint_root).expanduser()
        if checkpoint_root is not None
        else Path.home()
        / "trustforge_revision_2026_artifacts"
        / "paper01"
        / "diffusion"
    )
    destination = (
        root
        / request["dataset_id"]
        / f"seed{request['experimental_seed']}"
        / "checkpoints"
        / "production"
        / request["run_identity"]
    )
    _create_private_parents(destination.parent)
    published = publish_diffusion_checkpoint_bundle(
        prepared=prepared,
        destination=destination,
        source_commit=environment.actual_source_commit,
        dataset_provenance=result["dataset_provenance"],
    )
    return {
        "checkpoint_root": str(root),
        "checkpoint_directory": published["path"],
        "manifest_sha256": published["manifest_sha256"],
        "manifest_bytes": published["bytes"],
        "best_state_sha256": prepared["best"]["state_sha256"],
        "last_state_sha256": prepared["last"]["state_sha256"],
        "checkpoint_role_primary": "best",
    }


def build_summary(
    *,
    request,
    environment,
    binding,
    result,
    publication,
):
    """Return an identity summary using the governed result directly."""
    trained = result["training_result"]

    return {
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
        "revision_module_paths": binding.module_paths,
        "model_builder_path": binding.model_builder_path,
        "dataset_provenance": result["dataset_provenance"],
        "scientific_configuration":
            result["scientific_configuration"],
        "supplementary_execution_policy": {
            "objective": "epsilon_prediction_mse",
            "validation_metric": "validation_mse",
            "selection_direction": "minimize",
            "selection_tie_break": "earliest_epoch",
            "test_split_usage": "prohibited",
            "early_stopping": False,
            "retry_policy": "disabled_by_launcher",
            "resume_policy": "not_supported",
        },
        "training_result": {
            "completed_epochs": trained["completed_epochs"],
            "completed_updates": trained["completed_updates"],
            "best_epoch": trained["best_epoch"],
            "best_validation_mse":
                trained["best_validation_mse"],
        },
        "checkpoint_status": "PREPARED",
        "publication_status": "PUBLISHED",
        "checkpoint_publication": publication,
    }


def run(args):
    """Validate identity, enforce the gate, then delegate training."""
    request = validate_request(args)
    environment = require_environment(args)
    governed = _load_governed_module()

    # The gate must run before module binding, adapter construction or
    # any dataset/model access.
    require_diffusion_production_authorization(governed)

    binding = require_module_binding(
        repository_root=environment.repository_root,
    )
    adapter = build_adapter(args)

    result = governed.run_governed_diffusion_training(
        adapter=adapter,
        dataset_id=request["dataset_id"],
        experimental_seed=request["experimental_seed"],
    )

    publication = publish_governed_checkpoint(
        result=result,
        request=request,
        environment=environment,
        adapter=adapter,
        checkpoint_root=getattr(args, "checkpoint_root", None),
    )

    summary = build_summary(
        request=request,
        environment=environment,
        binding=binding,
        result=result,
        publication=publication,
    )

    print(
        "DIFFUSION_REVISION_PRODUCTION_RESULT "
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
