"""Production coordinator for VAE checkpoint preparation and publication.

Connects an in-memory governed VAE training result to the existing
best/last checkpoint bridge and the existing VAE generation/commit
publication lifecycle.

This module does not train, select, restore, synthesize or authorize
execution. It preserves the existing candidate-publication limitations,
including unverified GPFS crash durability.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

from trustforge.provenance import canonical_json_bytes

from .experimental_seed_policy import require_governed_experimental_seed
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from .variational_autoencoder_checkpoint_commit_publisher import (
    publish_candidate_vae_commit,
)
from .variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)
from .variational_autoencoder_checkpoint_committed_generation import (
    verify_committed_vae_generation,
)
from .variational_autoencoder_checkpoint_compatibility import (
    CANDIDATE_ARCHITECTURE_VERSION,
)
from .variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
    MODEL_FAMILY,
    PAPER_REVISION,
    SCHEMA_TYPE,
    SCHEMA_VERSION,
)
from .variational_autoencoder_checkpoint_publication_input import (
    validate_vae_publication_input,
)
from .variational_autoencoder_checkpoint_publication_policy import (
    COMMITTED_SYNCED,
)
from .variational_autoencoder_checkpoint_unpublished_transaction import (
    prepare_unpublished_vae_generation,
)
from .variational_autoencoder_model import (
    DECODER_FEATURE_CHANNELS,
    ENCODER_DENSE_UNITS,
    IMAGE_SHAPE,
    LATENT_DIM,
)
from .variational_autoencoder_observation import (
    KL_WEIGHT,
    OBSERVATION_VARIANCE,
)
from .variational_autoencoder_training_checkpoint_bridge import (
    prepare_vae_training_checkpoints,
)


VAE_OBSERVATION_VERSION = "gaussian_fixed_variance_1p0_v1"

SELECTION_POLICY_ID = "min_validation_total_loss_earliest_epoch_v1"
VALIDATION_SEED_POLICY_ID = "fixed_per_batch_across_epochs_v1"

SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ROLES = ("best", "last")


def _require_positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_source_commit(value):
    if (
        not isinstance(value, str)
        or SOURCE_COMMIT_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "source_commit must be a full lowercase SHA-1"
        )
    return value


def _require_class_mapping(value, *, class_count):
    if not isinstance(value, (tuple, list)):
        raise TypeError("class_mapping must be a sequence")

    mapping = list(value)

    if len(mapping) != class_count:
        raise ValueError("class_mapping length mismatch")

    if any(
        not isinstance(label, str)
        or not label
        or label != label.strip()
        for label in mapping
    ):
        raise ValueError(
            "class_mapping labels must be nonempty strings"
        )

    if len(set(mapping)) != len(mapping):
        raise ValueError("class_mapping labels must be unique")

    return mapping


def _checked_history(trained):
    if not isinstance(trained, dict):
        raise TypeError("training result must be a dictionary")

    if trained.get("checkpoint_status") != "NOT_PREPARED":
        raise ValueError("training checkpoint status mismatch")

    if trained.get("publication_status") != "NOT_PUBLISHED":
        raise ValueError("training publication status mismatch")

    history = trained.get("history")

    if not isinstance(history, list) or len(history) != 2:
        raise ValueError("expected exactly two completed epochs")

    if trained.get("epochs_completed") != 2:
        raise ValueError("training duration mismatch")

    previous_updates = 0

    for index, record in enumerate(history):
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("invalid training epoch sequence")

        updates = record.get("completed_updates")

        if type(updates) is not int or updates <= previous_updates:
            raise ValueError("invalid completed update sequence")

        previous_updates = updates

        validation = record.get("validation")

        if not isinstance(validation, dict):
            raise ValueError("missing validation record")

        loss = validation.get("total_loss")

        if (
            type(loss) not in (int, float)
            or not math.isfinite(float(loss))
        ):
            raise ValueError("invalid validation loss")

    if trained.get("completed_updates") != previous_updates:
        raise ValueError("final update count mismatch")

    if not isinstance(trained.get("best_weights"), dict):
        raise ValueError("missing best checkpoint weights")

    if not isinstance(trained.get("last_weights"), dict):
        raise ValueError("missing last checkpoint weights")

    return history


def _checked_best_epoch(trained, history):
    selection = trained.get("epoch_selection")

    if not isinstance(selection, dict):
        raise TypeError("missing epoch selection")

    best_epoch = selection.get("best_epoch")

    if type(best_epoch) is not int or best_epoch not in range(2):
        raise ValueError("invalid selected best epoch")

    expected_best = min(
        range(2),
        key=lambda index: history[index]["validation"]["total_loss"],
    )

    if best_epoch != expected_best:
        raise ValueError("selected best epoch mismatch")

    best_loss = float(
        history[best_epoch]["validation"]["total_loss"]
    )

    if selection.get("best_validation_total_loss") != best_loss:
        raise ValueError("selected best validation loss mismatch")

    return best_epoch


def _run_identity(source_commit, attempt):
    return f"commit{source_commit[:7]}_attempt{attempt:02d}"


def _manifest_evidence(
    *,
    governed_result,
    class_mapping,
    effective_configuration,
    source_commit,
    attempt,
    run_identity,
    role,
    epoch,
    history,
):
    record = history[epoch]

    configuration = dict(effective_configuration)
    configuration["source_commit"] = source_commit
    configuration["attempt"] = attempt
    configuration["run_identity"] = run_identity

    return {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE,
        "paper_revision": PAPER_REVISION,
        "model_family": MODEL_FAMILY,
        "dataset_id": governed_result["dataset_id"],
        "experimental_seed": governed_result["experimental_seed"],
        "class_mapping": list(class_mapping),
        "class_count": governed_result["class_count"],
        "image_shape": list(IMAGE_SHAPE),
        "latent_dim": LATENT_DIM,
        "encoder_dense_units": ENCODER_DENSE_UNITS,
        "decoder_feature_channels": DECODER_FEATURE_CHANNELS,
        "architecture_version": CANDIDATE_ARCHITECTURE_VERSION,
        "observation_version": VAE_OBSERVATION_VERSION,
        "observation_variance": OBSERVATION_VARIANCE,
        "kl_weight": KL_WEIGHT,
        "kl_annealing": False,
        "completed_epochs": epoch + 1,
        "completed_updates": record["completed_updates"],
        "checkpoint_role": role,
        "selection_policy_id": SELECTION_POLICY_ID,
        "validation_criterion": {
            "metric": "validation_total_loss",
            "direction": "minimize",
            "value": float(record["validation"]["total_loss"]),
        },
        "validation_seed_policy_id": VALIDATION_SEED_POLICY_ID,
        "effective_configuration": configuration,
    }


def _derive_generation_id(
    *,
    prepared,
    dataset_id,
    experimental_seed,
    attempt,
    source_commit,
    role,
):
    manifest_bytes = prepared["artifacts"][MANIFEST_FILENAME]
    manifest_sha256 = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    material = canonical_json_bytes({
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "attempt": attempt,
        "source_commit": source_commit,
        "checkpoint_role": role,
        "manifest_sha256": manifest_sha256,
    })

    generation_id = hashlib.sha256(
        material
    ).hexdigest()[:32]

    return validate_vae_generation_id(generation_id)


def _validate_publication_root(
    publication_root,
    *,
    repository_root,
):
    requested = Path(publication_root).expanduser()

    if not requested.name or requested.name in (".", ".."):
        raise ValueError(
            "publication root must name a new directory"
        )

    if requested.is_symlink():
        raise ValueError(
            "publication root must not be a symlink"
        )

    repository = Path(repository_root).expanduser().resolve()
    resolved = requested.parent.resolve() / requested.name

    if resolved == repository or repository in resolved.parents:
        raise ValueError(
            "publication root must be outside the source checkout"
        )

    if requested.exists():
        raise FileExistsError(
            "publication root already exists"
        )

    return requested


def _create_publication_root(requested):
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(
            "publication root already exists"
        )

    requested.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(requested, 0o700)

    generations = requested / "generations"
    commits = requested / "commits"

    generations.mkdir(mode=0o700)
    commits.mkdir(mode=0o700)
    os.chmod(generations, 0o700)
    os.chmod(commits, 0o700)

    root = requested.resolve()

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            "publication root is not a real directory"
        )

    return root


def _open_directory(path):
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    return os.open(path, flags)


def publish_governed_vae_checkpoints(
    *,
    governed_result,
    class_mapping,
    effective_configuration,
    publication_root,
    source_commit,
    attempt,
    repository_root,
):
    """Prepare and publish best/last checkpoints for one VAE result.

    Returns only after both generations have committed verification.
    Incomplete or indeterminate publication raises and leaves artifacts
    for explicit recovery. No production authorization is implied.
    """
    if not isinstance(governed_result, dict):
        raise TypeError("governed result must be a dictionary")

    if governed_result.get("checkpoint_status") != "NOT_PREPARED":
        raise ValueError("unexpected training checkpoint status")

    if governed_result.get("publication_status") != "NOT_PUBLISHED":
        raise ValueError("training result already reports publication")

    dataset_id = governed_result.get("dataset_id")

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    experimental_seed = require_governed_experimental_seed(
        governed_result.get("experimental_seed")
    )

    class_count = governed_result.get("class_count")

    if (
        type(class_count) is not int
        or class_count != DATASET_CLASS_COUNTS[dataset_id]
    ):
        raise ValueError("governed class count mismatch")

    mapping = _require_class_mapping(
        class_mapping,
        class_count=class_count,
    )

    source_commit = _require_source_commit(source_commit)
    attempt = _require_positive_int(attempt, "attempt")

    if not isinstance(effective_configuration, dict):
        raise TypeError(
            "effective_configuration must be a dictionary"
        )

    for field in ("batch_size", "learning_rate", "epochs"):
        if field not in effective_configuration:
            raise ValueError(
                "effective_configuration is missing " + field
            )

    if not isinstance(
        governed_result.get("dataset_provenance"),
        dict,
    ):
        raise ValueError("missing dataset provenance")

    trained = governed_result.get("training_result")
    history = _checked_history(trained)
    best_epoch = _checked_best_epoch(trained, history)

    run_identity = _run_identity(source_commit, attempt)

    requested_root = _validate_publication_root(
        publication_root,
        repository_root=repository_root,
    )

    evidence = {
        role: _manifest_evidence(
            governed_result=governed_result,
            class_mapping=mapping,
            effective_configuration=effective_configuration,
            source_commit=source_commit,
            attempt=attempt,
            run_identity=run_identity,
            role=role,
            epoch=(
                best_epoch
                if role == "best"
                else len(history) - 1
            ),
            history=history,
        )
        for role in ROLES
    }

    prepared = prepare_vae_training_checkpoints(
        governed_result=governed_result,
        class_mapping=mapping,
        manifest_evidence_by_role=evidence,
    )

    if set(prepared) != set(ROLES):
        raise RuntimeError("unexpected prepared checkpoint roles")

    for role in ROLES:
        validate_vae_publication_input(prepared[role])

        metadata = prepared[role]["metadata"]

        if metadata["observation_version"] != VAE_OBSERVATION_VERSION:
            raise RuntimeError(
                "approved observation version was not preserved"
            )

        if metadata["checkpoint_role"] != role:
            raise RuntimeError("prepared checkpoint role mismatch")

        if metadata["dataset_id"] != dataset_id:
            raise RuntimeError("prepared dataset identity mismatch")

        if metadata["experimental_seed"] != experimental_seed:
            raise RuntimeError("prepared seed identity mismatch")

        if metadata["class_mapping"] != mapping:
            raise RuntimeError("prepared class mapping mismatch")

        configuration = metadata["effective_configuration"]

        if configuration.get("source_commit") != source_commit:
            raise RuntimeError("prepared source identity mismatch")

        if configuration.get("attempt") != attempt:
            raise RuntimeError("prepared attempt identity mismatch")

        if configuration.get("run_identity") != run_identity:
            raise RuntimeError("prepared run identity mismatch")

    generation_ids = {
        role: _derive_generation_id(
            prepared=prepared[role],
            dataset_id=dataset_id,
            experimental_seed=experimental_seed,
            attempt=attempt,
            source_commit=source_commit,
            role=role,
        )
        for role in ROLES
    }

    if generation_ids["best"] == generation_ids["last"]:
        raise RuntimeError("best/last generation identity collision")

    root = _create_publication_root(requested_root)

    generations_fd = _open_directory(root / "generations")
    commits_fd = _open_directory(root / "commits")

    records = {}

    try:
        for role in ROLES:
            generation_id = generation_ids[role]
            role_prepared = prepared[role]
            generation_fd = None

            try:
                unpublished = prepare_unpublished_vae_generation(
                    generations_fd,
                    generation_id,
                    prepared=role_prepared,
                    requested_dataset_id=dataset_id,
                    requested_experimental_seed=experimental_seed,
                    requested_class_mapping=mapping,
                )

                generation_fd = unpublished["generation_fd"]

                published = publish_candidate_vae_commit(
                    generations_fd,
                    commits_fd,
                    generation_fd,
                    generation_id,
                    prepared=role_prepared,
                    requested_dataset_id=dataset_id,
                    requested_experimental_seed=experimental_seed,
                    requested_class_mapping=mapping,
                )

                if published.get("outcome") != COMMITTED_SYNCED:
                    raise RuntimeError(
                        "VAE checkpoint publication did not commit"
                    )

                verified = verify_committed_vae_generation(
                    root,
                    generation_id=generation_id,
                    requested_dataset_id=dataset_id,
                    requested_experimental_seed=experimental_seed,
                    requested_class_mapping=mapping,
                )

                if (
                    verified["commit_record"]
                    != published["commit_record"]
                ):
                    raise RuntimeError(
                        "committed VAE record mismatch"
                    )

                if (
                    verified["metadata"]
                    != role_prepared["metadata"]
                ):
                    raise RuntimeError(
                        "committed VAE metadata mismatch"
                    )

                records[role] = {
                    "generation_id": generation_id,
                    "generation_directory": str(
                        root / "generations" / generation_id
                    ),
                    "commit_filename":
                        published["commit_filename"],
                    "commit_record":
                        published["commit_record"],
                    "manifest_sha256":
                        published["commit_record"][
                            "manifest_sha256"
                        ],
                    "metadata": role_prepared["metadata"],
                    "source_commit": source_commit,
                    "attempt": attempt,
                    "run_identity": run_identity,
                }

            finally:
                if generation_fd is not None:
                    os.close(generation_fd)
                    generation_fd = None

    finally:
        os.close(generations_fd)
        os.close(commits_fd)

    best = records["best"]

    return {
        "publication_root": str(root),
        "source_commit": source_commit,
        "attempt": attempt,
        "run_identity": run_identity,
        "observation_version": VAE_OBSERVATION_VERSION,
        "publication_status": "PUBLISHED",
        "crash_durability_verified": False,
        "roles": records,
        "best_loader_directory": best["generation_directory"],
        "checkpoint_identity": {
            "dataset_id": dataset_id,
            "experimental_seed": experimental_seed,
            "class_mapping": list(mapping),
            "checkpoint_role": "best",
            "checkpoint_sha256": best["manifest_sha256"],
            "source_commit": source_commit,
        },
    }


__all__ = [
    "VAE_OBSERVATION_VERSION",
    "publish_governed_vae_checkpoints",
]
