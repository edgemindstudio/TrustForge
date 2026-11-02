"""Governed Paper 1 revision-2026 GAN runner.

Feasibility execution remains disabled pending explicit authorization.
No production execution path is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os

import numpy as np

from trustforge.models.paper01_revision_2026.gan_run_governance import (
    GANRunAccounting,
    GANRunPlan,
    plan_gan_run,
)

from trustforge.models.paper01_revision_2026.gan_exposure_accounting import (
    GANExposureAccounting,
    training_labels_from_class_positions,
)

from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)

from trustforge.models.paper01_revision_2026.gan_class_mapping import (
    resolve_gan_class_mapping,
)
from trustforge.models.paper01_revision_2026.gan_adapter_identity import (
    require_gan_adapter_identity,
)
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    require_gan_dataset_provenance,
)


# This is an authorization boundary, not a command-line option.
GAN_FEASIBILITY_TRAINING_ENABLED = False
GAN_PRODUCTION_TRAINING_ENABLED = False


@dataclass(frozen=True)
class GANRunPreparation:
    plan: GANRunPlan
    checkpoint_directory: Path


def prepare_gan_run(
    *,
    dataset,
    dataset_id,
    experimental_seed,
    class_mapping,
    checkpoint_directory,
):
    """Plan a run without constructing models or reading image batches."""
    require_gan_adapter_identity(
        dataset_id=dataset_id,
        adapter=dataset,
    )

    authoritative_mapping = resolve_gan_class_mapping(
        dataset_id=dataset_id,
        adapter=dataset,
    )

    if tuple(class_mapping) != authoritative_mapping:
        raise ValueError(
            "caller class mapping disagrees with governed dataset"
        )

    train_size = dataset.split_size("train")
    plan = plan_gan_run(
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        class_mapping=class_mapping,
        train_size=train_size,
    )

    directory = Path(checkpoint_directory).expanduser()

    if not directory.is_dir():
        raise ValueError("checkpoint directory must already exist")

    if (directory / "model_metadata.json").exists():
        raise FileExistsError("terminal checkpoint already exists")

    if list(directory.glob("model_state.ckpt.*")):
        raise FileExistsError("checkpoint state already exists")

    return GANRunPreparation(
        plan=plan,
        checkpoint_directory=directory,
    )


def execute_gan_feasibility_run(
    preparation: GANRunPreparation,
    *,
    dataset,
):
    """Execute only after a separate governance authorization.

    The present revision deliberately rejects every execution attempt.
    """
    if not GAN_FEASIBILITY_TRAINING_ENABLED:
        raise PermissionError(
            "GAN feasibility training is not authorized"
        )

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GAN training requires a Slurm compute node")

    if not isinstance(preparation, GANRunPreparation):
        raise TypeError("invalid governed GAN preparation")

    plan = preparation.plan

    require_gan_adapter_identity(
        dataset_id=plan.dataset_id,
        adapter=dataset,
    )

    if int(dataset.split_size("train")) != plan.train_size:
        raise ValueError("training split size changed after planning")

    if resolve_gan_class_mapping(
        dataset_id=plan.dataset_id,
        adapter=dataset,
    ) != plan.class_mapping:
        raise ValueError("dataset class mapping changed after planning")

    # Mandatory compute-node provenance boundary.
    # This does not authorize training.
    require_gan_dataset_provenance(
        dataset_id=plan.dataset_id,
        adapter=dataset,
    )

    import tensorflow as tf

    from trustforge.models.paper01_revision_2026.gan_data_integration import (
        GANTrainingData,
    )
    from trustforge.models.paper01_revision_2026.gan_training_bridge import (
        GANTrainingBridge,
    )
    from trustforge.models.paper01_revision_2026.gan_model_factory import (
        GANModelConfig,
        build_revision_gan,
    )
    from trustforge.models.paper01_revision_2026.gan_training_step import (
        gan_training_step,
    )
    from trustforge.models.paper01_revision_2026.gan_checkpoint_persistence import (
        persist_terminal_gan_checkpoint,
    )

    tf.keras.utils.set_random_seed(plan.experimental_seed)

    training_data = GANTrainingData(
        dataset=dataset,
        class_count=len(plan.class_mapping),
        seed=plan.experimental_seed,
    )
    bridge = GANTrainingBridge(
        training_data,
        seed=plan.experimental_seed,
    )

    models = build_revision_gan(
        GANModelConfig(num_classes=len(plan.class_mapping))
    )

    accounting = GANRunAccounting(plan)

    train_labels = training_labels_from_class_positions(
        training_data.sampler.class_positions,
        train_size=plan.train_size,
        class_count=len(plan.class_mapping),
    )

    exposure = GANExposureAccounting(
        train_labels=train_labels,
        class_count=len(plan.class_mapping),
    )

    last_losses = None

    # Any exception terminates the run before authoritative publication.
    for iteration in range(plan.planned_generator_iterations):
        batch = bridge.draw(plan.batch_size)

        losses = gan_training_step(
            **models,
            **batch.training_step_inputs(),
        )

        if (
            set(losses) != {
                "discriminator_real_loss",
                "discriminator_fake_loss",
                "generator_loss",
            }
            or not all(
                math.isfinite(float(value))
                for value in losses.values()
            )
        ):
            raise RuntimeError("invalid GAN iteration loss record")

        expected_g = accounting.generator_iterations + 1
        expected_d = accounting.discriminator_updates + 2

        actual_g = int(
            models["generator_optimizer"].iterations.numpy()
        )
        actual_d = int(
            models["discriminator_optimizer"].iterations.numpy()
        )

        if (actual_g, actual_d) != (expected_g, expected_d):
            raise RuntimeError("GAN optimizer update accounting mismatch")

        # Record exposure only after all three optimizer updates succeed.
        exposure.record_batch(
            positions=batch.train_positions,
            one_hot_labels=batch.real_labels,
        )

        accounting.record_completed_iteration()
        last_losses = dict(losses)

        if (
            iteration == 0
            or (iteration + 1) % 100 == 0
            or iteration + 1 == plan.planned_generator_iterations
        ):
            print(
                f"GAN_PROGRESS dataset={plan.dataset_id} "
                f"seed={plan.experimental_seed} "
                f"generator_iterations={accounting.generator_iterations} "
                f"discriminator_updates={accounting.discriminator_updates} "
                f"losses={last_losses}",
                flush=True,
            )

    accounting.require_terminal()

    exposure_report = exposure.report(
        expected_batches=accounting.generator_iterations,
        expected_draws=plan.nominal_real_exposures,
    )

    exposure_evidence, exposure_evidence_sha256 = (
        build_gan_exposure_evidence(
            plan=plan,
            exposure_report=exposure_report,
        )
    )

    metadata = persist_terminal_gan_checkpoint(
        preparation.checkpoint_directory,
        dataset_id=plan.dataset_id,
        experimental_seed=plan.experimental_seed,
        class_mapping=plan.class_mapping,
        planned_generator_iterations=plan.planned_generator_iterations,
        exposure_evidence=exposure_evidence,
        **models,
    )

    return {
        "metadata": metadata,
        "class_exposures": [
            item["draws"]
            for item in exposure_report["per_class"]
        ],
        "exposure_report": exposure_report,
        "exposure_evidence": exposure_evidence,
        "exposure_evidence_sha256": exposure_evidence_sha256,
        "last_losses": last_losses,
        "completed_generator_iterations": accounting.generator_iterations,
        "completed_discriminator_updates": accounting.discriminator_updates,
    }
