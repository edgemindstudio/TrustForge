"""Isolated TensorFlow checkpoint integrity test on a Slurm node.

Synthetic optimizer state only: no GAN training or production artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_persistence import (
    persist_terminal_gan_checkpoint,
    load_terminal_gan_checkpoint,
)
from trustforge.provenance import sha256_file


def snapshot(variables):
    return [np.asarray(v.numpy()).copy() for v in variables]


def assert_identical(expected, variables, name):
    actual = snapshot(variables)
    assert len(expected) == len(actual), f"{name}: count mismatch"
    for i, (a, b) in enumerate(zip(expected, actual)):
        if not np.array_equal(a, b):
            variable = variables[i]
            print(
                "STATE_MISMATCH "
                f"component={name} "
                f"index={i} "
                f"variable_name={variable.name} "
                f"variable_path={getattr(variable, 'path', 'N/A')} "
                f"dtype={variable.dtype} "
                f"shape={variable.shape} "
                f"expected={a.flatten()[:8].tolist()} "
                f"actual={b.flatten()[:8].tolist()}",
                flush=True,
            )
        np.testing.assert_array_equal(
            a, b, err_msg=f"{name}: variable {i}"
        )


def expect_rejection(action, name, exceptions):
    try:
        action()
    except exceptions as exc:
        print(
            f"{name}=PASS REJECTION={type(exc).__name__}",
            flush=True,
        )
    else:
        raise AssertionError(f"{name}: invalid operation accepted")


from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)
from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)
from trustforge.provenance import canonical_json_bytes


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("checkpoint test requires Slurm")

    tf.keras.utils.set_random_seed(2026)

    models = build_revision_gan(GANModelConfig(num_classes=2))
    g = models["generator"]
    d = models["discriminator"]
    go = models["generator_optimizer"]
    do = models["discriminator_optimizer"]

    go.build(g.trainable_variables)
    do.build(d.trainable_variables)

    # Distinct nonzero values test the optimizer-slot restoration.
    for optimizer, base in ((go, 0.125), (do, 0.375)):
        for i, variable in enumerate(optimizer.variables):
            if i == 0:
                continue
            variable.assign(
                tf.ones_like(variable) * (base + i * 0.001)
            )

    # Synthetic accounting fixture; no training iteration occurs.
    # Simulate the governed minimum terminal budget.
    go.iterations.assign(500)
    do.iterations.assign(1000)

    original = {
        "generator": snapshot(g.variables),
        "discriminator": snapshot(d.variables),
        "generator_optimizer": snapshot(go.variables),
        "discriminator_optimizer": snapshot(do.variables),
    }

    labels = ("benign", "malware")

    plan = plan_gan_run(
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=labels,
        train_size=5,
    )

    draws_per_class = plan.nominal_real_exposures // 2

    report = {
        "batches": plan.planned_generator_iterations,
        "total_real_draws": plan.nominal_real_exposures,
        "total_unique_positions": 5,
        "per_class": [
            {
                "class_id": class_id,
                "train_size": train_size,
                "draws": draws_per_class,
                "unique_positions": train_size,
                "repeated_draws": draws_per_class - train_size,
                "nominal_passes": draws_per_class / train_size,
                "unique_coverage": 1.0,
            }
            for class_id, train_size in enumerate((2, 3))
        ],
    }

    exposure_evidence, evidence_digest = build_gan_exposure_evidence(
        plan=plan,
        exposure_report=report,
    )

    with tempfile.TemporaryDirectory(
        prefix="paper01_gan_ckpt_test_"
    ) as temporary:
        root = Path(temporary)

        metadata = persist_terminal_gan_checkpoint(
            root,
            dataset_id="malex",
            experimental_seed=42,
            class_mapping=labels,
            planned_generator_iterations=500,
            exposure_evidence=exposure_evidence,
            generator=g,
            discriminator=d,
            generator_optimizer=go,
            discriminator_optimizer=do,
        )

        assert (root / "model_metadata.json").is_file()
        assert len(metadata["state_files"]) >= 2
        assert metadata["schema_version"] == "2.0"
        assert metadata["exposure_evidence_sha256"] == evidence_digest
        assert (
            (root / "gan_exposure_evidence.json").read_bytes()
            == canonical_json_bytes(exposure_evidence)
        )
        print("CHECKPOINT_BOUND_EXPOSURE_EVIDENCE=PASS", flush=True)

        for filename, digest in metadata["state_files"].items():
            assert sha256_file(root / filename) == digest

        print("CHECKPOINT_PUBLICATION_AND_HASHES=PASS", flush=True)

        def restore(**overrides):
            arguments = {
                "requested_dataset_id": "malex",
                "requested_experimental_seed": 42,
                "requested_class_mapping": labels,
            }
            arguments.update(overrides)
            return load_terminal_gan_checkpoint(root, **arguments)

        restored = restore()
        assert restored["exposure_evidence"] == exposure_evidence

        for name, variables in (
            ("generator", restored["generator"].variables),
            ("discriminator", restored["discriminator"].variables),
            (
                "generator_optimizer",
                restored["generator_optimizer"].variables,
            ),
            (
                "discriminator_optimizer",
                restored["discriminator_optimizer"].variables,
            ),
        ):
            assert_identical(original[name], variables, name)

        print("EXACT_MODEL_AND_OPTIMIZER_RESTORATION=PASS", flush=True)

        expect_rejection(
            lambda: restore(requested_dataset_id="ustc_tfc2016"),
            "WRONG_DATASET_REJECTED",
            (ValueError, TypeError),
        )

        expect_rejection(
            lambda: restore(requested_experimental_seed=43),
            "WRONG_SEED_REJECTED",
            (ValueError, TypeError),
        )

        expect_rejection(
            lambda: persist_terminal_gan_checkpoint(
                root,
                dataset_id="malex",
                experimental_seed=42,
                class_mapping=labels,
                planned_generator_iterations=500,
                exposure_evidence=exposure_evidence,
                generator=g,
                discriminator=d,
                generator_optimizer=go,
                discriminator_optimizer=do,
            ),
            "OVERWRITE_REJECTED",
            (FileExistsError, OSError),
        )

        victim = root / sorted(metadata["state_files"])[0]
        with victim.open("ab") as handle:
            handle.write(b"deliberate-test-corruption")

        expect_rejection(
            restore,
            "CORRUPTED_CHECKPOINT_REJECTED",
            (ValueError, RuntimeError, OSError),
        )

    print("TEMPORARY_CHECKPOINT_ARTIFACTS_REMOVED=YES", flush=True)
    print("GAN_CHECKPOINT_ROUNDTRIP_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
