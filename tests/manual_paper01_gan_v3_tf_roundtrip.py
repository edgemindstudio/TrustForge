"""Manual TensorFlow v3 round-trip smoke test.

SLURM COMPUTE NODE ONLY.
Uses initialized model state and simulated test evidence.
No dataset reads, gradients, optimizer updates, or GAN training.
Not valid as a scientific training artifact.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_persistence import (
    persist_terminal_gan_checkpoint_v3,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_restoration import (
    load_terminal_gan_checkpoint_v3,
)


def snapshot(models):
    result = {}
    for name, obj in models.items():
        result[name] = [
            np.array(variable.numpy(), copy=True)
            for variable in obj.variables
        ]
    return result


def compare_snapshots(before, after):
    assert set(before) == set(after)
    for name in before:
        original = before[name]
        restored = after[name]
        assert len(original) == len(restored), name
        for index, (left, right) in enumerate(
            zip(original, restored)
        ):
            np.testing.assert_array_equal(
                left, right,
                err_msg=f"{name} variable {index} mismatch",
            )


def fixture_arguments(root):
    """Reuse the established fixture, not its mocked checkpoint."""
    fixture_path = (
        Path(__file__).resolve().parent
        / "test_paper01_gan_checkpoint_v3_real_integration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "gan_v3_smoke_fixture", fixture_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    patcher = pytest.MonkeyPatch()
    try:
        _, arguments = module.prepared.__wrapped__(
            root, patcher
        )
    finally:
        patcher.undo()

    return arguments


def main():
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError(
            "Real Slurm compute-node allocation required"
        )

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    print("SMOKE_MODE=REAL_TENSORFLOW_NO_TRAINING", flush=True)
    print(f"SLURM_JOB_ID={job_id}", flush=True)
    print(f"TENSORFLOW_VERSION={tf.__version__}", flush=True)
    print("EVIDENCE_SOURCE=SIMULATED_TEST_FIXTURE", flush=True)
    print("SCIENTIFIC_TRAINING_EVIDENCE=NO", flush=True)

    with tempfile.TemporaryDirectory(
        prefix="trustforge_gan_v3_tf_smoke_"
    ) as temporary:
        root = Path(temporary)
        arguments = fixture_arguments(root)

        models = build_revision_gan(
            GANModelConfig(num_classes=2)
        )

        generator = models["generator"]
        discriminator = models["discriminator"]
        g_optimizer = models["generator_optimizer"]
        d_optimizer = models["discriminator_optimizer"]

        g_optimizer.build(generator.trainable_variables)
        d_optimizer.build(discriminator.trainable_variables)

        iterations = arguments["planned_generator_iterations"]

        # Assign bookkeeping counters without optimizer updates.
        g_optimizer.iterations.assign(iterations)
        d_optimizer.iterations.assign(2 * iterations)

        arguments.update(models)

        before = snapshot(models)

        metadata = persist_terminal_gan_checkpoint_v3(
            root, **arguments
        )

        print("REAL_TENSORFLOW_PUBLICATION=PASSED", flush=True)

        restored = load_terminal_gan_checkpoint_v3(
            root,
            requested_dataset_id="malex",
            requested_experimental_seed=42,
            requested_class_mapping=["benign", "malware"],
        )

        after = snapshot({
            name: restored[name] for name in models
        })

        compare_snapshots(before, after)

        assert restored["metadata"] == metadata
        assert (
            int(restored["generator_optimizer"].iterations.numpy())
            == iterations
        )
        assert (
            int(restored["discriminator_optimizer"].iterations.numpy())
            == 2 * iterations
        )

        print("REAL_TENSORFLOW_RESTORATION=PASSED", flush=True)
        print("MODEL_AND_OPTIMIZER_VARIABLES=EXACT_MATCH", flush=True)
        print("GAN_TRAINING_STEPS=0", flush=True)
        print("PRODUCTION_DATASETS_ACCESSED=NO", flush=True)
        print("SMOKE_RESULT=PASS", flush=True)


if __name__ == "__main__":
    main()
