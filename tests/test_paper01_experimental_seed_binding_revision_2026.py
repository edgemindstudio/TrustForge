"""Focused S2-05 production-boundary seed-binding tests.

These tests do not train models, read real data, publish checkpoints or
submit Slurm jobs.  They exercise only request-boundary rejection and
identity propagation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from trustforge.models.paper01_revision_2026.diffusion_governed_training import (
    run_governed_diffusion_training,
)
from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_governed_runner import (
    GANRunPreparation,
    execute_gan_production_run,
)
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_governed_training import (
    run_governed_vae_training,
)
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    SEEDS as AR_MAF_SEEDS,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_source_binding import (
    GOVERNED_EXPERIMENTAL_SEEDS as GMM_SEEDS,
    require_governed_experimental_seed as require_gmm_seed,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    EXPERIMENTAL_SEEDS as SYNTHESIS_SEEDS,
    latent_entropy,
    load_synthesis_recipe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "cicmaldroid2020"
CLASS_MAPPING = tuple(str(index) for index in range(5))


def gan_preparation(seed, tmp_path):
    plan = plan_gan_run(
        dataset_id=DATASET,
        experimental_seed=seed,
        class_mapping=CLASS_MAPPING,
        train_size=100,
    )
    return GANRunPreparation(
        plan=plan,
        checkpoint_directory=tmp_path,
    )


def rbm_runner():
    path = REPO_ROOT / "scripts/train_paper01_rbm_revision_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "paper01_rbm_seed_binding_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load RBM runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", range(42, 49))
def test_shared_validator_accepts_exact_universe(seed):
    assert require_governed_experimental_seed(seed) == seed


@pytest.mark.parametrize("seed", [41, 49, 0, -1, 100])
def test_shared_validator_rejects_outside_universe(seed):
    with pytest.raises(ValueError, match="42-48"):
        require_governed_experimental_seed(seed)


@pytest.mark.parametrize("seed", [True, False, 42.0, "42", None, np.int64(42)])
def test_shared_validator_rejects_non_python_integer(seed):
    with pytest.raises(TypeError):
        require_governed_experimental_seed(seed)


def test_gan_internal_planner_remains_generic():
    plan = plan_gan_run(
        dataset_id=DATASET,
        experimental_seed=41,
        class_mapping=CLASS_MAPPING,
        train_size=100,
    )
    assert plan.experimental_seed == 41


def test_gan_production_boundary_rejects_unapproved_seed(tmp_path):
    with pytest.raises(ValueError, match="42-48"):
        execute_gan_production_run(
            gan_preparation(41, tmp_path),
            dataset=None,
        )


def test_gan_production_boundary_accepts_seed_and_stops_at_gate(tmp_path):
    preparation = gan_preparation(43, tmp_path)
    assert preparation.plan.experimental_seed == 43
    with pytest.raises(PermissionError, match="not authorized"):
        execute_gan_production_run(
            preparation,
            dataset=None,
        )


def test_vae_production_boundary_rejects_unapproved_seed():
    with pytest.raises(ValueError, match="42-48"):
        run_governed_vae_training(
            adapter=None,
            dataset_id=DATASET,
            experimental_seed=41,
        )


def test_vae_production_boundary_accepts_seed_and_stops_at_gate():
    with pytest.raises(PermissionError, match="not authorized"):
        run_governed_vae_training(
            adapter=None,
            dataset_id=DATASET,
            experimental_seed=43,
        )


def test_vae_generic_runner_seed_helper_unchanged():
    from trustforge.models.paper01_revision_2026.variational_autoencoder_training_runner import (
        _bounded_seed,
    )

    assert _bounded_seed(0) == 0
    assert _bounded_seed(1000) == 1000


def test_diffusion_production_boundary_rejects_unapproved_seed():
    with pytest.raises(ValueError, match="42-48"):
        run_governed_diffusion_training(
            adapter=None,
            dataset_id=DATASET,
            experimental_seed=49,
        )


def test_diffusion_production_boundary_accepts_seed_and_stops_at_gate():
    with pytest.raises(PermissionError, match="not authorized"):
        run_governed_diffusion_training(
            adapter=None,
            dataset_id=DATASET,
            experimental_seed=47,
        )


def test_vae_production_boundary_requires_explicit_seed():
    with pytest.raises(TypeError):
        run_governed_vae_training(
            adapter=None,
            dataset_id=DATASET,
        )


def test_diffusion_production_boundary_requires_explicit_seed():
    with pytest.raises(TypeError):
        run_governed_diffusion_training(
            adapter=None,
            dataset_id=DATASET,
        )


def test_rbm_parser_does_not_supply_an_implicit_seed(tmp_path):
    runner = rbm_runner()
    args = runner.build_parser().parse_args(
        [
            "--dataset-id",
            DATASET,
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
        ]
    )
    assert not hasattr(args, "experimental_seed")
    assert runner.DEFAULT_EXPERIMENTAL_SEED == 42


def test_rbm_main_rejects_missing_production_seed(tmp_path):
    runner = rbm_runner()
    with pytest.raises(ValueError, match="experimental-seed is required"):
        runner.main(
            [
                "--dataset-id",
                DATASET,
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
            ]
        )


def test_rbm_launcher_rejects_unapproved_seed_before_slurm(tmp_path):
    runner = rbm_runner()
    with pytest.raises(ValueError, match="42-48"):
        runner.main(
            [
                "--dataset-id",
                DATASET,
                "--experimental-seed",
                "41",
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
            ]
        )


def test_rbm_launcher_accepts_approved_seed_and_stops_at_slurm(tmp_path):
    runner = rbm_runner()
    with pytest.raises(RuntimeError, match="Slurm"):
        runner.main(
            [
                "--dataset-id",
                DATASET,
                "--experimental-seed",
                "43",
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
            ]
        )


def test_rbm_slurm_launcher_requires_explicit_seed():
    text = (
        REPO_ROOT
        / "slurm/paper01_revision_2026/rbm_revision_checkpoint.slurm"
    ).read_text()
    assert ': "${EXPERIMENTAL_SEED:?' in text
    assert "EXPERIMENTAL_SEED:-42" not in text


def test_ar_maf_and_gmm_seed_policies_remain_unchanged():
    assert AR_MAF_SEEDS == tuple(range(42, 49))
    assert GMM_SEEDS == tuple(range(42, 49))
    assert require_gmm_seed(42) == 42
    with pytest.raises(ValueError, match="42-48"):
        require_gmm_seed(41)


def test_s2_02_synthesis_seed_contract_remains_unchanged():
    assert SYNTHESIS_SEEDS == tuple(range(42, 49))
    recipe = load_synthesis_recipe(REPO_ROOT)
    assert recipe.data["experimental_seeds"] == list(range(42, 49))
    with pytest.raises(ValueError, match="42-48"):
        latent_entropy(
            dataset_id=DATASET,
            family="gan",
            experimental_seed=41,
            class_id=0,
            sample_index=0,
        )
    with pytest.raises(ValueError, match="42-48"):
        latent_entropy(
            dataset_id=DATASET,
            family="vae",
            experimental_seed=49,
            class_id=0,
            sample_index=0,
        )


def test_no_extra_seeds_or_datasets_introduced():
    assert require_governed_experimental_seed.__module__.endswith(
        "experimental_seed_policy"
    )
    assert tuple(range(42, 49)) == AR_MAF_SEEDS == GMM_SEEDS
    assert tuple(range(42, 49)) == SYNTHESIS_SEEDS
