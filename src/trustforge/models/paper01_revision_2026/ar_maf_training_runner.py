"""Restricted AR/MAF training-to-checkpoint preparation.

General execution remains disabled. Explicit bounded integration permits
only CICMalDroid2020, seed 42, AR or MAF, under the complete selected recipe.
No checkpoint publication, synthesis or scheduler submission occurs here.
"""

import hashlib
from pathlib import Path
import subprocess

from .ar_maf_training_setup import prepare_training_setup
from .ar_maf_training_core import run_training_core
from .ar_maf_checkpoint_state import encode_model_snapshot
from .ar_maf_checkpoint_manifest import prepare_checkpoint_manifest
from .ar_maf_checkpoint_identity import canonical_json_bytes

TRAINING_EXECUTION_ENABLED = False
RECIPE_PATH = (
    "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
)


def capture_source_identity(repository_root):
    """Capture tracked Python source and selected recipe from a clean checkout.

    This records project source, not installed dependency identities or
    immutable filesystem contents. Runtime dependencies require separate
    environment evidence.
    """
    root = Path(repository_root)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
        )

    head = git("rev-parse", "HEAD").decode().strip()
    git("diff", "--quiet")
    git("diff", "--cached", "--quiet")
    names = git("ls-files", "-z").decode().split("\0")
    selected = sorted(
        name for name in names
        if name and (name.endswith(".py") or name == RECIPE_PATH)
    )
    if RECIPE_PATH not in selected:
        raise ValueError("selected recipe is not tracked")
    records = {}
    for name in selected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("source unavailable or symlinked: " + name)
        records[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return head, records


def run_training_and_prepare_checkpoint(
    *, repository_root, family, dataset_id, experimental_seed, adapter,
    bounded_integration=False,
):
    """Run the selected complete recipe and prepare unpublished bundle bytes."""
    if type(bounded_integration) is not bool:
        raise TypeError("bounded_integration must be boolean")
    if bounded_integration:
        if not (
            family in ("ar", "maf")
            and dataset_id == "cicmaldroid2020"
            and type(experimental_seed) is int
            and experimental_seed == 42
        ):
            raise RuntimeError("request outside bounded CIC seed42 scope")
    elif TRAINING_EXECUTION_ENABLED is not True:
        raise RuntimeError("AR/MAF training execution remains disabled")

    root = Path(repository_root)
    source_commit, source_files = capture_source_identity(root)
    recipe_raw = (root / RECIPE_PATH).read_bytes()
    if hashlib.sha256(recipe_raw).hexdigest() != source_files[RECIPE_PATH]:
        raise RuntimeError("recipe changed after source capture")

    setup = prepare_training_setup(
        family=family, dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        adapter=adapter, recipe_raw=recipe_raw,
    )
    optimizer = setup["optimizer"]
    if int(optimizer.iterations.numpy()) != 0:
        raise RuntimeError("training requires a fresh optimizer")

    trained = run_training_core(
        adapter=setup["adapter"],
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        model=setup["model"],
        training_step=setup["training_step"],
        validation_step=setup["validation_step"],
    )
    if (
        trained["completed_epochs"] != 2
        or trained["completed_training_steps"]
        != int(optimizer.iterations.numpy())
    ):
        raise RuntimeError("completed training and optimizer accounting disagree")

    evidence = setup["evidence"]
    for row in trained["history"]:
        if (
            row["training_examples"] != evidence["training_examples"]
            or row["validation_examples"] != evidence["validation_examples"]
        ):
            raise RuntimeError("training exposure differs from preflight counts")

    specification, best_raw = encode_model_snapshot(
        model=setup["model"], weights=trained["best_weights"],
    )
    last_specification, last_raw = encode_model_snapshot(
        model=setup["model"], weights=trained["last_weights"],
    )
    if last_specification != specification:
        raise RuntimeError("best/last tensor specifications differ")

    if capture_source_identity(root) != (source_commit, source_files):
        raise RuntimeError("source identity changed during execution")

    manifest = prepare_checkpoint_manifest(
        identity=evidence["identity"],
        recipe_raw=recipe_raw,
        history=trained["history"],
        specification=specification,
        best_raw=best_raw,
        last_raw=last_raw,
        source_commit=source_commit,
        source_files=source_files,
    )
    return {
        "manifest": manifest,
        "manifest_raw": canonical_json_bytes(manifest) + b"\n",
        "best_raw": best_raw,
        "last_raw": last_raw,
        "recipe_raw": recipe_raw,
        "publication_status": "NOT_PUBLISHED",
    }
