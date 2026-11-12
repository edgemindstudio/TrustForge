"""Exercise the paired control path with bounded fixture views and model state."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from trustforge.evaluation import paper01_classifier_tensorflow as classifier
from trustforge.evaluation.paper01_cic_pool import CICPoolView
from trustforge.evaluation.paper01_classifier_control import fit_classifier_control, state_digest
from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.evaluation.paper01_pool_consumption import validate_pool_consumption


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
).read_bytes()


class Adapter:
    def split_size(self, split):
        return 5

    def canonical_batch(self, split, positions):
        return (
            np.zeros((len(positions), 40, 40, 3), dtype=np.float32),
            np.eye(5, dtype=np.float32)[positions],
        )


class FixturePool(CICPoolView):
    size = 5

    def __init__(self, *, later_value=0):
        self.pool_identity = {
            "dataset_id": "cicmaldroid2020", "experimental_seed": 42,
            "family": "maf", "manifest_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64, "recipe_sha256": "d" * 64,
            "schema_version": 1,
        }
        self.reads = 0
        self.later_value = later_value

    @property
    def identity(self):
        return deepcopy(self.pool_identity)

    def preflight(self):
        pass

    def batch(self, positions):
        value = self.later_value if self.reads else 0
        self.reads += 1
        return (
            positions.copy(),
            np.full((len(positions), 40, 40, 3), value, dtype=np.float32),
            np.eye(5, dtype=np.float32)[positions],
        )


def views(pool):
    adapter = Adapter()
    kwargs = {
        "adapter": adapter, "dataset_id": "cicmaldroid2020",
        "provenance_sha256": "a" * 64,
    }
    real = RealSplitView(split="train", **kwargs)
    validation = RealSplitView(split="validation", **kwargs)
    return real, CombinedTrainingView(real=real, synthetic=pool), validation


class Model:
    def __init__(self):
        self.weights = [np.zeros(1, dtype=np.float32)]

    def get_weights(self):
        return [value.copy() for value in self.weights]

    def set_weights(self, values):
        self.weights = [value.copy() for value in values]


def binding(*, fault=None, **kwargs):
    model = Model()
    updates = 0

    def step(images, labels):
        updates_before = 2 if fault == "double" else 1
        nonlocal updates
        updates += updates_before
        model.weights[0][0] += 1
        return float("nan") if fault == "loss" else 1.0

    def state():
        values = model.get_weights()
        if fault == "state" and updates:
            values[0][0] = np.nan
        return [*values, np.array(updates, dtype=np.int64)]

    return {
        "model": model,
        "initial_weights_sha256": state_digest(model.get_weights()),
        "controls": {
            "train_step": step,
            "validation_step": lambda images, labels: 1.0,
            "get_weights": model.get_weights,
            "set_weights": model.set_weights,
            "get_execution_state": state,
            "optimizer_updates": lambda: updates,
        },
    }


def fit_pair(pool):
    real, augmented, validation = views(pool)
    with (
        patch.object(classifier, "require_compute_node"),
        patch.object(classifier, "build_classifier_binding", side_effect=binding),
    ):
        return classifier.prepare_paired_classifiers(
            real_training=real, augmented_training=augmented,
            validation=validation, evaluator_seed=42, recipe_raw=RECIPE,
        )["arms"]


class PoolConsumptionTests(unittest.TestCase):
    def test_paired_path_records_prefix_at_selected_weights(self):
        arms = fit_pair(FixturePool())
        self.assertNotIn("pool_consumption", arms["real_only"])
        augmented = arms["real_plus_synthetic"]
        record = augmented["pool_consumption"]
        self.assertNotIn("pool_consumption", augmented["report"])
        self.assertEqual(record["selected_weights_sha256"], state_digest(
            augmented["selected_weights"],
        ))
        self.assertEqual(record["selected_epoch_index"], 0)
        self.assertEqual(record["selected_prefix"]["optimizer_updates"], 1)
        self.assertEqual(record["selected_prefix"]["synthetic_examples"], 5)
        self.assertEqual(record["final_prefix"]["optimizer_updates"], 6)
        self.assertEqual(record["final_prefix"]["synthetic_examples"], 30)

    def test_later_batches_change_final_digest_only(self):
        normal = fit_pair(FixturePool())["real_plus_synthetic"]["pool_consumption"]
        changed = fit_pair(FixturePool(later_value=1))[
            "real_plus_synthetic"
        ]["pool_consumption"]
        self.assertEqual(normal["selected_prefix"], changed["selected_prefix"])
        self.assertNotEqual(
            normal["final_prefix"]["stream_sha256"],
            changed["final_prefix"]["stream_sha256"],
        )

    def test_read_and_failed_update_publish_no_record(self):
        for fault in ("double", "loss", "state"):
            with self.subTest(fault=fault):
                pool = FixturePool()
                _, augmented, validation = views(pool)
                augmented.batch(np.array([5], dtype=np.int64))
                result = {}
                with self.assertRaises((ValueError, RuntimeError)):
                    fit_classifier_control(
                        training=augmented, validation=validation,
                        evaluator_seed=42, recipe_raw=RECIPE,
                        pool_consumption=result, **binding(fault=fault)["controls"],
                    )
                self.assertGreater(pool.reads, 1)
                self.assertEqual(result, {})

    def test_conflicting_pool_identity_rejected(self):
        pool = FixturePool()
        _, augmented, _ = views(pool)
        result = fit_pair(pool)["real_plus_synthetic"]
        identity = augmented.identity
        identity["synthetic"]["manifest_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            validate_pool_consumption(
                result["pool_consumption"], training_identity=identity,
                training_size=augmented.size, training_report=result["report"],
            )
