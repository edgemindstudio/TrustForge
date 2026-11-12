"""Record bounded synthetic batch evidence from successful classifier updates."""

import hashlib
import json
import re

from .paper01_classifier_control import state_digest
from .paper01_combined_inputs import CombinedTrainingView


SCHEMA = "trustforge.paper01.augmented-pool-consumption.v1"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _binding(training_identity, evaluator_seed, training_size):
    real = training_identity["real"]
    pool = training_identity["synthetic"]
    if (
        training_identity["condition"] != "real_plus_synthetic"
        or training_identity["position_order"] != "real_then_synthetic"
        or pool["dataset_id"] != real["dataset_id"]
        or type(pool["experimental_seed"]) is not int
        or pool["experimental_seed"] != evaluator_seed
        or type(real["size"]) is not int
        or not 0 < real["size"] < training_size
    ):
        raise ValueError("conflicting synthetic pool identity")
    return {
        "schema": SCHEMA,
        "dataset_id": real["dataset_id"],
        "evaluator_seed": evaluator_seed,
        "arm": "real_plus_synthetic",
        "training_identity": training_identity,
        "synthetic_examples_per_epoch": training_size - real["size"],
    }


def validate_pool_consumption(record, *, training_identity, training_size, training_report):
    """Check identity and accounting against independent training expectations.

    The stream digests are recorded evidence. This check does not replay
    training, authenticate the producer, or establish generator lineage.
    """
    binding = _binding(training_identity, training_report["evaluator_seed"], training_size)
    selected = training_report["selected_epoch_index"]
    completed = training_report["completed_epochs"]
    history = training_report["history"]
    expected = {
        **binding,
        "classifier_recipe_sha256": training_report["recipe_sha256"],
        "selected_weights_sha256": training_report["selected_weights_sha256"],
        "selected_epoch_index": selected,
        "completed_epochs": completed,
    }
    if (
        type(record) is not dict
        or set(record) != set(expected) | {"selected_prefix", "final_prefix"}
        or any(_json(record[key]) != _json(value) for key, value in expected.items())
    ):
        raise ValueError("pool consumption binding mismatch")
    for name, epochs, updates in (
        ("selected_prefix", selected + 1, history[selected]["optimizer_updates"]),
        ("final_prefix", completed, training_report["optimizer_updates"]),
    ):
        prefix = record[name]
        if type(prefix) is not dict or set(prefix) != {
            "optimizer_updates", "synthetic_examples", "stream_sha256",
        }:
            raise ValueError("invalid consumption prefix")
        count = epochs * binding["synthetic_examples_per_epoch"]
        if (
            type(prefix["optimizer_updates"]) is not int
            or prefix["optimizer_updates"] != updates
            or type(prefix["synthetic_examples"]) is not int
            or prefix["synthetic_examples"] != count
            or type(prefix["stream_sha256"]) is not str
            or HEX64.fullmatch(prefix["stream_sha256"]) is None
        ):
            raise ValueError("invalid consumption accounting")
    if selected + 1 == completed and record["selected_prefix"] != record["final_prefix"]:
        raise ValueError("equal-length consumption prefixes differ")


class PoolConsumption:
    """Internal accumulator for one governed augmented training view."""

    def __init__(self, training, evaluator_seed):
        if not isinstance(training, CombinedTrainingView):
            raise TypeError("governed combined training view required")
        self._training = training
        self._identity = training.identity
        self._size = training.size
        self._binding = _binding(self._identity, evaluator_seed, self._size)
        self._offset = self._identity["real"]["size"]
        self._digest = hashlib.sha256((SCHEMA + "\n" + _json(self._binding)).encode())
        self._examples = 0

    def prepare_batch(self, positions, images, labels):
        mask = positions >= self._offset
        local = positions[mask] - self._offset
        if not local.size:
            return None
        return int(local.size), state_digest([local, images[mask], labels[mask]])

    def record_update(self, update, evidence):
        count, digest = evidence
        self._digest.update(update.to_bytes(8, "big"))
        self._digest.update(bytes.fromhex(digest))
        self._examples += count

    def prefix(self, updates):
        return {
            "optimizer_updates": updates,
            "synthetic_examples": self._examples,
            "stream_sha256": self._digest.hexdigest(),
        }

    def finish(self, report, selected_prefix):
        if self._training.identity != self._identity or self._training.size != self._size:
            raise ValueError("synthetic training identity changed")
        record = {
            **self._binding,
            "classifier_recipe_sha256": report["recipe_sha256"],
            "selected_weights_sha256": report["selected_weights_sha256"],
            "selected_epoch_index": report["selected_epoch_index"],
            "completed_epochs": report["completed_epochs"],
            "selected_prefix": selected_prefix,
            "final_prefix": self.prefix(report["optimizer_updates"]),
        }
        validate_pool_consumption(
            record, training_identity=self._identity, training_size=self._size,
            training_report=report,
        )
        return record
