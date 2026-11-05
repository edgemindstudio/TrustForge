"""Bind preflight evidence to canonical batches from the same adapter.

No model construction, training authorization or checkpoint publication.
Checks consistency of adapter interfaces, not independent label ground truth.
"""

import numpy as np

from .ar_maf_dataset_preflight import (
    preflight_dataset, adapter_class_mapping,
)
from .ar_maf_execution_inputs import DATASET_CLASSES, require_integer
from .variational_autoencoder_data_integration import canonical_vae_batch


class CheckedAdapterInputs:
    """Internal batch wrapper; direct construction does not establish preflight."""

    def __init__(self, *, adapter, dataset_id, class_mapping, split_counts):
        self._adapter = adapter
        self._dataset_id = dataset_id
        self._mapping = tuple(class_mapping)
        self._counts = dict(split_counts)
        self._whole_labels = {}

    def split_size(self, split):
        if split not in ("train", "validation"):
            raise ValueError("training inputs cannot access test split")
        observed = require_integer(
            self._adapter.split_size(split), split + "_size",
        )
        if observed != self._counts[split]:
            raise RuntimeError("adapter split size changed after preflight")
        return observed

    def canonical_batch(self, split, positions):
        size = self.split_size(split)
        mapping = adapter_class_mapping(
            dataset_id=self._dataset_id, adapter=self._adapter,
        )
        if mapping != self._mapping:
            raise RuntimeError("adapter class mapping changed after preflight")

        # The existing canonical validator checks positions, shapes, ranges,
        # finite values and strict one-hot labels on the original adapter.
        requested = np.array(positions, copy=True)
        images, onehot = canonical_vae_batch(
            adapter=self._adapter, split=split,
            logical_positions=requested,
            num_classes=DATASET_CLASSES[self._dataset_id],
        )

        if self._dataset_id in ("ustc_tfc2016", "cicmaldroid2020"):
            if split not in self._whole_labels:
                labels = np.asarray(self._adapter.labels(split))
                if labels.shape != (size,) or labels.dtype.kind not in "iu":
                    raise ValueError("invalid whole-split integer labels")
                if np.any(labels < 0) or np.any(
                    labels >= DATASET_CLASSES[self._dataset_id]
                ):
                    raise ValueError("integer labels outside class universe")
                self._whole_labels[split] = labels.copy()
            expected = self._whole_labels[split][requested]
        else:
            expected = np.asarray(
                self._adapter.labels(split, requested.copy())
            )

        if (
            expected.shape != (len(requested),)
            or expected.dtype.kind not in "iu"
            or np.any(expected < 0)
            or np.any(expected >= DATASET_CLASSES[self._dataset_id])
        ):
            raise ValueError("invalid requested integer labels")
        if not np.array_equal(np.argmax(onehot, axis=1), expected):
            raise RuntimeError("canonical and integer label interfaces disagree")
        return images, onehot


def preflight_and_bind_inputs(
    *, family, dataset_id, experimental_seed, adapter, recipe_raw,
):
    """Preflight and wrap the same adapter object without authorizing training."""
    evidence = preflight_dataset(
        family=family, dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        adapter=adapter, recipe_raw=recipe_raw,
    )
    checked = CheckedAdapterInputs(
        adapter=adapter, dataset_id=dataset_id,
        class_mapping=evidence["identity"]["class_mapping"],
        split_counts={
            "train": evidence["training_examples"],
            "validation": evidence["validation_examples"],
        },
    )
    return checked, evidence
