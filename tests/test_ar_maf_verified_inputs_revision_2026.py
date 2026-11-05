import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_verified_inputs as module


class Adapter:
    num_classes = 2

    def __init__(self, dataset="malex"):
        self.dataset = dataset
        self.class_to_id = {"benign": 0, "malware": 1}
        self.counts = {"train": 5, "validation": 3}
        self.integer = {
            split: np.arange(n, dtype=np.int32) % self.classes
            for split, n in self.counts.items()
        }
        self.wrong_onehot = False
        self.label_calls = 0

    @property
    def classes(self):
        return 9 if self.dataset == "ustc_tfc2016" else 2

    def split_size(self, split):
        return self.counts[split]

    def canonical_batch(self, split, positions):
        ids = self.integer[split][positions]
        if self.wrong_onehot:
            ids = (ids + 1) % self.classes
        return (
            np.zeros((len(positions), 40, 40, 3), dtype=np.float32),
            np.eye(self.classes, dtype=np.float32)[ids],
        )

    def labels(self, split, positions=None):
        self.label_calls += 1
        values = self.integer[split]
        return values if positions is None else values[positions]


def wrapped(adapter):
    mapping = (
        tuple(str(i) for i in range(9))
        if adapter.dataset == "ustc_tfc2016"
        else ("benign", "malware")
    )
    return module.CheckedAdapterInputs(
        adapter=adapter, dataset_id=adapter.dataset,
        class_mapping=mapping, split_counts=adapter.counts,
    )


@pytest.mark.parametrize("dataset", ["malex", "ustc_tfc2016"])
def test_matching_batch_and_positions_preserved(dataset):
    adapter = Adapter(dataset)
    checked = wrapped(adapter)
    positions = np.array([4, 1], dtype=np.int64)
    images, labels = checked.canonical_batch("train", positions)
    assert images.shape == (2, 40, 40, 3)
    np.testing.assert_array_equal(
        np.argmax(labels, axis=1), adapter.integer["train"][positions],
    )
    np.testing.assert_array_equal(positions, [4, 1])


@pytest.mark.parametrize("dataset", ["malex", "ustc_tfc2016"])
def test_label_disagreement_rejected(dataset):
    adapter = Adapter(dataset)
    adapter.wrong_onehot = True
    with pytest.raises(RuntimeError, match="interfaces disagree"):
        wrapped(adapter).canonical_batch(
            "train", np.array([0, 1], dtype=np.int64),
        )


def test_whole_split_reference_is_copied_and_reused():
    adapter = Adapter("ustc_tfc2016")
    checked = wrapped(adapter)
    positions = np.array([0, 1], dtype=np.int64)
    checked.canonical_batch("train", positions)
    checked.canonical_batch("train", positions)
    assert adapter.label_calls == 1
    adapter.integer["train"][0] = 1
    with pytest.raises(RuntimeError, match="interfaces disagree"):
        checked.canonical_batch("train", positions)


def test_changed_mapping_rejected():
    adapter = Adapter()
    checked = wrapped(adapter)
    adapter.class_to_id = {"benign": 0, "different": 1}
    with pytest.raises(ValueError, match="semantic mapping"):
        checked.canonical_batch(
            "train", np.array([0], dtype=np.int64),
        )


def test_changed_split_size_rejected():
    adapter = Adapter()
    checked = wrapped(adapter)
    adapter.counts["train"] = 6
    with pytest.raises(RuntimeError, match="split size changed"):
        checked.split_size("train")


def test_test_split_rejected():
    with pytest.raises(ValueError, match="test split"):
        wrapped(Adapter()).canonical_batch(
            "test", np.array([0], dtype=np.int64),
        )


@pytest.mark.parametrize("defect", ["float", "shape", "range"])
def test_invalid_integer_label_interface_rejected(defect):
    adapter = Adapter()
    checked = wrapped(adapter)

    def invalid(split, positions):
        if defect == "float":
            return np.zeros(len(positions), dtype=np.float32)
        if defect == "shape":
            return np.zeros((len(positions), 1), dtype=np.int32)
        return np.full(len(positions), 2, dtype=np.int32)

    adapter.labels = invalid
    with pytest.raises(ValueError, match="integer labels"):
        checked.canonical_batch(
            "train", np.array([0, 1], dtype=np.int64),
        )


def test_binding_preflights_and_wraps_same_adapter(monkeypatch):
    adapter = Adapter()
    calls = []

    def preflight(**kwargs):
        calls.append(kwargs["adapter"])
        return {
            "identity": {"class_mapping": ["benign", "malware"]},
            "training_examples": 5,
            "validation_examples": 3,
        }

    monkeypatch.setattr(module, "preflight_dataset", preflight)
    checked, evidence = module.preflight_and_bind_inputs(
        family="ar", dataset_id="malex", experimental_seed=42,
        adapter=adapter, recipe_raw=b"fixture",
    )
    assert calls == [adapter]
    assert checked._adapter is adapter
    assert checked.split_size("train") == evidence["training_examples"]


def test_preflight_failure_prevents_wrapper_creation(monkeypatch):
    def fail(**kwargs):
        raise ValueError("preflight rejected")

    monkeypatch.setattr(module, "preflight_dataset", fail)
    with pytest.raises(ValueError, match="preflight rejected"):
        module.preflight_and_bind_inputs(
            family="ar", dataset_id="malex", experimental_seed=42,
            adapter=Adapter(), recipe_raw=b"fixture",
        )
