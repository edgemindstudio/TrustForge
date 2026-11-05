from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_dataset_preflight as preflight


def recipe_bytes():
    return (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026/"
          "ar_maf_integration_recipe.json"
    ).read_bytes()


@pytest.mark.parametrize("dataset,count", [
    ("ustc_tfc2016", 9), ("cicmaldroid2020", 5),
])
def test_numeric_model_ids_are_preserved_without_semantic_inference(
    dataset, count,
):
    assert preflight.adapter_class_mapping(
        dataset_id=dataset, adapter=object(),
    ) == tuple(str(i) for i in range(count))


def test_malex_mapping_follows_ids_not_dictionary_order():
    adapter = SimpleNamespace(
        num_classes=2, class_to_id={"malware": 1, "benign": 0},
    )
    assert preflight.adapter_class_mapping(
        dataset_id="malex", adapter=adapter,
    ) == ("benign", "malware")


def test_malnet_mapping_follows_ids():
    mapping = {f"type_{i:02d}": i for i in reversed(range(47))}
    adapter = SimpleNamespace(num_classes=47, class_to_id=mapping)
    assert preflight.adapter_class_mapping(
        dataset_id="malnet_image", adapter=adapter,
    ) == tuple(f"type_{i:02d}" for i in range(47))


@pytest.mark.parametrize("mapping", [
    {"benign": 1, "malware": 0},
    {"benign": 0, "malware": 0},
    {"benign": 0, "malware": 2},
    {"benign": False, "malware": 1},
    {"benign": 0.0, "malware": 1},
    {"benign": 0},
    {"benign": 0, " ": 1},
    {0: 0, "malware": 1},
])
def test_invalid_malex_mapping_rejected(mapping):
    with pytest.raises((ValueError, TypeError)):
        preflight.adapter_class_mapping(
            dataset_id="malex",
            adapter=SimpleNamespace(num_classes=2, class_to_id=mapping),
        )


@pytest.mark.parametrize("count", [True, 2.0, 3])
def test_invalid_adapter_class_count_rejected(count):
    with pytest.raises((ValueError, TypeError)):
        preflight.adapter_class_mapping(
            dataset_id="malex",
            adapter=SimpleNamespace(
                num_classes=count,
                class_to_id={"benign": 0, "malware": 1},
            ),
        )


class NoAccess:
    def __getattr__(self, name):
        raise AssertionError("adapter accessed before input/runtime rejection")


def arguments():
    return {
        "family": "ar", "dataset_id": "malex",
        "experimental_seed": 42, "adapter": NoAccess(),
        "recipe_raw": recipe_bytes(),
    }


@pytest.mark.parametrize("key,value", [
    ("family", "gan"), ("dataset_id", "unknown"),
    ("experimental_seed", True), ("experimental_seed", 42.0),
    ("experimental_seed", 41), ("recipe_raw", b"changed"),
    ("recipe_raw", "not bytes"),
])
def test_invalid_request_rejected_before_adapter_access(key, value):
    values = arguments()
    values[key] = value
    with pytest.raises((ValueError, TypeError)):
        preflight.preflight_dataset(**values)


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "SLURM_JOB_NODELIST"])
def test_missing_allocation_rejected_before_adapter_access(
    monkeypatch, missing,
):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "compute_fixture")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "compute_fixture")
    with pytest.raises(RuntimeError, match="compute-node"):
        preflight.preflight_dataset(**arguments())


def test_login_host_rejected_even_with_allocation_variables(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "compute_fixture")
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "login2")
    with pytest.raises(RuntimeError, match="compute-node"):
        preflight.preflight_dataset(**arguments())
