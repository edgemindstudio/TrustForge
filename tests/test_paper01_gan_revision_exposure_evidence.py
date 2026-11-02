"""Tests for GAN terminal exposure evidence validation."""

from copy import deepcopy
import pytest

from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)
from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)


def fixture():
    plan = plan_gan_run(
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        train_size=5,
    )
    draws = plan.nominal_real_exposures // 2
    rows = []
    for class_id, size in enumerate((2, 3)):
        unique = size
        rows.append({
            "class_id": class_id,
            "train_size": size,
            "draws": draws,
            "unique_positions": unique,
            "repeated_draws": draws - unique,
            "nominal_passes": draws / size,
            "unique_coverage": 1.0,
        })
    report = {
        "batches": plan.planned_generator_iterations,
        "total_real_draws": plan.nominal_real_exposures,
        "total_unique_positions": 5,
        "per_class": rows,
    }
    return plan, report


def test_valid_evidence_and_deterministic_digest():
    plan, report = fixture()
    evidence, digest = build_gan_exposure_evidence(
        plan=plan, exposure_report=report
    )
    again, other_digest = build_gan_exposure_evidence(
        plan=plan, exposure_report=report
    )
    assert evidence == again
    assert digest == other_digest
    assert len(digest) == 64
    assert evidence["per_class"][1]["class_label"] == "malware"


@pytest.mark.parametrize(
    "field,value",
    [
        ("batches", 499),
        ("total_real_draws", 123),
        ("total_unique_positions", 4),
    ],
)
def test_inconsistent_totals_rejected(field, value):
    plan, report = fixture()
    report[field] = value
    with pytest.raises(ValueError):
        build_gan_exposure_evidence(plan=plan, exposure_report=report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_id", 1),
        ("train_size", 4),
        ("draws", 1),
        ("unique_positions", 4),
        ("repeated_draws", 0),
        ("nominal_passes", 0.0),
        ("unique_coverage", 0.5),
    ],
)
def test_inconsistent_class_evidence_rejected(field, value):
    plan, report = fixture()
    report["per_class"][0][field] = value
    with pytest.raises(ValueError):
        build_gan_exposure_evidence(plan=plan, exposure_report=report)


def test_does_not_modify_input():
    plan, report = fixture()
    original = deepcopy(report)
    build_gan_exposure_evidence(plan=plan, exposure_report=report)
    assert report == original
