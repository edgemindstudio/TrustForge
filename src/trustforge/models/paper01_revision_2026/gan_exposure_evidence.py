"""Validated scientific exposure evidence for revision-2026 GAN."""

from __future__ import annotations

import math

from trustforge.provenance import sha256_canonical_json
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    GANRunPlan,
)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def build_gan_exposure_evidence(*, plan, exposure_report):
    """Bind completed train-position exposure counts to a run plan."""
    if not isinstance(plan, GANRunPlan):
        raise TypeError("expected governed GANRunPlan")
    if not isinstance(exposure_report, dict):
        raise TypeError("expected exposure report")

    required = {
        "batches",
        "total_real_draws",
        "total_unique_positions",
        "per_class",
    }
    if set(exposure_report) != required:
        raise ValueError("exposure report fields mismatch")

    expected_draws = plan.nominal_real_exposures
    if _integer(exposure_report["batches"], "batches") != (
        plan.planned_generator_iterations
    ):
        raise ValueError("completed batch count mismatch")

    if _integer(
        exposure_report["total_real_draws"], "total_real_draws"
    ) != expected_draws:
        raise ValueError("total real draws mismatch")

    rows = exposure_report["per_class"]
    if not isinstance(rows, list) or len(rows) != len(plan.class_mapping):
        raise ValueError("per-class report length mismatch")

    normalized = []
    for class_id, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "class_id", "train_size", "draws",
            "unique_positions", "repeated_draws",
            "nominal_passes", "unique_coverage",
        }:
            raise ValueError("invalid per-class report fields")

        if _integer(row["class_id"], "class_id") != class_id:
            raise ValueError("class ordering mismatch")

        size = _integer(row["train_size"], "train_size", 1)
        draws = _integer(row["draws"], "draws")
        unique = _integer(row["unique_positions"], "unique_positions")
        repeats = _integer(row["repeated_draws"], "repeated_draws")

        if unique > size or unique > draws or repeats != draws - unique:
            raise ValueError("invalid duplicate or coverage accounting")

        for key, expected in (
            ("nominal_passes", draws / size),
            ("unique_coverage", unique / size),
        ):
            value = row[key]
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not math.isclose(value, expected, rel_tol=1e-12)
            ):
                raise ValueError(f"inconsistent {key}")

        normalized.append({
            "class_id": class_id,
            "class_label": plan.class_mapping[class_id],
            "train_size": size,
            "draws": draws,
            "unique_positions": unique,
            "repeated_draws": repeats,
            "nominal_passes": draws / size,
            "unique_coverage": unique / size,
        })

    if sum(row["train_size"] for row in normalized) != plan.train_size:
        raise ValueError("training split size mismatch")

    if sum(row["draws"] for row in normalized) != expected_draws:
        raise ValueError("class draw totals mismatch")

    total_unique = sum(row["unique_positions"] for row in normalized)
    if _integer(
        exposure_report["total_unique_positions"],
        "total_unique_positions",
    ) != total_unique:
        raise ValueError("unique position total mismatch")

    evidence = {
        "schema_type": "paper01_revision2026_gan_exposure_evidence",
        "schema_version": "1.0",
        "dataset_id": plan.dataset_id,
        "experimental_seed": plan.experimental_seed,
        "class_mapping": list(plan.class_mapping),
        "train_size": plan.train_size,
        "batch_size": plan.batch_size,
        "class_sampling_alpha": 0.85,
        "planned_generator_iterations": plan.planned_generator_iterations,
        "completed_generator_iterations": exposure_report["batches"],
        "completed_discriminator_updates": (
            plan.planned_discriminator_updates
        ),
        "total_real_draws": expected_draws,
        "total_unique_positions": total_unique,
        "per_class": normalized,
    }

    return evidence, sha256_canonical_json(evidence)
