"""Verify scientific exposure evidence bound to a GAN v2 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from trustforge.provenance import canonical_json_bytes, sha256_file
from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)

EVIDENCE_FILENAME = "gan_exposure_evidence.json"


def verify_gan_checkpoint_evidence(directory, metadata):
    """Verify the evidence file, its contents, and checkpoint identity."""
    if metadata.get("schema_version") != "2.0":
        raise ValueError("GAN evidence requires checkpoint schema v2")

    if metadata.get("exposure_evidence_file") != EVIDENCE_FILENAME:
        raise ValueError("invalid GAN evidence filename")

    path = Path(directory) / EVIDENCE_FILENAME

    if not path.is_file() or path.is_symlink():
        raise ValueError("GAN exposure evidence file is missing or unsafe")

    raw = path.read_bytes()
    if not raw:
        raise ValueError("empty GAN exposure evidence")

    digest = sha256_file(path)
    if digest != metadata.get("exposure_evidence_sha256"):
        raise ValueError("GAN exposure evidence integrity failure")

    try:
        evidence = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid GAN exposure evidence JSON") from exc

    if not isinstance(evidence, dict):
        raise ValueError("GAN exposure evidence must be an object")

    if raw != canonical_json_bytes(evidence):
        raise ValueError("GAN exposure evidence is not canonical JSON")

    required = {
        "schema_type",
        "schema_version",
        "dataset_id",
        "experimental_seed",
        "class_mapping",
        "train_size",
        "batch_size",
        "class_sampling_alpha",
        "planned_generator_iterations",
        "completed_generator_iterations",
        "completed_discriminator_updates",
        "total_real_draws",
        "total_unique_positions",
        "per_class",
    }

    if set(evidence) != required:
        raise ValueError("GAN exposure evidence schema mismatch")

    if (
        evidence["schema_type"]
        != "paper01_revision2026_gan_exposure_evidence"
        or evidence["schema_version"] != "1.0"
    ):
        raise ValueError("unsupported GAN exposure evidence schema")

    if (
        evidence["dataset_id"] != metadata["dataset_id"]
        or evidence["experimental_seed"] != metadata["experimental_seed"]
        or evidence["class_mapping"] != metadata["class_mapping"]
        or evidence["batch_size"] != metadata["batch_size"]
        or evidence["class_sampling_alpha"]
        != metadata["class_sampling_alpha"]
        or evidence["planned_generator_iterations"]
        != metadata["planned_generator_iterations"]
        or evidence["completed_generator_iterations"]
        != metadata["generator_iteration"]
        or evidence["completed_discriminator_updates"]
        != metadata["discriminator_updates"]
    ):
        raise ValueError("GAN evidence and checkpoint metadata disagree")

    plan = plan_gan_run(
        dataset_id=evidence["dataset_id"],
        experimental_seed=evidence["experimental_seed"],
        class_mapping=evidence["class_mapping"],
        train_size=evidence["train_size"],
    )

    if plan.planned_generator_iterations != (
        metadata["planned_generator_iterations"]
    ):
        raise ValueError("GAN evidence budget disagrees with governed plan")

    if not isinstance(evidence["per_class"], list):
        raise ValueError("invalid GAN evidence per-class records")

    report_rows = []
    for row in evidence["per_class"]:
        if not isinstance(row, dict):
            raise ValueError("invalid GAN evidence class record")
        stripped = dict(row)
        stripped.pop("class_label", None)
        report_rows.append(stripped)

    report = {
        "batches": evidence["completed_generator_iterations"],
        "total_real_draws": evidence["total_real_draws"],
        "total_unique_positions": evidence["total_unique_positions"],
        "per_class": report_rows,
    }

    regenerated, regenerated_digest = build_gan_exposure_evidence(
        plan=plan,
        exposure_report=report,
    )

    if regenerated != evidence:
        raise ValueError("GAN exposure evidence is internally inconsistent")

    if regenerated_digest != digest:
        raise ValueError("GAN exposure evidence digest mismatch")

    return evidence
