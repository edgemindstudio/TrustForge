"""CIC fitted-state-aware AR/MAF execution preparation and publication.

Retains the existing runner's authorization restrictions. No job submission.
Historical bundles are not migrated or attributed new preprocessing evidence.
"""

import hashlib
import json
from pathlib import Path
import platform
import sys

from . import ar_maf_training_runner as runner
from .ar_maf_checkpoint_identity import canonical_json_bytes
from .ar_maf_checkpoint_publication import publish_checkpoint_bundle
from .ar_maf_checkpoint_bundle import read_checkpoint_bundle
from .ar_maf_cic_execution_manifest import prepare_cic_execution_manifest


def _capture(*, repository_root, adapter, dataset_config_raw):
    """Capture the same adapter's current state after governed preflight."""
    if type(dataset_config_raw) is not bytes:
        raise TypeError("configuration must be exact bytes")
    config = json.loads(dataset_config_raw)
    if (
        type(config) is not dict
        or set(config) != {"dataset_id", "paths"}
        or config["dataset_id"] != "cicmaldroid2020"
        or type(config["paths"]) is not dict
        or set(config["paths"]) != {"source_csv", "feature_columns_path"}
    ):
        raise ValueError("invalid CIC configuration")
    for key, value in config["paths"].items():
        if type(value) is not str or not value.strip():
            raise ValueError("invalid CIC source path")
        if Path(value).expanduser().resolve() != getattr(adapter, key):
            raise ValueError("configuration differs from consumed adapter")

    from .gan_adapter_identity import require_gan_adapter_identity
    from .gan_dataset_provenance import require_gan_dataset_provenance
    require_gan_adapter_identity(dataset_id="cicmaldroid2020", adapter=adapter)
    verified = require_gan_dataset_provenance(
        dataset_id="cicmaldroid2020", adapter=adapter,
    )
    commit, sources = runner.capture_source_identity(repository_root)

    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from trustforge.data.paper01_revision_2026.cic_fitted_state import (
        capture_fitted_state,
    )
    return capture_fitted_state(
        scaler=adapter.scaler,
        training_indices=adapter.partitions.train,
        context={
            "source_preflight": verified,
            "configuration_sha256": hashlib.sha256(dataset_config_raw).hexdigest(),
            "feature_columns": list(adapter.feature_columns),
            "partition": {
                "seed": adapter.partition_seed,
                "ratios": [
                    adapter.train_fraction,
                    adapter.validation_fraction,
                    adapter.test_fraction,
                ],
            },
            "implementation": {
                "commit": commit, "tracked_source_files": sources,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "tensorflow": tf.__version__,
                "byteorder": sys.byteorder,
            },
        },
    )


def prepare_cic_execution(
    *, repository_root, family, experimental_seed, adapter,
    dataset_config_raw, bounded_integration=False,
):
    """Prepare an enforced envelope only after unchanged-state verification."""
    # Reject unauthorized requests before source hashing or adapter work.
    if type(bounded_integration) is not bool:
        raise TypeError("bounded_integration must be boolean")
    if bounded_integration:
        if not runner.bounded_cic_request_allowed(
            family=family, dataset_id="cicmaldroid2020",
            experimental_seed=experimental_seed,
        ):
            raise RuntimeError("request outside bounded CIC execution scope")
    elif runner.TRAINING_EXECUTION_ENABLED is not True:
        raise RuntimeError("AR/MAF training execution remains disabled")

    arguments = {
        "repository_root": repository_root,
        "adapter": adapter,
        "dataset_config_raw": dataset_config_raw,
    }
    before = _capture(**arguments)
    prepared = runner.run_training_and_prepare_checkpoint(
        repository_root=repository_root,
        family=family,
        dataset_id="cicmaldroid2020",
        experimental_seed=experimental_seed,
        adapter=adapter,
        bounded_integration=bounded_integration,
    )
    after = _capture(**arguments)
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise RuntimeError("CIC preprocessing or provenance changed during training")

    manifest = prepared["manifest"]
    identity = manifest["identity"]
    if (
        identity["family"] != family
        or identity["dataset_id"] != "cicmaldroid2020"
        or identity["experimental_seed"] != experimental_seed
    ):
        raise RuntimeError("prepared checkpoint differs from requested execution")

    requested = {
        "recipe_raw": prepared["recipe_raw"],
        "expected_identity": identity,
        "expected_source_commit": manifest["source_commit"],
        "expected_source_files": manifest["source_files"],
        "expected_fitted_state_sha256": before["sha256"],
    }
    fitted_raw = canonical_json_bytes(before) + b"\n"
    envelope_raw = prepare_cic_execution_manifest(
        checkpoint_manifest_raw=prepared["manifest_raw"],
        fitted_state_raw=fitted_raw,
        best_raw=prepared["best_raw"],
        last_raw=prepared["last_raw"],
        **requested,
    )
    return {
        "execution_manifest_raw": envelope_raw,
        "fitted_state_raw": fitted_raw,
        "best_raw": prepared["best_raw"],
        "last_raw": prepared["last_raw"],
        "requested": requested,
        "publication_status": "NOT_PUBLISHED",
    }


def publish_and_read_cic_execution(parent, generation_name, *, prepared):
    """Require the fitted-state-aware path for publication and immediate read."""
    requested = prepared["requested"]
    if not requested.get("expected_fitted_state_sha256"):
        raise ValueError("fitted-state expectation required")
    published = publish_checkpoint_bundle(
        parent, generation_name,
        manifest_raw=prepared["execution_manifest_raw"],
        best_raw=prepared["best_raw"],
        last_raw=prepared["last_raw"],
        **requested,
    )
    verified = read_checkpoint_bundle(
        Path(parent) / generation_name, **requested,
    )
    return {"publication": published, "verified": verified}
