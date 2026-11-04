
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf

import trustforge.models.paper01_revision_2026.gaussian_mixture_initialization as gi

from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    MIN_SCALE,
    MIN_VARIANCE,
    inverse_softplus_positive,
    transformed_scale,
)

root = Path.home() / "TrustForge"
work = Path.home() / "trustforge_pretraining_validation"

expected_head = "951bd900e3d8e60938b1c03bc5a0a18c45f6fa8d"
expected_init = "b704c5de2599389bdf3ac601bdba36e2028c965c5835c5000279f774080e293c"
expected_diag = "51af82fe203fc02f8cdb304bce7001f87b084a233a62ce8ceb869e32e632d51b"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

head = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    text=True,
).strip()

init_path = root / (
    "src/trustforge/models/paper01_revision_2026/"
    "gaussian_mixture_initialization.py"
)
diagnostic = work / "diagnose_gmm_malex_scale.py"

if (
    head != expected_head
    or sha(init_path) != expected_init
    or sha(diagnostic) != expected_diag
):
    raise SystemExit("FAIL: compute-node source identity changed")

checkpoint = (
    Path.home() / "trustforge_revision_2026_artifacts"
    / "paper01/gmm/malex/seed42/class0"
)
if (
    not checkpoint.is_dir()
    or checkpoint.is_symlink()
    or any(checkpoint.iterdir())
):
    raise SystemExit("FAIL: original checkpoint evidence changed")

print("GMM_RESIDUAL_PREFLIGHT: PASS", flush=True)

original_helper = gi._production_initial_scales
captured = False

def investigate(variance):
    global captured

    # First execute the actual governed helper.
    try:
        result = original_helper(variance)
    except RuntimeError as error:
        if "bounded GMM raw-scale correction" not in str(error):
            raise
        captured = True
    else:
        print("ORIGINAL_HELPER_UNEXPECTEDLY_PASSED", flush=True)
        return result

    variance = np.asarray(variance, dtype=np.float64)
    floored = np.maximum(
        variance, np.float64(MIN_VARIANCE)
    )
    floor = np.float64(MIN_SCALE)
    target = np.sqrt(floored, dtype=np.float64)

    exact_floor = target == floor
    target[exact_floor] = np.nextafter(
        floor, np.float64(np.inf)
    )

    offset = target - floor
    raw = inverse_softplus_positive(
        tf.constant(offset, dtype=tf.float64)
    ).numpy()

    def reconstruct(candidate):
        return transformed_scale(
            tf.constant(candidate, dtype=tf.float64)
        ).numpy()

    def distance(candidate):
        return gi._positive_float64_ulp_distance(
            reconstruct(candidate), target
        )

    baseline_distance = distance(raw)
    best_distance = baseline_distance.copy()
    best_steps = np.zeros(variance.size, dtype=np.int32)
    baseline = raw.copy()

    lower = baseline.copy()
    upper = baseline.copy()
    needs_repair = (
        baseline_distance > gi.MAX_SCALE_ROUNDTRIP_ULP
    )

    for step in range(1, 9):
        lower = np.nextafter(
            lower, np.float64(-np.inf)
        )
        upper = np.nextafter(
            upper, np.float64(np.inf)
        )

        for direction, neighbor in (
            (-1, lower), (1, upper)
        ):
            proposal = baseline.copy()
            proposal[needs_repair] = neighbor[needs_repair]

            candidate_distance = distance(proposal)
            improved = (
                needs_repair
                & (candidate_distance < best_distance)
            )

            best_distance[improved] = candidate_distance[improved]
            best_steps[improved] = direction * step

    unresolved = np.flatnonzero(
        best_distance > gi.MAX_SCALE_ROUNDTRIP_ULP
    )

    worst = sorted(
        unresolved.tolist(),
        key=lambda i: (-int(best_distance[i]), i)
    )[:20]

    witnesses = [
        {
            "dimension": int(i),
            "variance": float(variance[i]),
            "target": float(target[i]),
            "baseline_raw": float(raw[i]),
            "baseline_ulp": int(baseline_distance[i]),
            "best_ulp": int(best_distance[i]),
            "best_raw_steps": int(best_steps[i]),
        }
        for i in worst
    ]

    result = {
        "dataset": "malex",
        "class_id": 0,
        "seed": 42,
        "dimensions": int(variance.size),
        "floor_promotions": int(np.count_nonzero(exact_floor)),
        "baseline_max_ulp": int(baseline_distance.max()),
        "baseline_violations": int(np.count_nonzero(needs_repair)),
        "bounded_best_max_ulp": int(best_distance.max()),
        "bounded_remaining_violations": int(unresolved.size),
        "remaining_witnesses": witnesses,
    }

    print(
        "GMM_RESIDUAL_DIAGNOSTIC "
        + json.dumps(result, sort_keys=True, allow_nan=False),
        flush=True,
    )

    # Preserve the production failure rather than fabricating
    # successful initialization.
    raise RuntimeError(
        "EXPECTED_GMM_SCALE_RESIDUAL_CAPTURED"
    )

gi._production_initial_scales = investigate

try:
    runpy.run_path(str(diagnostic))
except RuntimeError as error:
    if (
        not captured
        or str(error) != "EXPECTED_GMM_SCALE_RESIDUAL_CAPTURED"
    ):
        raise
    print("GMM_RESIDUAL_CAPTURE: COMPLETE", flush=True)
    print("MODEL_INITIALIZATION: NOT COMPLETED", flush=True)
    print("OPTIMIZER_STEPS: 0", flush=True)
    print("CHECKPOINT_WRITTEN: NO", flush=True)
else:
    print("GMM_RESIDUAL_CAPTURE: NOT REQUIRED", flush=True)

print("===== COMMAND 297 DIAGNOSTIC COMPLETE =====", flush=True)
