
from pathlib import Path
import hashlib
import json
import runpy
import subprocess

root = Path.home() / "TrustForge"
work = Path.home() / "trustforge_pretraining_validation"

expected_head = "951bd900e3d8e60938b1c03bc5a0a18c45f6fa8d"
expected_diagnostic_sha = "51af82fe203fc02f8cdb304bce7001f87b084a233a62ce8ceb869e32e632d51b"
expected_initializer_sha = "b704c5de2599389bdf3ac601bdba36e2028c965c5835c5000279f774080e293c"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

head = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    text=True,
).strip()

diagnostic = work / "diagnose_gmm_malex_scale.py"
initializer_path = (
    root / "src/trustforge/models/paper01_revision_2026"
    / "gaussian_mixture_initialization.py"
)

if head != expected_head:
    raise SystemExit("FAIL: compute-node HEAD changed")

if digest(diagnostic) != expected_diagnostic_sha:
    raise SystemExit("FAIL: diagnostic fingerprint changed")

if digest(initializer_path) != expected_initializer_sha:
    raise SystemExit("FAIL: initializer fingerprint changed")

checkpoint = (
    Path.home()
    / "trustforge_revision_2026_artifacts"
    / "paper01/gmm/malex/seed42/class0"
)

if (
    not checkpoint.is_dir()
    or checkpoint.is_symlink()
    or any(checkpoint.iterdir())
):
    raise SystemExit("FAIL: original checkpoint evidence changed")

print("GMM_REAL_INITIALIZATION_PREFLIGHT: PASS", flush=True)

# Execute the existing real-data diagnostic, preserving its
# canonical sample and class-position variables in memory.
state = runpy.run_path(str(diagnostic))

if "images" not in state or "train_positions" not in state:
    raise SystemExit("FAIL: diagnostic sample bindings unavailable")

if int(state["train_positions"].size) != 125794:
    raise SystemExit("FAIL: unexpected MaleX class count")

from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    initialize_class_model_from_canonical_sample,
    _positive_float64_ulp_distance,
    MAX_SCALE_ROUNDTRIP_ULP,
)

import numpy as np

print("===== FULL REAL-DATA MODEL INITIALIZATION =====", flush=True)

model, diagnostics = initialize_class_model_from_canonical_sample(
    state["images"],
    class_count=125794,
    experimental_seed=42,
)

scales = model.scales().numpy()

if scales.shape != (5, 4800):
    raise RuntimeError("incorrect initialized model scale shape")

if not all(
    np.array_equal(scales[0], scales[i])
    for i in range(1, 5)
):
    raise RuntimeError("initialized component scales disagree")

print("REAL_MODEL_SCALE_SHAPE:", scales.shape, flush=True)
print("REAL_MODEL_SCALE_MIN:", float(scales.min()), flush=True)
print("REAL_MODEL_SCALE_MAX:", float(scales.max()), flush=True)
print("REAL_MODEL_CENTER_INDICES:", diagnostics.center_indices, flush=True)
print("REAL_MODEL_INITIALIZATION: PASS", flush=True)
print("OPTIMIZER_STEPS: 0", flush=True)
print("CHECKPOINT_WRITTEN: NO", flush=True)
