#!/usr/bin/env bash
#SBATCH --job-name=p1r1-inception-id
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
: "${TF_INCEPTION_ID_RUN_DIR:?Run directory required}"
: "${SLURM_JOB_ID:?Slurm allocation required}"

cd "$HOME/TrustForge"
[[ "$(git rev-parse HEAD)" == 99a98663508061334fd1c1658fc4c173bc2f9c9e ]]
git diff --quiet
git diff --cached --quiet

"$HOME/miniforge3/envs/gcs_py310/bin/python" -B -u <<'PY'
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import sysconfig

if socket.gethostname().split(".")[0].startswith("login"):
    raise SystemExit("STOP: compute node required")

cache = Path(os.environ.get("KERAS_HOME", str(Path.home() / ".keras")))
weights = cache / "models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
if weights.is_symlink() or not weights.is_file():
    raise SystemExit("STOP: cached weight artifact unavailable")

before = weights.stat()
md5 = hashlib.md5()
sha256 = hashlib.sha256()
with weights.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        md5.update(chunk)
        sha256.update(chunk)
after = weights.stat()

if (
    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
) != (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
):
    raise SystemExit("STOP: weight artifact changed during hashing")

expected_md5 = "bcbd6486424b2319ff4ef7d526e38f63"
if md5.hexdigest() != expected_md5:
    raise SystemExit("STOP: artifact differs from installed Keras checksum")

source = (
    Path(sysconfig.get_paths()["purelib"])
    / "keras/src/applications/inception_v3.py"
)
source_raw = source.read_bytes()
if expected_md5.encode() not in source_raw:
    raise SystemExit("STOP: installed source checksum specification changed")

report = {
    "scope": "INCEPTION_NO_TOP_WEIGHT_IDENTITY",
    "source_commit": "99a98663508061334fd1c1658fc4c173bc2f9c9e",
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "weights_path": str(weights),
    "weights_bytes": after.st_size,
    "weights_md5": md5.hexdigest(),
    "weights_sha256": sha256.hexdigest(),
    "installed_keras_checksum_matched": True,
    "keras_version": metadata.version("keras"),
    "tensorflow_version": metadata.version("tensorflow"),
    "keras_inception_source_path": str(source),
    "keras_inception_source_sha256": hashlib.sha256(source_raw).hexdigest(),
    "model_loaded": False,
    "features_extracted": False,
    "full_benchmark_go": False,
}
output = Path(os.environ["TF_INCEPTION_ID_RUN_DIR"]) / "report.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, sort_keys=True, allow_nan=False)
    handle.write("\n")
print(json.dumps(report, sort_keys=True), flush=True)
print("INCEPTION_WEIGHT_IDENTITY=PASS", flush=True)
PY

[[ "$(git rev-parse HEAD)" == 99a98663508061334fd1c1658fc4c173bc2f9c9e ]]
git diff --quiet
git diff --cached --quiet
