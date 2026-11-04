#!/usr/bin/env bash
#SBATCH --job-name=p1r1-gmm-zipmeta
#SBATCH --partition=talon
#SBATCH --nodelist=talon16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=p1r1-gmm-zipmeta.%j.out
#SBATCH --error=p1r1-gmm-zipmeta.%j.err

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

"$HOME/miniforge3/envs/gcs_py310/bin/python" - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import shutil

archive = (
    Path.home()
    / "gencys/data/MaleX_raw/malex-fyp-image-dataset.zip"
)

expected_archive_bytes = 87775065860

if not archive.is_file() or archive.is_symlink():
    raise SystemExit("FAIL: source archive unavailable")

if archive.stat().st_size != expected_archive_bytes:
    raise SystemExit("FAIL: archive size mismatch")

prefixes = {
    "benign": (
        "ben_bigram_imgs_zipped/"
        "byteplot_imgs_RxR/256/"
    ),
    "malware": (
        "mal_byteplot_imgs_zipped/"
        "byteplot_imgs_RxR/256/"
    ),
}

label_name = (
    "labels_zipped/"
    "MaleXv1-m864669-b179725_10062017.hdf5"
)

counts = {"benign": 0, "malware": 0, "labels": 0}
bytes_by_type = {key: 0 for key in counts}

with ZipFile(archive, "r", allowZip64=True) as zf:
    for entry in zf.infolist():
        name = entry.filename

        if entry.is_dir():
            continue

        if name == label_name:
            counts["labels"] += 1
            bytes_by_type["labels"] += entry.file_size
            continue

        for key, prefix in prefixes.items():
            if not name.startswith(prefix):
                continue

            remainder = name[len(prefix):]

            if (
                remainder
                and "/" not in remainder
                and remainder.endswith(".png")
            ):
                counts[key] += 1
                bytes_by_type[key] += entry.file_size

            break

print("===== MALEX ARCHIVE METADATA =====")
print("ARCHIVE_BYTES:", expected_archive_bytes)
print("BENIGN_COUNT:", counts["benign"])
print("MALWARE_COUNT:", counts["malware"])
print("LABEL_COUNT:", counts["labels"])

if counts != {
    "benign": 179725,
    "malware": 864669,
    "labels": 1,
}:
    raise SystemExit("FAIL: selected entry counts mismatch")

extracted_bytes = sum(bytes_by_type.values())
required_bytes = expected_archive_bytes + extracted_bytes

stage = Path("/tmp")
usage = shutil.disk_usage(stage)

margin_bytes = 64 * 1024**3
headroom = usage.free - required_bytes

for key, value in bytes_by_type.items():
    print(f"{key.upper()}_UNCOMPRESSED_BYTES:", value)

print("SELECTED_EXTRACTED_BYTES:", extracted_bytes)
print("ARCHIVE_PLUS_EXTRACTION_BYTES:", required_bytes)
print("NODE_AVAILABLE_BYTES:", usage.free)
print("REMAINING_BEFORE_MARGIN_BYTES:", headroom)
print("SAFETY_MARGIN_BYTES:", margin_bytes)

if headroom < margin_bytes:
    print("RESULT=NO_GO_INSUFFICIENT_HEADROOM")
    raise SystemExit(2)

print("RESULT=METADATA_CAPACITY_PASS")
print("ACTUAL_EXTRACTION_TESTED=NO")
print("ARCHIVE_COPIED=NO")
print("TRAINING_EXECUTED=NO")
PY
