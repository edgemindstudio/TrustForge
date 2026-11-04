from pathlib import Path
import numpy as np

from trustforge.data.paper01_revision_2026.ustc import (
    USTCRevision2026,
)
from trustforge.data.paper01_revision_2026.cicmaldroid import (
    CICMalDroidRevision2026,
)
from trustforge.data.paper01_revision_2026.malnet_image import (
    MalNetImageRevision2026,
)
from trustforge.data.paper01_revision_2026.malex import (
    MaleXRevision2026,
)

HOME = Path.home()
ROOT = HOME / "TrustForge"

SPLITS = ("train", "validation", "test")

EXPECTED = {
    "ustc": (72900, 8100, 9000, 9),
    "cicmaldroid2020": (8117, 1158, 2323, 5),
    "malnet_image": (883416, 126202, 252406, 47),
    "malex": (731049, 104435, 208872, 2),
}


def validate_dataset(name, adapter):
    print(f"\n===== DATASET: {name} =====", flush=True)

    train, validation, test, classes = EXPECTED[name]

    expected_counts = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split in SPLITS:
        count = adapter.split_size(split)
        expected = expected_counts[split]

        print(
            f"{split}: observed={count}, expected={expected}",
            flush=True,
        )

        assert count == expected, (
            f"{name}/{split}: count mismatch"
        )

        positions = np.asarray(
            [0, count - 1],
            dtype=np.int64,
        )

        images, labels = adapter.canonical_batch(
            split, positions
        )

        images = np.asarray(images)
        labels = np.asarray(labels)

        assert images.shape == (2, 40, 40, 3)
        assert images.dtype == np.float32
        assert np.all(np.isfinite(images))
        assert np.all((images >= 0.0) & (images <= 1.0))

        assert labels.shape == (2, classes)
        assert np.all(np.isfinite(labels))
        assert np.all((labels == 0.0) | (labels == 1.0))
        assert np.all(labels.sum(axis=1) == 1.0)

        print(
            f"{split}: CANONICAL BATCH PASS",
            flush=True,
        )

    print(f"DATASET {name}: PASS", flush=True)


def main():
    datasets = [
        (
            "ustc",
            lambda: USTCRevision2026(
                source_dir=HOME / "gencys/data/USTC-TFC2016_malware",
            ),
        ),
        (
            "cicmaldroid2020",
            lambda: CICMalDroidRevision2026(
                source_csv=HOME / (
                    "gencys/data/CICMalDroid2020_raw/"
                    "feature_vectors_syscalls_frequency_5_Cat.csv"
                ),
                feature_columns_path=HOME / (
                    "gencys/data/CICMalDroid2020_stage3/"
                    "feature_columns.json"
                ),
                validate_hashes=True,
            ),
        ),
        (
            "malnet_image",
            lambda: MalNetImageRevision2026(
                image_root=HOME / "gencys/data/MalNet-Image_raw/malnet-images",
                preparation_dir=HOME / (
                    "gencys/data/MalNet-Image_revision_2026/"
                    "type-1.0-prepared"
                ),
                mapping_path=ROOT / (
                    "studies/paper01_benchmark/audits/revision_2026/"
                    "malnet_image_type_mapping.json"
                ),
                validate_hashes=True,
                validate_production_contract=True,
            ),
        ),
        (
            "malex",
            lambda: MaleXRevision2026(
                label_archive=HOME / (
                    "gencys/data/MaleX_raw/extracted/"
                    "labels_zipped/"
                    "MaleXv1-m864669-b179725_10062017.hdf5"
                ),
                benign_byteplot_root=HOME / (
                    "gencys/data/MaleX_raw/extracted/"
                    "ben_bigram_imgs_zipped/byteplot_imgs_RxR/256"
                ),
                malware_byteplot_root=HOME / (
                    "gencys/data/MaleX_raw/extracted/"
                    "mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256"
                ),
                validate_hashes=True,
                validate_production_contract=True,
            ),
        ),
    ]

    for name, factory in datasets:
        print(f"\nINITIALIZING: {name}", flush=True)
        adapter = factory()
        validate_dataset(name, adapter)

    print("\nFOUR-DATASET REAL-DATA VALIDATION: PASS", flush=True)


if __name__ == "__main__":
    main()
