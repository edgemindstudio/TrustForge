from pathlib import Path

from trustforge.models.paper01_revision_2026.gan_adapter_identity import (
    require_gan_adapter_identity,
)
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    require_gan_dataset_provenance,
)
from trustforge.models.paper01_revision_2026.gan_v3_provenance_adapter import (
    build_gan_v3_provenance,
)


def main():
    # Dataset factories are local to main(), so use an explicit
    # AST/source-independent route through the existing constructors.
    # This avoids executing the earlier validation main().
    from trustforge.data.paper01_revision_2026.ustc import USTCRevision2026
    from trustforge.data.paper01_revision_2026.cicmaldroid import CICMalDroidRevision2026
    from trustforge.data.paper01_revision_2026.malnet_image import MalNetImageRevision2026
    from trustforge.data.paper01_revision_2026.malex import MaleXRevision2026

    home = Path.home()
    repo = home / "TrustForge"

    datasets = [
        (
            "ustc_tfc2016",
            lambda: USTCRevision2026(
                home / "gencys/data/USTC-TFC2016_malware"
            ),
        ),
        (
            "cicmaldroid2020",
            lambda: CICMalDroidRevision2026(
                home / "gencys/data/CICMalDroid2020_raw/feature_vectors_syscalls_frequency_5_Cat.csv",
                home / "gencys/data/CICMalDroid2020_stage3/feature_columns.json",
                validate_hashes=True,
            ),
        ),
        (
            "malnet_image",
            lambda: MalNetImageRevision2026(
                home / "gencys/data/MalNet-Image_raw/malnet-images",
                home / "gencys/data/MalNet-Image_revision_2026/type-1.0-prepared",
                repo / "studies/paper01_benchmark/audits/revision_2026/malnet_image_type_mapping.json",
                validate_hashes=True,
                validate_production_contract=True,
            ),
        ),
        (
            "malex",
            lambda: MaleXRevision2026(
                home / "gencys/data/MaleX_raw/extracted/labels_zipped/MaleXv1-m864669-b179725_10062017.hdf5",
                home / "gencys/data/MaleX_raw/extracted/ben_bigram_imgs_zipped/byteplot_imgs_RxR/256",
                home / "gencys/data/MaleX_raw/extracted/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256",
                validate_hashes=True,
                validate_production_contract=True,
            ),
        ),
    ]

    for dataset_id, factory in datasets:
        print(
            f"\nPREFLIGHT START: {dataset_id}",
            flush=True,
        )

        adapter = factory()

        identity = require_gan_adapter_identity(
            dataset_id=dataset_id,
            adapter=adapter,
        )

        verified = require_gan_dataset_provenance(
            dataset_id=dataset_id,
            adapter=adapter,
        )

        provenance = build_gan_v3_provenance(
            dataset_id=dataset_id,
            verified_preflight=verified,
        )

        assert provenance["dataset_id"] == dataset_id

        print("ADAPTER IDENTITY:", identity, flush=True)
        print(
            "VERIFIED CONTRACT:",
            verified["source_contract"],
            flush=True,
        )
        print(
            "PROVENANCE TYPE:",
            provenance.get("schema_type"),
            flush=True,
        )
        print(
            f"PREFLIGHT PASS: {dataset_id}",
            flush=True,
        )

    print(
        "\nFOUR-DATASET GOVERNED PROVENANCE PREFLIGHT: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
