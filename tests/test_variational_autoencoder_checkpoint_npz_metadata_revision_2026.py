"""Adversarial NPZ metadata regression tests.

All archives are synthetic and held in memory.
"""

from io import BytesIO
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_npz as codec,
)


SPECIFICATION = (
    {
        "key": "encoder_0000",
        "shape": (2, 3),
        "dtype": "float32",
        "trainable": True,
    },
)


def canonical_archive():
    state = {
        "encoder_0000": np.arange(
            6, dtype=np.float32
        ).reshape(2, 3),
    }
    return codec.encode_network_state(
        role="encoder",
        state=state,
        specification=SPECIFICATION,
    )


def decode(raw):
    return codec.decode_network_state(
        role="encoder",
        raw=raw,
        specification=SPECIFICATION,
    )


def rebuild_archive(
    *,
    comment=b"",
    permissions=0o600,
    member_comment=b"",
    extra=b"",
):
    original = canonical_archive()

    with ZipFile(BytesIO(original), "r") as source:
        payload = source.read("encoder_0000.npy")

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
        compression=ZIP_STORED,
    ) as target:
        info = ZipInfo("encoder_0000.npy")
        info.compress_type = ZIP_STORED
        info.create_system = 3
        info.external_attr = permissions << 16
        info.comment = member_comment
        info.extra = extra

        target.writestr(info, payload)
        target.comment = comment

    return output.getvalue()


def test_canonical_metadata_accepted():
    decoded = decode(canonical_archive())
    assert decoded["encoder_0000"].shape == (2, 3)


@pytest.mark.parametrize(
    "comment",
    [b"COMMENT", b"\x00" * 8],
)
def test_zip_archive_comment_rejected(comment):
    with pytest.raises(ValueError):
        decode(rebuild_archive(comment=comment))


def test_modified_member_permissions_rejected():
    with pytest.raises(ValueError):
        decode(rebuild_archive(permissions=0o644))


def test_member_comment_rejected():
    with pytest.raises(ValueError):
        decode(rebuild_archive(member_comment=b"EXTRA"))


def test_member_extra_field_rejected():
    with pytest.raises(ValueError):
        decode(
            rebuild_archive(
                extra=b"\x01\x00\x00\x00"
            )
        )


def test_corrupted_npy_magic_rejected():
    raw = canonical_archive()

    marker = b"\x93NUMPY"
    assert raw.count(marker) == 1

    corrupted = raw.replace(
        marker,
        b"\x93BADPY",
        1,
    )

    with pytest.raises(ValueError):
        decode(corrupted)


def test_corrupted_npy_header_rejected():
    raw = canonical_archive()

    marker = b"'descr': '<f4'"
    assert raw.count(marker) == 1

    corrupted = raw.replace(
        marker,
        b"'descr': '<f8'",
        1,
    )

    with pytest.raises(ValueError):
        decode(corrupted)
