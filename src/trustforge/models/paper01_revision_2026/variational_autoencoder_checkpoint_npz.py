"""Bounded NPZ state codec for Paper 1 revision-2026 VAE.

This module operates exclusively on in-memory bytes and explicit
tensor specifications. It does not access checkpoint paths, import
TensorFlow, mutate models, or authorize training.

Format candidate: ZIP_STORED members containing float32 NPY arrays.
"""

from __future__ import annotations

from io import BytesIO
import re
from zipfile import (
    BadZipFile,
    ZIP_STORED,
    ZipFile,
    ZipInfo,
)

import numpy as np


MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 48 * 1024 * 1024
MAX_MEMBERS = 64
MAX_HEADER_BYTES = 10000

EXPECTED_DTYPE = np.dtype("float32")
KEY_PATTERN = re.compile(r"^(encoder|decoder)_[0-9]{4}$")


def _specification(specification, *, role):
    """Validate an exact, ordered tensor specification."""
    if role not in ("encoder", "decoder"):
        raise ValueError("unsupported network role")

    if not isinstance(specification, (list, tuple)):
        raise TypeError("specification must be a sequence")

    if not 0 < len(specification) <= MAX_MEMBERS:
        raise ValueError("invalid specification length")

    validated = {}

    for index, entry in enumerate(specification):
        if type(entry) is not dict or set(entry) != {
            "key", "shape", "dtype", "trainable"
        }:
            raise ValueError("invalid tensor specification entry")

        key = entry["key"]
        expected_key = f"{role}_{index:04d}"

        if (
            type(key) is not str
            or KEY_PATTERN.fullmatch(key) is None
            or key != expected_key
        ):
            raise ValueError("invalid ordered tensor identifier")

        shape = entry["shape"]
        if (
            not isinstance(shape, (tuple, list))
            or not shape
            or any(type(d) is not int or d < 1 for d in shape)
        ):
            raise ValueError("invalid tensor shape specification")

        if entry["dtype"] != "float32":
            raise ValueError("unsupported tensor dtype specification")

        if type(entry["trainable"]) is not bool:
            raise ValueError("invalid tensor trainability specification")

        count = 1
        for dimension in shape:
            count *= dimension
            if count * EXPECTED_DTYPE.itemsize > MAX_MEMBER_BYTES:
                raise ValueError("tensor exceeds member limit")

        validated[key] = tuple(shape)

    return validated


def _validate_array(key, array, expected_shape):
    if type(array) is not np.ndarray:
        raise TypeError(f"tensor must be ndarray: {key}")

    if array.dtype != EXPECTED_DTYPE:
        raise TypeError(f"tensor dtype mismatch: {key}")

    if array.shape != expected_shape:
        raise ValueError(f"tensor shape mismatch: {key}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"nonfinite tensor: {key}")


def encode_network_state(*, role, state, specification):
    """Encode an exact tensor inventory to deterministic NPZ bytes."""
    expected = _specification(specification, role=role)

    if type(state) is not dict or set(state) != set(expected):
        raise ValueError("network tensor inventory mismatch")

    output = BytesIO()

    with ZipFile(output, mode="w", compression=ZIP_STORED) as archive:
        for key, shape in expected.items():
            array = state[key]
            _validate_array(key, array, shape)

            member_buffer = BytesIO()
            np.lib.format.write_array(
                member_buffer,
                np.ascontiguousarray(array),
                allow_pickle=False,
            )
            payload = member_buffer.getvalue()

            if len(payload) > MAX_MEMBER_BYTES:
                raise ValueError("encoded tensor exceeds member limit")

            info = ZipInfo(f"{key}.npy")
            info.compress_type = ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16

            archive.writestr(info, payload)

            if output.tell() > MAX_ARCHIVE_BYTES:
                raise ValueError("encoded archive exceeds limit")

    raw = output.getvalue()

    if not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
        raise ValueError("invalid encoded archive size")

    return raw


def decode_network_state(*, role, raw, specification):
    """Decode an exact, bounded, pickle-free NPZ state inventory."""
    expected = _specification(specification, role=role)

    if type(raw) is not bytes:
        raise TypeError("archive must be bytes")

    if not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
        raise ValueError("invalid archive size")

    result = {}

    try:
        with ZipFile(BytesIO(raw), mode="r") as archive:
            members = archive.infolist()

            if len(members) != len(expected):
                raise ValueError("archive member count mismatch")

            if len(members) > MAX_MEMBERS:
                raise ValueError("too many archive members")

            names = [member.filename for member in members]
            required = {f"{key}.npy" for key in expected}

            if len(names) != len(set(names)):
                raise ValueError("duplicate archive member")

            if set(names) != required:
                raise ValueError("unexpected or missing archive member")

            total = 0

            for member in members:
                if member.is_dir():
                    raise ValueError("directory member prohibited")

                if member.compress_type != ZIP_STORED:
                    raise ValueError("compressed archive member prohibited")

                if member.flag_bits & 0x1:
                    raise ValueError("encrypted archive member prohibited")

                if not 0 < member.file_size <= MAX_MEMBER_BYTES:
                    raise ValueError("invalid member size")

                if member.compress_size != member.file_size:
                    raise ValueError("invalid stored member size")

                total += member.file_size
                if total > MAX_TOTAL_MEMBER_BYTES:
                    raise ValueError("archive payload exceeds limit")

            for key, shape in expected.items():
                name = f"{key}.npy"
                member = archive.getinfo(name)

                with archive.open(member, "r") as handle:
                    payload = handle.read(MAX_MEMBER_BYTES + 1)

                if len(payload) != member.file_size:
                    raise ValueError("archive member size mismatch")

                stream = BytesIO(payload)
                array = np.load(
                    stream,
                    allow_pickle=False,
                    max_header_size=MAX_HEADER_BYTES,
                )

                if not isinstance(array, np.ndarray):
                    raise ValueError("invalid NPY array")

                if stream.tell() != len(payload):
                    raise ValueError("trailing NPY payload bytes")

                _validate_array(key, array, shape)
                result[key] = array.copy()

    except (BadZipFile, EOFError, OSError) as exc:
        raise ValueError("invalid NPZ archive") from exc

    # Require the entire archive to match the canonical
    # TrustForge NPZ representation, not merely contain
    # decodable tensor members.
    #
    # This rejects prefixed/trailing bytes, alternate ZIP
    # layouts, unexpected ZIP metadata, and other
    # noncanonical representations.
    canonical = encode_network_state(
        role=role,
        state=result,
        specification=specification,
    )

    if raw != canonical:
        raise ValueError(
            "noncanonical VAE checkpoint NPZ archive"
        )

    return result


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_MEMBER_BYTES",
    "MAX_TOTAL_MEMBER_BYTES",
    "MAX_MEMBERS",
    "encode_network_state",
    "decode_network_state",
]
