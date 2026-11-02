"""Synthetic regression tests for the VAE checkpoint NPZ codec."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import warnings

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_npz as codec,
)


def make_case(role="encoder"):
    specification = (
        {
            "key": f"{role}_0000",
            "shape": (2, 3),
            "dtype": "float32",
            "trainable": True,
        },
        {
            "key": f"{role}_0001",
            "shape": (3,),
            "dtype": "float32",
            "trainable": False,
        },
    )
    state = {
        f"{role}_0000": np.arange(6, dtype=np.float32).reshape(2, 3),
        f"{role}_0001": np.array([1., 2., 3.], dtype=np.float32),
    }
    return specification, state


def archive_with_members(members, compression=ZIP_STORED):
    output = BytesIO()
    with ZipFile(output, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return output.getvalue()


def npy_bytes(array):
    output = BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_valid_round_trip(role):
    specification, state = make_case(role)
    raw = codec.encode_network_state(
        role=role, state=state, specification=specification
    )
    restored = codec.decode_network_state(
        role=role, raw=raw, specification=specification
    )
    assert set(restored) == set(state)
    for key in state:
        np.testing.assert_array_equal(restored[key], state[key])


def test_encoding_is_deterministic():
    specification, state = make_case()
    first = codec.encode_network_state(
        role="encoder", state=state, specification=specification
    )
    second = codec.encode_network_state(
        role="encoder", state=state, specification=specification
    )
    assert first == second


@pytest.mark.parametrize("role", ["encoder", "decoder"])
def test_missing_tensor_rejected(role):
    specification, state = make_case(role)
    state.pop(f"{role}_0001")
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role=role, state=state, specification=specification
        )


def test_extra_tensor_rejected():
    specification, state = make_case()
    state["encoder_9999"] = np.zeros(1, np.float32)
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role="encoder", state=state, specification=specification
        )


@pytest.mark.parametrize("invalid", [np.float64, np.int32])
def test_wrong_tensor_dtype_rejected(invalid):
    specification, state = make_case()
    state["encoder_0000"] = state["encoder_0000"].astype(invalid)
    with pytest.raises(TypeError):
        codec.encode_network_state(
            role="encoder", state=state, specification=specification
        )


def test_wrong_tensor_shape_rejected():
    specification, state = make_case()
    state["encoder_0000"] = state["encoder_0000"].reshape(6)
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role="encoder", state=state, specification=specification
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_tensor_rejected(invalid):
    specification, state = make_case()
    state["encoder_0000"].flat[0] = invalid
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role="encoder", state=state, specification=specification
        )


def test_duplicate_archive_member_rejected():
    specification, state = make_case()
    payload = npy_bytes(state["encoder_0000"])
    other = npy_bytes(state["encoder_0001"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = archive_with_members([
            ("encoder_0000.npy", payload),
            ("encoder_0000.npy", payload),
            ("encoder_0001.npy", other),
        ])
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_missing_archive_member_rejected():
    specification, state = make_case()
    raw = archive_with_members([
        ("encoder_0000.npy", npy_bytes(state["encoder_0000"]))
    ])
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_unexpected_archive_member_rejected():
    specification, state = make_case()
    raw = archive_with_members([
        ("encoder_0000.npy", npy_bytes(state["encoder_0000"])),
        ("unexpected.npy", npy_bytes(state["encoder_0001"])),
    ])
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_compressed_member_rejected():
    specification, state = make_case()
    raw = archive_with_members([
        ("encoder_0000.npy", npy_bytes(state["encoder_0000"])),
        ("encoder_0001.npy", npy_bytes(state["encoder_0001"])),
    ], compression=ZIP_DEFLATED)
    with pytest.raises(ValueError, match="compressed"):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_invalid_npy_payload_rejected():
    specification, state = make_case()
    raw = archive_with_members([
        ("encoder_0000.npy", b"not-a-valid-npy-array"),
        ("encoder_0001.npy", npy_bytes(state["encoder_0001"])),
    ])
    with pytest.raises((ValueError, EOFError)):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_object_array_rejected():
    specification, state = make_case()
    output = BytesIO()
    np.save(
        output,
        np.array([{"unsafe": True}], dtype=object),
        allow_pickle=True,
    )
    raw = archive_with_members([
        ("encoder_0000.npy", output.getvalue()),
        ("encoder_0001.npy", npy_bytes(state["encoder_0001"])),
    ])
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_archive_size_limit_rejected(monkeypatch):
    specification, state = make_case()
    raw = codec.encode_network_state(
        role="encoder", state=state, specification=specification
    )
    monkeypatch.setattr(codec, "MAX_ARCHIVE_BYTES", 8)
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_member_size_limit_rejected(monkeypatch):
    specification, state = make_case()
    raw = codec.encode_network_state(
        role="encoder", state=state, specification=specification
    )
    monkeypatch.setattr(codec, "MAX_MEMBER_BYTES", 12)
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


def test_total_member_size_limit_rejected(monkeypatch):
    specification, state = make_case()
    raw = codec.encode_network_state(
        role="encoder", state=state, specification=specification
    )
    monkeypatch.setattr(codec, "MAX_TOTAL_MEMBER_BYTES", 16)
    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder", raw=raw, specification=specification
        )


@pytest.mark.parametrize("invalid", [None, [], "not-bytes"])
def test_nonbytes_archive_rejected(invalid):
    specification, _ = make_case()
    with pytest.raises(TypeError):
        codec.decode_network_state(
            role="encoder", raw=invalid, specification=specification
        )


def test_wrong_role_specification_rejected():
    specification, state = make_case("encoder")
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role="decoder", state=state, specification=specification
        )


def test_invalid_specification_dtype_rejected():
    specification, state = make_case()
    specification[0]["dtype"] = "float64"
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role="encoder", state=state, specification=specification
        )
