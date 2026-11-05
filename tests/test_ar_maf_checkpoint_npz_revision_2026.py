from io import BytesIO
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_npz as codec


def fixture(count=2):
    specification = [
        {
            "key": f"model_{i:04d}",
            "shape": [2],
            "dtype": "float32",
            "trainable": i % 2 == 0,
        }
        for i in range(count)
    ]
    state = {
        entry["key"]: np.array([i, i + 0.5], dtype=np.float32)
        for i, entry in enumerate(specification)
    }
    return specification, state


def encode(specification, state):
    return codec.encode_network_state(
        role="model", state=state, specification=specification,
    )


def decode(raw, specification):
    return codec.decode_network_state(
        role="model", raw=raw, specification=specification,
    )


@pytest.mark.parametrize("count", [2, 65, 512])
def test_exact_deterministic_round_trip(count):
    specification, state = fixture(count)
    raw = encode(specification, state)
    assert raw == encode(specification, state)
    restored = decode(raw, specification)
    assert list(restored) == list(state)
    for key in state:
        np.testing.assert_array_equal(restored[key], state[key])
        assert restored[key].dtype == np.float32
        assert not np.shares_memory(restored[key], state[key])


@pytest.mark.parametrize("defect", [
    "missing", "extra", "shape", "dtype", "nonfinite",
])
def test_invalid_state_rejected(defect):
    specification, state = fixture()
    if defect == "missing":
        del state["model_0001"]
    elif defect == "extra":
        state["unexpected"] = np.zeros(2, dtype=np.float32)
    elif defect == "shape":
        state["model_0000"] = np.zeros(3, dtype=np.float32)
    elif defect == "dtype":
        state["model_0000"] = state["model_0000"].astype(np.float64)
    else:
        state["model_0000"][0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        encode(specification, state)


def test_wrong_expected_shape_rejected_on_decode():
    specification, state = fixture()
    raw = encode(specification, state)
    specification[0]["shape"] = [3]
    with pytest.raises(ValueError):
        decode(raw, specification)


@pytest.mark.parametrize("alteration", ["prefix", "suffix", "truncate"])
def test_noncanonical_or_truncated_archive_rejected(alteration):
    specification, state = fixture()
    raw = encode(specification, state)
    if alteration == "prefix":
        raw = b"unexpected" + raw
    elif alteration == "suffix":
        raw += b"unexpected"
    else:
        raw = raw[:-20]
    with pytest.raises(ValueError):
        decode(raw, specification)


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "compressed"])
def test_invalid_zip_inventory_rejected(defect):
    specification, state = fixture()
    raw = encode(specification, state)
    with ZipFile(BytesIO(raw)) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    if defect == "missing":
        members.pop()
    elif defect == "extra":
        members.append(("unexpected.npy", members[0][1]))
    elif defect == "duplicate":
        members[1] = members[0]

    output = BytesIO()
    compression = ZIP_DEFLATED if defect == "compressed" else ZIP_STORED
    with ZipFile(output, "w", compression=compression) as archive:
        for name, payload in members:
            if defect == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    # Write both copies together below instead.
                    archive.writestr(name, payload)
                    archive.writestr(name, payload)
                break
            archive.writestr(name, payload)
    with pytest.raises(ValueError):
        decode(output.getvalue(), specification)


def test_archive_limit_enforced(monkeypatch):
    specification, state = fixture()
    raw = encode(specification, state)
    monkeypatch.setattr(codec, "MAX_ARCHIVE_BYTES", len(raw) - 1)
    with pytest.raises(ValueError):
        decode(raw, specification)
    with pytest.raises(ValueError):
        encode(specification, state)


def test_member_limit_enforced(monkeypatch):
    specification, state = fixture()
    monkeypatch.setattr(codec, "MAX_MEMBER_BYTES", 4)
    with pytest.raises(ValueError):
        encode(specification, state)


def test_tensor_count_limit_enforced():
    specification, state = fixture(513)
    with pytest.raises(ValueError):
        encode(specification, state)


@pytest.mark.parametrize("role", ["encoder", "decoder", "ar", "maf"])
def test_non_model_roles_rejected(role):
    specification, state = fixture()
    with pytest.raises(ValueError):
        codec.encode_network_state(
            role=role, state=state, specification=specification,
        )


def test_total_payload_limit_enforced_in_both_directions(monkeypatch):
    specification, state = fixture()
    raw = encode(specification, state)
    monkeypatch.setattr(codec, "MAX_TOTAL_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="payload exceeds"):
        encode(specification, state)
    with pytest.raises(ValueError, match="payload exceeds"):
        decode(raw, specification)
