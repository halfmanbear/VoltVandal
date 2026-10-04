import ctypes
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voltvandal.hardware import point_lock as lock


def state(version=2):
    value = lock.LockState()
    value.version = ctypes.sizeof(value) | (version << 16)
    value.count = 2
    value.entries[0].index = 6
    value.entries[1].index = 9
    value.entries[1].mode = 7
    value.entries[1].unknown2 = 1234
    return value


def test_reference_abi_sizes_and_offsets():
    assert ctypes.sizeof(lock.LockEntry) == 24
    assert ctypes.sizeof(lock.LockState) == 780
    assert lock.LockState.entries.offset == 12
    assert ctypes.sizeof(lock.VoltageStatus) == 76
    assert lock.VoltageStatus.voltage_uv.offset == 40


def test_getter_negotiates_only_known_versions(monkeypatch):
    versions = []

    def call(api, fid, handle, value):
        versions.append(value.version >> 16)
        if versions[-1] == 2:
            return -9
        value.count = 1
        value.entries[0].index = 6
        return 0

    monkeypatch.setattr(lock, "_call", call)
    assert lock._read(None, 1).version >> 16 == 1
    assert versions == [2, 1]


@pytest.mark.parametrize("status", [-1, -104, -175])
def test_getter_does_not_guess_after_other_errors(monkeypatch, status):
    call = Mock(return_value=status)
    monkeypatch.setattr(lock, "_call", call)
    with pytest.raises(lock.PointLockError):
        lock._read(None, 1)
    assert call.call_count == 1


@pytest.mark.parametrize("count", [0, 33, 0xFFFFFFFF])
def test_bad_counts_are_rejected(count):
    value = state()
    value.count = count
    with pytest.raises(lock.PointLockError):
        lock._core(value)


def test_write_requires_matching_readback(monkeypatch):
    expected, actual = state(), state()
    expected.entries[0].mode = 3
    expected.entries[0].voltage_uv = 900000
    monkeypatch.setattr(lock, "_call", Mock(return_value=0))
    monkeypatch.setattr(lock, "_read", Mock(return_value=actual))
    with pytest.raises(lock.PointLockError, match="readback"):
        lock._write_verified(None, 1, expected)


def mock_native(monkeypatch):
    from voltvandal.hardware import nvapi

    monkeypatch.setattr(nvapi, "_nvapi_init", Mock())
    monkeypatch.setattr(nvapi, "_get_handle", Mock(return_value=1))
    monkeypatch.setattr(nvapi, "_read_active_bins", Mock(return_value=(
        None, SimpleNamespace(clocks=[SimpleNamespace(voltageUV=900000)]), [0])))
    monkeypatch.setattr(lock, "_bus", Mock(return_value=3))
    return nvapi


def test_lock_preserves_other_controls(monkeypatch):
    mock_native(monkeypatch)
    original = state()
    snapshot = bytes(original).hex()
    monkeypatch.setattr(lock, "_read", Mock(return_value=original))
    write = Mock()
    monkeypatch.setattr(lock, "_write_verified", write)
    lock._native(dict(operation="lock", gpu=0, bus=3, snapshot=snapshot, voltage_uv=900000))
    assert (original.entries[0].mode, original.entries[0].voltage_uv) == (3, 900000)
    assert original.entries[1].mode == 7
    assert original.entries[1].unknown2 == 1234
    write.assert_called_once()


def test_unknown_voltage_never_writes(monkeypatch):
    mock_native(monkeypatch)
    original = state()
    monkeypatch.setattr(lock, "_read", Mock(return_value=original))
    write = Mock()
    monkeypatch.setattr(lock, "_write_verified", write)
    with pytest.raises(lock.PointLockError, match="exposed core curve bin"):
        lock._native(dict(operation="lock", gpu=0, bus=3,
                          snapshot=bytes(original).hex(), voltage_uv=975000))
    write.assert_not_called()


def test_restore_preserves_other_current_entries(monkeypatch):
    mock_native(monkeypatch)
    previous, current = state(), state()
    previous.entries[0].mode = 3
    previous.entries[0].voltage_uv = 875000
    current.entries[1].unknown2 = 999
    monkeypatch.setattr(lock, "_read", Mock(return_value=current))
    monkeypatch.setattr(lock, "_write_verified", Mock())
    lock._native(dict(operation="restore", gpu=0, bus=3, snapshot=bytes(previous).hex()))
    assert current.entries[0].voltage_uv == 875000
    assert current.entries[0].mode == 3
    assert current.entries[1].unknown2 == 999


def prepare_context(monkeypatch):
    previous = dict(gpu=0, bus=3, snapshot=bytes(state()).hex())
    monkeypatch.setattr(lock, "inspect_point_lock", Mock(return_value=previous))
    monkeypatch.setattr(lock, "check_monitor_identity", Mock(return_value="GPU-test"))
    request = Mock(return_value={})
    monkeypatch.setattr(lock, "_request", request)
    return request


@pytest.mark.parametrize("interruption", [None, RuntimeError("stress crashed"), KeyboardInterrupt()])
def test_context_restores_on_every_exit(monkeypatch, tmp_path, interruption):
    request = prepare_context(monkeypatch)
    journal = tmp_path / "recovery.json"

    def run():
        with lock.temporary_point_lock(0, 900000, journal) as bus:
            assert bus == 3
            assert journal.exists()
            if interruption is not None:
                raise interruption

    if interruption is None:
        run()
    else:
        with pytest.raises(type(interruption)):
            run()
    assert [call.args[0] for call in request.call_args_list] == ["lock", "restore"]
    assert not journal.exists()


def test_partial_apply_failure_also_restores(monkeypatch, tmp_path):
    request = prepare_context(monkeypatch)
    request.side_effect = [lock.PointLockError("readback mismatch"), {}]
    journal = tmp_path / "recovery.json"
    with pytest.raises(lock.PointLockError, match="readback"):
        with lock.temporary_point_lock(0, 900000, journal):
            pytest.fail("Must not run workload")
    assert request.call_count == 2
    assert not journal.exists()


def test_failed_restore_keeps_journal_and_blocks_new_lock(monkeypatch, tmp_path):
    request = prepare_context(monkeypatch)
    request.side_effect = [{}, lock.PointLockError("restore failed")]
    journal = tmp_path / "recovery.json"
    with pytest.raises(lock.PointLockError, match="restore failed"):
        with lock.temporary_point_lock(0, 900000, journal):
            pass
    assert json.loads(journal.read_text())["uuid"] == "GPU-test"
    with pytest.raises(lock.PointLockError, match="Unresolved"):
        with lock.temporary_point_lock(0, 900000, journal):
            pytest.fail("Must not overwrite recovery")
    assert request.call_count == 2


def test_worker_timeout_is_reported(monkeypatch):
    monkeypatch.setattr(lock.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("child", 1)))
    with pytest.raises(lock.PointLockError, match="timed out"):
        lock.inspect_point_lock(0)


def test_inspect_is_read_only(monkeypatch):
    api = mock_native(monkeypatch)
    monkeypatch.setattr(api, "_nvapi_enum_gpus", Mock(return_value=[1]))
    monkeypatch.setattr(api, "_qif", Mock())
    monkeypatch.setattr(lock, "_read", Mock(return_value=state()))
    write = Mock()
    monkeypatch.setattr(lock, "_write_verified", write)
    assert lock._native(dict(operation="inspect", gpu=0))["version"] == 2
    write.assert_not_called()


def test_negative_gpu_rejected_before_loading_driver(monkeypatch):
    api = mock_native(monkeypatch)
    with pytest.raises(lock.PointLockError, match="non-negative"):
        lock._native(dict(operation="inspect", gpu=-1))
    api._nvapi_init.assert_not_called()
