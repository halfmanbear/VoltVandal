"""Experimental point locking using known NVAPI layouts, isolated in subprocesses.

See doc/point-lock.md for sources, aliases and validation limits. Importing this
module does not load NVAPI. No signature fuzzing or speculative writes are used.
"""

import ctypes
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


GET_LOCK = 0xE440B867
SET_LOCK = 0x39442CFB
GET_VOLTAGE = 0x465F9BCF
GET_BUS = 0x1BE0B8E5
INCOMPATIBLE_VERSION = -9
CORE_INDEX = 6


class PointLockError(RuntimeError):
    pass


class LockEntry(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("index", "unknown1", "mode", "unknown2", "voltage_uv", "unknown3")]


class LockState(ctypes.Structure):
    _pack_ = 8
    _fields_ = [("version", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("count", ctypes.c_uint32), ("entries", LockEntry * 32)]


class VoltageStatus(ctypes.Structure):
    # NvAPIWrapper PrivateVoltageStatusV1: 76 bytes, voltage at offset 40.
    _pack_ = 8
    _fields_ = [("version", ctypes.c_uint32), ("unknown1", ctypes.c_uint32),
                ("unknown2", ctypes.c_uint32 * 8), ("voltage_uv", ctypes.c_uint32),
                ("unknown3", ctypes.c_uint32 * 8)]


def _call(api, function_id, handle, value):
    fn = api._qif(function_id, ctypes.c_int,
                  [ctypes.c_void_p, ctypes.POINTER(type(value))])
    return fn(ctypes.c_void_p(handle), ctypes.byref(value))


def _core(state):
    if not 1 <= state.count <= 32:
        raise PointLockError(f"Invalid lock entry count: {state.count}")
    matches = [state.entries[i] for i in range(state.count)
               if state.entries[i].index == CORE_INDEX]
    if len(matches) != 1:
        raise PointLockError("Driver did not expose one recognised core lock entry")
    return matches[0]


def _read(api, handle):
    for version in (2, 1):
        state = LockState()
        state.version = ctypes.sizeof(state) | (version << 16)
        rc = _call(api, GET_LOCK, handle, state)
        if rc == INCOMPATIBLE_VERSION:
            continue
        if rc != 0:
            raise PointLockError(f"Lock getter failed: {rc}")
        if state.version != ctypes.sizeof(state) | (version << 16):
            raise PointLockError("Driver returned an unexpected lock structure version")
        _core(state)
        return state
    raise PointLockError("Neither known lock structure version is supported")


def _write_verified(api, handle, state):
    expected = _core(state)
    expected_pair = (expected.mode, expected.voltage_uv)
    rc = _call(api, SET_LOCK, handle, state)
    if rc != 0:
        raise PointLockError(f"Lock setter failed: {rc}")
    actual = _core(_read(api, handle))
    if (actual.mode, actual.voltage_uv) != expected_pair:
        raise PointLockError("Lock readback did not match the requested mode and voltage")


def _bus(api, handle):
    value = ctypes.c_uint32()
    rc = _call(api, GET_BUS, handle, value)
    if rc != 0:
        raise PointLockError(f"GPU bus lookup failed: {rc}")
    return value.value


def _voltage(api, handle):
    value = VoltageStatus()
    value.version = ctypes.sizeof(value) | (1 << 16)
    rc = _call(api, GET_VOLTAGE, handle, value)
    if rc != 0 or not 400_000 <= value.voltage_uv <= 1_500_000:
        raise PointLockError(f"Current-voltage reading unavailable (status {rc})")
    return value.voltage_uv / 1000.0


def _native(request):
    from . import nvapi as api

    gpu = request["gpu"]
    if not isinstance(gpu, int) or gpu < 0:
        raise PointLockError("GPU index must be non-negative")
    api._nvapi_init()
    handle = api._get_handle(gpu)
    bus = _bus(api, handle)
    if "bus" in request and request["bus"] != bus:
        raise PointLockError("GPU identity changed; refusing point-lock operation")
    operation = request["operation"]
    if operation == "voltage":
        return {"voltage_mv": _voltage(api, handle)}

    state = _read(api, handle)
    entry = _core(state)
    if operation == "inspect":
        if entry.mode not in (0, 3):
            raise PointLockError(f"Unrecognised pre-existing lock mode: {entry.mode}")
        # Resolve the setter, but never invoke it in this read-only operation.
        api._qif(SET_LOCK, ctypes.c_int,
                 [ctypes.c_void_p, ctypes.POINTER(LockState)])
        buses = [_bus(api, h) for h in api._nvapi_enum_gpus()]
        if buses.count(bus) != 1:
            raise PointLockError("Ambiguous PCI bus identity; point locking unavailable")
        return {"gpu": gpu, "bus": bus, "version": state.version >> 16,
                "mode": entry.mode, "voltage_uv": entry.voltage_uv,
                "snapshot": bytes(state).hex()}

    if operation == "lock":
        voltage = request["voltage_uv"]
        if not isinstance(voltage, int) or voltage <= 0:
            raise PointLockError("Lock voltage must be positive integer microvolts")
        _, curve, indices = api._read_active_bins(handle)
        if voltage not in {curve.clocks[i].voltageUV for i in indices}:
            raise PointLockError("Requested voltage is not an exposed core curve bin")
        if bytes(state).hex() != request["snapshot"]:
            raise PointLockError("Lock state changed since snapshot; refusing to overwrite it")
        if entry.mode not in (0, 3):
            raise PointLockError(f"Unrecognised pre-existing lock mode: {entry.mode}")
        entry.mode, entry.voltage_uv, entry.unknown3 = 3, voltage, 0
    elif operation == "restore":
        raw = bytes.fromhex(request["snapshot"])
        if len(raw) != ctypes.sizeof(LockState):
            raise PointLockError("Invalid lock recovery snapshot size")
        previous = LockState.from_buffer_copy(raw)
        if previous.version != state.version:
            raise PointLockError("Lock structure version changed since snapshot")
        old_entry = _core(previous)
        if old_entry.mode not in (0, 3):
            raise PointLockError("Recovery snapshot contains an unrecognised lock mode")
        # Restore only the entry we own; preserve other current driver controls.
        ctypes.memmove(ctypes.addressof(entry), ctypes.addressof(old_entry),
                       ctypes.sizeof(LockEntry))
    else:
        raise PointLockError(f"Unknown point-lock operation: {operation}")
    _write_verified(api, handle, state)
    return {"mode": entry.mode, "voltage_uv": entry.voltage_uv}


def _request(operation, gpu, timeout=10.0, **kwargs):
    # An actual process timeout, unlike a daemon thread that can write later.
    source_root = str(Path(__file__).resolve().parents[2])
    script = ("import sys; sys.path.insert(0, sys.argv[1]); "
              "from voltvandal.hardware.point_lock import _worker; _worker()")
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, source_root],
            input=json.dumps({"operation": operation, "gpu": gpu, **kwargs}),
            capture_output=True, text=True, timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as ex:
        raise PointLockError(f"Point-lock {operation} timed out; child terminated") from ex
    if result.returncode:
        raise PointLockError(f"Point-lock {operation} child failed ({result.returncode}): "
                             f"{result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
    except ValueError as ex:
        raise PointLockError("Invalid response from point-lock child") from ex
    if "error" in response:
        raise PointLockError(response["error"])
    return response


def inspect_point_lock(gpu):
    """Read-only capability/snapshot inspection; never invokes a setter."""
    return _request("inspect", gpu)


def read_voltage_mv(gpu, bus):
    return _request("voltage", gpu, timeout=2.0, bus=bus)["voltage_mv"]


def check_monitor_identity(gpu, bus):
    import pynvml

    try:
        pynvml.nvmlInit()
    except Exception as ex:
        raise PointLockError(f"NVML identity check unavailable: {ex}") from ex
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        if pci.domain != 0 or pci.bus != bus:
            raise PointLockError("NVAPI/NVML GPU identity mismatch; point test refused")
        return str(pynvml.nvmlDeviceGetUUID(handle))
    except PointLockError:
        raise
    except Exception as ex:
        raise PointLockError(f"NVML identity check failed: {ex}") from ex
    finally:
        pynvml.nvmlShutdown()


def restore_point_lock(journal):
    journal = Path(journal)
    previous = json.loads(journal.read_text(encoding="utf-8"))
    uuid = check_monitor_identity(previous["gpu"], previous["bus"])
    if uuid != previous["uuid"]:
        raise PointLockError("GPU UUID differs from recovery journal")
    _request("restore", previous["gpu"], bus=previous["bus"],
             snapshot=previous["snapshot"])
    journal.unlink()  # Retain recovery data on any failure above.


@contextmanager
def temporary_point_lock(gpu, voltage_uv, journal):
    journal = Path(journal)
    if journal.exists():
        raise PointLockError(f"Unresolved point-lock recovery: {journal}")
    previous = inspect_point_lock(gpu)
    previous["uuid"] = check_monitor_identity(gpu, previous["bus"])
    # Exclusive creation prevents accidentally replacing recovery data.
    with journal.open("x", encoding="utf-8") as stream:
        json.dump(previous, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _request("lock", gpu, bus=previous["bus"], snapshot=previous["snapshot"],
                 voltage_uv=voltage_uv)
        yield previous["bus"]
    finally:
        # Also restore after partial setter failure, readback mismatch or Ctrl+C.
        restore_point_lock(journal)


def _worker():
    try:
        print(json.dumps(_native(json.load(sys.stdin))))
    except Exception as ex:
        print(json.dumps({"error": str(ex)}))
