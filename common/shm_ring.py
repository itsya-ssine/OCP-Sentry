"""ctypes wrapper around common/shm_ring.c — a lock-free (seqlock) shared
memory "latest frame" slot per camera. Optional, opt-in low-latency
alternative to routing raw frames through Redis between services/ingest
and services/inference when both run on the same host — Redis remains the
transport for detections/events regardless (low-volume, benefits from
persistence + consumer groups; this is only for the high-volume raw-frame
handoff). See shm_ring.h for the full design rationale.

NOT wired into ingest.py / inference_worker.py by default — this ships as
a standalone, tested building block. Swapping the frame transport is an
architecture change worth doing deliberately (see the docstring at the
bottom of this file for the integration sketch), not something to flip on
silently under an existing working deployment.

Requires libshmring.so to be built alongside this file:
    cc -O2 -fPIC -shared -o common/libshmring.so common/shm_ring.c -lrt -pthread
(Dockerfile.base should do this at image build time if/when this gets
wired in — see the integration notes at the bottom of this file.)
"""
from __future__ import annotations

import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).resolve().parent / "libshmring.so"

MAX_FRAME = 512 * 1024


class _Slot(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("frame_id", ctypes.c_uint64),
        ("ts", ctypes.c_double),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("jpg_len", ctypes.c_uint32),
        ("jpg", ctypes.c_ubyte * MAX_FRAME),
    ]


def _load_lib() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        raise FileNotFoundError(
            f"{_LIB_PATH} not found. Build it first:\n"
            f"  cc -O2 -fPIC -shared -o {_LIB_PATH} "
            f"{_LIB_PATH.with_name('shm_ring.c')} -lrt -pthread"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.shm_ring_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(_Slot))]
    lib.shm_ring_create.restype = ctypes.c_int
    lib.shm_ring_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(_Slot))]
    lib.shm_ring_open.restype = ctypes.c_int
    lib.shm_ring_close.argtypes = [ctypes.POINTER(_Slot)]
    lib.shm_ring_unlink.argtypes = [ctypes.c_char_p]
    lib.shm_ring_write.argtypes = [
        ctypes.POINTER(_Slot), ctypes.c_double, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_uint32,
    ]
    lib.shm_ring_read_if_newer.argtypes = [
        ctypes.POINTER(_Slot), ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
    ]
    lib.shm_ring_read_if_newer.restype = ctypes.c_int
    return lib


_lib = _load_lib()


def _shm_name(camera_id: str) -> bytes:
    # shm_open names must start with '/' and contain no other '/'.
    safe = camera_id.replace("/", "_")
    return f"/nvr_frame_{safe}".encode()


class FrameBusWriter:
    """Held by services/ingest, one per camera. Each write() overwrites the
    single latest-frame slot — there is no backlog by design."""

    def __init__(self, camera_id: str):
        self._camera_id = camera_id
        self._slot_ptr = ctypes.POINTER(_Slot)()
        rc = _lib.shm_ring_create(_shm_name(camera_id), ctypes.byref(self._slot_ptr))
        if rc != 0:
            raise OSError(f"shm_ring_create failed for camera {camera_id!r} (rc={rc})")

    def write(self, ts: float, jpg: bytes, width: int = 0, height: int = 0) -> None:
        if len(jpg) > MAX_FRAME:
            raise ValueError(f"frame ({len(jpg)} bytes) exceeds shm bus MAX_FRAME ({MAX_FRAME}); raise SHM_RING_MAX_FRAME in shm_ring.h and rebuild")
        _lib.shm_ring_write(self._slot_ptr, ctypes.c_double(ts), width, height, jpg, len(jpg))

    def close(self, unlink: bool = True) -> None:
        _lib.shm_ring_close(self._slot_ptr)
        if unlink:
            _lib.shm_ring_unlink(_shm_name(self._camera_id))


class FrameBusReader:
    """Held by services/inference, one per camera it's watching. Polls for
    a frame newer than the last one it saw."""

    def __init__(self, camera_id: str):
        self._slot_ptr = ctypes.POINTER(_Slot)()
        rc = _lib.shm_ring_open(_shm_name(camera_id), ctypes.byref(self._slot_ptr))
        if rc != 0:
            raise OSError(
                f"shm_ring_open failed for camera {camera_id!r} (rc={rc}) — "
                f"has services/ingest created this camera's slot yet? Retry/backoff "
                f"rather than treating this as fatal at startup."
            )
        self._last_seen = 0

    def read_latest(self):
        """Returns (frame_id, ts, jpg_bytes) if a newer frame is available,
        else None. Non-blocking — call in a poll loop."""
        frame_id = ctypes.c_uint64()
        ts = ctypes.c_double()
        width = ctypes.c_uint32()
        height = ctypes.c_uint32()
        jpg_len = ctypes.c_uint32()
        buf = ctypes.create_string_buffer(MAX_FRAME)

        got = _lib.shm_ring_read_if_newer(
            self._slot_ptr, self._last_seen,
            ctypes.byref(frame_id), ctypes.byref(ts),
            ctypes.byref(width), ctypes.byref(height),
            buf, ctypes.byref(jpg_len), MAX_FRAME,
        )
        if not got:
            return None
        self._last_seen = frame_id.value
        return frame_id.value, ts.value, buf.raw[: jpg_len.value]

    def close(self) -> None:
        _lib.shm_ring_close(self._slot_ptr)


# --- Integration sketch (not applied) ------------------------------------
#
# To actually route frames through this instead of Redis:
#
#   services/ingest/ingest.py, _detect_loop():
#     writer = FrameBusWriter(camera.id)
#     ... on each frame from mjpeg_split: writer.write(ts, frame)
#     (keep the existing r.xadd too, or drop it — but if inference no
#     longer reads frames:{camera_id} from Redis, stop writing it there
#     to avoid an ever-growing unread stream)
#
#   services/inference/inference_worker.py, main():
#     readers = {c.id: FrameBusReader(c.id) for c in cameras}
#     ... replace the r.xread(...) loop with a poll loop over readers,
#     e.g. round-robin with a short sleep when nothing's new, calling
#     reader.read_latest() per camera instead of blocking on Redis
#
# This is a real architecture change (frame delivery becomes "poll for
# latest" instead of "queue of everything sampled"), so it changes
# behavior under load: with Redis + XADD, a slow inference worker still
# eventually sees every sampled frame (up to frames_stream_maxlen); with
# this shared-memory bus, a slow inference worker skips straight to
# whatever's newest and silently drops frames in between — which is
# exactly the right tradeoff for a live detector (never fall behind and
# process stale video) but worth doing deliberately with that tradeoff in
# mind, not as a silent swap.
