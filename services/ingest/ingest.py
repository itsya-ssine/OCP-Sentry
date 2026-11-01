"""Per-camera ffmpeg ingest.

For every enabled camera this spawns two long-running ffmpeg subprocesses
in their own threads:

1. `_detect_loop`  — pulls the low-res `detect_url` stream, decodes it to a
   steady trickle of MJPEG frames at `detection.frame_sample_fps`, and
   XADDs each frame (as raw JPEG bytes + timestamp) onto the Redis stream
   `frames:{camera_id}` for services/inference/inference_worker.py to
   consume.

2. `_record_loop`  — pulls the full-res `record_url` stream and muxes it
   straight to disk (no re-encode) as fixed-length segment files under
   storage/segments/{camera_id}/, which services/events/event_manager.py
   later stitches into event clips.

Both loops wrap their ffmpeg subprocess in a restart-with-backoff loop so
a camera reboot or network blip doesn't take the whole ingest service
down — only that camera's feed drops out until it reconnects.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.config import Camera, Settings, load_cameras, load_settings  # noqa: E402
from common.redis_client import frames_stream_key, get_redis  # noqa: E402

RESTART_BACKOFF_SECONDS = 5
MJPEG_SPLIT_BIN = str(Path(__file__).resolve().parent / "mjpeg_split")


def _read_exact(stream, n: int) -> bytes | None:
    """Reads exactly n bytes from `stream`, or None on EOF/short read
    (the pipe closed mid-record — treat like any other stream-ended case)."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _log(camera_id: str, msg: str) -> None:
    print(f"[ingest:{camera_id}] {msg}", flush=True)


def _ffmpeg_input_args(url: str) -> list[str]:
    """Builds the ffmpeg args to open `url`, based on a scheme prefix.

    Supports real cameras (rtsp://...) as well as local test sources:
      - "v4l2:/dev/video0"     Linux webcam/capture device
      - "avfoundation:0"       macOS webcam (device index or name)
      - anything else is passed to ffmpeg as a bare -i (works for local
        video files, http(s) streams, etc.)
    """
    if url.startswith("rtsp://"):
        return ["-rtsp_transport", "tcp", "-i", url]
    if url.startswith("v4l2:"):
        device = url.split(":", 1)[1]
        return ["-f", "v4l2", "-framerate", "30", "-video_size", "1280x720", "-i", device]
    if url.startswith("avfoundation:"):
        device = url.split(":", 1)[1]
        return ["-f", "avfoundation", "-framerate", "30", "-i", device]
    if url.startswith(("http://", "https://", "rtmp://")):
        return ["-i", url]
    # Anything else is treated as a local file (e.g. a test clip standing
    # in for a camera). Without -re, ffmpeg reads it as fast as the CPU
    # allows instead of at playback speed, so a short clip finishes almost
    # instantly and this process's restart-backoff loop just relaunches it
    # every RESTART_BACKOFF_SECONDS. -stream_loop -1 -re makes it loop
    # forever, paced in real time, so it behaves like an always-on camera.
    return ["-stream_loop", "-1", "-re", "-i", url]


def _detect_loop(camera: Camera, settings: Settings) -> None:
    """Continuously (re)launch ffmpeg -> mjpeg_split (see mjpeg_split.c) and
    push each length-prefixed JPEG it emits onto this camera's Redis frame
    stream. The C helper does the SOI/EOI scanning; this function just
    does two fixed-size reads per frame and an XADD."""
    r = get_redis(settings.redis)
    stream_key = frames_stream_key(camera.id)

    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        *_ffmpeg_input_args(camera.detect_url),
        "-vf", f"fps={settings.detection.frame_sample_fps}",
        "-f", "mjpeg",
        "-q:v", "5",
        "pipe:1",
    ]

    while True:
        _log(camera.id, "starting detect stream ffmpeg -> mjpeg_split")
        ffmpeg_proc = None
        split_proc = None
        try:
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            split_proc = subprocess.Popen(
                [MJPEG_SPLIT_BIN],
                stdin=ffmpeg_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            # mjpeg_split now holds the read end of ffmpeg's stdout pipe;
            # dropping our reference here means ffmpeg gets SIGPIPE (and
            # exits promptly) if mjpeg_split dies first, instead of
            # ffmpeg's output silently piling up with nothing draining it.
            ffmpeg_proc.stdout.close()

            assert split_proc.stdout is not None
            while True:
                header = _read_exact(split_proc.stdout, 4)
                if header is None:
                    break
                (frame_len,) = struct.unpack("<I", header)
                frame = _read_exact(split_proc.stdout, frame_len)
                if frame is None:
                    break

                ts = time.time()
                try:
                    r.xadd(
                        stream_key,
                        {"ts": str(ts).encode(), "camera_id": camera.id.encode(), "jpg": frame},
                        maxlen=settings.redis.frames_stream_maxlen,
                        approximate=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log(camera.id, f"redis xadd failed: {exc}")
        except FileNotFoundError as exc:
            _log(camera.id, f"binary not found ({exc}) — is ffmpeg/mjpeg_split installed in this image?")
            time.sleep(RESTART_BACKOFF_SECONDS)
        except Exception as exc:  # noqa: BLE001
            _log(camera.id, f"detect loop error: {exc}")
        finally:
            # Make sure both ends of the pipeline actually die before we
            # loop around and spawn fresh ones — a stuck/zombie process
            # here would otherwise pile up across restarts.
            for proc in (split_proc, ffmpeg_proc):
                if proc is not None and proc.poll() is None:
                    proc.kill()
            for proc in (ffmpeg_proc, split_proc):
                if proc is not None:
                    proc.wait()

        _log(camera.id, f"detect stream ffmpeg exited, restarting in {RESTART_BACKOFF_SECONDS}s")
        time.sleep(RESTART_BACKOFF_SECONDS)


def _record_loop(camera: Camera, settings: Settings) -> None:
    """Continuously (re)launch ffmpeg to segment the full-res stream to disk,
    no re-encoding, so segments are cheap to produce and cheap to concat later."""
    out_dir = settings.storage.segments_path() / camera.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "%Y%m%d_%H%M%S.mp4")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        *_ffmpeg_input_args(camera.record_url),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(settings.storage.segment_duration_seconds),
        "-reset_timestamps", "1",
        "-strftime", "1",
        out_pattern,
    ]

    while True:
        _log(camera.id, "starting record stream ffmpeg")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
        except FileNotFoundError:
            _log(camera.id, "ffmpeg binary not found — is it installed in this image?")
        except Exception as exc:  # noqa: BLE001
            _log(camera.id, f"record loop error: {exc}")

        _log(camera.id, f"record stream ffmpeg exited, restarting in {RESTART_BACKOFF_SECONDS}s")
        time.sleep(RESTART_BACKOFF_SECONDS)


def _cleanup_loop(settings: Settings, cameras: list[Camera]) -> None:
    """Deletes segment files older than segment_retention_hours so disk
    usage stays bounded across 30 cameras recording continuously."""
    retention = settings.storage.segment_retention_hours * 3600
    while True:
        time.sleep(300)
        cutoff = time.time() - retention
        for camera in cameras:
            cam_dir = settings.storage.segments_path() / camera.id
            if not cam_dir.exists():
                continue
            for f in cam_dir.glob("*.mp4"):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass


def main() -> None:
    settings = load_settings()
    cameras = [c for c in load_cameras() if c.enabled]

    if not cameras:
        print("[ingest] no enabled cameras in config/cameras.yaml, exiting", flush=True)
        return

    print(f"[ingest] starting ingest for {len(cameras)} camera(s)", flush=True)

    threads = []
    for camera in cameras:
        t1 = threading.Thread(target=_detect_loop, args=(camera, settings), daemon=True)
        t1.start()
        threads.append(t1)

        if camera.record_enabled:
            t2 = threading.Thread(target=_record_loop, args=(camera, settings), daemon=True)
            t2.start()
            threads.append(t2)
        else:
            _log(camera.id, "record_enabled: false, skipping continuous recording "
                             "(events will still fire, just without a stitched clip)")

    cleanup = threading.Thread(target=_cleanup_loop, args=(settings, cameras), daemon=True)
    cleanup.start()
    threads.append(cleanup)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
