"""Detection debounce state machine + clip stitching + Postgres writes.

Consumes services/inference/inference_worker.py's `detections` Redis
stream (via a consumer group, so restarts don't lose or duplicate
in-flight detections) and, per (camera, label):

  - requires `events.debounce_frames` consecutive detections before
    opening an event, so a single false-positive frame doesn't create an
    alert;
  - keeps the event open as long as matching detections keep arriving;
  - closes it after `events.cooldown_seconds` of silence;
  - on close, stitches the relevant recorded segments (plus pre/post-roll)
    into a single clip with ffmpeg, saves the best-scoring frame as the
    thumbnail, writes the event row to Postgres, and publishes it on the
    `events` pub/sub channel for services/api/main.py to relay to
    WebSocket clients in real time.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import db  # noqa: E402
from common.config import Camera, Settings, cameras_by_id, load_cameras, load_settings  # noqa: E402
from common.redis_client import get_redis  # noqa: E402

CONSUMER_GROUP = "event_manager"
CONSUMER_NAME = "event_manager-1"


def _log(msg: str) -> None:
    print(f"[events] {msg}", flush=True)


@dataclass
class TrackState:
    streak: int = 0
    open: bool = False
    event_id: str = ""
    event_start_ts: float = 0.0
    last_ts: float = 0.0
    best_score: float = 0.0
    best_jpg: bytes = b""


def _ensure_group(r, stream: str, group: str) -> None:
    try:
        r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
    except Exception as exc:  # noqa: BLE001
        if "BUSYGROUP" not in str(exc):
            raise


def _reclaim_pending(r, stream: str, group: str, consumer: str) -> list:
    """On startup, re-deliver any entries that were XREADGROUP'd to a
    previous run of this consumer but never XACK'd (e.g. the process
    crashed between reading a batch and finishing/acking it). Without
    this, XREADGROUP's ">" id only ever returns *new* messages, so those
    stranded entries would sit in the group's pending list forever and
    never reach _finalize_event / get acked.

    Returns entries in the same [(entry_id, fields), ...] shape XREADGROUP
    would, ready to be folded into the normal processing loop.
    """
    reclaimed: list = []
    start_id = "0-0"
    while True:
        pending = r.xpending_range(stream, group, min=start_id, max="+", count=500)
        if not pending:
            break
        ids = [p["message_id"] for p in pending]
        claimed = r.xclaim(stream, group, consumer, min_idle_time=0, message_ids=ids)
        reclaimed.extend(claimed)
        if len(pending) < 500:
            break
        # advance past the last id we just paged through
        last_id = pending[-1]["message_id"]
        start_id = _next_stream_id(last_id)
    if reclaimed:
        _log(f"reclaimed {len(reclaimed)} pending entry(ies) left over from a previous run")
    return reclaimed


def _next_stream_id(entry_id: bytes | str) -> str:
    """Smallest stream id strictly greater than entry_id, for paging XPENDING."""
    if isinstance(entry_id, bytes):
        entry_id = entry_id.decode()
    ms, seq = entry_id.split("-")
    return f"{ms}-{int(seq) + 1}"


def _parse_segment_ts(path: Path) -> datetime | None:
    # segment files are named %Y%m%d_%H%M%S.mp4 by ingest.py
    try:
        return datetime.strptime(path.stem, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _stitch_clip(camera: Camera, start: datetime, end: datetime, settings: Settings, event_id: str) -> str | None:
    seg_dir = settings.storage.segments_path() / camera.id
    if not seg_dir.exists():
        return None

    segment_duration = timedelta(seconds=settings.storage.segment_duration_seconds)
    matching = []
    for f in sorted(seg_dir.glob("*.mp4")):
        seg_start = _parse_segment_ts(f)
        if seg_start is None:
            continue
        seg_end = seg_start + segment_duration
        if seg_end >= start and seg_start <= end:
            matching.append(f)

    if not matching:
        _log(f"no recorded segments found for {camera.id} in [{start}, {end}], skipping clip")
        return None

    clips_dir = settings.storage.clips_path() / camera.id
    clips_dir.mkdir(parents=True, exist_ok=True)
    out_path = clips_dir / f"{event_id}.mp4"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as list_file:
        for f in matching:
            list_file.write(f"file '{f.resolve()}'\n")
        list_path = list_file.name

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        _log(f"ffmpeg concat failed for {camera.id}: {exc}")
        return None
    finally:
        Path(list_path).unlink(missing_ok=True)

    relative = out_path.relative_to(settings.storage.clips_path().parent)
    return str(relative)


def _save_thumbnail(jpg_bytes: bytes, settings: Settings, event_id: str) -> str | None:
    if not jpg_bytes:
        return None
    thumb_path = settings.storage.thumbnails_path() / f"{event_id}.jpg"
    thumb_path.write_bytes(jpg_bytes)
    return str(thumb_path.relative_to(settings.storage.thumbnails_path().parent))


def _open_event(camera: Camera, label: str, state: TrackState, settings: Settings, r,
                 score: float, jpg: bytes) -> None:
    """Called the instant an event opens (debounce_frames consecutive
    detections seen). Immediately writes a 'open'-status row to Postgres
    and publishes it, so the frontend shows an alert right away instead of
    waiting for the subject to leave frame and the event to close — for a
    security camera, "someone is in frame right now" is the useful signal,
    not just the after-the-fact summary."""
    start_dt = datetime.fromtimestamp(state.event_start_ts - settings.events.pre_roll_seconds, tz=timezone.utc)
    thumb_path = _save_thumbnail(jpg, settings, state.event_id)

    conn = db.get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (id, camera_id, camera_name, label, score, start_time, end_time,
                                         status, clip_path, thumbnail_path)
                    VALUES (%s, %s, %s, %s, %s, %s, NULL, 'open', NULL, %s)
                    """,
                    (state.event_id, camera.id, camera.name, label, score, start_dt, thumb_path),
                )
    finally:
        db.put_conn(conn)

    payload = {
        "type": "opened",
        "id": state.event_id,
        "camera_id": camera.id,
        "camera_name": camera.name,
        "label": label,
        "score": score,
        "start_time": start_dt.isoformat(),
        "end_time": None,
        "status": "open",
        "clip_path": None,
        "thumbnail_path": thumb_path,
    }
    r.publish(settings.redis.events_channel, json.dumps(payload))
    _log(f"event opened: {camera.name} / {label}")


def _discard_event(state: TrackState, camera_id: str, label: str, duration: float, settings: Settings, r) -> None:
    """A track that opened but never reached min_event_duration_seconds
    (a brief flicker). It already has an 'open' row from _open_event —
    delete it and tell the frontend to drop the ongoing card rather than
    leaving a stub event lingering forever as 'open'."""
    _log(f"discarding sub-threshold-duration event {camera_id}/{label} ({duration:.1f}s)")
    conn = db.get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM events WHERE id = %s", (state.event_id,))
    finally:
        db.put_conn(conn)
    r.publish(settings.redis.events_channel, json.dumps({"type": "discarded", "id": state.event_id}))


def _finalize_event(camera: Camera, label: str, state: TrackState, settings: Settings, r) -> None:
    duration = state.last_ts - state.event_start_ts
    if duration < settings.events.min_event_duration_seconds:
        _discard_event(state, camera.id, label, duration, settings, r)
        return

    end_dt = datetime.fromtimestamp(state.last_ts + settings.events.post_roll_seconds, tz=timezone.utc)
    start_dt = datetime.fromtimestamp(state.event_start_ts - settings.events.pre_roll_seconds, tz=timezone.utc)

    clip_path = _stitch_clip(camera, start_dt, end_dt, settings, state.event_id)
    thumb_path = _save_thumbnail(state.best_jpg, settings, state.event_id)

    conn = db.get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE events
                    SET score = %s, end_time = %s, status = 'closed', clip_path = %s, thumbnail_path = %s
                    WHERE id = %s
                    """,
                    (state.best_score, end_dt, clip_path, thumb_path, state.event_id),
                )
    finally:
        db.put_conn(conn)

    payload = {
        "type": "closed",
        "id": state.event_id,
        "camera_id": camera.id,
        "camera_name": camera.name,
        "label": label,
        "score": state.best_score,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "status": "closed",
        "clip_path": clip_path,
        "thumbnail_path": thumb_path,
    }
    r.publish(settings.redis.events_channel, json.dumps(payload))
    _log(f"event closed: {camera.name} / {label} (score={state.best_score:.2f}, "
         f"duration={duration:.1f}s, clip={clip_path})")


def _process_entry(entry_id, fields, states: dict, settings: Settings, r, stream: str, cameras: dict) -> None:
    try:
        camera_id = fields[b"camera_id"].decode()
        label = fields[b"label"].decode()
        score = float(fields[b"score"].decode())
        ts = float(fields[b"ts"].decode())
        jpg = fields.get(b"jpg", b"")

        key = (camera_id, label)
        state = states.setdefault(key, TrackState())

        # Detections that arrive way outside the cooldown window
        # start a fresh streak rather than extending a stale one.
        if state.streak > 0 and (ts - state.last_ts) > settings.events.cooldown_seconds:
            state.streak = 0
            state.open = False

        state.streak += 1
        state.last_ts = ts
        if score > state.best_score:
            state.best_score = score
            state.best_jpg = jpg

        if not state.open and state.streak >= settings.events.debounce_frames:
            state.open = True
            state.event_id = str(uuid.uuid4())
            state.event_start_ts = ts - (settings.events.debounce_frames - 1) * (
                1.0 / max(settings.detection.frame_sample_fps, 0.1)
            )
            camera = cameras.get(camera_id)
            if camera is not None:
                _open_event(camera, label, state, settings, r, score, jpg)
            else:
                _log(f"event opened: {camera_id} / {label} (camera not in config, skipping alert)")
    except Exception as exc:  # noqa: BLE001
        _log(f"error processing detection entry: {exc}")
    finally:
        r.xack(stream, CONSUMER_GROUP, entry_id)


def main() -> None:
    settings = load_settings()
    cameras = cameras_by_id(load_cameras())

    db.ensure_schema(settings.postgres)
    db.init_pool(settings.postgres)

    r = get_redis(settings.redis)
    stream = settings.redis.detections_stream
    _ensure_group(r, stream, CONSUMER_GROUP)

    states: dict[tuple[str, str], TrackState] = {}

    # Recover anything left pending (delivered but never acked) from a
    # previous run of this consumer before joining the live stream, so a
    # crash mid-batch doesn't strand detections forever.
    for entry_id, fields in _reclaim_pending(r, stream, CONSUMER_GROUP, CONSUMER_NAME):
        _process_entry(entry_id, fields, states, settings, r, stream, cameras)

    _log("event manager started, watching detections stream")

    while True:
        try:
            resp = r.xreadgroup(CONSUMER_GROUP, CONSUMER_NAME, {stream: ">"}, count=20, block=2000)
        except Exception as exc:  # noqa: BLE001
            _log(f"xreadgroup failed: {exc}")
            time.sleep(1)
            resp = None

        now = time.time()

        if resp:
            for _stream_key, entries in resp:
                for entry_id, fields in entries:
                    _process_entry(entry_id, fields, states, settings, r, stream, cameras)

        # Sweep for open tracks that have gone quiet long enough to close,
        # independent of whether this iteration received new messages.
        for key, state in list(states.items()):
            if state.open and (now - state.last_ts) > settings.events.cooldown_seconds:
                camera_id, label = key
                camera = cameras.get(camera_id)
                if camera is not None:
                    _finalize_event(camera, label, state, settings, r)
                del states[key]


if __name__ == "__main__":
    main()
