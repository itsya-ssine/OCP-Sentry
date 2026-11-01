"""Shared config loader used by every service.

Instead of each service parsing config/*.yaml on its own, everything
routes through here: `load_cameras()` / `load_settings()` return typed
dataclasses, so a typo in settings.yaml fails fast at startup rather than
as a KeyError three services deep.

Environment variables override the connection-related settings.yaml
values (host/port/credentials for redis + postgres) so the same
config/settings.yaml can be used both on bare metal and inside
docker-compose, where service hostnames differ from localhost.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# common/../ -> project root. Works both bare-metal (repo checkout) and in
# containers, since every Dockerfile copies the project onto /app with the
# same relative layout and sets NVR_ROOT=/app.
PROJECT_ROOT = Path(os.environ.get("NVR_ROOT", Path(__file__).resolve().parent.parent))

CAMERAS_PATH = PROJECT_ROOT / "config" / "cameras.yaml"
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass
class Camera:
    id: str
    name: str
    detect_url: str          # low-res substream used for inference sampling
    record_url: str          # full-res stream used for continuous segment recording
    enabled: bool = True
    record_enabled: bool = True   # set False for sources that can't sustain two
                                   # concurrent opens (e.g. a single webcam device) —
                                   # detection still runs, events just won't have clips


@dataclass
class DetectionSettings:
    frame_sample_fps: float
    confidence_threshold: float
    per_class_thresholds: dict
    classes_of_interest: list


@dataclass
class EventSettings:
    debounce_frames: int
    cooldown_seconds: float
    min_event_duration_seconds: float
    pre_roll_seconds: float
    post_roll_seconds: float


@dataclass
class StorageSettings:
    segments_dir: str
    clips_dir: str
    thumbnails_dir: str
    segment_duration_seconds: int
    segment_retention_hours: int
    clip_retention_days: int

    def segments_path(self) -> Path:
        p = PROJECT_ROOT / self.segments_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def clips_path(self) -> Path:
        p = PROJECT_ROOT / self.clips_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def thumbnails_path(self) -> Path:
        p = PROJECT_ROOT / self.thumbnails_dir
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class RedisSettings:
    host: str
    port: int
    frames_stream_maxlen: int
    detections_stream: str
    events_channel: str


@dataclass
class PostgresSettings:
    host: str
    port: int
    db: str
    user: str
    password: str


@dataclass
class ModelSettings:
    path: str
    labelmap_path: str
    input_size: int
    batch_size: int

    def model_path(self) -> Path:
        return PROJECT_ROOT / self.path

    def labelmap_full_path(self) -> Path:
        return PROJECT_ROOT / self.labelmap_path


@dataclass
class ApiSettings:
    host: str
    port: int
    api_key: str
    cors_origins: list


@dataclass
class Settings:
    detection: DetectionSettings
    events: EventSettings
    storage: StorageSettings
    redis: RedisSettings
    postgres: PostgresSettings
    model: ModelSettings
    api: ApiSettings


def load_cameras(path: Optional[Path] = None) -> list[Camera]:
    path = path or CAMERAS_PATH
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cameras = []
    for entry in raw.get("cameras", []):
        cameras.append(Camera(
            id=str(entry["id"]),
            name=entry.get("name", entry["id"]),
            detect_url=entry["detect_url"],
            record_url=entry.get("record_url", entry["detect_url"]),
            enabled=entry.get("enabled", True),
            record_enabled=entry.get("record_enabled", True),
        ))
    return cameras


def load_settings(path: Optional[Path] = None) -> Settings:
    path = path or SETTINGS_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    d = raw["detection"]
    detection = DetectionSettings(
        frame_sample_fps=d["frame_sample_fps"],
        confidence_threshold=d["confidence_threshold"],
        per_class_thresholds=d.get("per_class_thresholds", {}),
        classes_of_interest=d.get("classes_of_interest", []),
    )

    e = raw["events"]
    events = EventSettings(
        debounce_frames=e["debounce_frames"],
        cooldown_seconds=e["cooldown_seconds"],
        min_event_duration_seconds=e["min_event_duration_seconds"],
        pre_roll_seconds=e["pre_roll_seconds"],
        post_roll_seconds=e["post_roll_seconds"],
    )

    s = raw["storage"]
    storage = StorageSettings(
        segments_dir=s["segments_dir"],
        clips_dir=s["clips_dir"],
        thumbnails_dir=s["thumbnails_dir"],
        segment_duration_seconds=s["segment_duration_seconds"],
        segment_retention_hours=s["segment_retention_hours"],
        clip_retention_days=s["clip_retention_days"],
    )

    r = raw["redis"]
    redis_settings = RedisSettings(
        host=os.environ.get("REDIS_HOST", r["host"]),
        port=int(os.environ.get("REDIS_PORT", r["port"])),
        frames_stream_maxlen=r["frames_stream_maxlen"],
        detections_stream=r["detections_stream"],
        events_channel=r["events_channel"],
    )

    p = raw["postgres"]
    # Password is deliberately NOT read from settings.yaml (a file that
    # tends to get committed to source control) — it must come from the
    # POSTGRES_PASSWORD environment variable (see .env.example). We don't
    # raise here if it's missing: load_settings() is called by every
    # service, including ingest/inference, which never touch Postgres at
    # all. Instead, common/db.py raises a clear error the moment something
    # that actually needs a DB connection (api, events) tries to use it.
    postgres = PostgresSettings(
        host=os.environ.get("POSTGRES_HOST", p["host"]),
        port=int(os.environ.get("POSTGRES_PORT", p["port"])),
        db=os.environ.get("POSTGRES_DB", p["db"]),
        user=os.environ.get("POSTGRES_USER", p["user"]),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )

    m = raw["model"]
    model = ModelSettings(
        path=m["path"],
        labelmap_path=m["labelmap_path"],
        input_size=m["input_size"],
        batch_size=m["batch_size"],
    )

    a = raw["api"]
    # Same reasoning as postgres password: don't raise here (ingest,
    # inference, and events never touch this), just pass through whatever
    # is set. services/api/main.py — the only thing that actually checks
    # requests against this key — validates it's present and long enough
    # at its own startup.
    api_key = os.environ.get("API_KEY", "")
    cors_env = os.environ.get("CORS_ORIGINS")
    if cors_env is not None:
        cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
    else:
        cors_origins = a.get("cors_origins", [])
    api = ApiSettings(host=a["host"], port=a["port"], api_key=api_key, cors_origins=cors_origins)

    return Settings(detection=detection, events=events, storage=storage,
                     redis=redis_settings, postgres=postgres, model=model, api=api)


def load_labelmap(settings: Settings) -> list[str]:
    path = settings.model.labelmap_full_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Labelmap not found at {path}. Run the training/ pipeline "
            f"(training/export.py) first to produce config/model/custom_model.onnx "
            f"and config/model/custom_labelmap.txt."
        )
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def cameras_by_id(cameras: list[Camera]) -> dict[str, Camera]:
    return {c.id: c for c in cameras}
