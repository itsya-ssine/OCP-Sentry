"""FastAPI REST + WebSocket alert API.

- GET  /api/cameras          camera list from config/cameras.yaml
- GET  /api/events           paginated/filterable event history (Postgres)
- GET  /api/events/{id}      single event
- GET  /api/health           liveness/readiness probe
- WS   /ws/alerts             live feed: every new event finalized by
                              services/events/event_manager.py is relayed
                              here in real time via Redis pub/sub
- /clips/*, /thumbnails/*    static file serving for the frontend player
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import hmac

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import db  # noqa: E402
from common.config import load_cameras, load_settings  # noqa: E402

settings = load_settings()
cameras = load_cameras()

# Validated here (not in common/config.py) because load_settings() is
# shared by every service — ingest/inference never check requests against
# this key, only this service does, so only this service should refuse to
# start over it.
if not settings.api.api_key:
    raise RuntimeError(
        "API_KEY environment variable is not set. Copy .env.example to "
        ".env and set a long random API_KEY there — every REST/WS "
        "request must present it (see README security section)."
    )
if len(settings.api.api_key) < 16:
    raise RuntimeError(
        "API_KEY is too short (must be at least 16 characters). Generate "
        "one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )

app = FastAPI(title="NVR Alert API")
app.add_middleware(
    CORSMiddleware,
    # No "*" here: this API serves live camera footage and event history,
    # so wildcarding origins/methods/headers is the wrong default for
    # anything privacy-sensitive. Configure real origins via
    # settings.api.cors_origins / the CORS_ORIGINS env var — left empty by
    # default since the frontend is same-origin behind nginx (nginx.conf)
    # and doesn't need CORS at all.
    allow_origins=settings.api.cors_origins,
    allow_methods=["GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def _check_api_key(candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    # constant-time compare so response timing can't be used to guess the
    # key one byte at a time
    return hmac.compare_digest(candidate, settings.api.api_key)


# Applied as middleware (rather than a per-route Depends) so it also covers
# the /clips and /thumbnails StaticFiles mounts below — those serve actual
# camera footage and shouldn't be reachable by anyone who can merely reach
# the API's port, and StaticFiles has no dependency-injection hook of its
# own. /api/health stays open so Docker's healthcheck (and load balancers)
# don't need a credential just to ask "are you up".
_PUBLIC_PATHS = {"/api/health"}


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path not in _PUBLIC_PATHS:
        candidate = request.headers.get("x-api-key") or request.query_params.get("api_key")
        if not _check_api_key(candidate):
            return JSONResponse(status_code=HTTP_401_UNAUTHORIZED, content={"detail": "missing or invalid API key"})
    return await call_next(request)


class EventOut(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    label: str
    score: float
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "closed"
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    created_at: datetime


class CameraOut(BaseModel):
    id: str
    name: str
    enabled: bool


@app.on_event("startup")
async def startup() -> None:
    db.ensure_schema(settings.postgres)
    db.init_pool(settings.postgres)

    app.state.redis = aioredis.Redis(host=settings.redis.host, port=settings.redis.port, decode_responses=True)
    app.state.ws_clients: set[WebSocket] = set()
    app.state.pubsub_task = asyncio.create_task(_relay_events())

    # Static file mounts for clip/thumbnail playback. Directories are
    # created up front so StaticFiles doesn't fail on a fresh checkout
    # before any events exist yet.
    settings.storage.clips_path()
    settings.storage.thumbnails_path()
    app.mount("/clips", StaticFiles(directory=str(settings.storage.clips_path())), name="clips")
    app.mount("/thumbnails", StaticFiles(directory=str(settings.storage.thumbnails_path())), name="thumbnails")


@app.on_event("shutdown")
async def shutdown() -> None:
    app.state.pubsub_task.cancel()
    await app.state.redis.close()


async def _relay_events() -> None:
    """Subscribes once to the events pub/sub channel and fans each message
    out to every currently-connected WebSocket client."""
    pubsub = app.state.redis.pubsub()
    await pubsub.subscribe(settings.redis.events_channel)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            dead = set()
            for ws in app.state.ws_clients:
                try:
                    await ws.send_text(message["data"])
                except Exception:  # noqa: BLE001
                    dead.add(ws)
            app.state.ws_clients -= dead
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe(settings.redis.events_channel)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cameras", response_model=list[CameraOut])
async def list_cameras():
    return [CameraOut(id=c.id, name=c.name, enabled=c.enabled) for c in cameras]


def _query_events(camera_id: Optional[str], label: Optional[str], since: Optional[datetime],
                   limit: int, offset: int) -> list[tuple]:
    """Blocking psycopg2 work for list_events(). Run via asyncio.to_thread
    so a slow query doesn't stall the event loop (and every other request
    / the WebSocket relay) — psycopg2 has no async API of its own."""
    clauses = []
    params: list = []
    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if label:
        clauses.append("label = %s")
        params.append(label)
    if since:
        clauses.append("start_time >= %s")
        params.append(since)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    query = f"""
        SELECT id, camera_id, camera_name, label, score, start_time, end_time,
               clip_path, thumbnail_path, status, created_at
        FROM events
        {where}
        ORDER BY start_time DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        db.put_conn(conn)


def _query_event(event_id: UUID) -> Optional[tuple]:
    """Blocking psycopg2 work for get_event() — see _query_events() docstring."""
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, camera_id, camera_name, label, score, start_time, end_time,
                       clip_path, thumbnail_path, status, created_at
                FROM events WHERE id = %s
                """,
                (str(event_id),),
            )
            return cur.fetchone()
    finally:
        db.put_conn(conn)


@app.get("/api/events", response_model=list[EventOut])
async def list_events(
    camera_id: Optional[str] = None,
    label: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    rows = await asyncio.to_thread(_query_events, camera_id, label, since, limit, offset)
    return [
        EventOut(
            id=str(row[0]), camera_id=row[1], camera_name=row[2], label=row[3], score=row[4],
            start_time=row[5], end_time=row[6], clip_path=row[7], thumbnail_path=row[8],
            status=row[9], created_at=row[10],
        )
        for row in rows
    ]


@app.get("/api/events/{event_id}", response_model=EventOut)
async def get_event(event_id: UUID):
    row = await asyncio.to_thread(_query_event, event_id)

    if row is None:
        raise HTTPException(status_code=404, detail="event not found")

    return EventOut(
        id=str(row[0]), camera_id=row[1], camera_name=row[2], label=row[3], score=row[4],
        start_time=row[5], end_time=row[6], clip_path=row[7], thumbnail_path=row[8],
        status=row[9], created_at=row[10],
    )


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    # api_key_middleware above only wraps "http"-scope requests — Starlette
    # never runs HTTP middleware for the websocket handshake — so the same
    # check is repeated explicitly here. Browsers can't set custom headers
    # on a WS handshake, hence the query-param form.
    candidate = websocket.query_params.get("api_key")
    if not _check_api_key(candidate):
        await websocket.close(code=4401)
        return

    await websocket.accept()
    app.state.ws_clients.add(websocket)
    try:
        while True:
            # Client -> server messages aren't used; just keep the
            # connection alive and detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        app.state.ws_clients.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
