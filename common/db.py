"""Postgres schema + a small shared connection pool.

Both `services/events/event_manager.py` (writer) and `services/api/main.py`
(reader) import this so the schema is defined in exactly one place.
"""
from __future__ import annotations

import time

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from common.config import PostgresSettings

_pool: pg_pool.SimpleConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id TEXT NOT NULL,
    camera_name TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'closed',
    clip_path TEXT,
    thumbnail_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_start_time ON events (start_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events (camera_id);
CREATE INDEX IF NOT EXISTS idx_events_label ON events (label);
"""

# Applied every startup, right after SCHEMA, so an existing install (created
# before the live-alert-on-open feature) picks up the new nullable
# end_time / status column without a manual migration step. Both
# statements are idempotent — safe to run against an already-migrated table.
MIGRATIONS = """
ALTER TABLE events ALTER COLUMN end_time DROP NOT NULL;
ALTER TABLE events ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'closed';
"""


def _raw_connect(pg: PostgresSettings):
    if not pg.password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. Copy "
            ".env.example to .env, set a strong password there, and make "
            "sure it's loaded (docker compose reads ./.env automatically)."
        )
    return psycopg2.connect(
        host=pg.host, port=pg.port, dbname=pg.db, user=pg.user, password=pg.password,
    )


def wait_for_postgres(pg: PostgresSettings, retries: int = 30, delay: float = 2.0) -> None:
    """Postgres takes a few seconds to accept connections after container start;
    services that depend on it retry instead of crash-looping."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = _raw_connect(pg)
            conn.close()
            return
        except psycopg2.OperationalError as exc:
            last_err = exc
            print(f"[db] postgres not ready yet (attempt {attempt}/{retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to postgres after {retries} attempts") from last_err


def ensure_schema(pg: PostgresSettings) -> None:
    wait_for_postgres(pg)
    conn = _raw_connect(pg)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")  # gen_random_uuid()
            cur.execute(SCHEMA)
            cur.execute(MIGRATIONS)
    finally:
        conn.close()


def init_pool(pg: PostgresSettings, minconn: int = 1, maxconn: int = 10) -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(
            minconn, maxconn,
            host=pg.host, port=pg.port, dbname=pg.db, user=pg.user, password=pg.password,
        )
    return _pool


def get_conn():
    assert _pool is not None, "init_pool() must be called before get_conn()"
    return _pool.getconn()


def put_conn(conn) -> None:
    assert _pool is not None
    _pool.putconn(conn)
