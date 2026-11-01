"""Tiny helper so every service builds its Redis connection the same way.

decode_responses is deliberately False: the frame/detection streams carry
raw JPEG bytes alongside text fields, so callers decode the text fields
themselves (see services/inference/inference_worker.py and
services/events/event_manager.py for the pattern) rather than fighting
redis-py's blanket decoding.
"""
from __future__ import annotations

import redis

from common.config import RedisSettings


def get_redis(settings: RedisSettings) -> redis.Redis:
    return redis.Redis(host=settings.host, port=settings.port, decode_responses=False)


def frames_stream_key(camera_id: str) -> str:
    return f"frames:{camera_id}"
