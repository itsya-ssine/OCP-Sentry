"""Inference worker: consumes sampled frames from Redis and runs the
fine-tuned YOLO ONNX model (person/fire/smoke), pushing detections above
threshold onto a shared Redis stream for services/events/event_manager.py.

"Batched" here means each loop iteration reads a batch of pending frames
across all camera streams in one XREAD call, not that they're packed into
a single ONNX forward pass — the model was exported with NMS baked in
(train_config.yaml -> export.nms: true), which makes per-image output
length ragged, so frames are run through the session one at a time in the
batch loop. This keeps the postprocessing simple and correct; swapping in
true tensor batching later is a drop-in change to `_run_batch`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.config import Settings, load_cameras, load_labelmap, load_settings  # noqa: E402
from common.redis_client import frames_stream_key, get_redis  # noqa: E402

THUMBNAIL_MAX_WIDTH = 480


def _log(msg: str) -> None:
    print(f"[inference] {msg}", flush=True)


def letterbox(img: np.ndarray, new_size: int, color=(114, 114, 114)):
    """Resize+pad to a square input while preserving aspect ratio (standard
    YOLO preprocessing). Returns the padded image plus the scale factor and
    (left, top) padding needed to map boxes back to original coordinates."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def preprocess(img: np.ndarray, input_size: int):
    padded, r, left, top = letterbox(img, input_size)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # HWC -> CHW -> NCHW
    return np.ascontiguousarray(blob), r, left, top


def postprocess(output: np.ndarray, r: float, left: int, top: int, orig_w: int, orig_h: int,
                 labelmap: list[str], settings: Settings):
    """`output` is the (N, 6) [x1,y1,x2,y2,conf,cls] array for one image,
    already NMS'd by the exported model. Zero-confidence rows are export
    padding and are skipped. Boxes are un-letterboxed back to original
    pixel coordinates and clipped to the frame."""
    detections = []
    for row in output:
        x1, y1, x2, y2, conf, cls = row[:6]
        if conf <= 0:
            continue
        cls_idx = int(cls)
        if cls_idx < 0 or cls_idx >= len(labelmap):
            continue
        label = labelmap[cls_idx]
        threshold = settings.detection.per_class_thresholds.get(label, settings.detection.confidence_threshold)
        if conf < threshold:
            continue
        if settings.detection.classes_of_interest and label not in settings.detection.classes_of_interest:
            continue

        x1 = (x1 - left) / r
        x2 = (x2 - left) / r
        y1 = (y1 - top) / r
        y2 = (y2 - top) / r
        x1 = max(0.0, min(x1, orig_w))
        x2 = max(0.0, min(x2, orig_w))
        y1 = max(0.0, min(y1, orig_h))
        y2 = max(0.0, min(y2, orig_h))

        detections.append({"label": label, "score": float(conf), "box": (x1, y1, x2, y2)})
    return detections


def make_thumbnail(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    if w > THUMBNAIL_MAX_WIDTH:
        scale = THUMBNAIL_MAX_WIDTH / w
        img = cv2.resize(img, (THUMBNAIL_MAX_WIDTH, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("failed to encode thumbnail")
    return buf.tobytes()


def load_session(settings: Settings) -> ort.InferenceSession:
    model_path = settings.model.model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run the training/ pipeline "
            f"(training/train.py then training/export.py) first — export.py "
            f"copies the ONNX model into config/model/ automatically."
        )
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(model_path), providers=providers)


def _select_shard(cameras: list, settings: Settings) -> list:
    """Splits `cameras` across N horizontally-scaled replicas of this
    service, selected by SHARD_INDEX / SHARD_COUNT env vars (both default
    to a single shard, i.e. today's all-in-one-process behavior).

    One inference process running all 30 cameras sequentially is a real
    CPU bottleneck at that camera count. To scale out, run several
    replicas of the inference service (e.g. in docker-compose, `deploy:
    replicas: 3` plus SHARD_COUNT=3, SHARD_INDEX=0/1/2 per replica) and
    each one will pick up a disjoint ~1/N of the cameras by hashing the
    camera id — no coordination between replicas needed since the split
    is deterministic.
    """
    import hashlib
    import os

    shard_count = max(1, int(os.environ.get("SHARD_COUNT", "1")))
    shard_index = int(os.environ.get("SHARD_INDEX", "0"))
    if shard_count == 1:
        return cameras
    if not (0 <= shard_index < shard_count):
        raise ValueError(f"SHARD_INDEX={shard_index} out of range for SHARD_COUNT={shard_count}")

    def shard_of(camera_id: str) -> int:
        digest = hashlib.sha1(camera_id.encode()).hexdigest()
        return int(digest, 16) % shard_count

    return [c for c in cameras if shard_of(c.id) == shard_index]


def main() -> None:
    settings = load_settings()
    all_cameras = [c for c in load_cameras() if c.enabled]
    cameras = _select_shard(all_cameras, settings)
    labelmap = load_labelmap(settings)

    if not cameras:
        _log("no cameras assigned to this shard (check SHARD_INDEX/SHARD_COUNT), exiting")
        return

    session = load_session(settings)
    input_name = session.get_inputs()[0].name

    r = get_redis(settings.redis)
    stream_keys = {frames_stream_key(c.id): "$" for c in cameras}  # "$" = only new frames from now on

    _log(f"watching {len(cameras)}/{len(all_cameras)} camera stream(s), classes={labelmap}")

    while True:
        try:
            resp = r.xread(stream_keys, count=settings.model.batch_size, block=1000)
        except Exception as exc:  # noqa: BLE001
            _log(f"redis xread failed: {exc}")
            time.sleep(1)
            continue

        if not resp:
            continue

        for stream_key_bytes, entries in resp:
            stream_key = stream_key_bytes.decode()
            for entry_id, fields in entries:
                stream_keys[stream_key] = entry_id
                try:
                    camera_id = fields[b"camera_id"].decode()
                    ts = fields[b"ts"].decode()
                    jpg_bytes = fields[b"jpg"]

                    img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    h, w = img.shape[:2]

                    blob, scale, left, top = preprocess(img, settings.model.input_size)
                    raw_out = session.run(None, {input_name: blob})[0]
                    out = raw_out[0] if raw_out.ndim == 3 else raw_out

                    detections = postprocess(out, scale, left, top, w, h, labelmap, settings)
                    if not detections:
                        continue

                    thumb = make_thumbnail(img)
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        r.xadd(
                            settings.redis.detections_stream,
                            {
                                "camera_id": camera_id.encode(),
                                "ts": ts.encode(),
                                "label": det["label"].encode(),
                                "score": str(det["score"]).encode(),
                                "x1": str(x1).encode(),
                                "y1": str(y1).encode(),
                                "x2": str(x2).encode(),
                                "y2": str(y2).encode(),
                                "jpg": thumb,
                            },
                            maxlen=10000,
                            approximate=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    _log(f"error processing frame from {stream_key}: {exc}")


if __name__ == "__main__":
    main()
