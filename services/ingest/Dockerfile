# Ingest service: per-camera ffmpeg detect-frame + segment-recording pipelines.
FROM nvr-base:latest

WORKDIR /app
COPY config/ /app/config/
COPY services/ingest/ingest.py /app/services/ingest/ingest.py
COPY services/ingest/mjpeg_split.c /app/services/ingest/mjpeg_split.c

# Compile the MJPEG frame-splitter (see mjpeg_split.c) that sits between
# ffmpeg and ingest.py in the detect-stream pipeline, replacing what used
# to be a pure-Python byte-scan on every single sampled frame. gcc is only
# needed at build time — the compiled binary has no runtime dependency on
# it, so it's removed again in the same layer to keep the image slim.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && cc -O2 -o /app/services/ingest/mjpeg_split /app/services/ingest/mjpeg_split.c \
    && apt-get purge -y --auto-remove gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN chown -R nvr:nvr /app/config /app/services

# RTSP cameras (the normal production case) need no special device access,
# but the webcam test path (config/cameras.webcam.example.yaml +
# docker-compose's `devices: - /dev/video0` mapping) opens a host device
# node, which is normally root:video 0660. Add nvr to a video group so it
# can open that device without running as root. 44 is Debian/Ubuntu's
# default video GID; if `docker compose up` fails with "Permission denied"
# opening /dev/video0, check `stat -c '%g' /dev/video0` on the host and
# rebuild with `--build-arg VIDEO_GID=<that number>`.
ARG VIDEO_GID=44
RUN groupadd -g "${VIDEO_GID}" video_host 2>/dev/null || true \
    && usermod -a -G "${VIDEO_GID}" nvr

USER nvr

CMD ["python", "services/ingest/ingest.py"]
