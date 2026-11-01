# Event manager: debounce state machine, clip stitching, Postgres writes.
FROM nvr-base:latest

WORKDIR /app
COPY config/ /app/config/
COPY services/events/event_manager.py /app/services/events/event_manager.py
RUN chown -R nvr:nvr /app/config /app/services

USER nvr

CMD ["python", "services/events/event_manager.py"]
